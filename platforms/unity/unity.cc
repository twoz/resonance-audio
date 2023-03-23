/*
Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS-IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "platforms/unity/unity.h"

#include <algorithm>
#include <memory>
#include <future>

#include "base/audio_buffer.h"
#include "base/constants_and_types.h"
#include "base/logging.h"
#include "base/misc_math.h"
#include "graph/resonance_audio_api_impl.h"
#include "platforms/common/room_effects_utils.h"
#include "dsp/fir_filter.h"
#include "utils/planar_interleaved_conversion.h"

#if !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
#include "utils/ogg_vorbis_recorder.h"
#endif  // !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))

#include "NeuralAcoustics.h"

namespace vraudio {
namespace unity {

namespace {

// Output channels must be stereo for the ResonanceAudio system to run properly.
const size_t kNumOutputChannels = 2;

#if !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
// Maximum number of buffers allowed to record a soundfield, which is set to ~5
// minutes (depending on the sampling rate and the number of frames per buffer).
const size_t kMaxNumRecordBuffers = 15000;

// Record compression quality.
const float kRecordQuality = 1.0f;
#endif  // !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))

// Stores the necessary components for the ResonanceAudio system. Methods called
// from the native implementation below must check the validity of this
// instance.
struct ResonanceAudioSystem {
  ResonanceAudioSystem(int sample_rate, size_t num_channels,
                       size_t frames_per_buffer)
      : api(CreateResonanceAudioApi(num_channels, frames_per_buffer,
                                    sample_rate)) {
#if !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
    is_recording_soundfield = false;
    soundfield_recorder.reset(
        new OggVorbisRecorder(sample_rate, kNumFirstOrderAmbisonicChannels,
                              frames_per_buffer, kMaxNumRecordBuffers));
#endif  // !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
  }

  // ResonanceAudio API instance to communicate with the internal system.
  std::unique_ptr<ResonanceAudioApi> api;

  // Default room properties, which effectively disable the room effects.
  ReflectionProperties null_reflection_properties;
  ReverbProperties null_reverb_properties;

#if !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
  // Denotes whether the soundfield recording is currently in progress.
  bool is_recording_soundfield;

  // First-order ambisonic soundfield recorder.
  std::unique_ptr<OggVorbisRecorder> soundfield_recorder;
#endif  // !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
};

class NeuralAcousticsRenderer {
  public:
    NeuralAcousticsRenderer(int sampleRate)
      : m_net(std::make_shared<sdk::NeuralAcoustics>(
          sdk::createModelApartment1("/home/twozn/dev/spatial-audio-sdk/auxiliary/"))) 
      {
        m_irResampler.SetRateAndNumChannels(22500, sampleRate, 2);
      }

    void process(AudioBuffer& inOut) {
      const auto length = inOut.num_frames();
      CHECK_EQ(inOut.num_channels(), 2);
      // TODO Update only when transform changes?
      if (updateImpulseResponses() || length != m_fir.samplesPerBuffer) {
        m_fir.left = std::make_unique<vraudio::FirFilter>(m_ir[0], length);
        m_fir.right = std::make_unique<vraudio::FirFilter>(m_ir[1], length);
        m_fir.samplesPerBuffer = length;
      }
      // TODO don't allocate
      AudioBuffer inCopy{2, length};
      std::copy(inOut[0].begin(), inOut[0].end(), inCopy[0].begin());
      std::copy(inOut[1].begin(), inOut[1].end(), inCopy[1].begin());
      inOut.Clear();
      m_fir.left->Process(inCopy[0], &inOut[0]);
      m_fir.right->Process(inCopy[1], &inOut[1]);
    }

    sdk::NeuralAcoustics& net() { return *m_net; }

  private:
    bool updateImpulseResponses() {
      constexpr auto kTimeout = std::chrono::milliseconds{10};

      if (updatePredictedSpectrogram(kTimeout)) {
        torch::Tensor irLeft = spectrogramToImpulseResponse(m_currentSpec[0]);
        torch::Tensor irRight = spectrogramToImpulseResponse(m_currentSpec[1]);
        CHECK_EQ(irLeft.size(0), irRight.size(0));

        auto irL = irLeft.accessor<float, 1>();
        auto irR = irRight.accessor<float, 1>();
        const auto irLength = irL.size(0);
        vraudio::AudioBuffer tmpIr{2, static_cast<size_t>(irLength)};
        auto tmpIrL = tmpIr[0];
        auto tmpIrR = tmpIr[1];
        // TODO linear memcpy?
        for (auto i = 0; i < irLength; ++i) {
          tmpIrL[i] = irL[i];
          tmpIrR[i] = irR[i];
        }
        m_ir = vraudio::AudioBuffer(2, m_irResampler.GetMaxOutputLength(irLength));
        // Resample from 22500 kHz
        m_irResampler.Process(tmpIr, &m_ir);
        return true;
      }
      return false;
    }

    bool updatePredictedSpectrogram(std::chrono::milliseconds timeout) {
      // TODO Terrible hack for now, launch the prediction in a background
      if (!m_netPending) {
        m_specFuture = std::async(std::launch::async, [net = m_net]() {
          return net->predictSpectrogram();
        });
        m_netPending = true;
      }
      if (m_netPending) {
        if (auto status = m_specFuture.wait_for(timeout); status == std::future_status::ready) {
          m_currentSpec = m_specFuture.get();
          m_netPending = false;
          return true;
        }
      }
      return false;
    }

    static torch::Tensor spectrogramToImpulseResponse(const torch::Tensor& spec) {
      // Random uniform phase
      constexpr int n_fft = 512;
      constexpr int64_t win_length = 512;
      constexpr int64_t hop_length = 128;
      
      // Inverse log, the spectrograms net is trained is log(abs(real) + 1e3)
      // net_wav = get_wave(np.clip(np.exp(net_out)-1e-3, 0.0, 10000.00))
      // TODO clip?
      return (spec.exp() - 1e-3).istft(n_fft, hop_length, win_length, torch::hann_window(win_length));
    }

    std::shared_ptr<sdk::NeuralAcoustics> m_net;
    std::future<torch::Tensor> m_specFuture;
    torch::Tensor m_currentSpec;
    bool m_netPending{false};

    vraudio::AudioBuffer m_ir;
    vraudio::Resampler m_irResampler;
    struct {
      std::unique_ptr<vraudio::FirFilter> left;
      std::unique_ptr<vraudio::FirFilter> right;
      size_t samplesPerBuffer{0};
    } m_fir;
};

// Singleton |ResonanceAudioSystem| instance to communicate with the internal
// API.
static std::shared_ptr<ResonanceAudioSystem> resonance_audio = nullptr;
static std::shared_ptr<NeuralAcousticsRenderer> neural_renderer = nullptr;

}  // namespace

void Initialize(int sample_rate, size_t num_channels,
                size_t frames_per_buffer) {
  CHECK_GE(sample_rate, 0);
  CHECK_EQ(num_channels, kNumOutputChannels);
  CHECK_GE(frames_per_buffer, 0);
  resonance_audio = std::make_shared<ResonanceAudioSystem>(
      sample_rate, num_channels, frames_per_buffer);
  neural_renderer = std::make_shared<NeuralAcousticsRenderer>(sample_rate);
}

void Shutdown() { resonance_audio.reset(); }

void ProcessListener(size_t num_frames, float* output) {
  CHECK(output != nullptr);

  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy == nullptr) {
    return;
  }

  if (!resonance_audio_copy->api->FillInterleavedOutputBuffer(
          kNumOutputChannels, num_frames, output)) {
    // No valid output was rendered, fill the output buffer with zeros.
    const size_t buffer_size_samples = kNumOutputChannels * num_frames;
    CHECK(!vraudio::DoesIntegerMultiplicationOverflow<size_t>(
        kNumOutputChannels, num_frames, buffer_size_samples));

    std::fill(output, output + buffer_size_samples, 0.0f);
  }

  // TODO Cache buffer
  // vraudio::AudioBuffer planarOutput{2, num_frames};
  // // Deinterleave
  // vraudio::PlanarFromInterleaved(output, num_frames, kNumOutputChannels,
  //  {planarOutput[0].begin(), planarOutput[1].begin()}, num_frames);

  // neural_renderer->process(planarOutput);
  // // Interleave back
  // vraudio::FillExternalBuffer(planarOutput, output, num_frames, kNumOutputChannels);

#if !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
  if (resonance_audio_copy->is_recording_soundfield) {
    // Record output into soundfield.
    auto* const resonance_audio_api_impl =
        static_cast<ResonanceAudioApiImpl*>(resonance_audio_copy->api.get());
    const auto* soundfield_buffer =
        resonance_audio_api_impl->GetAmbisonicOutputBuffer();
    std::unique_ptr<AudioBuffer> record_buffer(
        new AudioBuffer(kNumFirstOrderAmbisonicChannels, num_frames));
    if (soundfield_buffer != nullptr) {
      for (size_t ch = 0; ch < kNumFirstOrderAmbisonicChannels; ++ch) {
        (*record_buffer)[ch] = (*soundfield_buffer)[ch];
      }
    } else {
      // No output received, fill the record buffer with zeros.
      record_buffer->Clear();
    }
    resonance_audio_copy->soundfield_recorder->AddInput(
        std::move(record_buffer));
  }
#endif  // !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
}

void SetListenerGain(float gain) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetMasterVolume(gain);
  }
}

void SetListenerStereoSpeakerMode(bool enable_stereo_speaker_mode) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetStereoSpeakerMode(enable_stereo_speaker_mode);
  }
}

void SetListenerTransform(float px, float py, float pz, float qx, float qy,
                          float qz, float qw) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetHeadPosition(px, py, pz);
    resonance_audio_copy->api->SetHeadRotation(qx, qy, qz, qw);
  }

  auto neural_copy = neural_renderer;
  if (neural_copy) {
    // TODO Convert quaternion head rotation to the neural acoustics orientation..
    // Set always forward for now
    neural_copy->net().setListenerTransform(sdk::NeuralAcoustics::Orientation::Forward, {px, py});
  }
}

ResonanceAudioApi::SourceId CreateSoundfield(int num_channels) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    return resonance_audio_copy->api->CreateAmbisonicSource(num_channels);
  }
  return ResonanceAudioApi::kInvalidSourceId;
}

ResonanceAudioApi::SourceId CreateSoundObject(RenderingMode rendering_mode) {
  SourceId id = ResonanceAudioApi::kInvalidSourceId;
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    id = resonance_audio_copy->api->CreateSoundObjectSource(rendering_mode);
    resonance_audio_copy->api->SetSourceDistanceModel(
        id, DistanceRolloffModel::kNone, 0.0f, 0.0f);
  }
  return id;
}

void DestroySource(ResonanceAudioApi::SourceId id) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->DestroySource(id);
  }
}

void ProcessSource(ResonanceAudioApi::SourceId id, size_t num_channels,
                   size_t num_frames, float* input) {
  CHECK(input != nullptr);

  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetInterleavedBuffer(id, input, num_channels,
                                                    num_frames);
  }
}

void SetSourceDirectivity(ResonanceAudioApi::SourceId id, float alpha,
                          float order) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSoundObjectDirectivity(id, alpha, order);
  }
}

void SetSourceDistanceAttenuation(ResonanceAudioApi::SourceId id,
                                  float distance_attenuation) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSourceDistanceAttenuation(
        id, distance_attenuation);
  }
}

void SetSourceGain(ResonanceAudioApi::SourceId id, float gain) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSourceVolume(id, gain);
  }
}

void SetSourceListenerDirectivity(ResonanceAudioApi::SourceId id, float alpha,
                                  float order) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSoundObjectListenerDirectivity(id, alpha,
                                                                 order);
  }
}

void SetSourceNearFieldEffectGain(ResonanceAudioApi::SourceId id,
                                  float near_field_effect_gain) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSoundObjectNearFieldEffectGain(
        id, near_field_effect_gain);
  }
}

void SetSourceOcclusionIntensity(ResonanceAudioApi::SourceId id,
                                 float intensity) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSoundObjectOcclusionIntensity(id, intensity);
  }
}

void SetSourceRoomEffectsGain(ResonanceAudioApi::SourceId id,
                              float room_effects_gain) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSourceRoomEffectsGain(id, room_effects_gain);
  }
}

void SetSourceSpread(int id, float spread_deg) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSoundObjectSpread(id, spread_deg);
  }
}

void SetSourceTransform(int id, float px, float py, float pz, float qx,
                        float qy, float qz, float qw) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy != nullptr) {
    resonance_audio_copy->api->SetSourcePosition(id, px, py, pz);
    resonance_audio_copy->api->SetSourceRotation(id, qx, qy, qz, qw);
  }
  // TODO Support multiple sources
  auto neural_copy = neural_renderer;
  if (neural_copy) {
    neural_copy->net().setSourcePosition({px, py});
  }
}

void SetRoomProperties(RoomProperties* room_properties, float* rt60s) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy == nullptr) {
    return;
  }
  if (room_properties == nullptr) {
    resonance_audio_copy->api->SetReflectionProperties(
        resonance_audio_copy->null_reflection_properties);
    resonance_audio_copy->api->SetReverbProperties(
        resonance_audio_copy->null_reverb_properties);
    return;
  }

  const auto reflection_properties =
      ComputeReflectionProperties(*room_properties);
  resonance_audio_copy->api->SetReflectionProperties(reflection_properties);
  const auto reverb_properties =
      (rt60s == nullptr)
          ? ComputeReverbProperties(*room_properties)
          : ComputeReverbPropertiesFromRT60s(
                rt60s, room_properties->reverb_brightness,
                room_properties->reverb_time, room_properties->reverb_gain);
  resonance_audio_copy->api->SetReverbProperties(reverb_properties);
}

#if !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))
bool StartSoundfieldRecorder() {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy == nullptr) {
    return false;
  }
  if (resonance_audio_copy->is_recording_soundfield) {
    LOG(ERROR) << "Another soundfield recording already in progress";
    return false;
  }

  resonance_audio_copy->is_recording_soundfield = true;
  return true;
}

bool StopSoundfieldRecorderAndWriteToFile(const char* file_path,
                                          bool seamless) {
  auto resonance_audio_copy = resonance_audio;
  if (resonance_audio_copy == nullptr) {
    return false;
  }
  if (!resonance_audio_copy->is_recording_soundfield) {
    LOG(ERROR) << "No recorded soundfield found";
    return false;
  }

  resonance_audio_copy->is_recording_soundfield = false;
  if (file_path == nullptr) {
    resonance_audio_copy->soundfield_recorder->Reset();
    return false;
  }
  resonance_audio_copy->soundfield_recorder->WriteToFile(
      file_path, kRecordQuality, seamless);
  return true;
}
#endif  // !(defined(PLATFORM_ANDROID) || defined(PLATFORM_IOS))

}  // namespace unity
}  // namespace vraudio
