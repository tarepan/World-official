//-----------------------------------------------------------------------------
// Copyright 2012 Masanori Morise
// Author: mmorise [at] meiji.ac.jp (Masanori Morise)
// Last update: 2021/02/15
//
// Voice synthesis based on f0, spectrogram and aperiodicity.
// forward_real_fft, inverse_real_fft and minimum_phase are used to speed up.
//-----------------------------------------------------------------------------
#include "world/synthesis.h"

#include <math.h>

#include "world/common.h"
#include "world/constantnumbers.h"
#include "world/matlabfunctions.h"

namespace {


/**
 * Synthesize a white noise as frequency-domain representation.
 *
 * @param noise_size       - Length [sample] of noise
 * @param fft_size         - Analysis/Synthesis FFT size
 * @param forward_real_fft - Output, FFT container containing the white noise
 * @param randn_state      - Random state
 */
static void GetNoiseSpectrum(int noise_size, int fft_size, const ForwardRealFFT *forward_real_fft, RandnState *randn_state) {
  // Generate white noise with uint random number
  double average = 0.0;
  for (int i = 0; i < noise_size; ++i) {
    forward_real_fft->waveform[i] = randn(randn_state);
    average += forward_real_fft->waveform[i];
  }
  average /= noise_size;

  // Amplitude normalization
  for (int i = 0; i < noise_size; ++i)
    forward_real_fft->waveform[i] -= average;

  // Zero padding
  for (int i = noise_size; i < fft_size; ++i)
    forward_real_fft->waveform[i] = 0.0;

  // Wave-to-Spec
  fft_execute(forward_real_fft->forward_fft);
}


/**
 * Synthesize an aperiodic component (colored noise series).
 *
 * @param noise_size                          - Length [sample] of noise
 * @param fft_size                            - Analysis/Synthesis FFT size
 * @param spectrum           :: (F=fft_size,) - Linear-power envelope   spectrum, effective size is `fft_size//2+1`
 * @param aperiodic_ratio    :: (F=fft_size,) - Linear-power aperiodity spectrum, effective size is `fft_size//2+1`, 0<value<1
 * @param current_vuv                         - VUV of the frame
 * @param forward_real_fft                    - FFT  container
 * @param inverse_real_fft                    - iFFT container
 * @param minimum_phase                       - MinPhase container
 * @param aperiodic_response :: (T=fft_size,) - Output, waveform of an aperiodic fragment
 * @param randn_state                         - Random state
 */
static void GetAperiodicResponse(int noise_size, int fft_size, const double *spectrum, const double *aperiodic_ratio, double current_vuv,
    const ForwardRealFFT *forward_real_fft, const InverseRealFFT *inverse_real_fft,
    const MinimumPhaseAnalysis *minimum_phase, double *aperiodic_response, RandnState *randn_state) {
  // Generate a white noise spectrum
  GetNoiseSpectrum(noise_size, fft_size, forward_real_fft, randn_state);

  // |H(ω)| to |H_a(ω)| to minimum phase H_a(ω)
  if (current_vuv != 0.0)
    for (int i = 0; i <= minimum_phase->fft_size / 2; ++i)
      minimum_phase->log_spectrum[i] = log(spectrum[i] * aperiodic_ratio[i]) / 2.0;
  else
    for (int i = 0; i <= minimum_phase->fft_size / 2; ++i)
      minimum_phase->log_spectrum[i] = log(spectrum[i]) / 2.0;
  GetMinimumPhaseSpectrum(minimum_phase);

  // Convolution in frequency domain
  for (int i = 0; i <= fft_size / 2; ++i) {
    //                                                  F(noise_fir)                        F(noise_series)                              F(noise_fir)                          F(noise_series)
    inverse_real_fft->spectrum[i][0] = minimum_phase->minimum_phase_spectrum[i][0] * forward_real_fft->spectrum[i][0] - minimum_phase->minimum_phase_spectrum[i][1] * forward_real_fft->spectrum[i][1];
    inverse_real_fft->spectrum[i][1] = minimum_phase->minimum_phase_spectrum[i][0] * forward_real_fft->spectrum[i][1] + minimum_phase->minimum_phase_spectrum[i][1] * forward_real_fft->spectrum[i][0];
  }

  // Freq-to-Wave
  fft_execute(inverse_real_fft->inverse_fft);
  fftshift(inverse_real_fft->waveform, fft_size, aperiodic_response);
}

//-----------------------------------------------------------------------------
// RemoveDCComponent()
//-----------------------------------------------------------------------------
static void RemoveDCComponent(const double *periodic_response, int fft_size,
    const double *dc_remover, double *new_periodic_response) {
  double dc_component = 0.0;
  for (int i = fft_size / 2; i < fft_size; ++i)
    dc_component += periodic_response[i];
  for (int i = 0; i < fft_size / 2; ++i)
    new_periodic_response[i] = -dc_component * dc_remover[i];
  for (int i = fft_size / 2; i < fft_size; ++i)
    new_periodic_response[i] -= dc_component * dc_remover[i];
}

//-----------------------------------------------------------------------------
// GetSpectrumWithFractionalTimeShift() calculates a periodic spectrum with
// the fractional time shift under 1/fs.
//-----------------------------------------------------------------------------
static void GetSpectrumWithFractionalTimeShift(int fft_size,
    double coefficient, const InverseRealFFT *inverse_real_fft) {
  double re, im, re2, im2;
  for (int i = 0; i <= fft_size / 2; ++i) {
    re = inverse_real_fft->spectrum[i][0];
    im = inverse_real_fft->spectrum[i][1];
    re2 = cos(coefficient * i);
    im2 = sqrt(1.0 - re2 * re2);  // sin(pshift)

    inverse_real_fft->spectrum[i][0] = re * re2 + im * im2;
    inverse_real_fft->spectrum[i][1] = im * re2 - re * im2;
  }
}

/**
 * Generate waveform segment (= impulse response) of periodic component.
 *
 * @param fft_size                               - Analysis/Synthesis FFT size
 * @param spectrum              :: (F=fft_size,) - Linear-power envelope   spectrum, effective size is `fft_size//2+1`
 * @param aperiodic_ratio       :: (F=fft_size,) - Linear-power aperiodity spectrum, effective size is `fft_size//2+1`, 0<value<1
 * @param current_vuv                            - Voice/UnVoice ratio of this waveform segment
 * @param inverse_real_fft                       - iFFT container
 * @param minimum_phase                          - MinPhase container
 * @param dc_remover
 * @param fractional_time_shift                  - Time shift [sec] of frame's pulse from coarse time
 * @param fs                                     - Sampling rate
 * @param periodic_response     :: (T=fft_size,) - Output waveform segment, size is defined by caller
 */
static void GetPeriodicResponse(int fft_size, const double *spectrum, const double *aperiodic_ratio, double current_vuv, const InverseRealFFT *inverse_real_fft,
    const MinimumPhaseAnalysis *minimum_phase, const double *dc_remover, double fractional_time_shift, int fs, double *periodic_response) {

  // Unvoiced, so pass
  if (current_vuv <= 0.5 || aperiodic_ratio[0] > 0.999) {
    for (int i = 0; i < fft_size; ++i) periodic_response[i] = 0.0;
    return;
  }

  // |H(ω)| to |H_p(ω)| to minimum phase H_p(ω)
  for (int i = 0; i <= minimum_phase->fft_size / 2; ++i) minimum_phase->log_spectrum[i] = log(spectrum[i] * (1.0 - aperiodic_ratio[i]) + world::kMySafeGuardMinimum) / 2.0;
  GetMinimumPhaseSpectrum(minimum_phase);
  for (int i = 0; i <= fft_size / 2; ++i) {
    inverse_real_fft->spectrum[i][0] = minimum_phase->minimum_phase_spectrum[i][0];
    inverse_real_fft->spectrum[i][1] = minimum_phase->minimum_phase_spectrum[i][1];
  }

  // apply fractional time delay of `fractional_time_shift` seconds using linear phase shift
  double coefficient = 2.0 * world::kPi * fractional_time_shift * fs / fft_size;
  GetSpectrumWithFractionalTimeShift(fft_size, coefficient, inverse_real_fft);

  // Generate waveform segment
  fft_execute(inverse_real_fft->inverse_fft);
  fftshift(inverse_real_fft->waveform, fft_size, periodic_response);
  RemoveDCComponent(periodic_response, fft_size, dc_remover, periodic_response);
}


/**
 * Calculate spectrum at pulse position.
 *
 * @param current_time                             - Coarse time [sec] of frame's pulse
 * @param frame_period                             - Period [sec] of the frame
 * @param f0_length                                - Length of frame fo contour
 * @param spectrogram       :: (L, F=fft_size/2+1) - Linear-power envelope spectrogram
 * @param fft_size                                 - Analysis/Synthesis FFT size
 * @param spectral_envelope :: (F=fft_size,)       - Output, linear-power envelope spectrum at pulse position, effective size is `fft_size//2+1`
 */
static void GetSpectralEnvelope(double current_time, double frame_period, int f0_length, const double * const *spectrogram, int fft_size, double *spectral_envelope) {
  // Frame numbers
  int current_frame_floor = MyMinInt(f0_length - 1, static_cast<int>(floor(current_time / frame_period)));
  int current_frame_ceil  = MyMinInt(f0_length - 1, static_cast<int>( ceil(current_time / frame_period)));

  // Interpolate spectrums in time, in linear-power domain
  double interpolation = current_time / frame_period - current_frame_floor;
  if (current_frame_floor == current_frame_ceil) for (int i = 0; i <= fft_size / 2; ++i) spectral_envelope[i] = fabs(spectrogram[current_frame_floor][i]);
  else for (int i = 0; i <= fft_size / 2; ++i)
      spectral_envelope[i] = (1.0 - interpolation) * fabs(spectrogram[current_frame_floor][i]) + interpolation * fabs(spectrogram[current_frame_ceil][i]);
}


/**
 * Calculate aperiodicity spectrum at pulse position.
 *
 * @param current_time                        - Coarse time [sec] of frame's pulse
 * @param frame_period                        - Period [sec] of the frame
 * @param f0_length                           - Length of frame fo contour
 * @param aperiodicity :: (L, F=fft_size/2+1) - Linear-amplitude aperiodicity spectrogram, 0<=value<=1
 * @param fft_size                            - Analysis/Synthesis FFT size
 * @param aperiodic_spectrum :: (F=fft_size,) - Output, linear-power aperiodicity spectrum at pulse position, effective size is `fft_size//2+1`, 0<value<1
 */
static void GetAperiodicRatio(double current_time, double frame_period, int f0_length, const double * const *aperiodicity, int fft_size, double *aperiodic_spectrum) {
  // Frame numbers
  int current_frame_floor = MyMinInt(f0_length - 1, static_cast<int>(floor(current_time / frame_period)));
  int current_frame_ceil  = MyMinInt(f0_length - 1, static_cast<int>( ceil(current_time / frame_period)));

  // Interpolate aperiodic spectrums in time, in Linear-amplitude domain, then convert to linear-power spectrum
  double interpolation = current_time / frame_period - current_frame_floor;
  if (current_frame_floor == current_frame_ceil)
    for (int i = 0; i <= fft_size / 2; ++i)
      aperiodic_spectrum[i] = pow(GetSafeAperiodicity(aperiodicity[current_frame_floor][i]), 2.0);
  else
    for (int i = 0; i <= fft_size / 2; ++i)
      aperiodic_spectrum[i] = pow((1.0 - interpolation) * GetSafeAperiodicity(aperiodicity[current_frame_floor][i]) +
                                          interpolation * GetSafeAperiodicity(aperiodicity[current_frame_ceil][i]),
                                  2.0);
}


/**
 * Calculates a periodic and aperiodic response at a time.
 *
 * @param current_vuv                                  - VUV flag of the frame
 * @param noise_size                                   - Length [sample] of noise
 * @param spectrogram           :: (L, F=fft_size/2+1) - Linear-power envelope spectrogram
 * @param fft_size                                     - Analysis/Synthesis FFT size
 * @param aperiodicity          :: (L, F=fft_size/2+1) - Linear-amplitude aperiodicity spectrogram, 0<=value<=1
 * @param f0_length                                    - Length of frame fo contour
 * @param frame_period                                 - Period [sec] of the frame
 * @param current_time                                 - Coarse time [sec] of frame's pulse
 * @param fractional_time_shift                        - Time shift [sec] of frame's pulse from `current_time`
 * @param fs                                           - Sampling rate
 * @param forward_real_fft                             -  FFT container
 * @param inverse_real_fft                             - iFFT container
 * @param minimum_phase                                - MinPhase container
 * @param dc_remover
 * @param response              :: (T=fft_size,)       - Output waveform fragment, size is already defined by caller
 * @param randn_state                                  - Random state
 */
static void GetOneFrameSegment(double current_vuv, int noise_size,
    const double * const *spectrogram, int fft_size, const double * const *aperiodicity, int f0_length, double frame_period,
    double current_time, double fractional_time_shift, int fs, const ForwardRealFFT *forward_real_fft, const InverseRealFFT *inverse_real_fft,
    const MinimumPhaseAnalysis *minimum_phase, const double *dc_remover, double *response, RandnState* randn_state) {

  // Initialize
  double *aperiodic_response = new double[fft_size]; // waveform of an aperiodic fragment
  double *periodic_response  = new double[fft_size]; // waveform of a   periodic fragment
  double *spectral_envelope  = new double[fft_size]; // Linear-power envelope   spectrum at pulse position, effective size is `fft_size//2+1`
  double *aperiodic_ratio    = new double[fft_size]; // Linear-power aperiodity spectrum at pulse position, effective size is `fft_size//2+1`, 0<value<1

  // Interpolate spectrums in time
  GetSpectralEnvelope(current_time, frame_period, f0_length, spectrogram,  fft_size, spectral_envelope);
  GetAperiodicRatio(  current_time, frame_period, f0_length, aperiodicity, fft_size, aperiodic_ratio);

  // Synthesize impulses
  GetPeriodicResponse(fft_size, spectral_envelope, aperiodic_ratio, current_vuv, inverse_real_fft, minimum_phase, dc_remover, fractional_time_shift, fs, periodic_response);
  GetAperiodicResponse(noise_size, fft_size, spectral_envelope, aperiodic_ratio, current_vuv, forward_real_fft, inverse_real_fft, minimum_phase, aperiodic_response, randn_state);

  // Update all elements of the waveform with normalization
  double sqrt_noise_size = sqrt(static_cast<double>(noise_size));
  for (int i = 0; i < fft_size; ++i)
    response[i] = (periodic_response[i] * sqrt_noise_size + aperiodic_response[i]) / fft_size;

  // Clean up
  delete[] spectral_envelope;
  delete[] aperiodic_ratio;
  delete[] periodic_response;
  delete[] aperiodic_response;
}


/**
 * Generate time axis, floored fo contour and VUV series.
 *
 * @param f0               :: (L=f0_length,)   - Frame fo contour
 * @param f0_length                            - Length of `f0`
 * @param fs                                   - Sampling rate
 * @param y_length                             - Length of the waveform
 * @param frame_period                         - Period of a frame [sec]
 * @param lowest_f0                            - fo under which judged as Unvoiced (fo=0)
 * @param time_axis        :: (T=y_length,)    - Output, times of data-points. The value is `i/fs`.
 * @param coarse_time_axis :: (L=f0_length+1,) - Output, times of frame data-points. The value is `i*frame_period`.
 * @param coarse_f0        :: (L=f0_length+1,) - Output, floored frame fo contour
 * @param coarse_vuv       :: (L=f0_length+1,) - Output, frame VUV series (0.0|1.0)
 */
static void GetTemporalParametersForTimeBase(const double *f0, int f0_length,
    int fs, int y_length, double frame_period, double lowest_f0, double *time_axis, double *coarse_time_axis, double *coarse_f0, double *coarse_vuv) {

  for (int i = 0; i < y_length; ++i)
    time_axis[i] = i / static_cast<double>(fs);

  for (int i = 0; i < f0_length; ++i) {
    // frame time axis
    coarse_time_axis[i] = i * frame_period;
    // floored fo contour
    coarse_f0[i] = f0[i] < lowest_f0 ? 0.0 : f0[i];
    // VUV flag series
    coarse_vuv[i] = coarse_f0[i] == 0.0 ? 0.0 : 1.0;
  }
  // Tail values for interpolation
  coarse_time_axis[f0_length] = f0_length * frame_period;
  coarse_f0[f0_length]  =  coarse_f0[f0_length - 1] * 2 -  coarse_f0[f0_length - 2];
  coarse_vuv[f0_length] = coarse_vuv[f0_length - 1] * 2 - coarse_vuv[f0_length - 2];
}


/**
 * Calculate pulse position from instantaneous fo.
 *
 * @param interpolated_f0             :: (T=y_length,) - Instantaneous fo series
 * @param time_axis                   :: (T=y_length,) - times of data-points. [0/fs, 1/fs, 2/fs, ..., (L-1)/fs]
 * @param y_length                                     - Length of the waveform
 * @param fs                                           - Sampling rate
 * @param pulse_locations             :: (T=y_length)  - Output, Times of pulse sample [sec],          e.g. [30/fs, 80/fs, 110/fs, ...]
 * @param pulse_locations_index       :: (T=y_length)  - Output, Indice of pulse sample,               e.g. [30,    80,    110,    ...]
 * @param pulse_locations_time_shift  :: (T=y_length)  - Output, Shift for exact pulse position [sec], e.g. [0.001, 0.003, 0.000,  ...]
 *
 * @return                                             - The number of pulses, equal to effective length of `pulse_locations*`
 */
static int GetPulseLocationsForTimeBase(const double *interpolated_f0,
    const double *time_axis, int y_length, int fs, double *pulse_locations, int *pulse_locations_index, double *pulse_locations_time_shift) {

  // Initialization
  double *total_phase    = new double[y_length]; // Accumulation of phase progression
  double *wrap_phase     = new double[y_length]; // Wrapped `total_phase`
  double *wrap_phase_abs = new double[y_length - 1]; // Wrapped-Phase difference between samples

  // Instantaneous frequency to φ(t)
  total_phase[0] = 2.0 * world::kPi * interpolated_f0[0] / fs;
  wrap_phase[0] = fmod(total_phase[0], 2.0 * world::kPi);
  for (int i = 1; i < y_length; ++i) {
    total_phase[i] = total_phase[i - 1] + 2.0 * world::kPi * interpolated_f0[i] / fs;
    wrap_phase[i] = fmod(total_phase[i], 2.0 * world::kPi);
    wrap_phase_abs[i - 1] = fabs(wrap_phase[i] - wrap_phase[i - 1]);
  }

  int number_of_pulses = 0;
  for (int i = 0; i < y_length - 1; ++i) {
    if (wrap_phase_abs[i] > world::kPi) {
      // zero-crossing as pulse position
      pulse_locations[number_of_pulses] = time_axis[i];
      pulse_locations_index[number_of_pulses] = i;

      // calculate the time shift in seconds between exact fractional pulse
      // position and the integer pulse position (sample i)
      // as we don't have access to the exact pulse position, we infer it
      // from the point between sample i and sample i + 1 where the
      // accummulated phase cross a multiple of 2pi
      // this point is found by solving y1 + x * (y2 - y1) = 0 for x, where y1
      // and y2 are the phases corresponding to sample i and i + 1, offset so
      // they cross zero; x >= 0
      double y1 = wrap_phase[i] - 2.0 * world::kPi;
      double y2 = wrap_phase[i + 1];
      double x = -y1 / (y2 - y1);
      pulse_locations_time_shift[number_of_pulses] = x / fs;

      ++number_of_pulses;
    }
  }

  // Clean up
  delete[] wrap_phase_abs;
  delete[] wrap_phase;
  delete[] total_phase;

  return number_of_pulses;
}


/**
 * Generate excitation pulses and UVUs.
 *
 * @param f0                         :: (L=f0_length,) - Frame fo contour
 * @param f0_length                                    - Length of `f0`
 * @param fs                                           - Sampling rate
 * @param frame_period                                 - Period of a frame [sec]
 * @param y_length                                     - Length of the waveform
 * @param lowest_f0                                    - fo under which judged as Unvoiced (fo=0)
 * @param pulse_locations            :: (T=y_length)   - Output, Times of pulse sample [sec],          e.g. [30/fs, 80/fs, 110fs, ...]
 * @param pulse_locations_index      :: (T=y_length)   - Output, Indice of pulse sample in time axis,  e.g. [30,    80,    110,   ...]
 * @param pulse_locations_time_shift :: (T=y_length)   - Output, Shift for exact pulse position [sec], e.g. [0.001, 0.003, 0.000, ...]
 * @param interpolated_vuv           :: (T=y_length)   - Output, sample-wise VUV series
 *
 * @return                                             - The number of pulses, equal to effective length of `pulse_locations*`
 */
static int GetTimeBase(const double *f0, int f0_length, int fs, double frame_period, int y_length, double lowest_f0,
    double *pulse_locations, int *pulse_locations_index, double *pulse_locations_time_shift, double *interpolated_vuv) {

  // Initialization
  double *time_axis        = new double[y_length];      // times of data-points. [0/fs, 1/fs, 2/fs, ..., (L-1)/fs]
  double *coarse_time_axis = new double[f0_length + 1]; // times of frame data-points. [0*frame_period, 1*frame_period, ...,]
  double *coarse_f0        = new double[f0_length + 1]; // Floored frame fo contour
  double *coarse_vuv       = new double[f0_length + 1]; // Frame VUV series (0.0|1.0)
  double *interpolated_f0  = new double[y_length];      // Instantaneous fo series

  GetTemporalParametersForTimeBase(f0, f0_length, fs, y_length, frame_period, lowest_f0, time_axis, coarse_time_axis, coarse_f0, coarse_vuv);

  // Upsample from frame scale to sample scale
  interp1(coarse_time_axis, coarse_f0,  f0_length + 1, time_axis, y_length, interpolated_f0);
  interp1(coarse_time_axis, coarse_vuv, f0_length + 1, time_axis, y_length, interpolated_vuv);
  // Correct interpolated values
  for (int i = 0; i < y_length; ++i) {
    interpolated_vuv[i] = interpolated_vuv[i] > 0.5  ?               1.0 : 0.0;
    interpolated_f0[i]  = interpolated_vuv[i] == 0.0 ? world::kDefaultF0 : interpolated_f0[i];
  }

  int number_of_pulses = GetPulseLocationsForTimeBase(interpolated_f0, time_axis, y_length, fs, pulse_locations, pulse_locations_index, pulse_locations_time_shift);

  // Clean up
  delete[] coarse_vuv;
  delete[] coarse_f0;
  delete[] coarse_time_axis;
  delete[] time_axis;
  delete[] interpolated_f0;

  return number_of_pulses;
}


static void GetDCRemover(int fft_size, double *dc_remover) {
  double dc_component = 0.0;
  for (int i = 0; i < fft_size / 2; ++i) {
    dc_remover[i] = 0.5 -
      0.5 * cos(2.0 * world::kPi * (i + 1.0) / (1.0 + fft_size));
    dc_remover[fft_size - i - 1] = dc_remover[i];
    dc_component += dc_remover[i] * 2.0;
  }
  for (int i = 0; i < fft_size / 2; ++i) {
    dc_remover[i] /= dc_component;
    dc_remover[fft_size - i - 1] = dc_remover[i];
  }
}

}  // namespace

/**
 * Synthesize a speech with pitch-synchronous impulse and noise FIR filtering.
 *
 * @param f0           :: (L=f0_length,)      - Frame fo contour
 * @param f0_length                           - Length of the `f0`
 * @param spectrogram  :: (L, F=fft_size/2+1) - Linear-power     envelope     spectrogram
 * @param aperiodicity :: (L, F=fft_size/2+1) - Linear-amplitude aperiodicity spectrogram, 0<=value<=1
 * @param fft_size                            - Analysis/Synthesis FFT size
 * @param frame_period                        - Period of a frame, synced with analysis [msec]
 * @param fs                                  - Sampling rate
 * @param y_length                            - Length of the waveform `y`
 * @param y            :: (T=y_length,)       - Generated waveform
 */
void Synthesis(const double *f0, int f0_length, const double * const *spectrogram, const double * const *aperiodicity,
    int fft_size, double frame_period, int fs, int y_length, double *y) {
  RandnState randn_state = {};
  randn_reseed(&randn_state);

  // Initialization
  double *impulse_response = new double[fft_size]; // waveform segment corresponding to a pulse
  for (int i = 0; i < y_length; ++i) y[i] = 0.0;
  //// MinPhase container
  MinimumPhaseAnalysis minimum_phase = {0};
  InitializeMinimumPhaseAnalysis(fft_size, &minimum_phase);
  //// iFFT container
  InverseRealFFT inverse_real_fft = {0};
  InitializeInverseRealFFT(fft_size, &inverse_real_fft);
  //// FFT container
  ForwardRealFFT forward_real_fft = {0};
  InitializeForwardRealFFT(fft_size, &forward_real_fft);
  //// Impulse and VUV
  double *pulse_locations            = new double[y_length]; // Times of pulse sample [sec],          effective length is `number_of_pulses`. e.g. [30/fs, 80/fs, 110fs, ...]
  int    *pulse_locations_index      = new    int[y_length]; // Indice of pulse sample in time axis,  effective length is `number_of_pulses`. e.g. [30,    80,    110,   ...]
  double *pulse_locations_time_shift = new double[y_length]; // Shift for exact pulse position [sec], effective length is `number_of_pulses`. e.g. [0.001, 0.003, 0.000, ...]
  double *interpolated_vuv           = new double[y_length]; // Sample-wise VUV series

  // Calculate impulses and VUV
  // `lowest_f0` is based on analysis/synthesis FFT size
  int number_of_pulses = GetTimeBase(f0, f0_length, fs, frame_period / 1000.0,
      y_length, fs / fft_size + 1.0, pulse_locations, pulse_locations_index, pulse_locations_time_shift, interpolated_vuv);

  double *dc_remover = new double[fft_size];
  GetDCRemover(fft_size, dc_remover);

  frame_period /= 1000.0; // [msec] -> [sec]
  int noise_size; // [sample]
  int offset; // offset length [sample] of overlap-add
  int lower_limit, upper_limit;
  for (int i = 0; i < number_of_pulses; ++i) {

    // Generate a periodic finite impulse response and a corresponding aperiodic waveform
    // Set noise_size as inter-pulse length
    noise_size = pulse_locations_index[MyMinInt(number_of_pulses - 1, i + 1)] - pulse_locations_index[i];
    GetOneFrameSegment(interpolated_vuv[pulse_locations_index[i]], noise_size,
        spectrogram, fft_size, aperiodicity, f0_length, frame_period,
        pulse_locations[i], pulse_locations_time_shift[i], fs,
        &forward_real_fft, &inverse_real_fft, &minimum_phase, dc_remover,
        impulse_response, &randn_state);

    // Add the impulse response to the whole waveform (Overlap-Add)
    offset = pulse_locations_index[i] - fft_size / 2 + 1;
    // Clip out of range
    lower_limit = MyMaxInt(0, -offset); // >=0
    upper_limit = MyMinInt(fft_size, y_length - offset); // <=fft_size
    for (int j = lower_limit; j < upper_limit; ++j) {
      y[j + offset] += impulse_response[j];
    }
  }

  // Clean up
  delete[] dc_remover;
  delete[] pulse_locations;
  delete[] pulse_locations_index;
  delete[] pulse_locations_time_shift;
  delete[] interpolated_vuv;
  DestroyMinimumPhaseAnalysis(&minimum_phase);
  DestroyInverseRealFFT(&inverse_real_fft);
  DestroyForwardRealFFT(&forward_real_fft);
  delete[] impulse_response;
}
