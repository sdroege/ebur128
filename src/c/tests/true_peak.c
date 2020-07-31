#include <float.h>
#include <limits.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef void * interpolator;
extern interpolator* interp_create_c(unsigned int taps, unsigned int factor, unsigned int channels);
extern void interp_destroy_c(interpolator* interp);
extern size_t interp_process_c(interpolator* interp, size_t frames, float* in, float* out);

#define EBUR128_MAX(a, b) (((a) > (b)) ? (a) : (b))

typedef struct {
  interpolator *interp;
  unsigned int interp_factor;
  unsigned int channels;
  float* resampler_buffer_input;
  size_t resampler_buffer_input_frames;
  float* resampler_buffer_output;
  size_t resampler_buffer_output_frames;
} true_peak;

true_peak* true_peak_create_c(unsigned int rate, unsigned int channels) {
  unsigned int samples_in_100ms = ((rate + 5) / 10);
  true_peak* tp = (true_peak*) calloc(1, sizeof(true_peak));
  tp->channels = channels;

  if (rate < 96000) {
    tp->interp = interp_create_c(49, 4, channels);
    tp->interp_factor = 4;
  } else if (rate < 192000) {
    tp->interp = interp_create_c(49, 2, channels);
    tp->interp_factor = 2;
  } else {
    free(tp);
    return NULL;
  }

  tp->resampler_buffer_input_frames = samples_in_100ms * 4;
  tp->resampler_buffer_input = (float*) malloc(
      tp->resampler_buffer_input_frames * channels * sizeof(float));

  tp->resampler_buffer_output_frames =
      tp->resampler_buffer_input_frames * tp->interp_factor;
  tp->resampler_buffer_output = (float*) malloc(
      tp->resampler_buffer_output_frames * channels * sizeof(float));

  return tp;
}

void true_peak_destroy_c(true_peak* tp) {
  if (!tp)
    return;

  free(tp->resampler_buffer_input);
  free(tp->resampler_buffer_output);
  free(tp);
}

#define EBUR128_CHECK_TRUE_PEAK(type, min_scale, max_scale)                                    \
void true_peak_check_##type##_c(true_peak* tp, size_t frames, const type* src, double* peaks) { \
    static double scaling_factor =                                                             \
        EBUR128_MAX(-((double) (min_scale)), (double) (max_scale));                            \
  size_t c, i, frames_out;                                                                     \
                                                                                               \
  for (i = 0; i < frames; ++i) {                                                               \
    for (c = 0; c < tp->channels; ++c) {                                                       \
      tp->resampler_buffer_input[i * tp->channels + c] =                                       \
          (float) ((double) src[i * tp->channels + c] / scaling_factor);                       \
    }                                                                                          \
  }                                                                                            \
                                                                                               \
  frames_out =                                                                                 \
      interp_process_c(tp->interp, frames, tp->resampler_buffer_input,                         \
                     tp->resampler_buffer_output);                                             \
                                                                                               \
  for (i = 0; i < frames_out; ++i) {                                                           \
    for (c = 0; c < tp->channels; ++c) {                                                       \
      double val =                                                                             \
          (double) tp->resampler_buffer_output[i * tp->channels + c];                          \
                                                                                               \
      if (EBUR128_MAX(val, -val) > peaks[c]) {                                                 \
        peaks[c] = EBUR128_MAX(val, -val);                                                     \
      }                                                                                        \
    }                                                                                          \
  }                                                                                            \
}

EBUR128_CHECK_TRUE_PEAK(short, SHRT_MIN, SHRT_MAX);
EBUR128_CHECK_TRUE_PEAK(int, INT_MIN, INT_MAX);
EBUR128_CHECK_TRUE_PEAK(float, -1.0, 1.0);
EBUR128_CHECK_TRUE_PEAK(double, -1.0, 1.0);
