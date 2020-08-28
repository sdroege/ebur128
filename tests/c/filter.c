#include <float.h>
#include <limits.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef void * true_peak;
extern true_peak* true_peak_create_c(unsigned int rate, unsigned int channels);
extern void true_peak_destroy_c(true_peak* tp);
extern void true_peak_check_short_c(true_peak* tp, size_t frames, const int16_t* src, double* peaks);
extern void true_peak_check_int_c(true_peak* tp, size_t frames, const int32_t* src, double* peaks);
extern void true_peak_check_float_c(true_peak* tp, size_t frames, const float* src, double* peaks);
extern void true_peak_check_double_c(true_peak* tp, size_t frames, const double* src, double* peaks);

#define FILTER_STATE_SIZE 5
#define EBUR128_UNUSED 0

#define EBUR128_MAX(a, b) (((a) > (b)) ? (a) : (b))

/** BS.1770 filter state. */
typedef double filter_state[FILTER_STATE_SIZE];

typedef struct {
  unsigned int rate;
  unsigned int channels;

  /** BS.1770 filter coefficients (numerator). */
  double b[5];
  /** BS.1770 filter coefficients (denominator). */
  double a[5];
  /** one filter_state per channel. */
  filter_state* v;

  /** Maximum sample peak, one per channel */
  int calculate_sample_peak;
  double* sample_peak;
  /** Maximum true peak, one per channel */
  true_peak *tp;
  double* true_peak;
} filter;

filter * filter_create_c(unsigned int rate, unsigned int channels, int calculate_sample_peak, int calculate_true_peak) {
  filter* f = (filter*) calloc(1, sizeof(filter));
  int i, j;

  f->rate = rate;
  f->channels = channels;

  double f0 = 1681.974450955533;
  double G = 3.999843853973347;
  double Q = 0.7071752369554196;

  double K = tan(M_PI * f0 / (double) f->rate);
  double Vh = pow(10.0, G / 20.0);
  double Vb = pow(Vh, 0.4996667741545416);

  double pb[3] = { 0.0, 0.0, 0.0 };
  double pa[3] = { 1.0, 0.0, 0.0 };
  double rb[3] = { 1.0, -2.0, 1.0 };
  double ra[3] = { 1.0, 0.0, 0.0 };

  double a0 = 1.0 + K / Q + K * K;
  pb[0] = (Vh + Vb * K / Q + K * K) / a0;
  pb[1] = 2.0 * (K * K - Vh) / a0;
  pb[2] = (Vh - Vb * K / Q + K * K) / a0;
  pa[1] = 2.0 * (K * K - 1.0) / a0;
  pa[2] = (1.0 - K / Q + K * K) / a0;

  /* fprintf(stderr, "%.14f %.14f %.14f %.14f %.14f\n",
                     b1[0], b1[1], b1[2], a1[1], a1[2]); */

  f0 = 38.13547087602444;
  Q = 0.5003270373238773;
  K = tan(M_PI * f0 / (double) f->rate);

  ra[1] = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K);
  ra[2] = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K);

  /* fprintf(stderr, "%.14f %.14f\n", a2[1], a2[2]); */

  f->b[0] = pb[0] * rb[0];
  f->b[1] = pb[0] * rb[1] + pb[1] * rb[0];
  f->b[2] = pb[0] * rb[2] + pb[1] * rb[1] + pb[2] * rb[0];
  f->b[3] = pb[1] * rb[2] + pb[2] * rb[1];
  f->b[4] = pb[2] * rb[2];

  f->a[0] = pa[0] * ra[0];
  f->a[1] = pa[0] * ra[1] + pa[1] * ra[0];
  f->a[2] = pa[0] * ra[2] + pa[1] * ra[1] + pa[2] * ra[0];
  f->a[3] = pa[1] * ra[2] + pa[2] * ra[1];
  f->a[4] = pa[2] * ra[2];

  f->v = (filter_state*) malloc(channels * sizeof(filter_state));
  for (i = 0; i < (int) channels; ++i) {
    for (j = 0; j < FILTER_STATE_SIZE; ++j) {
      f->v[i][j] = 0.0;
    }
  }

  f->calculate_sample_peak = calculate_sample_peak;
  f->sample_peak = (double*) malloc(channels * sizeof(double));

  f->tp = calculate_true_peak ? true_peak_create_c(rate, channels) : NULL;
  f->true_peak = (double*) malloc(channels * sizeof(double));

  for (i = 0; i < (int) channels; ++i) {
    f->sample_peak[i] = 0.0;
    f->true_peak[i] = 0.0;
  }

  return f;
}

void filter_destroy_c(filter* f) {
  free(f->sample_peak);
  true_peak_destroy_c(f->tp);
  free(f->true_peak);
  free(f);
}

const double* filter_sample_peak_c(const filter *f) {
  return f->sample_peak;
}

const double* filter_true_peak_c(const filter *f) {
  return f->true_peak;
}

void filter_reset_peaks_c(filter *f) {
  int i;

  for (i = 0; i < (int) f->channels; ++i) {
    f->sample_peak[i] = 0.0;
    f->true_peak[i] = 0.0;
  }
}

#if defined(__SSE2_MATH__) || defined(_M_X64) || _M_IX86_FP >= 2
#include <xmmintrin.h>
#define TURN_ON_FTZ                                                            \
  unsigned int mxcsr = _mm_getcsr();                                           \
  _mm_setcsr(mxcsr | _MM_FLUSH_ZERO_ON);
#define TURN_OFF_FTZ _mm_setcsr(mxcsr);
#define FLUSH_MANUALLY
#else
#warning "manual FTZ is being used, please enable SSE2 (-msse2 -mfpmath=sse)"
#define TURN_ON_FTZ
#define TURN_OFF_FTZ
#define FLUSH_MANUALLY                                             \
  f->v[c][4] = fabs(f->v[c][4]) < DBL_MIN ? 0.0 : f->v[c][4];      \
  f->v[c][3] = fabs(f->v[c][3]) < DBL_MIN ? 0.0 : f->v[c][3];      \
  f->v[c][2] = fabs(f->v[c][2]) < DBL_MIN ? 0.0 : f->v[c][2];      \
  f->v[c][1] = fabs(f->v[c][1]) < DBL_MIN ? 0.0 : f->v[c][1];
#endif

#define EBUR128_FILTER(type, min_scale, max_scale)                             \
  void filter_process_##type##_c(filter* f, size_t frames, const type* src,    \
                                    double* audio_data, const unsigned int* channel_map) { \
    static double scaling_factor =                                             \
        EBUR128_MAX(-((double) (min_scale)), (double) (max_scale));            \
                                                                               \
    size_t i, c;                                                               \
                                                                               \
    TURN_ON_FTZ                                                                \
                                                                               \
    if (f->calculate_sample_peak) {                                            \
      for (c = 0; c < f->channels; ++c) {                                      \
        double max = 0.0;                                                      \
        for (i = 0; i < frames; ++i) {                                         \
          double cur = (double) src[i * f->channels + c];                      \
          if (EBUR128_MAX(cur, -cur) > max) {                                  \
            max = EBUR128_MAX(cur, -cur);                                      \
          }                                                                    \
        }                                                                      \
        max /= scaling_factor;                                                 \
        if (max > f->sample_peak[c]) {                                         \
          f->sample_peak[c] = max;                                             \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    if (f->tp) {                                                               \
      true_peak_check_##type##_c(f->tp, frames, src, f->true_peak);            \
    }                                                                          \
    for (c = 0; c < f->channels; ++c) {                                        \
      if (channel_map[c] == EBUR128_UNUSED) {                                  \
        continue;                                                              \
      }                                                                        \
      for (i = 0; i < frames; ++i) {                                           \
        f->v[c][0] =                                                           \
            (double) ((double) src[i * f->channels + c] / scaling_factor) -    \
            f->a[1] * f->v[c][1] - /**/                                        \
            f->a[2] * f->v[c][2] - /**/                                        \
            f->a[3] * f->v[c][3] - /**/                                        \
            f->a[4] * f->v[c][4];                                              \
        audio_data[i * f->channels + c] = /**/                                 \
            f->b[0] * f->v[c][0] + /**/                                        \
            f->b[1] * f->v[c][1] + /**/                                        \
            f->b[2] * f->v[c][2] + /**/                                        \
            f->b[3] * f->v[c][3] + /**/                                        \
            f->b[4] * f->v[c][4];                                              \
        f->v[c][4] = f->v[c][3];                                               \
        f->v[c][3] = f->v[c][2];                                               \
        f->v[c][2] = f->v[c][1];                                               \
        f->v[c][1] = f->v[c][0];                                               \
      }                                                                        \
      FLUSH_MANUALLY                                                           \
    }                                                                          \
    TURN_OFF_FTZ                                                               \
  }

EBUR128_FILTER(short, SHRT_MIN, SHRT_MAX)
EBUR128_FILTER(int, INT_MIN, INT_MAX)
EBUR128_FILTER(float, -1.0f, 1.0f)
EBUR128_FILTER(double, -1.0, 1.0)
