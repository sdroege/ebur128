#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include <ebur128.h>

#define assert_int_eq(a, b) do { \
  if ((a) != (b)) { \
    fprintf(stderr, "%s:%d: assertion failed: \"%s\" (%d) != \"%s\" (%d)\n", __FILE__, __LINE__, #a, (a), #b, (b)); \
    exit(-1); \
  } \
} while(0);

#define assert_double_eq(a, b) do { \
  if (fabs((a) - (b)) >= 0.000001) { \
    fprintf(stderr, "%s:%d: assertion failed: \"%s\" (%lf) != \"%s\" (%lf)\n", __FILE__, __LINE__, #a, (a), #b, (b)); \
    exit(-1); \
  } \
} while(0);

int main(int argc, char **argv) {
  ebur128_state *s;
  float *data;
  size_t i;
  float acc, step;
  double val;
  const unsigned long sample_rate = 48000;
  const int channels = 2;
  const size_t num_frames = sample_rate * 5;

  s = ebur128_init(channels, sample_rate, EBUR128_MODE_M | EBUR128_MODE_S | EBUR128_MODE_I | EBUR128_MODE_LRA | EBUR128_MODE_SAMPLE_PEAK | EBUR128_MODE_TRUE_PEAK);

  data = malloc(num_frames * channels * sizeof (*data));
  acc = 0.0;
  step = 2.0 * M_PI * 440.0 / sample_rate;
  for (i = 0; i < num_frames; i++) {
    size_t j;

    val = sinf(acc);

    for (j = 0; j < channels; j++)
      data[i*channels + j] = val;

    acc += step;
  }

  assert_int_eq(ebur128_add_frames_float(s, data, num_frames), EBUR128_SUCCESS);

  assert_int_eq(ebur128_loudness_global(s, &val), EBUR128_SUCCESS);
  assert_double_eq(val, -0.6826039914165554);

  assert_int_eq(ebur128_loudness_momentary(s, &val), EBUR128_SUCCESS);
  assert_double_eq(val, -0.6813325598268921);

  assert_int_eq(ebur128_loudness_shortterm(s, &val), EBUR128_SUCCESS);
  assert_double_eq(val, -0.6827591715100236);

  assert_int_eq(ebur128_loudness_window(s, 1, &val), EBUR128_SUCCESS);
  assert_double_eq(val, -0.8742956620008693);

  assert_int_eq(ebur128_loudness_range(s, &val), EBUR128_SUCCESS);
  assert_double_eq(val, 0.00006921150169403312);

  assert_int_eq(ebur128_relative_threshold(s, &val), EBUR128_SUCCESS);
  assert_double_eq(val, -10.682603991416554);

  for (i = 0; i < channels; i++) {
    assert_int_eq(ebur128_sample_peak(s, i, &val), EBUR128_SUCCESS);
    assert_double_eq(val, 1.0);
    assert_int_eq(ebur128_prev_sample_peak(s, i, &val), EBUR128_SUCCESS);
    assert_double_eq(val, 1.0);

    assert_int_eq(ebur128_true_peak(s, i, &val), EBUR128_SUCCESS);
    assert_double_eq(val, 1.0008491277694702);
    assert_int_eq(ebur128_prev_true_peak(s, i, &val), EBUR128_SUCCESS);
    assert_double_eq(val, 1.0008491277694702);
  }

  free(data);
  ebur128_destroy(&s);

  return 0;
}
