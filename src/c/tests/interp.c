#include <float.h>
#include <limits.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERROR(condition, errorcode, goto_point)                          \
  if ((condition)) {                                                           \
    errcode = (errorcode);                                                     \
    goto goto_point;                                                           \
  }

#define ALMOST_ZERO 0.000001

typedef struct {
  unsigned int count;  /* Number of coefficients in this subfilter */
  unsigned int* index; /* Delay index of corresponding filter coeff */
  double* coeff;       /* List of subfilter coefficients */
} interp_filter;

typedef struct {         /* Data structure for polyphase FIR interpolator */
  unsigned int factor;   /* Interpolation factor of the interpolator */
  unsigned int taps;     /* Taps (prefer odd to increase zero coeffs) */
  unsigned int channels; /* Number of channels */
  unsigned int delay;    /* Size of delay buffer */
  interp_filter* filter; /* List of subfilters (one for each factor) */
  float** z;             /* List of delay buffers (one for each channel) */
  unsigned int zi;       /* Current delay buffer index */
} interpolator_c;

interpolator_c*
interp_create_c(unsigned int taps, unsigned int factor, unsigned int channels) {
  int errcode; /* unused */
  interpolator_c* interp;
  unsigned int j;

  (void) errcode;

  interp = (interpolator_c*) calloc(1, sizeof(interpolator_c));
  CHECK_ERROR(!interp, 0, exit);

  interp->taps = taps;
  interp->factor = factor;
  interp->channels = channels;
  interp->delay = (interp->taps + interp->factor - 1) / interp->factor;

  /* Initialize the filter memory
   * One subfilter per interpolation factor. */
  interp->filter =
      (interp_filter*) calloc(interp->factor, sizeof(*interp->filter));
  CHECK_ERROR(!interp->filter, 0, free_interp);

  for (j = 0; j < interp->factor; j++) {
    interp->filter[j].index =
        (unsigned int*) calloc(interp->delay, sizeof(unsigned int));
    interp->filter[j].coeff = (double*) calloc(interp->delay, sizeof(double));
    CHECK_ERROR(!interp->filter[j].index || !interp->filter[j].coeff, 0,
                free_filter_index_coeff);
  }

  /* One delay buffer per channel. */
  interp->z = (float**) calloc(interp->channels, sizeof(float*));
  CHECK_ERROR(!interp->z, 0, free_filter_index_coeff);
  for (j = 0; j < interp->channels; j++) {
    interp->z[j] = (float*) calloc(interp->delay, sizeof(float));
    CHECK_ERROR(!interp->z[j], 0, free_filter_z);
  }

  /* Calculate the filter coefficients */
  for (j = 0; j < interp->taps; j++) {
    /* Calculate sinc */
    double m = (double) j - (double) (interp->taps - 1) / 2.0;
    double c = 1.0;
    if (fabs(m) > ALMOST_ZERO) {
      c = sin(m * M_PI / interp->factor) / (m * M_PI / interp->factor);
    }
    /* Apply Hanning window */
    c *= 0.5 * (1 - cos(2 * M_PI * j / (interp->taps - 1)));

    if (fabs(c) > ALMOST_ZERO) { /* Ignore any zero coeffs. */
      /* Put the coefficient into the correct subfilter */
      unsigned int f = j % interp->factor;
      unsigned int t = interp->filter[f].count++;
      interp->filter[f].coeff[t] = c;
      interp->filter[f].index[t] = j / interp->factor;
    }
  }
  return interp;

free_filter_z:
  for (j = 0; j < interp->channels; j++) {
    free(interp->z[j]);
  }
  free(interp->z);
free_filter_index_coeff:
  for (j = 0; j < interp->factor; j++) {
    free(interp->filter[j].index);
    free(interp->filter[j].coeff);
  }
  free(interp->filter);
free_interp:
  free(interp);
exit:
  return NULL;
}

void interp_destroy_c(interpolator_c* interp) {
  unsigned int j = 0;
  if (!interp) {
    return;
  }
  for (j = 0; j < interp->factor; j++) {
    free(interp->filter[j].index);
    free(interp->filter[j].coeff);
  }
  free(interp->filter);
  for (j = 0; j < interp->channels; j++) {
    free(interp->z[j]);
  }
  free(interp->z);
  free(interp);
}

size_t
interp_process_c(interpolator_c* interp, size_t frames, float* in, float* out) {
  size_t frame = 0;
  unsigned int chan = 0;
  unsigned int f = 0;
  unsigned int t = 0;
  unsigned int out_stride = interp->channels * interp->factor;
  float* outp = 0;
  double acc = 0;
  double c = 0;

  for (frame = 0; frame < frames; frame++) {
    for (chan = 0; chan < interp->channels; chan++) {
      /* Add sample to delay buffer */
      interp->z[chan][interp->zi] = *in++;
      /* Apply coefficients */
      outp = out + chan;
      for (f = 0; f < interp->factor; f++) {
        acc = 0.0;
        for (t = 0; t < interp->filter[f].count; t++) {
          int i = (int) interp->zi - (int) interp->filter[f].index[t];
          if (i < 0) {
            i += (int) interp->delay;
          }
          c = interp->filter[f].coeff[t];
          acc += (double) interp->z[chan][i] * c;
        }
        *outp = (float) acc;
        outp += interp->channels;
      }
    }
    out += out_stride;
    interp->zi++;
    if (interp->zi == interp->delay) {
      interp->zi = 0;
    }
  }

  return frames * interp->factor;
}
