/* See COPYING file for copyright and license details. */

#include "ebur128.h"

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
#define EBUR128_MAX(a, b) (((a) > (b)) ? (a) : (b))

#define ALMOST_ZERO 0.000001

/* Rust implementations */
typedef void * filter;
extern filter* filter_create(unsigned int rate, unsigned int channels, int calculate_sample_peak, int calculate_true_peak);
extern void filter_destroy(filter* f);
extern void filter_process_short(filter* f, size_t frames, const int16_t* src, double* dest, const int* channel_map);
extern void filter_process_int(filter* f, size_t frames, const int32_t* src, double* dest, const int* channel_map);
extern void filter_process_float(filter* f, size_t frames, const float* src, double* dest, const int* channel_map);
extern void filter_process_double(filter* f, size_t frames, const double* src, double* dest, const int* channel_map);
extern const double* filter_sample_peak(const filter *f);
extern const double* filter_true_peak(const filter *f);
extern void filter_reset_peaks(filter *f);

typedef void * history;
extern history* history_create(int use_histogram, size_t max);
extern void history_add(history* hist, double energy);
extern void history_set_max_size(history* hist, size_t max);
extern double history_gated_loudness(const history* hist);
extern double history_relative_threshold(const history* hist);
extern double history_loudness_range(const history* hist);
extern void history_destroy(history *hist);

struct ebur128_state_internal {
  /** Filtered audio data (used as ring buffer). */
  double* audio_data;
  /** Size of audio_data array. */
  size_t audio_data_frames;
  /** Current index for audio_data. */
  size_t audio_data_index;
  /** How many frames are needed for a gating block. Will correspond to 400ms
   *  of audio at initialization, and 100ms after the first block (75% overlap
   *  as specified in the 2011 revision of BS1770). */
  unsigned long needed_frames;
  /** The channel map. Has as many elements as there are channels. */
  int* channel_map;
  /** How many samples fit in 100ms (rounded). */
  unsigned long samples_in_100ms;

  filter* f;
  history* block_energy_history;
  history* short_term_block_energy_history;

  /** Keeps track of when a new short term block is needed. */
  size_t short_term_frame_counter;
  /** Maximum sample peak, one per channel */
  double* sample_peak;
  /** Maximum true peak, one per channel */
  double* true_peak;
  /** The maximum window duration in ms. */
  unsigned long window;
  unsigned long history;
};

static double relative_gate = -10.0;

/* Those will be calculated when initializing the library */
static double relative_gate_factor;
static double minus_twenty_decibels;

static int ebur128_init_channel_map(ebur128_state* st) {
  size_t i;
  st->d->channel_map = (int*) malloc(st->channels * sizeof(int));
  if (!st->d->channel_map) {
    return EBUR128_ERROR_NOMEM;
  }
  if (st->channels == 4) {
    st->d->channel_map[0] = EBUR128_LEFT;
    st->d->channel_map[1] = EBUR128_RIGHT;
    st->d->channel_map[2] = EBUR128_LEFT_SURROUND;
    st->d->channel_map[3] = EBUR128_RIGHT_SURROUND;
  } else if (st->channels == 5) {
    st->d->channel_map[0] = EBUR128_LEFT;
    st->d->channel_map[1] = EBUR128_RIGHT;
    st->d->channel_map[2] = EBUR128_CENTER;
    st->d->channel_map[3] = EBUR128_LEFT_SURROUND;
    st->d->channel_map[4] = EBUR128_RIGHT_SURROUND;
  } else {
    for (i = 0; i < st->channels; ++i) {
      switch (i) {
        case 0: st->d->channel_map[i] = EBUR128_LEFT; break;
        case 1: st->d->channel_map[i] = EBUR128_RIGHT; break;
        case 2: st->d->channel_map[i] = EBUR128_CENTER; break;
        case 3: st->d->channel_map[i] = EBUR128_UNUSED; break;
        case 4: st->d->channel_map[i] = EBUR128_LEFT_SURROUND; break;
        case 5: st->d->channel_map[i] = EBUR128_RIGHT_SURROUND; break;
        default: st->d->channel_map[i] = EBUR128_UNUSED; break;
      }
    }
  }
  return EBUR128_SUCCESS;
}

void ebur128_get_version(int* major, int* minor, int* patch) {
  *major = EBUR128_VERSION_MAJOR;
  *minor = EBUR128_VERSION_MINOR;
  *patch = EBUR128_VERSION_PATCH;
}

#define VALIDATE_MAX_CHANNELS (64)
#define VALIDATE_MAX_SAMPLERATE (2822400)
#define VALIDATE_MAX_WINDOW                                                    \
  ((3ul << 30) / VALIDATE_MAX_SAMPLERATE / VALIDATE_MAX_CHANNELS /             \
   sizeof(double))

#define VALIDATE_CHANNELS_AND_SAMPLERATE(err)                                  \
  do {                                                                         \
    if (channels == 0 || channels > VALIDATE_MAX_CHANNELS) {                   \
      return (err);                                                            \
    }                                                                          \
                                                                               \
    if (samplerate < 16 || samplerate > VALIDATE_MAX_SAMPLERATE) {             \
      return (err);                                                            \
    }                                                                          \
  } while (0);

void
ebur128_libinit(void) {
  /* initialize static constants */
  relative_gate_factor = pow(10.0, relative_gate / 10.0);
  minus_twenty_decibels = pow(10.0, -20.0 / 10.0);
}

ebur128_state*
ebur128_init(unsigned int channels, unsigned long samplerate, int mode) {
  int errcode;
  ebur128_state* st;
  unsigned int i;
  size_t j;

  VALIDATE_CHANNELS_AND_SAMPLERATE(NULL);

  st = (ebur128_state*) malloc(sizeof(ebur128_state));
  CHECK_ERROR(!st, 0, exit)
  st->d = (struct ebur128_state_internal*) malloc(
      sizeof(struct ebur128_state_internal));
  CHECK_ERROR(!st->d, 0, free_state)
  st->channels = channels;
  errcode = ebur128_init_channel_map(st);
  CHECK_ERROR(errcode, 0, free_internal)

  st->d->sample_peak = (double*) malloc(channels * sizeof(double));
  CHECK_ERROR(!st->d->sample_peak, 0, free_channel_map)
  st->d->true_peak = (double*) malloc(channels * sizeof(double));
  CHECK_ERROR(!st->d->true_peak, 0, free_sample_peak)
  for (i = 0; i < channels; ++i) {
    st->d->sample_peak[i] = 0.0;
    st->d->true_peak[i] = 0.0;
  }

  st->d->history = ULONG_MAX;
  st->samplerate = samplerate;
  st->d->samples_in_100ms = (st->samplerate + 5) / 10;
  st->mode = mode;
  if ((mode & EBUR128_MODE_S) == EBUR128_MODE_S) {
    st->d->window = 3000;
  } else if ((mode & EBUR128_MODE_M) == EBUR128_MODE_M) {
    st->d->window = 400;
  } else {
    goto free_true_peak;
  }
  st->d->audio_data_frames = st->samplerate * st->d->window / 1000;
  if (st->d->audio_data_frames % st->d->samples_in_100ms) {
    /* round up to multiple of samples_in_100ms */
    st->d->audio_data_frames =
        (st->d->audio_data_frames + st->d->samples_in_100ms) -
        (st->d->audio_data_frames % st->d->samples_in_100ms);
  }
  st->d->audio_data = (double*) malloc(st->d->audio_data_frames * st->channels *
                                       sizeof(double));
  CHECK_ERROR(!st->d->audio_data, 0, free_true_peak)
  for (j = 0; j < st->d->audio_data_frames * st->channels; ++j) {
    st->d->audio_data[j] = 0.0;
  }

  st->d->block_energy_history = history_create(mode & EBUR128_MODE_HISTOGRAM ? 1 : 0, st->d->history / 100);
  st->d->short_term_block_energy_history = history_create(mode & EBUR128_MODE_HISTOGRAM ? 1 : 0, st->d->history / 3000);
  st->d->short_term_frame_counter = 0;

  st->d->f = filter_create(st->samplerate, st->channels, (st->mode & EBUR128_MODE_SAMPLE_PEAK) == EBUR128_MODE_SAMPLE_PEAK, (st->mode & EBUR128_MODE_TRUE_PEAK) == EBUR128_MODE_TRUE_PEAK);

  /* the first block needs 400ms of audio data */
  st->d->needed_frames = st->d->samples_in_100ms * 4;
  /* start at the beginning of the buffer */
  st->d->audio_data_index = 0;

  return st;

free_true_peak:
  free(st->d->true_peak);
free_sample_peak:
  free(st->d->sample_peak);
free_channel_map:
  free(st->d->channel_map);
free_internal:
  free(st->d);
free_state:
  free(st);
exit:
  return NULL;
}

void ebur128_destroy(ebur128_state** st) {
  history_destroy((*st)->d->short_term_block_energy_history);
  history_destroy((*st)->d->block_energy_history);
  filter_destroy((*st)->d->f);
  free((*st)->d->audio_data);
  free((*st)->d->channel_map);
  free((*st)->d->sample_peak);
  free((*st)->d->true_peak);
  free((*st)->d);
  free(*st);
  *st = NULL;
}

static double ebur128_energy_to_loudness(double energy) {
  return 10 * (log(energy) / log(10.0)) - 0.691;
}

static int ebur128_calc_gating_block(ebur128_state* st,
                                     size_t frames_per_block,
                                     double* optional_output) {
  size_t i, c;
  double sum = 0.0;
  double channel_sum;
  for (c = 0; c < st->channels; ++c) {
    if (st->d->channel_map[c] == EBUR128_UNUSED) {
      continue;
    }
    channel_sum = 0.0;
    if (st->d->audio_data_index < frames_per_block * st->channels) {
      for (i = 0; i < st->d->audio_data_index / st->channels; ++i) {
        channel_sum += st->d->audio_data[i * st->channels + c] *
                       st->d->audio_data[i * st->channels + c];
      }
      for (i = st->d->audio_data_frames -
               (frames_per_block - st->d->audio_data_index / st->channels);
           i < st->d->audio_data_frames; ++i) {
        channel_sum += st->d->audio_data[i * st->channels + c] *
                       st->d->audio_data[i * st->channels + c];
      }
    } else {
      for (i = st->d->audio_data_index / st->channels - frames_per_block;
           i < st->d->audio_data_index / st->channels; ++i) {
        channel_sum += st->d->audio_data[i * st->channels + c] *
                       st->d->audio_data[i * st->channels + c];
      }
    }
    if (st->d->channel_map[c] == EBUR128_Mp110 ||
        st->d->channel_map[c] == EBUR128_Mm110 ||
        st->d->channel_map[c] == EBUR128_Mp060 ||
        st->d->channel_map[c] == EBUR128_Mm060 ||
        st->d->channel_map[c] == EBUR128_Mp090 ||
        st->d->channel_map[c] == EBUR128_Mm090) {
      channel_sum *= 1.41;
    } else if (st->d->channel_map[c] == EBUR128_DUAL_MONO) {
      channel_sum *= 2.0;
    }
    sum += channel_sum;
  }

  sum /= (double) frames_per_block;

  if (optional_output) {
    *optional_output = sum;
    return EBUR128_SUCCESS;
  }

  history_add(st->d->block_energy_history, sum);

  return EBUR128_SUCCESS;
}

int ebur128_set_channel(ebur128_state* st,
                        unsigned int channel_number,
                        int value) {
  if (channel_number >= st->channels) {
    return EBUR128_ERROR_INVALID_CHANNEL_INDEX;
  }
  if (value == EBUR128_DUAL_MONO &&
      (st->channels != 1 || channel_number != 0)) {
    fprintf(stderr, "EBUR128_DUAL_MONO only works with mono files!\n");
    return EBUR128_ERROR_INVALID_CHANNEL_INDEX;
  }
  st->d->channel_map[channel_number] = value;
  return EBUR128_SUCCESS;
}

int ebur128_change_parameters(ebur128_state* st,
                              unsigned int channels,
                              unsigned long samplerate) {
  int errcode = EBUR128_SUCCESS;
  size_t j;

  /* This is needed to suppress a clang-tidy warning. */
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif
#if __has_builtin(__builtin_unreachable)
  if (st->channels == 0) {
    __builtin_unreachable();
  }
#endif

  VALIDATE_CHANNELS_AND_SAMPLERATE(EBUR128_ERROR_NOMEM);

  if (channels == st->channels && samplerate == st->samplerate) {
    return EBUR128_ERROR_NO_CHANGE;
  }

  free(st->d->audio_data);
  st->d->audio_data = NULL;

  if (channels != st->channels) {
    unsigned int i;

    free(st->d->channel_map);
    st->d->channel_map = NULL;
    free(st->d->sample_peak);
    st->d->sample_peak = NULL;
    free(st->d->true_peak);
    st->d->true_peak = NULL;
    st->channels = channels;

    errcode = ebur128_init_channel_map(st);
    CHECK_ERROR(errcode, EBUR128_ERROR_NOMEM, exit)

    st->d->sample_peak = (double*) malloc(channels * sizeof(double));
    CHECK_ERROR(!st->d->sample_peak, EBUR128_ERROR_NOMEM, exit)
    st->d->true_peak = (double*) malloc(channels * sizeof(double));
    CHECK_ERROR(!st->d->true_peak, EBUR128_ERROR_NOMEM, exit)
    for (i = 0; i < channels; ++i) {
      st->d->sample_peak[i] = 0.0;
      st->d->true_peak[i] = 0.0;
    }
  }
  if (samplerate != st->samplerate) {
    st->samplerate = samplerate;
    st->d->samples_in_100ms = (st->samplerate + 5) / 10;
  }

  /* If we're here, either samplerate or channels
   * have changed. Re-init filter. */
  filter_destroy(st->d->f);
  st->d->f = filter_create(st->samplerate, st->channels, (st->mode & EBUR128_MODE_SAMPLE_PEAK) == EBUR128_MODE_SAMPLE_PEAK, (st->mode & EBUR128_MODE_TRUE_PEAK) == EBUR128_MODE_TRUE_PEAK);

  st->d->audio_data_frames = st->samplerate * st->d->window / 1000;
  if (st->d->audio_data_frames % st->d->samples_in_100ms) {
    /* round up to multiple of samples_in_100ms */
    st->d->audio_data_frames =
        (st->d->audio_data_frames + st->d->samples_in_100ms) -
        (st->d->audio_data_frames % st->d->samples_in_100ms);
  }
  st->d->audio_data = (double*) malloc(st->d->audio_data_frames * st->channels *
                                       sizeof(double));
  CHECK_ERROR(!st->d->audio_data, EBUR128_ERROR_NOMEM, exit)
  for (j = 0; j < st->d->audio_data_frames * st->channels; ++j) {
    st->d->audio_data[j] = 0.0;
  }

  /* the first block needs 400ms of audio data */
  st->d->needed_frames = st->d->samples_in_100ms * 4;
  /* start at the beginning of the buffer */
  st->d->audio_data_index = 0;
  /* reset short term frame counter */
  st->d->short_term_frame_counter = 0;

exit:
  return errcode;
}

int ebur128_set_max_window(ebur128_state* st, unsigned long window) {
  int errcode = EBUR128_SUCCESS;
  size_t j;

  if ((st->mode & EBUR128_MODE_S) == EBUR128_MODE_S && window < 3000) {
    window = 3000;
  } else if ((st->mode & EBUR128_MODE_M) == EBUR128_MODE_M && window < 400) {
    window = 400;
  }

  if (window == st->d->window) {
    return EBUR128_ERROR_NO_CHANGE;
  }

  if (window >= VALIDATE_MAX_WINDOW) {
    return EBUR128_ERROR_NOMEM;
  }

  st->d->window = window;
  free(st->d->audio_data);
  st->d->audio_data = NULL;
  st->d->audio_data_frames = st->samplerate * st->d->window / 1000;
  if (st->d->audio_data_frames % st->d->samples_in_100ms) {
    /* round up to multiple of samples_in_100ms */
    st->d->audio_data_frames =
        (st->d->audio_data_frames + st->d->samples_in_100ms) -
        (st->d->audio_data_frames % st->d->samples_in_100ms);
  }
  st->d->audio_data = (double*) malloc(st->d->audio_data_frames * st->channels *
                                       sizeof(double));
  CHECK_ERROR(!st->d->audio_data, EBUR128_ERROR_NOMEM, exit)
  for (j = 0; j < st->d->audio_data_frames * st->channels; ++j) {
    st->d->audio_data[j] = 0.0;
  }

  /* the first block needs 400ms of audio data */
  st->d->needed_frames = st->d->samples_in_100ms * 4;
  /* start at the beginning of the buffer */
  st->d->audio_data_index = 0;
  /* reset short term frame counter */
  st->d->short_term_frame_counter = 0;

exit:
  return errcode;
}

int ebur128_set_max_history(ebur128_state* st, unsigned long history) {
  if ((st->mode & EBUR128_MODE_LRA) == EBUR128_MODE_LRA && history < 3000) {
    history = 3000;
  } else if ((st->mode & EBUR128_MODE_M) == EBUR128_MODE_M && history < 400) {
    history = 400;
  }
  if (history == st->d->history) {
    return EBUR128_ERROR_NO_CHANGE;
  }
  st->d->history = history;

  history_set_max_size(st->d->block_energy_history, st->d->history / 100);
  history_set_max_size(st->d->short_term_block_energy_history, st->d->history / 3000);

  return EBUR128_SUCCESS;
}

static int ebur128_energy_shortterm(ebur128_state* st, double* out);
#define EBUR128_ADD_FRAMES(type)                                               \
  int ebur128_add_frames_##type(ebur128_state* st, const type* src,            \
                                size_t frames) {                               \
    size_t src_index = 0;                                                      \
    unsigned int c = 0;                                                        \
    filter_reset_peaks(st->d->f);                                              \
    while (frames > 0) {                                                       \
      if (frames >= st->d->needed_frames) {                                    \
        filter_process_##type(st->d->f, st->d->needed_frames, src + src_index, st->d->audio_data + st->d->audio_data_index, st->d->channel_map);      \
        src_index += st->d->needed_frames * st->channels;                      \
        frames -= st->d->needed_frames;                                        \
        st->d->audio_data_index += st->d->needed_frames * st->channels;        \
        /* calculate the new gating block */                                   \
        if ((st->mode & EBUR128_MODE_I) == EBUR128_MODE_I) {                   \
          if (ebur128_calc_gating_block(st, st->d->samples_in_100ms * 4,       \
                                        NULL)) {                               \
            return EBUR128_ERROR_NOMEM;                                        \
          }                                                                    \
        }                                                                      \
        if ((st->mode & EBUR128_MODE_LRA) == EBUR128_MODE_LRA) {               \
          st->d->short_term_frame_counter += st->d->needed_frames;             \
          if (st->d->short_term_frame_counter ==                               \
              st->d->samples_in_100ms * 30) {                                  \
            double st_energy;                                                  \
            if (ebur128_energy_shortterm(st, &st_energy) == EBUR128_SUCCESS) { \
              history_add(st->d->short_term_block_energy_history,              \
                    st_energy);                                                \
            }                                                                  \
            st->d->short_term_frame_counter = st->d->samples_in_100ms * 20;    \
          }                                                                    \
        }                                                                      \
        /* 100ms are needed for all blocks besides the first one */            \
        st->d->needed_frames = st->d->samples_in_100ms;                        \
        /* reset audio_data_index when buffer full */                          \
        if (st->d->audio_data_index ==                                         \
            st->d->audio_data_frames * st->channels) {                         \
          st->d->audio_data_index = 0;                                         \
        }                                                                      \
      } else {                                                                 \
        filter_process_##type(st->d->f, frames, src + src_index, st->d->audio_data + st->d->audio_data_index, st->d->channel_map);      \
        st->d->audio_data_index += frames * st->channels;                      \
        if ((st->mode & EBUR128_MODE_LRA) == EBUR128_MODE_LRA) {               \
          st->d->short_term_frame_counter += frames;                           \
        }                                                                      \
        st->d->needed_frames -= (unsigned long) frames;                        \
        frames = 0;                                                            \
      }                                                                        \
    }                                                                          \
    for (c = 0; c < st->channels; c++) {                                       \
      const double *prev_sample_peak = filter_sample_peak(st->d->f);           \
      const double *prev_true_peak = filter_true_peak(st->d->f);               \
      if (prev_sample_peak[c] > st->d->sample_peak[c]) {                       \
        st->d->sample_peak[c] = prev_sample_peak[c];                           \
      }                                                                        \
      if (prev_true_peak[c] > st->d->true_peak[c]) {                          \
        st->d->true_peak[c] = prev_true_peak[c];                               \
      }                                                                        \
    }                                                                          \
    return EBUR128_SUCCESS;                                                    \
  }

EBUR128_ADD_FRAMES(short)
EBUR128_ADD_FRAMES(int)
EBUR128_ADD_FRAMES(float)
EBUR128_ADD_FRAMES(double)

int ebur128_relative_threshold(ebur128_state* st, double* out) {
  if ((st->mode & EBUR128_MODE_I) != EBUR128_MODE_I) {
    return EBUR128_ERROR_INVALID_MODE;
  }

  *out = history_relative_threshold(st->d->block_energy_history);

  return EBUR128_SUCCESS;
}

int ebur128_loudness_global(ebur128_state* st, double* out) {
  if ((st->mode & EBUR128_MODE_I) != EBUR128_MODE_I) {
    return EBUR128_ERROR_INVALID_MODE;
  }

  *out = history_gated_loudness(st->d->block_energy_history);

  return EBUR128_SUCCESS;
}

static int ebur128_energy_in_interval(ebur128_state* st,
                                      size_t interval_frames,
                                      double* out) {
  if (interval_frames > st->d->audio_data_frames) {
    return EBUR128_ERROR_INVALID_MODE;
  }
  ebur128_calc_gating_block(st, interval_frames, out);
  return EBUR128_SUCCESS;
}

static int ebur128_energy_shortterm(ebur128_state* st, double* out) {
  return ebur128_energy_in_interval(st, st->d->samples_in_100ms * 30, out);
}

int ebur128_loudness_momentary(ebur128_state* st, double* out) {
  double energy;
  int error;

  error = ebur128_energy_in_interval(st, st->d->samples_in_100ms * 4, &energy);
  if (error) {
    return error;
  }

  if (energy <= 0.0) {
    *out = -HUGE_VAL;
    return EBUR128_SUCCESS;
  }

  *out = ebur128_energy_to_loudness(energy);
  return EBUR128_SUCCESS;
}

int ebur128_loudness_shortterm(ebur128_state* st, double* out) {
  double energy;
  int error;

  error = ebur128_energy_shortterm(st, &energy);
  if (error) {
    return error;
  }

  if (energy <= 0.0) {
    *out = -HUGE_VAL;
    return EBUR128_SUCCESS;
  }

  *out = ebur128_energy_to_loudness(energy);
  return EBUR128_SUCCESS;
}

int ebur128_loudness_window(ebur128_state* st,
                            unsigned long window,
                            double* out) {
  double energy;
  size_t interval_frames;
  int error;

  if (window >= VALIDATE_MAX_WINDOW) {
    return EBUR128_ERROR_INVALID_MODE;
  }

  interval_frames = st->samplerate * window / 1000;
  error = ebur128_energy_in_interval(st, interval_frames, &energy);
  if (error) {
    return error;
  }

  if (energy <= 0.0) {
    *out = -HUGE_VAL;
    return EBUR128_SUCCESS;
  }

  *out = ebur128_energy_to_loudness(energy);
  return EBUR128_SUCCESS;
}

/* EBU - TECH 3342 */
int ebur128_loudness_range(ebur128_state* st, double* out) {
  if ((st->mode & EBUR128_MODE_LRA) != EBUR128_MODE_LRA) {
    return EBUR128_ERROR_INVALID_MODE;
  }

  *out = history_loudness_range(st->d->short_term_block_energy_history);

  return EBUR128_SUCCESS;
}

int ebur128_sample_peak(ebur128_state* st,
                        unsigned int channel_number,
                        double* out) {
  if ((st->mode & EBUR128_MODE_SAMPLE_PEAK) != EBUR128_MODE_SAMPLE_PEAK) {
    return EBUR128_ERROR_INVALID_MODE;
  }

  if (channel_number >= st->channels) {
    return EBUR128_ERROR_INVALID_CHANNEL_INDEX;
  }

  *out = st->d->sample_peak[channel_number];
  return EBUR128_SUCCESS;
}

int ebur128_prev_sample_peak(ebur128_state* st,
                             unsigned int channel_number,
                             double* out) {
  if ((st->mode & EBUR128_MODE_SAMPLE_PEAK) != EBUR128_MODE_SAMPLE_PEAK) {
    return EBUR128_ERROR_INVALID_MODE;
  }

  if (channel_number >= st->channels) {
    return EBUR128_ERROR_INVALID_CHANNEL_INDEX;
  }

  *out = filter_sample_peak(st->d->f)[channel_number];
  return EBUR128_SUCCESS;
}

int ebur128_true_peak(ebur128_state* st,
                      unsigned int channel_number,
                      double* out) {
  if ((st->mode & EBUR128_MODE_TRUE_PEAK) != EBUR128_MODE_TRUE_PEAK) {
    return EBUR128_ERROR_INVALID_MODE;
  }

  if (channel_number >= st->channels) {
    return EBUR128_ERROR_INVALID_CHANNEL_INDEX;
  }

  *out = EBUR128_MAX(st->d->true_peak[channel_number],
                     st->d->sample_peak[channel_number]);
  return EBUR128_SUCCESS;
}

int ebur128_prev_true_peak(ebur128_state* st,
                           unsigned int channel_number,
                           double* out) {
  if ((st->mode & EBUR128_MODE_TRUE_PEAK) != EBUR128_MODE_TRUE_PEAK) {
    return EBUR128_ERROR_INVALID_MODE;
  }

  if (channel_number >= st->channels) {
    return EBUR128_ERROR_INVALID_CHANNEL_INDEX;
  }

  *out = EBUR128_MAX(filter_true_peak(st->d->f)[channel_number],
                     filter_sample_peak(st->d->f)[channel_number]);
  return EBUR128_SUCCESS;
}
