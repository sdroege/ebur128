#include <float.h>
#include <limits.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double calc_gating_block_c(size_t frames_per_block, const double *audio_data, size_t audio_data_frames, size_t audio_data_index, unsigned int *channel_map, size_t channels) {
  size_t i, c;
  double sum = 0.0;
  double channel_sum;
  for (c = 0; c < channels; ++c) {
    if (channel_map[c] == 0 /* EBUR128_UNUSED */) {
      continue;
    }
    channel_sum = 0.0;
    if (audio_data_index < frames_per_block * channels) {
      for (i = 0; i < audio_data_index / channels; ++i) {
        channel_sum += audio_data[i * channels + c] *
                       audio_data[i * channels + c];
      }
      for (i = audio_data_frames -
               (frames_per_block - audio_data_index / channels);
           i < audio_data_frames; ++i) {
        channel_sum += audio_data[i * channels + c] *
                       audio_data[i * channels + c];
      }
    } else {
      for (i = audio_data_index / channels - frames_per_block;
           i < audio_data_index / channels; ++i) {
        channel_sum += audio_data[i * channels + c] *
                       audio_data[i * channels + c];
      }
    }
    if (channel_map[c] == 4 /* EBUR128_Mp110 */ ||
        channel_map[c] == 5 /* EBUR128_Mm110 */ ||
        channel_map[c] == 9 /* EBUR128_Mp060 */ ||
        channel_map[c] == 10 /* EBUR128_Mm060 */ ||
        channel_map[c] == 11 /* EBUR128_Mp090 */ ||
        channel_map[c] == 12 /* EBUR128_Mm090 */) {
      channel_sum *= 1.41;
    } else if (channel_map[c] == 6 /* EBUR128_DUAL_MONO */) {
      channel_sum *= 2.0;
    }
    sum += channel_sum;
  }

  sum /= (double) frames_per_block;

  return sum;
}
