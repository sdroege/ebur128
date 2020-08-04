#include <float.h>
#include <limits.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "queue.h"

STAILQ_HEAD(ebur128_double_queue, ebur128_dq_entry);
struct ebur128_dq_entry {
  double z;
  STAILQ_ENTRY(ebur128_dq_entry) entries;
};

static double minus_twenty_decibels;
static double relative_gate = -10.0;
static double relative_gate_factor;
static double histogram_energies[1000];
static double histogram_energy_boundaries[1001];

typedef struct {
  int use_histogram;

  struct ebur128_double_queue block_list;
  unsigned long block_list_max;
  unsigned long block_list_size;

  unsigned long* block_energy_histogram;
} history;

static double ebur128_energy_to_loudness(double energy) {
  return 10 * (log(energy) / log(10.0)) - 0.691;
}

static size_t find_histogram_index(double energy) {
  size_t index_min = 0;
  size_t index_max = 1000;
  size_t index_mid;

  do {
    index_mid = (index_min + index_max) / 2;
    if (energy >= histogram_energy_boundaries[index_mid]) {
      index_min = index_mid;
    } else {
      index_max = index_mid;
    }
  } while (index_max - index_min != 1);

  return index_min;
}

void history_init_c(void) {
  int i;

  relative_gate_factor = pow(10.0, relative_gate / 10.0);
  minus_twenty_decibels = pow(10.0, -20.0 / 10.0);

  histogram_energy_boundaries[0] = pow(10.0, (-70.0 + 0.691) / 10.0);
  for (i = 0; i < 1000; ++i) {
    histogram_energies[i] =
        pow(10.0, ((double) i / 10.0 - 69.95 + 0.691) / 10.0);
  }
  for (i = 1; i < 1001; ++i) {
    histogram_energy_boundaries[i] =
        pow(10.0, ((double) i / 10.0 - 70.0 + 0.691) / 10.0);
  }
}

history* history_create_c(int use_histogram, size_t max) {
  int i;
  history *hist = calloc(1, sizeof(history));

  hist->use_histogram = use_histogram;

 if (hist->use_histogram) {
    hist->block_energy_histogram =
        (unsigned long*) malloc(1000 * sizeof(unsigned long));
    for (i = 0; i < 1000; ++i) {
      hist->block_energy_histogram[i] = 0;
    }
  } else {
    hist->block_energy_histogram = NULL;
  }

  STAILQ_INIT(&hist->block_list);
  hist->block_list_size = 0;
  hist->block_list_max = max;

  return hist;
}

void history_add_c(history* hist, double energy) {
  if (energy >= histogram_energy_boundaries[0]) {
    if (hist->use_histogram) {
      ++hist->block_energy_histogram[find_histogram_index(energy)];
    } else {
      struct ebur128_dq_entry* block;
      if (hist->block_list_size == hist->block_list_max) {
        block = STAILQ_FIRST(&hist->block_list);
        STAILQ_REMOVE_HEAD(&hist->block_list, entries);
      } else {
        block =
            (struct ebur128_dq_entry*) malloc(sizeof(struct ebur128_dq_entry));
        hist->block_list_size++;
      }
      block->z = energy;
      STAILQ_INSERT_TAIL(&hist->block_list, block, entries);
    }
  }
}

void history_set_max_size_c(history* hist, size_t max) {
  hist->block_list_max = max;
  while (hist->block_list_size > hist->block_list_max) {
    struct ebur128_dq_entry* block = STAILQ_FIRST(&hist->block_list);
    STAILQ_REMOVE_HEAD(&hist->block_list, entries);
    free(block);
    hist->block_list_size--;
  }
}

static void history_calc_relative_threshold(const history* hist, size_t* above_thresh_counter, double* relative_threshold) {
  struct ebur128_dq_entry* it;
  size_t i;

  if (hist->use_histogram) {
    for (i = 0; i < 1000; ++i) {
      *relative_threshold +=
        hist->block_energy_histogram[i] * histogram_energies[i];
      *above_thresh_counter += hist->block_energy_histogram[i];
    }
  } else {
    STAILQ_FOREACH(it, &hist->block_list, entries) {
      ++*above_thresh_counter;
      *relative_threshold += it->z;
    }
  }
}

double history_gated_loudness_c(const history* hist) {
  struct ebur128_dq_entry* it;
  double gated_loudness = 0.0;
  double relative_threshold = 0.0;
  size_t above_thresh_counter = 0;
  size_t j, start_index;

  history_calc_relative_threshold(hist, &above_thresh_counter, &relative_threshold);
  if (!above_thresh_counter) {
    return -HUGE_VAL;
  }

  relative_threshold /= (double) above_thresh_counter;
  relative_threshold *= relative_gate_factor;

  above_thresh_counter = 0;
  if (relative_threshold < histogram_energy_boundaries[0]) {
    start_index = 0;
  } else {
    start_index = find_histogram_index(relative_threshold);
    if (relative_threshold > histogram_energies[start_index]) {
      ++start_index;
    }
  }

  if (hist->use_histogram) {
    for (j = start_index; j < 1000; ++j) {
      gated_loudness +=
          hist->block_energy_histogram[j] * histogram_energies[j];
      above_thresh_counter += hist->block_energy_histogram[j];
    }
  } else {
    STAILQ_FOREACH(it, &hist->block_list, entries) {
      if (it->z >= relative_threshold) {
        ++above_thresh_counter;
        gated_loudness += it->z;
      }
    }
  }

  if (!above_thresh_counter) {
    return -HUGE_VAL;
  }
  gated_loudness /= (double) above_thresh_counter;

  return ebur128_energy_to_loudness(gated_loudness);
}

double history_relative_threshold_c(const history* hist) {
  double relative_threshold = 0.0;
  size_t above_thresh_counter = 0;

  history_calc_relative_threshold(hist, &above_thresh_counter,
                                  &relative_threshold);

  if (!above_thresh_counter) {
    return -70.0;
  }

  relative_threshold /= (double) above_thresh_counter;
  relative_threshold *= relative_gate_factor;

  return ebur128_energy_to_loudness(relative_threshold);
}

static int history_double_cmp(const void* p1, const void* p2) {
  const double* d1 = (const double*) p1;
  const double* d2 = (const double*) p2;
  return (*d1 > *d2) - (*d1 < *d2);
}

double history_loudness_range_c(const history* hist) {
  size_t j;
  struct ebur128_dq_entry* it;
  double* stl_vector;
  size_t stl_size;
  double* stl_relgated;
  size_t stl_relgated_size;
  double stl_power, stl_integrated;
  /* High and low percentile energy */
  double h_en, l_en;

  if (hist->use_histogram) {
    unsigned long hist_[1000] = { 0 };
    size_t percentile_low, percentile_high;
    size_t index;

    stl_size = 0;
    stl_power = 0.0;
    for (j = 0; j < 1000; ++j) {
      hist_[j] += hist->block_energy_histogram[j];
      stl_size += hist->block_energy_histogram[j];
      stl_power += hist->block_energy_histogram[j] *
                   histogram_energies[j];
    }
    if (!stl_size) {
      return 0.0;
    }

    stl_power /= stl_size;
    stl_integrated = minus_twenty_decibels * stl_power;

    if (stl_integrated < histogram_energy_boundaries[0]) {
      index = 0;
    } else {
      index = find_histogram_index(stl_integrated);
      if (stl_integrated > histogram_energies[index]) {
        ++index;
      }
    }
    stl_size = 0;
    for (j = index; j < 1000; ++j) {
      stl_size += hist_[j];
    }
    if (!stl_size) {
      return 0.0;
    }

    percentile_low = (size_t)((stl_size - 1) * 0.1 + 0.5);
    percentile_high = (size_t)((stl_size - 1) * 0.95 + 0.5);

    stl_size = 0;
    j = index;
    while (stl_size <= percentile_low) {
      stl_size += hist_[j++];
    }
    l_en = histogram_energies[j - 1];
    while (stl_size <= percentile_high) {
      stl_size += hist_[j++];
    }
    h_en = histogram_energies[j - 1];

    return ebur128_energy_to_loudness(h_en) - ebur128_energy_to_loudness(l_en);
  }

  stl_size = 0;
    STAILQ_FOREACH(it, &hist->block_list, entries) {
      ++stl_size;
    }
  if (!stl_size) {
    return 0.0;
  }
  stl_vector = (double*) malloc(stl_size * sizeof(double));

  j = 0;
    STAILQ_FOREACH(it, &hist->block_list, entries) {
      stl_vector[j] = it->z;
      ++j;
    }
  qsort(stl_vector, stl_size, sizeof(double), history_double_cmp);
  stl_power = 0.0;
  for (j = 0; j < stl_size; ++j) {
    stl_power += stl_vector[j];
  }
  stl_power /= (double) stl_size;
  stl_integrated = minus_twenty_decibels * stl_power;

  stl_relgated = stl_vector;
  stl_relgated_size = stl_size;
  while (stl_relgated_size > 0 && *stl_relgated < stl_integrated) {
    ++stl_relgated;
    --stl_relgated_size;
  }

  if (stl_relgated_size) {
    h_en = stl_relgated[(size_t)((stl_relgated_size - 1) * 0.95 + 0.5)];
    l_en = stl_relgated[(size_t)((stl_relgated_size - 1) * 0.1 + 0.5)];
    free(stl_vector);
    return ebur128_energy_to_loudness(h_en) - ebur128_energy_to_loudness(l_en);
  } else {
    free(stl_vector);
    return 0.0;
  }
}

void history_destroy_c(history *hist) {
  struct ebur128_dq_entry* entry;
  free(hist->block_energy_histogram);
  while (!STAILQ_EMPTY(&hist->block_list)) {
    entry = STAILQ_FIRST(&hist->block_list);
    STAILQ_REMOVE_HEAD(&hist->block_list, entries);
    free(entry);
  }
  free(hist);
}
