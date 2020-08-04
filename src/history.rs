use std::collections::VecDeque;

// FIXME: Make const once powf() is a const function
lazy_static::lazy_static! {
    static ref HISTOGRAM_ENERGIES: [f64; 1000] = {
        let mut energies = [0.0; 1000];

        for (i, o) in energies.iter_mut().enumerate() {
            *o = f64::powf(10.0, (i as f64 / 10.0 - 69.95 + 0.691) / 10.0);
        }

        energies
    };

    static ref HISTOGRAM_ENERGY_BOUNDARIES: [f64; 1001] = {
        let mut boundaries = [0.0; 1001];

        for (i, o) in boundaries.iter_mut().enumerate() {
            *o = f64::powf(10.0, (i as f64 / 10.0 - 70.0 + 0.691) / 10.0);
        }

        boundaries
    };
}

fn find_histogram_index(energy: f64) -> usize {
    let mut min = 0;
    let mut max = 1000;

    // Binary search
    loop {
        let mid = (min + max) / 2;
        if energy >= HISTOGRAM_ENERGY_BOUNDARIES[mid] {
            min = mid;
        } else {
            max = mid;
        }

        if max - min == 1 {
            break;
        }
    }

    min
}

pub struct Histogram(Box<[u64; 1000]>);

impl Histogram {
    fn new() -> Self {
        Histogram(Box::new([0; 1000]))
    }

    fn add(&mut self, energy: f64) {
        let idx = find_histogram_index(energy);
        self.0[idx] += 1;
    }

    fn calc_relative_threshold(&self) -> (u64, f64) {
        let mut above_thresh_counter = 0;
        let mut relative_threshold = 0.0;

        for (count, energy) in self.0.iter().zip(HISTOGRAM_ENERGIES.iter()) {
            relative_threshold += *count as f64 * *energy;
            above_thresh_counter += *count;
        }

        (above_thresh_counter, relative_threshold)
    }

    fn loudness_range(&self) -> f64 {
        let mut size = 0;
        let mut power = 0.0;

        for (count, energy) in self.0.iter().zip(HISTOGRAM_ENERGIES.iter()) {
            size += *count;
            power += *count as f64 * *energy;
        }

        if size == 0 {
            return 0.0;
        }

        power /= size as f64;
        let minus_twenty_decibels = f64::powf(10.0, -20.0 / 10.0);
        let integrated = minus_twenty_decibels * power;

        let index = if integrated < HISTOGRAM_ENERGY_BOUNDARIES[0] {
            0
        } else {
            let index = find_histogram_index(integrated);
            if integrated > HISTOGRAM_ENERGIES[index] {
                index + 1
            } else {
                index
            }
        };
        let size = self.0[index..].iter().sum::<u64>();
        if size == 0 {
            return 0.0;
        }

        let percentile_low = ((size - 1) as f64 * 0.1 + 0.5) as u64;
        let percentile_high = ((size - 1) as f64 * 0.95 + 0.5) as u64;

        // TODO: Use an iterator here, maybe something around Iterator::scan()
        let mut j = index;
        let mut size = 0;
        while size <= percentile_low {
            size += self.0[j];
            j += 1;
        }
        let l_en = HISTOGRAM_ENERGIES[j - 1];

        while size <= percentile_high {
            size += self.0[j];
            j += 1;
        }
        let h_en = HISTOGRAM_ENERGIES[j - 1];

        energy_to_loudness(h_en) - energy_to_loudness(l_en)
    }
}

// TODO: Would ideally use a linked-list based queue of fixed-size queues
// to not require a huge contiguous allocation
pub struct Queue {
    queue: VecDeque<f64>,
    max: usize,
}

impl Queue {
    fn new(max: usize) -> Self {
        Queue {
            queue: VecDeque::with_capacity(std::cmp::min(max, 5000)),
            max,
        }
    }

    fn add(&mut self, energy: f64) {
        // Remove last element to keep the size
        if self.max == self.queue.len() {
            self.queue.pop_front();
        }
        self.queue.push_back(energy);
    }

    fn set_max_size(&mut self, max: usize) {
        if self.queue.len() < max {
            // FIXME: Use shrink() once stabilized
            self.queue.resize(max, 0.0);
            self.queue.shrink_to_fit();
        }
        self.max = max;
    }

    fn calc_relative_threshold(&self) -> (u64, f64) {
        (self.queue.len() as u64, self.queue.iter().sum::<f64>())
    }

    fn loudness_range(&self) -> f64 {
        if self.queue.is_empty() {
            return 0.0;
        }

        let (v1, v2) = self.queue.as_slices();
        let mut vec = Vec::with_capacity(self.queue.len());
        vec.extend_from_slice(v1);
        vec.extend_from_slice(v2);

        vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let power = vec.iter().sum::<f64>() / vec.len() as f64;
        let minus_twenty_decibels = f64::powf(10.0, -20.0 / 10.0);
        let integrated = minus_twenty_decibels * power;

        // TODO: Use iterators here or otherwise get rid of bounds checks
        let mut relgated = 0;
        let mut relgated_size = vec.len();
        while relgated_size > 0 && vec[relgated] < integrated {
            relgated += 1;
            relgated_size -= 1;
        }

        if relgated_size > 0 {
            let h_en = vec[relgated + ((relgated_size - 1) as f64 * 0.95 + 0.5) as usize];
            let l_en = vec[relgated + ((relgated_size - 1) as f64 * 0.1 + 0.5) as usize];

            energy_to_loudness(h_en) - energy_to_loudness(l_en)
        } else {
            0.0
        }
    }
}

pub enum History {
    Queue(Queue),
    Histogram(Histogram),
}

impl History {
    pub fn new(use_histogram: bool, max: usize) -> Self {
        if use_histogram {
            History::Histogram(Histogram::new())
        } else {
            History::Queue(Queue::new(max))
        }
    }

    pub fn add(&mut self, energy: f64) {
        if energy < HISTOGRAM_ENERGY_BOUNDARIES[0] {
            return;
        }

        match self {
            History::Histogram(ref mut h) => h.add(energy),
            History::Queue(ref mut q) => q.add(energy),
        }
    }

    pub fn set_max_size(&mut self, max: usize) {
        match self {
            History::Histogram(_) => (),
            History::Queue(ref mut q) => q.set_max_size(max),
        }
    }

    fn calc_relative_threshold(&self) -> (u64, f64) {
        match self {
            History::Histogram(ref h) => h.calc_relative_threshold(),
            History::Queue(ref q) => q.calc_relative_threshold(),
        }
    }

    pub fn gated_loudness(&self) -> f64 {
        let (above_thresh_counter, relative_threshold) = self.calc_relative_threshold();

        if above_thresh_counter == 0 {
            return std::f64::MIN;
        }

        let relative_gate = -10.0;
        let relative_gate_factor = f64::powf(10.0, relative_gate / 10.0);
        let relative_threshold =
            (relative_threshold / above_thresh_counter as f64) * relative_gate_factor;

        let (above_thresh_counter, gated_loudness) = match self {
            History::Histogram(ref h) => {
                let start_index = if relative_threshold < HISTOGRAM_ENERGY_BOUNDARIES[0] {
                    0
                } else {
                    let start_index = find_histogram_index(relative_threshold);
                    if relative_threshold > HISTOGRAM_ENERGIES[start_index] {
                        start_index + 1
                    } else {
                        start_index
                    }
                };

                let mut above_thresh_counter = 0;
                let mut gated_loudness = 0.0;
                for (count, energy) in h.0[start_index..]
                    .iter()
                    .zip(HISTOGRAM_ENERGIES[start_index..].iter())
                {
                    gated_loudness += *count as f64 * *energy;
                    above_thresh_counter += *count;
                }

                (above_thresh_counter, gated_loudness)
            }
            History::Queue(ref q) => {
                let mut above_thresh_counter = 0;
                let mut gated_loudness = 0.0;

                for v in q.queue.iter() {
                    if *v >= relative_threshold {
                        above_thresh_counter += 1;
                        gated_loudness += *v;
                    }
                }

                (above_thresh_counter, gated_loudness)
            }
        };

        if above_thresh_counter == 0 {
            return std::f64::MIN;
        }

        energy_to_loudness(gated_loudness / above_thresh_counter as f64)
    }

    pub fn relative_threshold(&self) -> f64 {
        let (above_thresh_counter, relative_threshold) = self.calc_relative_threshold();

        if above_thresh_counter == 0 {
            return -70.0;
        }

        let relative_gate = -10.0;
        let relative_gate_factor = f64::powf(10.0, relative_gate / 10.0);
        let relative_threshold =
            (relative_threshold / above_thresh_counter as f64) * relative_gate_factor;

        energy_to_loudness(relative_threshold)
    }

    pub fn loudness_range(&self) -> f64 {
        match self {
            History::Histogram(ref h) => h.loudness_range(),
            History::Queue(ref q) => q.loudness_range(),
        }
    }
}

fn energy_to_loudness(energy: f64) -> f64 {
    // The non-test version is faster and more accurate but gives
    // slightly different results than the C version and fails the
    // tests...
    #[cfg(feature = "internal-tests")]
    {
        10.0 * (f64::ln(energy) / f64::ln(10.0)) - 0.691
    }
    #[cfg(not(feature = "internal-tests"))]
    {
        10.0 * f64::ln(energy) - 0.691
    }
}

#[no_mangle]
pub unsafe extern "C" fn history_create(use_histogram: i32, max: usize) -> *mut History {
    Box::into_raw(Box::new(History::new(use_histogram != 0, max)))
}

#[no_mangle]
pub unsafe extern "C" fn history_add(history: *mut History, energy: f64) {
    let history = &mut *history;
    history.add(energy);
}

#[no_mangle]
pub unsafe extern "C" fn history_set_max_size(history: *mut History, max: usize) {
    let history = &mut *history;
    history.set_max_size(max);
}

#[no_mangle]
pub unsafe extern "C" fn history_gated_loudness(history: *const History) -> f64 {
    let history = &*history;
    history.gated_loudness()
}

#[no_mangle]
pub unsafe extern "C" fn history_relative_threshold(history: *const History) -> f64 {
    let history = &*history;
    history.relative_threshold()
}

#[no_mangle]
pub unsafe extern "C" fn history_loudness_range(history: *const History) -> f64 {
    let history = &*history;
    history.loudness_range()
}

#[no_mangle]
pub unsafe extern "C" fn history_destroy(history: *mut History) {
    drop(Box::from_raw(history));
}

#[cfg(feature = "internal-tests")]
use std::os::raw::c_void;

#[cfg(feature = "internal-tests")]
extern "C" {
    pub fn history_init_c();

    pub fn history_create_c(use_histogram: i32, max: usize) -> *mut c_void;

    pub fn history_add_c(history: *mut c_void, energy: f64);

    pub fn history_set_max_size_c(history: *mut c_void, max: usize);

    pub fn history_gated_loudness_c(history: *const c_void) -> f64;

    pub fn history_relative_threshold_c(history: *const c_void) -> f64;

    pub fn history_loudness_range_c(history: *const c_void) -> f64;

    pub fn history_destroy_c(history: *mut c_void);
}

#[cfg(all(test, feature = "internal-tests"))]
mod tests {
    use super::*;
    use float_cmp::approx_eq;
    use quickcheck_macros::quickcheck;
    use std::num::NonZeroU16;

    #[derive(Clone, Copy, Debug)]
    struct Energy(f64);

    impl quickcheck::Arbitrary for Energy {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            use rand::Rng;

            Energy(g.gen_range(-5.0, 1200.0))
        }
    }

    fn init() {
        use std::sync::Once;

        static START: Once = Once::new();

        START.call_once(|| unsafe { history_init_c() });
    }

    #[quickcheck]
    fn compare_c_impl_gated_loudness(
        energy: Vec<Energy>,
        use_histogram: bool,
        max: NonZeroU16,
    ) -> Result<(), String> {
        init();

        let mut hist = History::new(use_histogram, max.get() as usize);
        for e in &energy {
            hist.add(e.0);
        }

        let val = hist.gated_loudness();

        let val_c = unsafe {
            let hist_c = history_create_c(if use_histogram { 1 } else { 0 }, max.get() as usize);
            for e in &energy {
                history_add_c(hist_c, e.0);
            }

            let val = history_gated_loudness_c(hist_c);
            history_destroy_c(hist_c);
            val
        };

        if !approx_eq!(f64, val, val_c, ulps = 2) {
            Err(format!("{} != {}", val, val_c))
        } else {
            Ok(())
        }
    }

    #[quickcheck]
    fn compare_c_impl_relative_threshold(
        energy: Vec<Energy>,
        use_histogram: bool,
        max: NonZeroU16,
    ) -> Result<(), String> {
        init();

        let mut hist = History::new(use_histogram, max.get() as usize);
        for e in &energy {
            hist.add(e.0);
        }

        let val = hist.relative_threshold();

        let val_c = unsafe {
            let hist_c = history_create_c(if use_histogram { 1 } else { 0 }, max.get() as usize);
            for e in &energy {
                history_add_c(hist_c, e.0);
            }

            let val = history_relative_threshold_c(hist_c);
            history_destroy_c(hist_c);
            val
        };

        if !approx_eq!(f64, val, val_c, ulps = 2) {
            Err(format!("{} != {}", val, val_c))
        } else {
            Ok(())
        }
    }

    #[quickcheck]
    fn compare_c_impl_loudness_range(
        energy: Vec<Energy>,
        use_histogram: bool,
        max: NonZeroU16,
    ) -> Result<(), String> {
        init();

        let mut hist = History::new(use_histogram, max.get() as usize);
        for e in &energy {
            hist.add(e.0);
        }

        let val = hist.loudness_range();

        let val_c = unsafe {
            let hist_c = history_create_c(if use_histogram { 1 } else { 0 }, max.get() as usize);
            for e in &energy {
                history_add_c(hist_c, e.0);
            }

            let val = history_loudness_range_c(hist_c);
            history_destroy_c(hist_c);
            val
        };

        if !approx_eq!(f64, val, val_c, ulps = 2) {
            Err(format!("{} != {}", val, val_c))
        } else {
            Ok(())
        }
    }
}
