// Copyright (c) 2011 Jan Kokemüller
// Copyright (c) 2020 Sebastian Dröge <sebastian@centricular.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

use crate::interp::Interp;

#[derive(Debug)]
pub struct TruePeak {
    interp: Interp,
    interp_factor: usize,
    rate: u32,
    channels: u32,
    buffer_input: Vec<f32>,
    buffer_output: Vec<f32>,
}

impl TruePeak {
    pub fn new(rate: u32, channels: u32) -> Option<Self> {
        let samples_in_100ms = (rate + 5) / 10;

        let (interp, interp_factor) = if rate < 96_000 {
            (Interp::new(49, 4, channels), 4)
        } else if rate < 192_000 {
            (Interp::new(49, 2, channels), 2)
        } else {
            return None;
        };

        let buffer_input = vec![0.0; 4 * samples_in_100ms as usize * channels as usize];
        let buffer_output = vec![0.0; buffer_input.len() * interp_factor];

        Some(Self {
            interp,
            interp_factor,
            rate,
            channels,
            buffer_input,
            buffer_output,
        })
    }

    // FIXME: Use f32 for storage
    pub fn check_true_peak<T: AsF32>(&mut self, src: &[T], peaks: &mut [f64]) {
        assert!(src.len() <= self.buffer_input.len());
        assert!(peaks.len() == self.channels as usize);

        if src.is_empty() {
            return;
        }

        // Deinterleave and convert to f32 for the resampler
        let frames = src.len() / self.channels as usize;
        for (s, i) in src.chunks_exact(self.channels as usize).enumerate() {
            for (c, i) in i.iter().enumerate() {
                self.buffer_input[frames * c + s] = i.as_f32();
            }
        }

        self.interp.process(
            &self.buffer_input[..(src.len())],
            &mut self.buffer_output[..(src.len() * self.interp_factor)],
        );

        // Find the maximum
        for (c, o) in self.buffer_output[..(frames * self.channels as usize * self.interp_factor)]
            .chunks_exact(frames * self.interp_factor)
            .enumerate()
        {
            let mut max = 0.0;
            for v in o {
                max = f32_max(max, v.abs());
            }
            peaks[c] = f64_max(max as f64, peaks[c]);
        }
    }
}

fn f32_max(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

fn f64_max(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}

pub trait AsF32: Copy {
    fn as_f32(self) -> f32;
}

impl AsF32 for i16 {
    fn as_f32(self) -> f32 {
        self as f32 / (-(std::i16::MIN as f32))
    }
}

impl AsF32 for i32 {
    fn as_f32(self) -> f32 {
        self as f32 / (-(std::i32::MIN as f32))
    }
}

impl AsF32 for f32 {
    fn as_f32(self) -> f32 {
        self
    }
}

impl AsF32 for f64 {
    fn as_f32(self) -> f32 {
        self as f32
    }
}

#[cfg(feature = "c-tests")]
use std::os::raw::c_void;

#[cfg(feature = "c-tests")]
extern "C" {
    pub fn true_peak_create_c(rate: u32, channels: u32) -> *mut c_void;
    pub fn true_peak_check_short_c(
        tp: *mut c_void,
        frames: usize,
        src: *const i16,
        peaks: *mut f64,
    );
    pub fn true_peak_check_int_c(tp: *mut c_void, frames: usize, src: *const i32, peaks: *mut f64);
    pub fn true_peak_check_float_c(
        tp: *mut c_void,
        frames: usize,
        src: *const f32,
        peaks: *mut f64,
    );
    pub fn true_peak_check_double_c(
        tp: *mut c_void,
        frames: usize,
        src: *const f64,
        peaks: *mut f64,
    );
    pub fn true_peak_destroy_c(tp: *mut c_void);
}

#[cfg(feature = "c-tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::Signal;
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn compare_c_impl_i16(signal: Signal<i16>) -> quickcheck::TestResult {
        use float_cmp::approx_eq;

        if signal.rate >= 192_000 {
            return quickcheck::TestResult::discard();
        }

        // Maximum of 400ms but our input is up to 5000ms, so distribute it evenly
        // by shrinking accordingly.
        let len = signal.data.len() / signal.channels as usize;
        let len = std::cmp::min(2 * len / 25, 4 * ((signal.rate as usize + 5) / 10));

        let mut peaks = vec![0.0f64; signal.channels as usize];
        let mut peaks_c = vec![0.0f64; signal.channels as usize];

        {
            let mut tp = TruePeak::new(signal.rate, signal.channels).unwrap();
            tp.check_true_peak(
                &signal.data[0..(len * signal.channels as usize)],
                &mut peaks,
            );
        }

        unsafe {
            let tp = true_peak_create_c(signal.rate, signal.channels);
            assert!(!tp.is_null());
            true_peak_check_short_c(tp, len, signal.data.as_ptr(), peaks_c.as_mut_ptr());
            true_peak_destroy_c(tp);
        }

        for (i, (r, c)) in peaks.iter().zip(peaks_c.iter()).enumerate() {
            assert!(
                approx_eq!(f64, *r, *c, ulps = 2),
                "Rust and C implementation differ at channel {}: {} != {}",
                i,
                r,
                c
            );
        }

        quickcheck::TestResult::passed()
    }

    #[quickcheck]
    fn compare_c_impl_i32(signal: Signal<i32>) -> quickcheck::TestResult {
        use float_cmp::approx_eq;

        if signal.rate >= 192_000 {
            return quickcheck::TestResult::discard();
        }

        // Maximum of 400ms but our input is up to 5000ms, so distribute it evenly
        // by shrinking accordingly.
        let len = signal.data.len() / signal.channels as usize;
        let len = std::cmp::min(2 * len / 25, 4 * ((signal.rate as usize + 5) / 10));

        let mut peaks = vec![0.0f64; signal.channels as usize];
        let mut peaks_c = vec![0.0f64; signal.channels as usize];

        {
            let mut tp = TruePeak::new(signal.rate, signal.channels).unwrap();
            tp.check_true_peak(
                &signal.data[0..(len * signal.channels as usize)],
                &mut peaks,
            );
        }

        unsafe {
            let tp = true_peak_create_c(signal.rate, signal.channels);
            assert!(!tp.is_null());
            true_peak_check_int_c(tp, len, signal.data.as_ptr(), peaks_c.as_mut_ptr());
            true_peak_destroy_c(tp);
        }

        for (i, (r, c)) in peaks.iter().zip(peaks_c.iter()).enumerate() {
            assert!(
                approx_eq!(f64, *r, *c, ulps = 2),
                "Rust and C implementation differ at channel {}: {} != {}",
                i,
                r,
                c
            );
        }

        quickcheck::TestResult::passed()
    }

    #[quickcheck]
    fn compare_c_impl_f32(signal: Signal<f32>) -> quickcheck::TestResult {
        use float_cmp::approx_eq;

        if signal.rate >= 192_000 {
            return quickcheck::TestResult::discard();
        }

        // Maximum of 400ms but our input is up to 5000ms, so distribute it evenly
        // by shrinking accordingly.
        let len = signal.data.len() / signal.channels as usize;
        let len = std::cmp::min(2 * len / 25, 4 * ((signal.rate as usize + 5) / 10));

        let mut peaks = vec![0.0f64; signal.channels as usize];
        let mut peaks_c = vec![0.0f64; signal.channels as usize];

        {
            let mut tp = TruePeak::new(signal.rate, signal.channels).unwrap();
            tp.check_true_peak(
                &signal.data[0..(len * signal.channels as usize)],
                &mut peaks,
            );
        }

        unsafe {
            let tp = true_peak_create_c(signal.rate, signal.channels);
            assert!(!tp.is_null());
            true_peak_check_float_c(tp, len, signal.data.as_ptr(), peaks_c.as_mut_ptr());
            true_peak_destroy_c(tp);
        }

        for (i, (r, c)) in peaks.iter().zip(peaks_c.iter()).enumerate() {
            assert!(
                approx_eq!(f64, *r, *c, ulps = 2),
                "Rust and C implementation differ at channel {}: {} != {}",
                i,
                r,
                c
            );
        }

        quickcheck::TestResult::passed()
    }

    #[quickcheck]
    fn compare_c_impl_f64(signal: Signal<f64>) -> quickcheck::TestResult {
        use float_cmp::approx_eq;

        if signal.rate >= 192_000 {
            return quickcheck::TestResult::discard();
        }

        // Maximum of 400ms but our input is up to 5000ms, so distribute it evenly
        // by shrinking accordingly.
        let len = signal.data.len() / signal.channels as usize;
        let len = std::cmp::min(2 * len / 25, 4 * ((signal.rate as usize + 5) / 10));

        let mut peaks = vec![0.0f64; signal.channels as usize];
        let mut peaks_c = vec![0.0f64; signal.channels as usize];

        {
            let mut tp = TruePeak::new(signal.rate, signal.channels).unwrap();
            tp.check_true_peak(
                &signal.data[0..(len * signal.channels as usize)],
                &mut peaks,
            );
        }

        unsafe {
            let tp = true_peak_create_c(signal.rate, signal.channels);
            assert!(!tp.is_null());
            true_peak_check_double_c(tp, len, signal.data.as_ptr(), peaks_c.as_mut_ptr());
            true_peak_destroy_c(tp);
        }

        for (i, (r, c)) in peaks.iter().zip(peaks_c.iter()).enumerate() {
            assert!(
                approx_eq!(f64, *r, *c, ulps = 2),
                "Rust and C implementation differ at channel {}: {} != {}",
                i,
                r,
                c
            );
        }

        quickcheck::TestResult::passed()
    }
}
