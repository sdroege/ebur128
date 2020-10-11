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

/// True peak measurement.
#[derive(Debug)]
pub struct TruePeak {
    /// Interpolator/resampler.
    interp: Interp,
    /// Configured sample rate.
    rate: u32,
    /// Configured number of channels.
    channels: u32,
    /// Input buffer to which to processed data is first copied. This allows for 400ms
    /// samples per channel, non-interleaved/planar.
    buffer_input: Box<[f32]>,
    /// Output buffer for the resampler. This allows for 400ms * resample factor samples.
    buffer_output: Box<[f32]>,
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

        let buffer_input =
            vec![0.0; 4 * samples_in_100ms as usize * channels as usize].into_boxed_slice();
        let buffer_output = vec![0.0; buffer_input.len() * interp_factor].into_boxed_slice();

        Some(Self {
            interp,
            rate,
            channels,
            buffer_input,
            buffer_output,
        })
    }

    pub fn reset(&mut self) {
        self.interp.reset();
    }

    pub fn check_true_peak<'a, T: crate::AsF32 + 'a, S: crate::Samples<'a, T>>(
        &mut self,
        src: &S,
        peaks: &mut [f64],
    ) {
        assert!(src.channels() == self.channels as usize);
        assert!(src.frames() * self.channels as usize <= self.buffer_input.len());
        assert!(
            src.frames() * self.channels as usize * self.interp.get_factor()
                <= self.buffer_output.len()
        );
        assert!(self.buffer_input.len() * self.interp.get_factor() == self.buffer_output.len());
        assert!(peaks.len() == self.channels as usize);

        let frames = src.frames();

        if frames == 0 {
            return;
        }

        let in_len = frames * self.channels as usize;
        let interp_factor = self.interp.get_factor();
        let out_len = frames * self.channels as usize * interp_factor;

        // Deinterleave and convert to f32 for the resampler
        for (c, dest) in self.buffer_input[..in_len]
            .chunks_exact_mut(frames)
            .enumerate()
        {
            assert!(c < src.channels());

            src.foreach_sample_zipped(c, dest.iter_mut(), |src, dest| {
                *dest = src.as_f32_scaled();
            });
        }

        self.interp.process(
            &self.buffer_input[..in_len],
            &mut self.buffer_output[..out_len],
        );

        // Find the maximum
        for (c, o) in self.buffer_output[..out_len]
            .chunks_exact(frames * interp_factor)
            .enumerate()
        {
            assert!(c < self.channels as usize);

            let mut max = 0.0;
            for v in o {
                let v = v.abs();
                if v > max {
                    max = v;
                }
            }

            if max as f64 > peaks[c] {
                peaks[c] = max as f64;
            }
        }
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
    use float_eq::assert_float_eq;
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn compare_c_impl_i16(signal: Signal<i16>) -> quickcheck::TestResult {
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
                &crate::Interleaved::new(
                    &signal.data[0..(len * signal.channels as usize)],
                    signal.channels as usize,
                )
                .unwrap(),
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
            assert_float_eq!(
                *r,
                *c,
                // For a performance-boost, interpolation-filter is defined as f32, causing lower precision
                abs <= 0.000002,
                "Rust and C implementation differ at channel {}",
                i,
            );
        }

        quickcheck::TestResult::passed()
    }

    #[quickcheck]
    fn compare_c_impl_i32(signal: Signal<i32>) -> quickcheck::TestResult {
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
                &crate::Interleaved::new(
                    &signal.data[0..(len * signal.channels as usize)],
                    signal.channels as usize,
                )
                .unwrap(),
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
            assert_float_eq!(
                *r,
                *c,
                // For a performance-boost, interpolation-filter is defined as f32, causing lower precision
                abs <= 0.000002,
                "Rust and C implementation differ at channel {}",
                i
            );
        }

        quickcheck::TestResult::passed()
    }

    #[quickcheck]
    fn compare_c_impl_f32(signal: Signal<f32>) -> quickcheck::TestResult {
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
                &crate::Interleaved::new(
                    &signal.data[0..(len * signal.channels as usize)],
                    signal.channels as usize,
                )
                .unwrap(),
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
            assert_float_eq!(
                *r,
                *c,
                // For a performance-boost, interpolation-filter is defined as f32, causing lower precision
                abs <= 0.000002,
                "Rust and C implementation differ at channel {}",
                i
            );
        }

        quickcheck::TestResult::passed()
    }

    #[quickcheck]
    fn compare_c_impl_f64(signal: Signal<f64>) -> quickcheck::TestResult {
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
                &crate::Interleaved::new(
                    &signal.data[0..(len * signal.channels as usize)],
                    signal.channels as usize,
                )
                .unwrap(),
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
            assert_float_eq!(
                *r,
                *c,
                // For a performance-boost, interpolation-filter is defined as f32, causing lower precision
                abs <= 0.000002,
                "Rust and C implementation differ at channel {}",
                i
            );
        }

        quickcheck::TestResult::passed()
    }
}
