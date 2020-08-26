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

use std::f64;

/// Data structure for polyphase FIR interpolator
#[derive(Debug)]
pub struct Interp {
    /// Interpolation factor of the interpolator
    factor: usize,
    /// Taps (prefer odd to increase zero coeffs)
    taps: usize,
    /// Number of channels
    channels: u32,
    /// Size of delay buffer
    delay: usize,
    /// List of subfilters (one for each factor)
    // TODO: Try with SmallVec
    filter: Vec<Filter>,
    /// List of delay buffers (one for each channel)
    z: Vec<f32>,
    /// Current delay buffer index
    zi: usize,
}

#[derive(Debug)]
struct Filter {
    /// List of subfilter coefficients and corresponding delay indices
    coeff: Vec<(f64, usize)>,
}

impl Interp {
    #[allow(clippy::many_single_char_names)]
    pub fn new(taps: usize, factor: usize, channels: u32) -> Self {
        let delay = (taps + factor - 1) / factor;

        // Initialize the filter memory
        // One subfilter per interpolation factor.
        let mut filter = Vec::with_capacity(factor);

        for _ in 0..factor {
            let f = Filter { coeff: vec![] };
            filter.push(f);
        }

        // One delay buffer per channel.
        let z = vec![0.0; delay * channels as usize];

        // Calculate the filter coefficients.
        for j in 0..taps {
            const ALMOST_ZERO: f64 = 0.000001;

            // Calculate Hanning window
            let w = 0.5 * (1.0 - f64::cos(2.0 * f64::consts::PI * j as f64 / (taps - 1) as f64));

            // Calculate sinc and apply hanning window
            let m = j as f64 - (taps - 1) as f64 / 2.0;
            let c = if m.abs() > ALMOST_ZERO {
                w * f64::sin(m * f64::consts::PI / factor as f64)
                    / (m * f64::consts::PI / factor as f64)
            } else {
                w
            };

            // Ignore any zero coeffs.
            if c.abs() > ALMOST_ZERO {
                // Put the coefficient into the correct subfilter
                let f = j % factor;

                let f = &mut filter[f];
                f.coeff.push((c, j / factor));
            }
        }

        Interp {
            factor,
            taps,
            channels,
            delay,
            filter,
            z,
            zi: 0,
        }
    }

    pub fn get_factor(&self) -> usize {
        self.factor
    }

    pub fn process(&mut self, src: &[f32], dst: &mut [f32]) {
        assert!(src.len() * self.factor == dst.len());
        assert!(self.z.len() == self.delay * self.channels as usize);
        assert!(self.filter.len() == self.factor);
        assert!(self.zi < self.delay);

        if src.is_empty() {
            return;
        }

        let frames = src.len() / self.channels as usize;

        for (src, (dst, z)) in src.chunks_exact(frames).zip(
            dst.chunks_exact_mut(frames * self.factor)
                .zip(self.z.chunks_exact_mut(self.delay)),
        ) {
            let mut zi = self.zi;

            for (src, dst) in src.iter().zip(dst.chunks_exact_mut(self.factor)) {
                // Add sample to delay buffer
                //
                // TODO Ringbuffer without bounds checks for z/zi
                //
                // Safety: zi is checked to be between 0 and self.delay
                *unsafe { z.get_unchecked_mut(zi) } = *src;

                // Apply coefficients
                for (filter, dst) in self.filter.iter().zip(dst.iter_mut()) {
                    let mut acc = 0.0;
                    for (c, index) in &filter.coeff {
                        let mut i = zi as i32 - *index as i32;
                        if i < 0 {
                            i += self.delay as i32;
                        }
                        // Safety: zi is checked to be between 0 and self.delay
                        acc += *unsafe { z.get_unchecked(i as usize) } as f64 * c;
                    }

                    *dst = acc as f32;
                }

                zi += 1;
                if zi == self.delay {
                    zi = 0;
                }
            }
        }

        self.zi = (self.zi + frames) % self.delay;
    }
}

#[cfg(feature = "c-tests")]
use std::os::raw::c_void;

#[cfg(feature = "c-tests")]
extern "C" {
    pub fn interp_create_c(taps: u32, factor: u32, channels: u32) -> *mut c_void;
    pub fn interp_process_c(
        interp: *mut c_void,
        frames: usize,
        src: *const f32,
        dst: *mut f32,
    ) -> usize;
    pub fn interp_destroy_c(interp: *mut c_void);
}

#[cfg(feature = "c-tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::Signal;
    use float_eq::assert_float_eq;
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn compare_c_impl(signal: Signal<f32>) -> quickcheck::TestResult {
        let frames = signal.data.len() / signal.channels as usize;
        let factor = if signal.rate < 96_000 {
            4
        } else if signal.rate < 192_000 {
            2
        } else {
            return quickcheck::TestResult::discard();
        };

        let mut data_out = vec![0.0f32; signal.data.len() * factor];
        let mut data_out_c = vec![0.0f32; signal.data.len() * factor];

        {
            // Need to deinterleave the input and interleave the output
            let mut data_in_tmp = vec![0.0f32; signal.data.len()];
            let mut data_out_tmp = vec![0.0f32; signal.data.len() * factor];

            for (c, out) in data_in_tmp.chunks_exact_mut(frames).enumerate() {
                for (s, out) in out.iter_mut().enumerate() {
                    *out = signal.data[signal.channels as usize * s + c];
                }
            }

            let mut interp = Interp::new(49, factor, signal.channels);
            interp.process(&data_in_tmp, &mut data_out_tmp);

            for (c, i) in data_out_tmp.chunks_exact(frames * factor).enumerate() {
                for (s, i) in i.iter().enumerate() {
                    data_out[signal.channels as usize * s + c] = *i;
                }
            }
        }

        unsafe {
            let interp = interp_create_c(49, factor as u32, signal.channels);
            interp_process_c(
                interp,
                frames,
                signal.data.as_ptr(),
                data_out_c.as_mut_ptr(),
            );
            interp_destroy_c(interp);
        }

        for (i, (r, c)) in data_out.iter().zip(data_out_c.iter()).enumerate() {
            assert_float_eq!(
                *r,
                *c,
                ulps <= 2,
                "Rust and C implementation differ at sample {}",
                i
            );
        }

        quickcheck::TestResult::passed()
    }
}
