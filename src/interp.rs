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

trait Interpolator: std::fmt::Debug {
    fn process(&mut self, src: &[f32], dst: &mut [f32]);
    fn reset(&mut self);
    fn get_factor(&self) -> usize;
}

#[derive(Debug)]
pub struct Interp(Box<dyn Interpolator>);

impl Interp {
    pub fn new(taps: usize, factor: usize, channels: u32) -> Self {
        let imp: Box<dyn Interpolator> = match (taps, factor, channels) {
            (49, 2, 1) => Box::new(specialized::Interp2F::<[f32; 1]>::new()),
            (49, 2, 2) => Box::new(specialized::Interp2F::<[f32; 2]>::new()),
            (49, 2, 4) => Box::new(specialized::Interp2F::<[f32; 4]>::new()),
            (49, 2, 6) => Box::new(specialized::Interp2F::<[f32; 6]>::new()),
            (49, 2, 8) => Box::new(specialized::Interp2F::<[f32; 8]>::new()),
            (49, 4, 1) => Box::new(specialized::Interp4F::<[f32; 1]>::new()),
            (49, 4, 2) => Box::new(specialized::Interp4F::<[f32; 2]>::new()),
            (49, 4, 4) => Box::new(specialized::Interp4F::<[f32; 4]>::new()),
            (49, 4, 6) => Box::new(specialized::Interp4F::<[f32; 6]>::new()),
            (49, 4, 8) => Box::new(specialized::Interp4F::<[f32; 8]>::new()),
            (taps, factor, channels) => Box::new(generic::Interp::new(taps, factor, channels)),
        };
        Self(imp)
    }

    pub fn process(&mut self, src: &[f32], dst: &mut [f32]) {
        self.0.process(src, dst)
    }

    pub fn reset(&mut self) {
        self.0.reset()
    }

    pub fn get_factor(&self) -> usize {
        self.0.get_factor()
    }
}

mod generic {
    use smallvec::SmallVec;
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
        filter: SmallVec<[Filter; 4]>,
        /// List of delay buffers (one for each channel)
        z: Box<[f32]>,
        /// Current delay buffer index
        zi: usize,
    }

    #[derive(Debug)]
    struct Filter {
        /// List of subfilter coefficients and corresponding delay indices
        coeff: Box<[(f64, usize)]>,
    }

    impl Interp {
        #[allow(clippy::many_single_char_names)]
        pub fn new(taps: usize, factor: usize, channels: u32) -> Self {
            let delay = (taps + factor - 1) / factor;

            // Initialize the filter memory
            // One subfilter per interpolation factor.
            let mut filter = SmallVec::<[_; 4]>::new();

            for _ in 0..factor {
                filter.push(vec![]);
            }

            // One delay buffer per channel.
            let z = vec![0.0; delay * channels as usize];

            // Calculate the filter coefficients.
            for j in 0..taps {
                const ALMOST_ZERO: f64 = 0.000001;

                // Calculate Hanning window
                let w =
                    0.5 * (1.0 - f64::cos(2.0 * f64::consts::PI * j as f64 / (taps - 1) as f64));

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
                    f.push((c, j / factor));
                }
            }

            Interp {
                factor,
                taps,
                channels,
                delay,
                filter: filter
                    .into_iter()
                    .map(|f| Filter {
                        coeff: f.into_boxed_slice(),
                    })
                    .collect(),
                z: z.into_boxed_slice(),
                zi: 0,
            }
        }
    }

    impl super::Interpolator for Interp {
        fn get_factor(&self) -> usize {
            self.factor
        }

        fn reset(&mut self) {
            // TODO: Use slice::fill() once stabilized
            for v in &mut *self.z {
                *v = 0.0;
            }
            self.zi = 0;
        }

        fn process(&mut self, src: &[f32], dst: &mut [f32]) {
            assert!(src.len().checked_mul(self.factor) == Some(dst.len()));
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
                        for (c, index) in &*filter.coeff {
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
}

/// A trait to be generic over number of channels in a of frame
///
/// TODO: Might want to use dasp-frame instead here, but needs
/// coordination with `Samples` trait
trait Frame: Sized + Copy {
    const CHANNELS: usize;

    fn scale_add(&mut self, other: &Self, coeff: f32);
    fn from_planar(slice: &[f32], stride: usize) -> Self;
}

type MonoFrame32 = [f32; 1];
type StereoFrame32 = [f32; 2];
type QuadFrame32 = [f32; 4];
type SurroundFrame32 = [f32; 6];
type Surround8Frame32 = [f32; 8];

impl Frame for MonoFrame32 {
    const CHANNELS: usize = 1;

    #[inline(always)]
    fn scale_add(&mut self, other: &Self, coeff: f32) {
        self[0] += other[0] * coeff;
    }

    #[inline(always)]
    fn from_planar(slice: &[f32], _stride: usize) -> Self {
        [slice[0]]
    }
}

impl Frame for StereoFrame32 {
    const CHANNELS: usize = 2;

    #[inline(always)]
    fn scale_add(&mut self, other: &Self, coeff: f32) {
        self[0] += other[0] * coeff;
        self[1] += other[1] * coeff;
    }

    #[inline(always)]
    fn from_planar(slice: &[f32], stride: usize) -> Self {
        [slice[0], slice[stride]]
    }
}

impl Frame for QuadFrame32 {
    const CHANNELS: usize = 4;

    #[inline(always)]
    fn scale_add(&mut self, other: &Self, coeff: f32) {
        self[0] += other[0] * coeff;
        self[1] += other[1] * coeff;
        self[2] += other[2] * coeff;
        self[3] += other[3] * coeff;
    }

    #[inline(always)]
    fn from_planar(slice: &[f32], stride: usize) -> Self {
        [
            slice[0],
            slice[stride],
            slice[2 * stride],
            slice[3 * stride],
        ]
    }
}

impl Frame for SurroundFrame32 {
    const CHANNELS: usize = 6;

    #[inline(always)]
    fn scale_add(&mut self, other: &Self, coeff: f32) {
        self[0] += other[0] * coeff;
        self[1] += other[1] * coeff;
        self[2] += other[2] * coeff;
        self[3] += other[3] * coeff;
        self[4] += other[4] * coeff;
        self[5] += other[5] * coeff;
    }

    #[inline(always)]
    fn from_planar(slice: &[f32], stride: usize) -> Self {
        [
            slice[0],
            slice[stride],
            slice[2 * stride],
            slice[3 * stride],
            slice[4 * stride],
            slice[5 * stride],
        ]
    }
}

impl Frame for Surround8Frame32 {
    const CHANNELS: usize = 8;

    #[inline(always)]
    fn scale_add(&mut self, other: &Self, coeff: f32) {
        self[0] += other[0] * coeff;
        self[1] += other[1] * coeff;
        self[2] += other[2] * coeff;
        self[3] += other[3] * coeff;
        self[4] += other[4] * coeff;
        self[5] += other[5] * coeff;
        self[6] += other[6] * coeff;
        self[7] += other[7] * coeff;
    }

    #[inline(always)]
    fn from_planar(slice: &[f32], stride: usize) -> Self {
        [
            slice[0],
            slice[stride],
            slice[2 * stride],
            slice[3 * stride],
            slice[4 * stride],
            slice[5 * stride],
            slice[6 * stride],
            slice[7 * stride],
        ]
    }
}

mod specialized {
    use super::Frame;
    use std::f64::consts::PI;

    const ALMOST_ZERO: f64 = 0.000001;
    const TAPS: usize = 48;

    const FACTOR4: usize = 4;
    const FACTOR4_INPUT_LENGTH: usize = TAPS / FACTOR4;
    const FACTOR2: usize = 2;
    const FACTOR2_INPUT_LENGTH: usize = TAPS / FACTOR2;

    #[derive(Debug)]
    pub(super) struct Interp4F<F: Frame> {
        filter: [[f32; FACTOR4]; FACTOR4_INPUT_LENGTH],
        buffer: [F; FACTOR4_INPUT_LENGTH],
        buffer_pos: usize,
    }

    impl<F> Interp4F<F>
    where
        F: Frame + Default,
    {
        pub(super) fn new() -> Self {
            let mut filter: [[_; FACTOR4]; FACTOR4_INPUT_LENGTH] = Default::default();
            for (j, coeff) in filter
                .iter_mut()
                .map(|x| x.iter_mut())
                .flatten()
                .enumerate()
            {
                let j = j as f64;
                // Calculate Hanning window, with one tap ignored. (Last tap is zero anyways, and we want to hit
                // an even multiple of 48)
                let window = (TAPS - 1 + 1) as f64;
                let w = 0.5 * (1.0 - f64::cos(2.0 * PI * j / window));

                // Calculate sinc and apply hanning window
                let m = j - window / 2.0;
                *coeff = if m.abs() > ALMOST_ZERO {
                    w * f64::sin(m * PI / FACTOR4 as f64) / (m * PI / FACTOR4 as f64)
                } else {
                    w
                } as f32;
            }

            Self {
                filter,
                buffer: Default::default(),
                buffer_pos: (FACTOR4_INPUT_LENGTH) - 1,
            }
        }

        pub(super) fn push(&mut self, frame: &F) -> [F; FACTOR4] {
            // Write in Frames in reverse, to enable forward-scanning with filter
            self.buffer_pos = (self.buffer_pos + self.buffer.len() - 1) % self.buffer.len();
            self.buffer[self.buffer_pos] = *frame;

            let mut output: [F; FACTOR4] = Default::default();

            let mut filterp = 0;

            for input_frame in &self.buffer[self.buffer_pos..] {
                let filter_coeffs = &self.filter[filterp];
                for (output_frame, coeff) in Iterator::zip(output.iter_mut(), filter_coeffs) {
                    output_frame.scale_add(input_frame, *coeff);
                }
                filterp += 1;
            }
            for input_frame in &self.buffer[..self.buffer_pos] {
                let filter_coeffs = &self.filter[filterp];
                for (output_frame, coeff) in Iterator::zip(output.iter_mut(), filter_coeffs) {
                    output_frame.scale_add(input_frame, *coeff);
                }
                filterp += 1;
            }

            output
        }
    }

    impl<F> super::Interpolator for Interp4F<F>
    where
        F: Frame + std::fmt::Debug + Default + AsRef<[f32]>,
    {
        fn process(&mut self, src: &[f32], dst: &mut [f32]) {
            assert_eq!(0, src.len() % F::CHANNELS);
            assert_eq!(src.len() * FACTOR4, dst.len());
            let frames = src.len() / F::CHANNELS;

            for i in 0..frames {
                let res = self.push(&F::from_planar(&src[i..], frames));
                for c in 0..F::CHANNELS {
                    for (f, frame) in res.iter().enumerate() {
                        dst[c * frames * FACTOR4 + i * FACTOR4 + f] = frame.as_ref()[c];
                    }
                }
            }
        }

        fn reset(&mut self) {
            self.buffer = Default::default();
        }

        fn get_factor(&self) -> usize {
            4
        }
    }

    #[derive(Debug)]
    pub(super) struct Interp2F<F: Frame> {
        filter: [[f32; FACTOR2]; FACTOR2_INPUT_LENGTH],
        buffer: [F; FACTOR2_INPUT_LENGTH],
        buffer_pos: usize,
    }

    impl<F> Interp2F<F>
    where
        F: Frame + Default,
    {
        pub(super) fn new() -> Self {
            let mut filter: [[_; FACTOR2]; FACTOR2_INPUT_LENGTH] = Default::default();
            for (j, coeff) in filter
                .iter_mut()
                .map(|x| x.iter_mut())
                .flatten()
                .enumerate()
            {
                let j = j as f64;
                // Calculate Hanning window, with one tap ignored. (Last tap is zero anyways, and we want to hit
                // an even multiple of 48)
                let window = (TAPS - 1 + 1) as f64;
                let w = 0.5 * (1.0 - f64::cos(2.0 * PI * j / window));

                // Calculate sinc and apply hanning window
                let m = j - window / 2.0;
                *coeff = if m.abs() > ALMOST_ZERO {
                    w * f64::sin(m * PI / FACTOR2 as f64) / (m * PI / FACTOR2 as f64)
                } else {
                    w
                } as f32;
            }

            Self {
                filter,
                buffer: Default::default(),
                buffer_pos: (FACTOR2_INPUT_LENGTH) - 1,
            }
        }

        pub(super) fn push(&mut self, frame: &F) -> [F; FACTOR2] {
            // Write in Frames in reverse, to enable forward-scanning with filter
            self.buffer_pos = (self.buffer_pos + self.buffer.len() - 1) % self.buffer.len();
            self.buffer[self.buffer_pos] = *frame;

            let mut output: [F; FACTOR2] = Default::default();

            let mut filterp = 0;

            for input_frame in &self.buffer[self.buffer_pos..] {
                let filter_coeffs = &self.filter[filterp];
                for (output_frame, coeff) in Iterator::zip(output.iter_mut(), filter_coeffs) {
                    output_frame.scale_add(input_frame, *coeff);
                }
                filterp += 1;
            }
            for input_frame in &self.buffer[..self.buffer_pos] {
                let filter_coeffs = &self.filter[filterp];
                for (output_frame, coeff) in Iterator::zip(output.iter_mut(), filter_coeffs) {
                    output_frame.scale_add(input_frame, *coeff);
                }
                filterp += 1;
            }

            output
        }
    }

    impl<F> super::Interpolator for Interp2F<F>
    where
        F: Frame + std::fmt::Debug + Default + AsRef<[f32]>,
    {
        fn process(&mut self, src: &[f32], dst: &mut [f32]) {
            assert_eq!(0, src.len() % F::CHANNELS);
            assert_eq!(src.len() * FACTOR2, dst.len());
            let frames = src.len() / F::CHANNELS;

            for i in 0..frames {
                let res = self.push(&F::from_planar(&src[i..], frames));
                for c in 0..F::CHANNELS {
                    for (f, frame) in res.iter().enumerate() {
                        dst[c * frames * FACTOR2 + i * FACTOR2 + f] = frame.as_ref()[c];
                    }
                }
            }
        }

        fn reset(&mut self) {
            self.buffer = Default::default();
        }

        fn get_factor(&self) -> usize {
            2
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internally_consistent() {
        let factor = 4;
        let src = (0..1024)
            .map(|i| (i as f32 - 512.) / 512.)
            .collect::<Vec<_>>();
        let mut dst_stereo = vec![0.; src.len() * factor];
        let mut dst_dual_mono = vec![0.; src.len() * factor];

        Interp::new(49, factor, 2).process(&src, &mut dst_stereo);
        Interp::new(49, factor, 1).process(&src[..512], &mut dst_dual_mono[..512 * factor]);
        Interp::new(49, factor, 1).process(&src[512..], &mut dst_dual_mono[512 * factor..]);

        assert_eq!(dst_stereo, dst_dual_mono);
    }
}

#[cfg(feature = "c-tests")]
#[cfg(test)]
mod c_tests {
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
                // For a performance-boost, filter is defined as f32, causing slightly lower precision
                abs <= 0.000002,
                "Rust and C implementation differ at sample {}",
                i
            );
        }

        quickcheck::TestResult::passed()
    }
}
