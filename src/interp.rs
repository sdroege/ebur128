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

use crate::utils::FrameAccumulator;
use std::f64::consts::PI;

const ALMOST_ZERO: f64 = 0.000001;
const TAPS: usize = 48;

// Workaround for missing const-generics
trait ArrayBuf<Item>: std::borrow::BorrowMut<[Item]> {
    const SIZE: usize;
}

impl<Item> ArrayBuf<Item> for [Item; 24] {
    const SIZE: usize = 24;
}

impl<Item> ArrayBuf<Item> for [Item; 12] {
    const SIZE: usize = 12;
}

/// A circular buffer offering fixed-length continous views into data
/// This is enabled by writing data twice, also to a "shadow"-buffer following the primary buffer,
/// The tradeoff is writing all data twice, the gain is giving the compiler continuous view with
/// predictable length into the data, unlocking some more optimizations
#[derive(Clone, Debug)]
struct RollingBuffer<A, T> {
    buf: [T; TAPS],
    position: usize,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: ArrayBuf<T>, T: Default + Copy> RollingBuffer<A, T> {
    fn new() -> Self {
        assert!(A::SIZE * 2 <= TAPS);

        let buf: [T; TAPS] = [Default::default(); TAPS];

        Self {
            buf,
            position: A::SIZE,
            _phantom: Default::default(),
        }
    }

    #[inline(always)]
    fn push_front(&mut self, v: T) {
        if self.position == 0 {
            self.position = A::SIZE - 1;
        } else {
            self.position -= 1;
        }
        unsafe {
            *self.buf.get_unchecked_mut(self.position) = v;
            *self.buf.get_unchecked_mut(self.position + A::SIZE) = v;
        }
    }
}

impl<A, T> AsRef<A> for RollingBuffer<A, T> {
    #[inline(always)]
    fn as_ref(&self) -> &A {
        unsafe { &*(self.buf.get_unchecked(self.position) as *const T as *const A) }
    }
}

macro_rules! interp_impl {
    ( $name:ident, $factor:expr ) => {
        #[derive(Debug, Clone)]
        pub struct $name<F: FrameAccumulator> {
            filter: [[f32; $factor]; (TAPS / $factor)],
            buffer: RollingBuffer<[F; TAPS / $factor], F>,
        }

        impl<F> Default for $name<F>
        where
            F: FrameAccumulator + Default,
        {
            fn default() -> Self {
                Self::new()
            }
        }

        impl<F> $name<F>
        where
            F: FrameAccumulator + Default,
        {
            pub fn new() -> Self {
                let mut filter: [[_; $factor]; (TAPS / $factor)] = Default::default();
                for (j, coeff) in filter
                    .iter_mut()
                    .map(|x| x.iter_mut())
                    .flatten()
                    .enumerate()
                {
                    let j = j as f64;
                    // Calculate Hanning window,
                    let window = TAPS + 1;
                    // Ignore one tap. (Last tap is zero anyways, and we want to hit an even multiple of 48)
                    let window = (window - 1) as f64;
                    let w = 0.5 * (1.0 - f64::cos(2.0 * PI * j / window));

                    // Calculate sinc and apply hanning window
                    let m = j - window / 2.0;
                    *coeff = if m.abs() > ALMOST_ZERO {
                        w * f64::sin(m * PI / $factor as f64) / (m * PI / $factor as f64)
                    } else {
                        w
                    } as f32;
                }

                Self {
                    filter,
                    buffer: RollingBuffer::new(),
                }
            }

            pub fn interpolate(&mut self, frame: F) -> [F; $factor] {
                // Write in Frames in reverse, to enable forward-scanning with filter
                self.buffer.push_front(frame);

                let mut output: [F; $factor] = Default::default();

                let buf = self.buffer.as_ref();

                for (filter_coeffs, input_frame) in Iterator::zip(self.filter.iter(), buf) {
                    for (output_frame, coeff) in Iterator::zip(output.iter_mut(), filter_coeffs) {
                        output_frame.scale_add(input_frame, *coeff);
                    }
                }

                output
            }

            pub fn reset(&mut self) {
                self.buffer = RollingBuffer::new();
            }
        }
    };
}

interp_impl!(Interp2F, 2);
interp_impl!(Interp4F, 4);

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
mod c_tests {
    use super::*;
    use crate::tests::Signal;
    use float_eq::assert_float_eq;
    use quickcheck_macros::quickcheck;

    fn process_rust(data_in: &[f32], data_out: &mut [f32], factor: usize, channels: usize) {
        macro_rules! process_specialized {
            ( $interp:ident, $channels:expr ) => {{
                let mut interp = $interp::new();
                let (_, data_in, _) = unsafe { data_in.align_to::<[f32; $channels]>() };
                let (_, data_out, _) = unsafe { data_out.align_to_mut::<[f32; $channels]>() };
                for (input_frame, output_frames) in
                    Iterator::zip(data_in.into_iter(), data_out.chunks_exact_mut(factor))
                {
                    output_frames.copy_from_slice(&interp.interpolate(*input_frame));
                }
            }};
        }

        macro_rules! process_generic {
            ( $interp:ident, $channels:expr ) => {{
                let mut interp = vec![$interp::<[f32; 1]>::new(); $channels];
                let frames = data_in.len() / channels;
                for frame in 0..frames {
                    for channel in 0..$channels {
                        let in_sample = data_in[(frame * $channels) + channel];
                        for (o, [output_sample]) in
                            interp[channel].interpolate([in_sample]).iter().enumerate()
                        {
                            let output_frame = frame * factor + o;
                            data_out[(output_frame * $channels) + channel] = *output_sample;
                        }
                    }
                }
            }};
        }

        match (factor, channels) {
            (2, 1) => process_specialized!(Interp2F, 1),
            (2, 2) => process_specialized!(Interp2F, 2),
            (2, 4) => process_specialized!(Interp2F, 4),
            (2, 6) => process_specialized!(Interp2F, 6),
            (2, 8) => process_specialized!(Interp2F, 8),
            (4, 1) => process_specialized!(Interp4F, 1),
            (4, 2) => process_specialized!(Interp4F, 2),
            (4, 4) => process_specialized!(Interp4F, 4),
            (4, 6) => process_specialized!(Interp4F, 6),
            (4, 8) => process_specialized!(Interp4F, 8),
            (2, c) => process_generic!(Interp2F, c),
            (4, c) => process_generic!(Interp4F, c),
            _ => unimplemented!(),
        }
    }

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

        process_rust(
            &signal.data,
            &mut data_out,
            factor,
            signal.channels as usize,
        );

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
                abs <= 0.000004,
                "Rust and C implementation differ at sample {}",
                i
            );
        }

        quickcheck::TestResult::passed()
    }
}
