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
