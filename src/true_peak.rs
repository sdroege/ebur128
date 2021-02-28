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

use crate::interp::{Interp2F, Interp4F};
use crate::utils::{FrameAccumulator, Sample};
use dasp_frame::Frame;

use UpsamplingScanner::*;

#[derive(Debug)]
enum UpsamplingScanner {
    Mono2F(Interp2F<[f32; 1]>),
    Stereo2F(Interp2F<[f32; 2]>),
    Quad2F(Interp2F<[f32; 4]>),
    Surround2F(Interp2F<[f32; 6]>),
    OctoSurround2F(Interp2F<[f32; 8]>),
    Mono4F(Interp4F<[f32; 1]>),
    Stereo4F(Interp4F<[f32; 2]>),
    Quad4F(Interp4F<[f32; 4]>),
    Surround4F(Interp4F<[f32; 6]>),
    OctoSurround4F(Interp4F<[f32; 8]>),
    Generic2F(Box<[Interp2F<[f32; 1]>]>),
    Generic4F(Box<[Interp4F<[f32; 1]>]>),
}

impl UpsamplingScanner {
    fn new(rate: u32, channels: u32) -> Option<Self> {
        enum Factor {
            Four,
            Two,
        }
        let interp_factor = if rate < 96_000 {
            Factor::Four
        } else if rate < 192_000 {
            Factor::Two
        } else {
            return None;
        };

        Some(match (channels as usize, interp_factor) {
            (1, Factor::Two) => Mono2F(Interp2F::new()),
            (2, Factor::Two) => Stereo2F(Interp2F::new()),
            (4, Factor::Two) => Quad2F(Interp2F::new()),
            (6, Factor::Two) => Surround2F(Interp2F::new()),
            (8, Factor::Two) => OctoSurround2F(Interp2F::new()),
            (1, Factor::Four) => Mono4F(Interp4F::new()),
            (2, Factor::Four) => Stereo4F(Interp4F::new()),
            (4, Factor::Four) => Quad4F(Interp4F::new()),
            (6, Factor::Four) => Surround4F(Interp4F::new()),
            (8, Factor::Four) => OctoSurround4F(Interp4F::new()),
            (c, Factor::Two) => Generic2F(vec![Interp2F::new(); c].into()),
            (c, Factor::Four) => Generic4F(vec![Interp4F::new(); c].into()),
        })
    }

    pub fn check_true_peak<'a, T: Sample + 'a, S: crate::Samples<'a, T>>(
        &mut self,
        src: &S,
        peaks: &mut [f64],
    ) {
        macro_rules! tp_specialized_impl {
            ( $channels:expr, $interpolator:expr ) => {{
                const CHANNELS: usize = $channels;
                assert!(src.channels() == CHANNELS && peaks.len() == CHANNELS);
                let mut tmp_peaks = <[f32; CHANNELS]>::from_fn(|i| peaks[i] as f32);

                src.foreach_frame(|frame: [T; CHANNELS]| {
                    let frame_f32: [f32; CHANNELS] = Frame::map(frame, |s| s.to_sample::<f32>());
                    for new_frame in &$interpolator.interpolate(frame_f32) {
                        tmp_peaks.retain_max_samples(&Frame::map(*new_frame, |s| s.abs()));
                    }
                });
                for (dst, src) in Iterator::zip(peaks.into_iter(), &tmp_peaks) {
                    *dst = *src as f64;
                }
            }};
        }

        macro_rules! tp_generic_impl {
            ( $interpolators:expr ) => {{
                assert!(src.channels() == $interpolators.len() && src.channels() == peaks.len());
                for (c, (interpolator, channel_peak)) in
                    Iterator::zip($interpolators.iter_mut(), peaks.iter_mut()).enumerate()
                {
                    src.foreach_sample(c, move |s| {
                        for [new_sample] in &interpolator.interpolate([s.to_sample::<f32>()]) {
                            let new_sample = new_sample.abs() as f64;
                            if new_sample > *channel_peak {
                                *channel_peak = new_sample;
                            }
                        }
                    });
                }
            }};
        }

        match self {
            Mono2F(interpolator) => tp_specialized_impl!(1, interpolator),
            Stereo2F(interpolator) => tp_specialized_impl!(2, interpolator),
            Quad2F(interpolator) => tp_specialized_impl!(4, interpolator),
            Surround2F(interpolator) => tp_specialized_impl!(6, interpolator),
            OctoSurround2F(interpolator) => tp_specialized_impl!(8, interpolator),
            Mono4F(interpolator) => tp_specialized_impl!(1, interpolator),
            Stereo4F(interpolator) => tp_specialized_impl!(2, interpolator),
            Quad4F(interpolator) => tp_specialized_impl!(4, interpolator),
            Surround4F(interpolator) => tp_specialized_impl!(6, interpolator),
            OctoSurround4F(interpolator) => tp_specialized_impl!(8, interpolator),
            Generic2F(interpolators) => tp_generic_impl!(interpolators),
            Generic4F(interpolators) => tp_generic_impl!(interpolators),
        }
    }

    fn reset(&mut self) {
        match self {
            Mono2F(interpolator) => interpolator.reset(),
            Stereo2F(interpolator) => interpolator.reset(),
            Quad2F(interpolator) => interpolator.reset(),
            Surround2F(interpolator) => interpolator.reset(),
            OctoSurround2F(interpolator) => interpolator.reset(),
            Mono4F(interpolator) => interpolator.reset(),
            Stereo4F(interpolator) => interpolator.reset(),
            Quad4F(interpolator) => interpolator.reset(),
            Surround4F(interpolator) => interpolator.reset(),
            OctoSurround4F(interpolator) => interpolator.reset(),
            Generic2F(interpolators) => interpolators.iter_mut().for_each(Interp2F::reset),
            Generic4F(interpolators) => interpolators.iter_mut().for_each(Interp4F::reset),
        }
    }
}

/// True peak measurement.
#[derive(Debug)]
pub struct TruePeak {
    /// Interpolator/resampler.
    interp: UpsamplingScanner,
}

impl TruePeak {
    pub fn new(rate: u32, channels: u32) -> Option<Self> {
        UpsamplingScanner::new(rate, channels).map(|interp| Self { interp })
    }

    pub fn reset(&mut self) {
        self.interp.reset();
    }

    pub fn check_true_peak<'a, T: Sample + 'a, S: crate::Samples<'a, T>>(
        &mut self,
        src: &S,
        peaks: &mut [f64],
    ) {
        self.interp.check_true_peak(src, peaks)
    }
}
