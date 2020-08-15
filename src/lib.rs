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

//!  Implementation of the [EBU R128 loudness standard](https://tech.ebu.ch/docs/r/r128.pdf).
//!
//!  The European Broadcasting Union Loudness Recommendation (EBU R128) informs broadcasters how
//!  they can analyze and normalize audio so that each piece of audio sounds roughly the same
//!  volume to the human ear.
//!
//!  This crate provides an API which analyzes audio and outputs perceived loudness. The results
//!  can then be used to normalize volume during playback.
//!
//!  Features:
//!   * Implements M, S and I modes
//!   * Implements loudness range measurement (EBU - TECH 3342)
//!   * True peak scanning
//!   * Supports all samplerates by recalculation of the filter coefficients

mod ebur128;
pub use self::ebur128::*;

#[cfg(feature = "internal-tests")]
pub mod interp;
#[cfg(not(feature = "internal-tests"))]
pub(crate) mod interp;

#[cfg(feature = "internal-tests")]
pub mod true_peak;
#[cfg(not(feature = "internal-tests"))]
pub(crate) mod true_peak;

#[cfg(feature = "internal-tests")]
pub mod history;
#[cfg(not(feature = "internal-tests"))]
pub(crate) mod history;

#[cfg(feature = "internal-tests")]
pub mod filter;
#[cfg(not(feature = "internal-tests"))]
pub(crate) mod filter;

#[cfg(feature = "capi")]
pub mod capi;

pub(crate) fn energy_to_loudness(energy: f64) -> f64 {
    // The non-test version is faster and more accurate but gives
    // slightly different results than the C version and fails the
    // tests because of that.
    #[cfg(test)]
    {
        10.0 * (f64::ln(energy) / f64::ln(10.0)) - 0.691
    }
    #[cfg(not(test))]
    {
        10.0 * f64::log10(energy) - 0.691
    }
}

#[cfg(test)]
mod tests {
    #[derive(Clone, Debug)]
    pub struct Signal<T: FromF32> {
        pub data: Vec<T>,
        pub channels: u32,
        pub rate: u32,
    }

    pub trait FromF32: Copy + Clone + std::fmt::Debug + Send + Sync + 'static {
        fn from_f32(val: f32) -> Self;
    }

    impl FromF32 for i16 {
        fn from_f32(val: f32) -> Self {
            (val * (std::i16::MAX - 1) as f32) as i16
        }
    }

    impl FromF32 for i32 {
        fn from_f32(val: f32) -> Self {
            (val * (std::i32::MAX - 1) as f32) as i32
        }
    }

    impl FromF32 for f32 {
        fn from_f32(val: f32) -> Self {
            val
        }
    }

    impl FromF32 for f64 {
        fn from_f32(val: f32) -> Self {
            val as f64
        }
    }

    impl<T: FromF32 + quickcheck::Arbitrary> quickcheck::Arbitrary for Signal<T> {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            use rand::Rng;

            let channels = g.gen_range(1, 16);
            let rate = g.gen_range(16_000, 224_000);
            let num_frames = (rate as f64 * g.gen_range(0.0, 5.0)) as usize;

            let max = g.gen_range(0.0, 1.0);
            let freqs = [
                g.gen_range(20.0, 16_000.0),
                g.gen_range(20.0, 16_000.0),
                g.gen_range(20.0, 16_000.0),
                g.gen_range(20.0, 16_000.0),
            ];
            let volumes = [
                g.gen_range(0.0, 1.0),
                g.gen_range(0.0, 1.0),
                g.gen_range(0.0, 1.0),
                g.gen_range(0.0, 1.0),
            ];
            let volume_scale = 1.0 / volumes.iter().sum::<f32>();
            let mut accumulators = [0.0; 4];
            let steps = [
                2.0 * std::f32::consts::PI * freqs[0] / rate as f32,
                2.0 * std::f32::consts::PI * freqs[1] / rate as f32,
                2.0 * std::f32::consts::PI * freqs[2] / rate as f32,
                2.0 * std::f32::consts::PI * freqs[3] / rate as f32,
            ];

            let mut data = vec![T::from_f32(0.0); num_frames * channels as usize];
            for frame in data.chunks_exact_mut(channels as usize) {
                let val = max
                    * (f32::sin(accumulators[0]) * volumes[0]
                        + f32::sin(accumulators[1]) * volumes[1]
                        + f32::sin(accumulators[2]) * volumes[2]
                        + f32::sin(accumulators[3]) * volumes[3])
                    / volume_scale;

                for sample in frame.iter_mut() {
                    *sample = T::from_f32(val);
                }

                for (acc, step) in accumulators.iter_mut().zip(steps.iter()) {
                    *acc += step;
                }
            }

            Signal {
                data,
                channels,
                rate,
            }
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            SignalShrinker::boxed(self.clone())
        }
    }

    struct SignalShrinker<A: FromF32> {
        seed: Signal<A>,
        /// How many elements to take
        size: usize,
        /// Whether we tried with one channel already
        tried_one_channel: bool,
    }

    impl<A: FromF32 + quickcheck::Arbitrary> SignalShrinker<A> {
        fn boxed(seed: Signal<A>) -> Box<dyn Iterator<Item = Signal<A>>> {
            let channels = seed.channels;
            Box::new(SignalShrinker {
                seed,
                size: 0,
                tried_one_channel: channels == 1,
            })
        }
    }

    impl<A> Iterator for SignalShrinker<A>
    where
        A: FromF32 + quickcheck::Arbitrary,
    {
        type Item = Signal<A>;
        fn next(&mut self) -> Option<Signal<A>> {
            if self.size < self.seed.data.len() {
                // Generate a smaller vector by removing size elements
                let xs1 = if self.tried_one_channel {
                    Vec::from(&self.seed.data[..self.size])
                } else {
                    self.seed
                        .data
                        .iter()
                        .cloned()
                        .step_by(self.seed.channels as usize)
                        .take(self.size)
                        .collect()
                };

                if self.size == 0 {
                    self.size = if self.tried_one_channel {
                        self.seed.channels as usize
                    } else {
                        1
                    };
                } else {
                    self.size *= 2;
                }

                Some(Signal {
                    data: xs1,
                    channels: if self.tried_one_channel {
                        self.seed.channels
                    } else {
                        1
                    },
                    rate: self.seed.rate,
                })
            } else if !self.tried_one_channel {
                self.tried_one_channel = true;
                self.size = 0;
                self.next()
            } else {
                None
            }
        }
    }
}
