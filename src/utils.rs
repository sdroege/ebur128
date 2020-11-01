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

/// Convert linear energy to logarithmic loudness.
pub fn energy_to_loudness(energy: f64) -> f64 {
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

/// Trait for abstracting over interleaved and planar samples.
pub trait Samples<'a, T: 'a>: Sized {
    /// Call the given closure for each sample of the given channel.
    // FIXME: Workaround for TrustedLen / TrustedRandomAccess being unstable
    // and because of that we wouldn't get nice optimizations
    fn foreach_sample(&self, channel: usize, func: impl FnMut(&'a T));

    /// Call the given closure for each sample of the given channel.
    // FIXME: Workaround for TrustedLen / TrustedRandomAccess being unstable
    // and because of that we wouldn't get nice optimizations
    fn foreach_sample_zipped<U>(
        &self,
        channel: usize,
        iter: impl Iterator<Item = U>,
        func: impl FnMut(&'a T, U),
    );

    /// Number of frames.
    fn frames(&self) -> usize;

    /// Number of channels.
    fn channels(&self) -> usize;

    /// Split into two at the given sample.
    fn split_at(self, sample: usize) -> (Self, Self);
}

/// Struct representing interleaved samples.
pub struct Interleaved<'a, T> {
    /// Interleaved sample data.
    data: &'a [T],
    /// Number of channels.
    channels: usize,
}

impl<'a, T> Interleaved<'a, T> {
    /// Create a new wrapper around the interleaved channels and do a sanity check.
    pub fn new(data: &'a [T], channels: usize) -> Result<Self, crate::Error> {
        if channels == 0 {
            return Err(crate::Error::NoMem);
        }

        if data.len() % channels != 0 {
            return Err(crate::Error::NoMem);
        }

        Ok(Interleaved { data, channels })
    }
}

impl<'a, T> Samples<'a, T> for Interleaved<'a, T> {
    #[inline]
    fn foreach_sample(&self, channel: usize, mut func: impl FnMut(&'a T)) {
        assert!(channel < self.channels);

        for v in self.data.chunks_exact(self.channels) {
            func(&v[channel])
        }
    }

    #[inline]
    fn foreach_sample_zipped<U>(
        &self,
        channel: usize,
        iter: impl Iterator<Item = U>,
        mut func: impl FnMut(&'a T, U),
    ) {
        assert!(channel < self.channels);

        for (v, u) in self.data.chunks_exact(self.channels).zip(iter) {
            func(&v[channel], u)
        }
    }

    #[inline]
    fn frames(&self) -> usize {
        self.data.len() / self.channels
    }

    #[inline]
    fn channels(&self) -> usize {
        self.channels
    }

    #[inline]
    fn split_at(self, sample: usize) -> (Self, Self) {
        assert!(sample * self.channels <= self.data.len());

        let (fst, snd) = self.data.split_at(sample * self.channels);
        (
            Interleaved {
                data: fst,
                channels: self.channels,
            },
            Interleaved {
                data: snd,
                channels: self.channels,
            },
        )
    }
}

/// Struct representing interleaved samples.
pub struct Planar<'a, T> {
    data: &'a [&'a [T]],
    start: usize,
    end: usize,
}

impl<'a, T> Planar<'a, T> {
    /// Create a new wrapper around the planar channels and do a sanity check.
    pub fn new(data: &'a [&'a [T]]) -> Result<Self, crate::Error> {
        if data.is_empty() {
            return Err(crate::Error::NoMem);
        }

        if data.iter().any(|d| data[0].len() != d.len()) {
            return Err(crate::Error::NoMem);
        }

        Ok(Planar {
            data,
            start: 0,
            end: data[0].len(),
        })
    }
}

impl<'a, T> Samples<'a, T> for Planar<'a, T> {
    #[inline]
    fn foreach_sample(&self, channel: usize, mut func: impl FnMut(&'a T)) {
        assert!(channel < self.data.len());

        for v in &self.data[channel][self.start..self.end] {
            func(v)
        }
    }

    #[inline]
    fn foreach_sample_zipped<U>(
        &self,
        channel: usize,
        iter: impl Iterator<Item = U>,
        mut func: impl FnMut(&'a T, U),
    ) {
        assert!(channel < self.data.len());

        for (v, u) in self.data[channel][self.start..self.end].iter().zip(iter) {
            func(v, u)
        }
    }

    #[inline]
    fn frames(&self) -> usize {
        self.end - self.start
    }

    #[inline]
    fn channels(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn split_at(self, sample: usize) -> (Self, Self) {
        assert!(self.start + sample <= self.end);

        (
            Planar {
                data: self.data,
                start: self.start,
                end: self.start + sample,
            },
            Planar {
                data: self.data,
                start: self.start + sample,
                end: self.end,
            },
        )
    }
}

pub trait Sample:
    dasp_sample::Sample + dasp_sample::Duplex<f32> + dasp_sample::Duplex<f64>
{
    const MAX_AMPLITUDE: f64;

    fn as_f64_raw(self) -> f64;
}

impl Sample for f32 {
    const MAX_AMPLITUDE: f64 = 1.0;

    #[inline(always)]
    fn as_f64_raw(self) -> f64 {
        self as f64
    }
}
impl Sample for f64 {
    const MAX_AMPLITUDE: f64 = 1.0;

    #[inline(always)]
    fn as_f64_raw(self) -> f64 {
        self as f64
    }
}
impl Sample for i16 {
    const MAX_AMPLITUDE: f64 = -(Self::MIN as f64);

    #[inline(always)]
    fn as_f64_raw(self) -> f64 {
        self as f64
    }
}
impl Sample for i32 {
    const MAX_AMPLITUDE: f64 = -(Self::MIN as f64);

    #[inline(always)]
    fn as_f64_raw(self) -> f64 {
        self as f64
    }
}

#[cfg(test)]
pub mod tests {
    use dasp_sample::{FromSample, Sample};

    #[derive(Clone, Debug)]
    pub struct Signal<T: FromSample<f32>> {
        pub data: Vec<T>,
        pub channels: u32,
        pub rate: u32,
    }

    impl<T: Sample + FromSample<f32> + quickcheck::Arbitrary> quickcheck::Arbitrary for Signal<T> {
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

            let mut data = vec![T::from_sample(0.0f32); num_frames * channels as usize];
            for frame in data.chunks_exact_mut(channels as usize) {
                let val = max
                    * (f32::sin(accumulators[0]) * volumes[0]
                        + f32::sin(accumulators[1]) * volumes[1]
                        + f32::sin(accumulators[2]) * volumes[2]
                        + f32::sin(accumulators[3]) * volumes[3])
                    / volume_scale;

                for sample in frame.iter_mut() {
                    *sample = T::from_sample(val);
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

    struct SignalShrinker<A: FromSample<f32>> {
        seed: Signal<A>,
        /// How many elements to take
        size: usize,
        /// Whether we tried with one channel already
        tried_one_channel: bool,
    }

    impl<A: FromSample<f32> + quickcheck::Arbitrary> SignalShrinker<A> {
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
        A: FromSample<f32> + quickcheck::Arbitrary,
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
