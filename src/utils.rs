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

use dasp_frame::Frame;

/// Convert linear energy to logarithmic loudness.
pub fn energy_to_loudness(energy: f64) -> f64 {
    10.0 * f64::log10(energy) - 0.691
}

/// Trait for abstracting over interleaved and planar samples.
pub trait Samples<'a, S: Sample + 'a>: Sized {
    /// Call the given closure for each sample of the given channel.
    // FIXME: Workaround for TrustedLen / TrustedRandomAccess being unstable
    // and because of that we wouldn't get nice optimizations
    fn foreach_sample(&self, channel: usize, func: impl FnMut(&'a S));

    /// Call the given closure for each sample of the given channel.
    // FIXME: Workaround for TrustedLen / TrustedRandomAccess being unstable
    // and because of that we wouldn't get nice optimizations
    fn foreach_sample_zipped<U>(
        &self,
        channel: usize,
        iter: impl Iterator<Item = U>,
        func: impl FnMut(&'a S, U),
    );

    fn foreach_frame<F: Frame<Sample = S>>(&self, func: impl FnMut(F));

    /// Number of frames.
    fn frames(&self) -> usize;

    /// Number of channels.
    fn channels(&self) -> usize;

    /// Split into two at the given sample.
    fn split_at(self, sample: usize) -> (Self, Self);
}

/// Struct representing interleaved samples.
pub struct Interleaved<'a, S> {
    /// Interleaved sample data.
    data: &'a [S],
    /// Number of channels.
    channels: usize,
}

impl<'a, S> Interleaved<'a, S> {
    /// Create a new wrapper around the interleaved channels and do a sanity check.
    pub fn new(data: &'a [S], channels: usize) -> Result<Self, crate::Error> {
        if channels == 0 {
            return Err(crate::Error::NoMem);
        }

        if data.len() % channels != 0 {
            return Err(crate::Error::NoMem);
        }

        Ok(Interleaved { data, channels })
    }
}

impl<'a, S: Sample> Samples<'a, S> for Interleaved<'a, S> {
    #[inline]
    fn foreach_sample(&self, channel: usize, mut func: impl FnMut(&'a S)) {
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
        mut func: impl FnMut(&'a S, U),
    ) {
        assert!(channel < self.channels);

        for (v, u) in Iterator::zip(self.data.chunks_exact(self.channels), iter) {
            func(&v[channel], u)
        }
    }

    #[inline]
    fn foreach_frame<F: Frame<Sample = S>>(&self, mut func: impl FnMut(F)) {
        assert_eq!(F::CHANNELS, self.channels);
        for f in self.data.chunks_exact(self.channels) {
            func(F::from_samples(&mut f.iter().copied()).unwrap());
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
pub struct Planar<'a, S> {
    data: &'a [&'a [S]],
    start: usize,
    end: usize,
}

impl<'a, S> Planar<'a, S> {
    /// Create a new wrapper around the planar channels and do a sanity check.
    pub fn new(data: &'a [&'a [S]]) -> Result<Self, crate::Error> {
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

impl<'a, S: Sample> Samples<'a, S> for Planar<'a, S> {
    #[inline]
    fn foreach_sample(&self, channel: usize, mut func: impl FnMut(&'a S)) {
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
        mut func: impl FnMut(&'a S, U),
    ) {
        assert!(channel < self.data.len());

        for (v, u) in Iterator::zip(self.data[channel][self.start..self.end].iter(), iter) {
            func(v, u)
        }
    }

    #[inline]
    fn foreach_frame<F: Frame<Sample = S>>(&self, mut func: impl FnMut(F)) {
        let channels = self.data.len();
        assert_eq!(F::CHANNELS, channels);
        for f in self.start..self.end {
            func(F::from_fn(|c| self.data[c][f]));
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

/// An extension-trait to accumulate samples into a frame
pub trait FrameAccumulator: Frame {
    fn scale_add(&mut self, other: &Self, coeff: f32);
    fn retain_max_samples(&mut self, other: &Self);
}

impl<F: Frame, S> FrameAccumulator for F
where
    S: SampleAccumulator + std::fmt::Debug,
    F: IndexMut<Target = S>,
{
    #[inline(always)]
    fn scale_add(&mut self, other: &Self, coeff: f32) {
        for i in 0..Self::CHANNELS {
            self.index_mut(i).scale_add(*other.index(i), coeff);
        }
    }

    fn retain_max_samples(&mut self, other: &Self) {
        for i in 0..Self::CHANNELS {
            let this = self.index_mut(i);
            let other = other.index(i);
            if *other > *this {
                *this = *other;
            }
        }
    }
}

// Required since std::ops::IndexMut seem to not be implemented for arrays
// making FrameAcc hard to implement for auto-vectorization
// IndexMut seems to be coming to stdlib, when https://github.com/rust-lang/rust/pull/74989
// implemented, this trait can be removed
pub trait IndexMut {
    type Target;
    fn index_mut(&mut self, i: usize) -> &mut Self::Target;
    fn index(&self, i: usize) -> &Self::Target;
}

macro_rules! index_mut_impl {
    ( $channels:expr ) => {
        impl<T: SampleAccumulator> IndexMut for [T; $channels] {
            type Target = T;
            #[inline(always)]
            fn index_mut(&mut self, i: usize) -> &mut Self::Target {
                &mut self[i]
            }
            #[inline(always)]
            fn index(&self, i: usize) -> &Self::Target {
                &self[i]
            }
        }
    };
}

index_mut_impl!(1);
index_mut_impl!(2);
index_mut_impl!(4);
index_mut_impl!(6);
index_mut_impl!(8);

pub trait SampleAccumulator: Sample {
    fn scale_add(&mut self, other: Self, coeff: f32);
}

impl SampleAccumulator for f32 {
    #[inline(always)]
    fn scale_add(&mut self, other: Self, coeff: f32) {
        #[cfg(feature = "precision-true-peak")]
        {
            *self = other.mul_add(coeff, *self);
        }
        #[cfg(not(feature = "precision-true-peak"))]
        {
            *self += other * coeff
        }
    }
}
