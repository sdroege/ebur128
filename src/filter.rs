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

use std::fmt;

use crate::ebur128::Channel;
use crate::utils::Sample;

/// BS.1770 filter and optional sample/true peak measurement context.
pub struct Filter {
    channels: u32,
    /// BS.1770 filter coefficients (numerator).
    b: [f64; 5],
    /// BS.1770 filter coefficients (denominator).
    a: [f64; 5],
    /// One filter state per channel.
    filter_state: Box<[[f64; 5]]>,

    /// Whether to measure sample peak.
    calculate_sample_peak: bool,
    /// Previously measured sample peak.
    sample_peak: Box<[f64]>,

    /// True peak measurement if enabled.
    tp: Option<crate::true_peak::TruePeak>,
    /// Previously measured true peak.
    true_peak: Box<[f64]>,
}

impl fmt::Debug for Filter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Filter")
            .field("channels", &self.channels)
            .field("b", &self.b)
            .field("a", &self.a)
            .field("filter_state", &self.filter_state)
            .field("calculate_sample_peak", &self.calculate_sample_peak)
            .field("sample_peak", &self.sample_peak)
            .field("calculate_true_peak", &self.tp.is_some())
            .field("true_peak", &self.true_peak)
            .finish()
    }
}

#[allow(non_snake_case)]
fn filter_coefficients(rate: f64) -> ([f64; 5], [f64; 5]) {
    let f0 = 1681.974450955533;
    let G = 3.999843853973347;
    let Q = 0.7071752369554196;

    let K = f64::tan(std::f64::consts::PI * f0 / rate);
    let Vh = f64::powf(10.0, G / 20.0);
    let Vb = f64::powf(Vh, 0.4996667741545416);

    let mut pb = [0.0, 0.0, 0.0];
    let mut pa = [1.0, 0.0, 0.0];
    let rb = [1.0, -2.0, 1.0];
    let mut ra = [1.0, 0.0, 0.0];

    let a0 = 1.0 + K / Q + K * K;
    pb[0] = (Vh + Vb * K / Q + K * K) / a0;
    pb[1] = 2.0 * (K * K - Vh) / a0;
    pb[2] = (Vh - Vb * K / Q + K * K) / a0;
    pa[1] = 2.0 * (K * K - 1.0) / a0;
    pa[2] = (1.0 - K / Q + K * K) / a0;

    let f0 = 38.13547087602444;
    let Q = 0.5003270373238773;
    let K = f64::tan(std::f64::consts::PI * f0 / rate);

    ra[1] = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K);
    ra[2] = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K);

    (
        // Numerator
        [
            pb[0] * rb[0],
            pb[0] * rb[1] + pb[1] * rb[0],
            pb[0] * rb[2] + pb[1] * rb[1] + pb[2] * rb[0],
            pb[1] * rb[2] + pb[2] * rb[1],
            pb[2] * rb[2],
        ],
        // Denominator
        [
            pa[0] * ra[0],
            pa[0] * ra[1] + pa[1] * ra[0],
            pa[0] * ra[2] + pa[1] * ra[1] + pa[2] * ra[0],
            pa[1] * ra[2] + pa[2] * ra[1],
            pa[2] * ra[2],
        ],
    )
}

impl Filter {
    pub fn new(
        rate: u32,
        channels: u32,
        calculate_sample_peak: bool,
        calculate_true_peak: bool,
    ) -> Self {
        assert!(rate > 0);
        assert!(channels > 0);

        let (b, a) = filter_coefficients(rate as f64);

        let tp = if calculate_true_peak {
            crate::true_peak::TruePeak::new(rate, channels)
        } else {
            None
        };

        Filter {
            channels,
            b,
            a,
            filter_state: vec![[0.0; 5]; channels as usize].into_boxed_slice(),
            calculate_sample_peak,
            sample_peak: vec![0.0; channels as usize].into_boxed_slice(),
            tp,
            true_peak: vec![0.0; channels as usize].into_boxed_slice(),
        }
    }

    pub fn reset_peaks(&mut self) {
        for v in &mut *self.sample_peak {
            *v = 0.0;
        }

        for v in &mut *self.true_peak {
            *v = 0.0;
        }
    }

    pub fn reset(&mut self) {
        self.reset_peaks();

        for f in &mut *self.filter_state {
            // TODO: Use slice::fill() once stabilized
            for v in &mut *f {
                *v = 0.0;
            }
        }

        if let Some(ref mut tp) = self.tp {
            tp.reset();
        }
    }

    pub fn sample_peak(&self) -> &[f64] {
        &*self.sample_peak
    }

    pub fn true_peak(&self) -> &[f64] {
        &*self.true_peak
    }

    pub fn process<'a, T: Sample + 'a, S: crate::Samples<'a, T>>(
        &mut self,
        src: &S,
        dest: &mut [f64],
        dest_index: usize,
        channel_map: &[crate::ebur128::Channel],
    ) {
        assert!(dest.len() % self.channels as usize == 0);
        assert!(channel_map.len() == self.channels as usize);
        assert!(src.channels() == self.channels as usize);
        assert!(self.filter_state.len() == self.channels as usize);

        ftz::with_ftz(|ftz| {
            if self.calculate_sample_peak {
                assert!(self.sample_peak.len() == self.channels as usize);

                for (c, sample_peak) in self.sample_peak.iter_mut().enumerate() {
                    let mut max = 0.0;

                    assert!(c < src.channels());

                    src.foreach_sample(c, |sample| {
                        let v = sample.as_f64_raw().abs();
                        if v > max {
                            max = v;
                        }
                    });

                    max /= T::MAX_AMPLITUDE;
                    if max > *sample_peak {
                        *sample_peak = max;
                    }
                }
            }

            if let Some(ref mut tp) = self.tp {
                assert!(self.true_peak.len() == self.channels as usize);
                tp.check_true_peak(src, &mut *self.true_peak);
            }

            let dest_stride = dest.len() / self.channels as usize;
            assert!(dest_index + src.frames() <= dest_stride);

            for (c, (channel_map, dest)) in
                Iterator::zip(channel_map.iter(), dest.chunks_exact_mut(dest_stride)).enumerate()
            {
                if *channel_map == crate::ebur128::Channel::Unused {
                    continue;
                }

                assert!(c < src.channels());

                let Filter {
                    ref mut filter_state,
                    ref a,
                    ref b,
                    ..
                } = *self;
                let filter_state = &mut filter_state[c];

                src.foreach_sample_zipped(c, dest[dest_index..].iter_mut(), |src, dest| {
                    filter_state[0] = (*src).to_sample::<f64>()
                        - a[1] * filter_state[1]
                        - a[2] * filter_state[2]
                        - a[3] * filter_state[3]
                        - a[4] * filter_state[4];
                    *dest = b[0] * filter_state[0]
                        + b[1] * filter_state[1]
                        + b[2] * filter_state[2]
                        + b[3] * filter_state[3]
                        + b[4] * filter_state[4];

                    filter_state[4] = filter_state[3];
                    filter_state[3] = filter_state[2];
                    filter_state[2] = filter_state[1];
                    filter_state[1] = filter_state[0];
                });

                if ftz.is_none() {
                    for v in filter_state {
                        if v.abs() < std::f64::EPSILON {
                            *v = 0.0;
                        }
                    }
                }
            }
        });
    }

    pub fn calc_gating_block(
        frames_per_block: usize,
        audio_data: &[f64],
        audio_data_index: usize,
        channel_map: &[Channel],
    ) -> f64 {
        let mut sum = 0.0;

        let channels = channel_map.len();
        assert!(audio_data.len() % channels == 0);
        let audio_data_stride = audio_data.len() / channels;
        assert!(audio_data_index <= audio_data_stride);

        for (c, (channel, audio_data)) in Iterator::zip(
            channel_map.iter(),
            audio_data.chunks_exact(audio_data_stride),
        )
        .enumerate()
        {
            if *channel == Channel::Unused {
                continue;
            }

            assert!(c < channels);
            assert!(audio_data_index <= audio_data.len());

            let mut channel_sum = 0.0;

            // XXX: Don't use channel_sum += sum() here because that gives slightly different
            // results than the C version because of rounding errors
            if audio_data_index < frames_per_block {
                for frame in &audio_data[..audio_data_index] {
                    channel_sum += *frame * *frame;
                }

                for frame in &audio_data[(audio_data.len() - frames_per_block + audio_data_index)..]
                {
                    channel_sum += *frame * *frame;
                }
            } else {
                for frame in &audio_data[(audio_data_index - frames_per_block)..audio_data_index] {
                    channel_sum += *frame * *frame;
                }
            }

            match channel {
                Channel::LeftSurround
                | Channel::RightSurround
                | Channel::Mp060
                | Channel::Mm060
                | Channel::Mp090
                | Channel::Mm090 => {
                    channel_sum *= 1.41;
                }
                Channel::DualMono => {
                    channel_sum *= 2.0;
                }
                _ => (),
            }

            sum += channel_sum;
        }

        sum /= frames_per_block as f64;

        sum
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2"
))]
mod ftz {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm_getcsr, _mm_setcsr, _MM_FLUSH_ZERO_ON};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm_getcsr, _mm_setcsr, _MM_FLUSH_ZERO_ON};

    pub struct Ftz(u32);

    impl Ftz {
        unsafe fn new() -> Self {
            let csr = _mm_getcsr();
            _mm_setcsr(csr | _MM_FLUSH_ZERO_ON);
            Ftz(csr)
        }
    }

    impl Drop for Ftz {
        fn drop(&mut self) {
            unsafe {
                _mm_setcsr(self.0);
            }
        }
    }

    pub fn with_ftz<F: FnOnce(Option<&Ftz>) -> T, T>(func: F) -> T {
        // Safety: MXCSR is unset in any case when Ftz goes out of scope and the closure also can't
        // mem::forget() it to prevent running the Drop impl.
        unsafe {
            let ftz = Ftz::new();
            func(Some(&ftz))
        }
    }
}

#[cfg(not(any(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2"
)),))]
mod ftz {
    pub enum Ftz {}

    pub fn with_ftz<F: FnOnce(Option<&Ftz>) -> T, T>(func: F) -> T {
        func(None)
    }
}
