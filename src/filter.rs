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

pub struct Filter {
    channels: u32,
    // BS.1770 filter coefficients (numerator).
    b: [f64; 5],
    // BS.1770 filter coefficients (denominator).
    a: [f64; 5],
    // One filter state per channel.
    filter_state: Vec<[f64; 5]>,

    calculate_sample_peak: bool,
    sample_peak: Vec<f64>,

    tp: Option<crate::true_peak::TruePeak>,
    true_peak: Vec<f64>,
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
            filter_state: vec![[0.0; 5]; channels as usize],
            calculate_sample_peak,
            sample_peak: vec![0.0; channels as usize],
            tp,
            true_peak: vec![0.0; channels as usize],
        }
    }

    pub fn reset_peaks(&mut self) {
        for v in &mut self.sample_peak {
            *v = 0.0;
        }

        for v in &mut self.true_peak {
            *v = 0.0;
        }
    }

    pub fn sample_peak(&self) -> &[f64] {
        &*self.sample_peak
    }

    pub fn true_peak(&self) -> &[f64] {
        &*self.true_peak
    }

    pub fn process<T: AsF64>(
        &mut self,
        src: &[T],
        dest: &mut [f64],
        channel_map: &[crate::ebur128::Channel],
    ) {
        let ftz = ftz::Ftz::new();

        assert!(src.len() == dest.len());
        assert!(src.len() % self.channels as usize == 0);
        assert!(dest.len() % self.channels as usize == 0);
        assert!(src.len() / self.channels as usize == dest.len() / self.channels as usize);
        assert!(channel_map.len() == self.channels as usize);
        assert!(self.filter_state.len() == self.channels as usize);

        // TODO: Deinterleaving into a &mut [f64] as a first step seems beneficial and
        // would also prevent the deinterleaving that now happens inside check_true_peak()
        // anyway

        if self.calculate_sample_peak {
            assert!(self.sample_peak.len() == self.channels as usize);

            for frame in src.chunks_exact(self.channels as usize) {
                for (c, sample) in frame.iter().enumerate() {
                    let v = sample.as_f64().abs();
                    if v > self.sample_peak[c] {
                        self.sample_peak[c] = v;
                    }
                }
            }
        }

        if let Some(ref mut tp) = self.tp {
            assert!(self.true_peak.len() == self.channels as usize);
            tp.check_true_peak(src, &mut *self.true_peak);
        }

        for (c, channel_map) in channel_map.iter().enumerate() {
            if *channel_map == crate::ebur128::Channel::Unused {
                continue;
            }

            let filter_state = &mut self.filter_state[c];
            for (src, dest) in src
                .chunks_exact(self.channels as usize)
                .zip(dest.chunks_exact_mut(self.channels as usize))
            {
                filter_state[0] = src[c].as_f64()
                    - self.a[1] * filter_state[1]
                    - self.a[2] * filter_state[2]
                    - self.a[3] * filter_state[3]
                    - self.a[4] * filter_state[4];
                dest[c] = self.b[0] * filter_state[0]
                    + self.b[1] * filter_state[1]
                    + self.b[2] * filter_state[2]
                    + self.b[3] * filter_state[3]
                    + self.b[4] * filter_state[4];

                filter_state[4] = filter_state[3];
                filter_state[3] = filter_state[2];
                filter_state[2] = filter_state[1];
                filter_state[1] = filter_state[0];
            }

            if ftz.is_none() {
                for v in filter_state {
                    if v.abs() < f64::EPSILON {
                        *v = 0.0;
                    }
                }
            }
        }
    }

    pub fn calc_gating_block(
        frames_per_block: usize,
        audio_data: &[f64],
        audio_data_index: usize,
        channel_map: &[Channel],
    ) -> f64 {
        let mut sum = 0.0;

        let channels = channel_map.len();
        assert!(audio_data_index <= audio_data.len());
        assert!(audio_data.len() % channels == 0);
        assert!(audio_data_index % channels == 0);

        for (c, channel) in channel_map.iter().enumerate() {
            if *channel == Channel::Unused {
                continue;
            }

            let mut channel_sum = 0.0;

            // XXX: Don't use channel_sum += sum() here because that gives slightly different
            // results than the C version because of rounding errors
            if audio_data_index < frames_per_block * channels {
                for frame in audio_data[..audio_data_index].chunks_exact(channels) {
                    channel_sum += frame[c] * frame[c];
                }

                for frame in audio_data
                    [(audio_data.len() - frames_per_block * channels + audio_data_index)..]
                    .chunks_exact(channels)
                {
                    channel_sum += frame[c] * frame[c];
                }
            } else {
                for frame in audio_data
                    [(audio_data_index - frames_per_block * channels)..audio_data_index]
                    .chunks_exact(channels)
                {
                    channel_sum += frame[c] * frame[c];
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

pub trait AsF64: crate::true_peak::AsF32 + Copy + PartialOrd {
    fn as_f64(self) -> f64;
}

impl AsF64 for i16 {
    fn as_f64(self) -> f64 {
        self as f64 / -(std::i16::MIN as f64)
    }
}

impl AsF64 for i32 {
    fn as_f64(self) -> f64 {
        self as f64 / -(std::i32::MIN as f64)
    }
}

impl AsF64 for f32 {
    fn as_f64(self) -> f64 {
        self as f64
    }
}

impl AsF64 for f64 {
    fn as_f64(self) -> f64 {
        self
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
        pub fn new() -> Option<Self> {
            unsafe {
                let csr = _mm_getcsr();

                _mm_setcsr(csr | _MM_FLUSH_ZERO_ON);

                Some(Ftz(csr))
            }
        }
    }

    impl Drop for Ftz {
        fn drop(&mut self) {
            unsafe {
                _mm_setcsr(self.0);
            }
        }
    }
}

#[cfg(not(any(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2"
)),))]
mod ftz {
    pub struct Ftz;

    impl Ftz {
        pub fn new() -> Option<Self> {
            None
        }
    }
}

#[cfg(feature = "c-tests")]
use std::os::raw::c_void;

#[cfg(feature = "c-tests")]
extern "C" {
    pub fn filter_create_c(
        rate: u32,
        channels: u32,
        calculate_sample_peak: i32,
        calculate_true_peak: i32,
    ) -> *mut c_void;
    pub fn filter_reset_peaks_c(filter: *mut c_void);
    pub fn filter_sample_peak_c(filter: *const c_void) -> *const f64;
    pub fn filter_true_peak_c(filter: *const c_void) -> *const f64;
    pub fn filter_process_short_c(
        filter: *mut c_void,
        frames: usize,
        src: *const i16,
        dest: *mut f64,
        channel_map: *const u32,
    );
    pub fn filter_process_int_c(
        filter: *mut c_void,
        frames: usize,
        src: *const i32,
        dest: *mut f64,
        channel_map: *const u32,
    );
    pub fn filter_process_float_c(
        filter: *mut c_void,
        frames: usize,
        src: *const f32,
        dest: *mut f64,
        channel_map: *const u32,
    );
    pub fn filter_process_double_c(
        filter: *mut c_void,
        frames: usize,
        src: *const f64,
        dest: *mut f64,
        channel_map: *const u32,
    );
    pub fn filter_destroy_c(filter: *mut c_void);

    pub fn calc_gating_block_c(
        frames_per_block: usize,
        audio_data: *const f64,
        audio_data_frames: usize,
        audio_data_index: usize,
        channel_map: *const u32,
        channels: usize,
    ) -> f64;
}

#[cfg(feature = "c-tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::Signal;
    use float_eq::assert_float_eq;
    use quickcheck_macros::quickcheck;

    fn compare_results(
        calculate_sample_peak: bool,
        calculate_true_peak: bool,
        sp: &[f64],
        tp: &[f64],
        sp_c: &[f64],
        tp_c: &[f64],
        data_out: &[f64],
        data_out_c: &[f64],
    ) {
        if calculate_sample_peak {
            for (i, (r, c)) in sp.iter().zip(sp_c.iter()).enumerate() {
                assert_float_eq!(
                    *r,
                    *c,
                    ulps <= 2,
                    "Rust and C implementation differ at sample peak {}",
                    i
                );
            }
        }

        if calculate_true_peak {
            for (i, (r, c)) in tp.iter().zip(tp_c.iter()).enumerate() {
                assert_float_eq!(
                    *r,
                    *c,
                    ulps <= 2,
                    "Rust and C implementation differ at true peak {}",
                    i
                );
            }
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
    }

    #[quickcheck]
    fn compare_c_impl_i16(
        signal: Signal<i16>,
        calculate_sample_peak: bool,
        calculate_true_peak: bool,
    ) {
        // Maximum of 400ms but our input is up to 5000ms, so distribute it evenly
        // by shrinking accordingly.
        let frames = signal.data.len() / signal.channels as usize;
        let frames = std::cmp::min(2 * frames / 25, 4 * ((signal.rate as usize + 5) / 10));

        let mut data_out = vec![0.0f64; frames * signal.channels as usize];
        let mut data_out_c = vec![0.0f64; frames * signal.channels as usize];

        let channel_map_c = vec![1; signal.channels as usize];
        let channel_map = vec![Channel::Left; signal.channels as usize];

        let (sp, tp) = {
            let mut f = Filter::new(
                signal.rate,
                signal.channels,
                calculate_sample_peak,
                calculate_true_peak,
            );
            f.process(
                &signal.data[..(frames * signal.channels as usize)],
                &mut data_out,
                &channel_map,
            );

            (Vec::from(f.sample_peak()), Vec::from(f.true_peak()))
        };

        let (sp_c, tp_c) = unsafe {
            use std::slice;

            let f = filter_create_c(
                signal.rate,
                signal.channels,
                if calculate_sample_peak { 1 } else { 0 },
                if calculate_true_peak { 1 } else { 0 },
            );
            filter_process_short_c(
                f,
                frames,
                signal.data[..(frames * signal.channels as usize)].as_ptr(),
                data_out_c.as_mut_ptr(),
                channel_map_c.as_ptr(),
            );

            let sp = Vec::from(slice::from_raw_parts(
                filter_sample_peak_c(f),
                signal.channels as usize,
            ));
            let tp = Vec::from(slice::from_raw_parts(
                filter_true_peak_c(f),
                signal.channels as usize,
            ));
            filter_destroy_c(f);
            (sp, tp)
        };

        compare_results(
            calculate_sample_peak,
            calculate_true_peak,
            &sp,
            &tp,
            &sp_c,
            &tp_c,
            &data_out,
            &data_out_c,
        );
    }

    #[quickcheck]
    fn compare_c_impl_i32(
        signal: Signal<i32>,
        calculate_sample_peak: bool,
        calculate_true_peak: bool,
    ) {
        // Maximum of 400ms but our input is up to 5000ms, so distribute it evenly
        // by shrinking accordingly.
        let frames = signal.data.len() / signal.channels as usize;
        let frames = std::cmp::min(2 * frames / 25, 4 * ((signal.rate as usize + 5) / 10));

        let mut data_out = vec![0.0f64; frames * signal.channels as usize];
        let mut data_out_c = vec![0.0f64; frames * signal.channels as usize];

        let channel_map_c = vec![1; signal.channels as usize];
        let channel_map = vec![Channel::Left; signal.channels as usize];

        let (sp, tp) = {
            let mut f = Filter::new(
                signal.rate,
                signal.channels,
                calculate_sample_peak,
                calculate_true_peak,
            );
            f.process(
                &signal.data[..(frames * signal.channels as usize)],
                &mut data_out,
                &channel_map,
            );

            (Vec::from(f.sample_peak()), Vec::from(f.true_peak()))
        };

        let (sp_c, tp_c) = unsafe {
            use std::slice;

            let f = filter_create_c(
                signal.rate,
                signal.channels,
                if calculate_sample_peak { 1 } else { 0 },
                if calculate_true_peak { 1 } else { 0 },
            );
            filter_process_int_c(
                f,
                frames,
                signal.data[..(frames * signal.channels as usize)].as_ptr(),
                data_out_c.as_mut_ptr(),
                channel_map_c.as_ptr(),
            );

            let sp = Vec::from(slice::from_raw_parts(
                filter_sample_peak_c(f),
                signal.channels as usize,
            ));
            let tp = Vec::from(slice::from_raw_parts(
                filter_true_peak_c(f),
                signal.channels as usize,
            ));
            filter_destroy_c(f);
            (sp, tp)
        };

        compare_results(
            calculate_sample_peak,
            calculate_true_peak,
            &sp,
            &tp,
            &sp_c,
            &tp_c,
            &data_out,
            &data_out_c,
        );
    }

    #[quickcheck]
    fn compare_c_impl_f32(
        signal: Signal<f32>,
        calculate_sample_peak: bool,
        calculate_true_peak: bool,
    ) {
        // Maximum of 400ms but our input is up to 5000ms, so distribute it evenly
        // by shrinking accordingly.
        let frames = signal.data.len() / signal.channels as usize;
        let frames = std::cmp::min(2 * frames / 25, 4 * ((signal.rate as usize + 5) / 10));

        let mut data_out = vec![0.0f64; frames * signal.channels as usize];
        let mut data_out_c = vec![0.0f64; frames * signal.channels as usize];

        let channel_map_c = vec![1; signal.channels as usize];
        let channel_map = vec![Channel::Left; signal.channels as usize];

        let (sp, tp) = {
            let mut f = Filter::new(
                signal.rate,
                signal.channels,
                calculate_sample_peak,
                calculate_true_peak,
            );
            f.process(
                &signal.data[..(frames * signal.channels as usize)],
                &mut data_out,
                &channel_map,
            );

            (Vec::from(f.sample_peak()), Vec::from(f.true_peak()))
        };

        let (sp_c, tp_c) = unsafe {
            use std::slice;

            let f = filter_create_c(
                signal.rate,
                signal.channels,
                if calculate_sample_peak { 1 } else { 0 },
                if calculate_true_peak { 1 } else { 0 },
            );
            filter_process_float_c(
                f,
                frames,
                signal.data[..(frames * signal.channels as usize)].as_ptr(),
                data_out_c.as_mut_ptr(),
                channel_map_c.as_ptr(),
            );

            let sp = Vec::from(slice::from_raw_parts(
                filter_sample_peak_c(f),
                signal.channels as usize,
            ));
            let tp = Vec::from(slice::from_raw_parts(
                filter_true_peak_c(f),
                signal.channels as usize,
            ));
            filter_destroy_c(f);
            (sp, tp)
        };

        compare_results(
            calculate_sample_peak,
            calculate_true_peak,
            &sp,
            &tp,
            &sp_c,
            &tp_c,
            &data_out,
            &data_out_c,
        );
    }

    #[quickcheck]
    fn compare_c_impl_f64(
        signal: Signal<f64>,
        calculate_sample_peak: bool,
        calculate_true_peak: bool,
    ) {
        // Maximum of 400ms but our input is up to 5000ms, so distribute it evenly
        // by shrinking accordingly.
        let frames = signal.data.len() / signal.channels as usize;
        let frames = std::cmp::min(2 * frames / 25, 4 * ((signal.rate as usize + 5) / 10));

        let mut data_out = vec![0.0f64; frames * signal.channels as usize];
        let mut data_out_c = vec![0.0f64; frames * signal.channels as usize];

        let channel_map_c = vec![1; signal.channels as usize];
        let channel_map = vec![Channel::Left; signal.channels as usize];

        let (sp, tp) = {
            let mut f = Filter::new(
                signal.rate,
                signal.channels,
                calculate_sample_peak,
                calculate_true_peak,
            );
            f.process(
                &signal.data[..(frames * signal.channels as usize)],
                &mut data_out,
                &channel_map,
            );

            (Vec::from(f.sample_peak()), Vec::from(f.true_peak()))
        };

        let (sp_c, tp_c) = unsafe {
            use std::slice;

            let f = filter_create_c(
                signal.rate,
                signal.channels,
                if calculate_sample_peak { 1 } else { 0 },
                if calculate_true_peak { 1 } else { 0 },
            );
            filter_process_double_c(
                f,
                frames,
                signal.data[..(frames * signal.channels as usize)].as_ptr(),
                data_out_c.as_mut_ptr(),
                channel_map_c.as_ptr(),
            );

            let sp = Vec::from(slice::from_raw_parts(
                filter_sample_peak_c(f),
                signal.channels as usize,
            ));
            let tp = Vec::from(slice::from_raw_parts(
                filter_true_peak_c(f),
                signal.channels as usize,
            ));
            filter_destroy_c(f);
            (sp, tp)
        };

        compare_results(
            calculate_sample_peak,
            calculate_true_peak,
            &sp,
            &tp,
            &sp_c,
            &tp_c,
            &data_out,
            &data_out_c,
        );
    }

    #[derive(Clone, Debug)]
    struct GatingBlock {
        frames_per_block: usize,
        audio_data: Vec<f64>,
        audio_data_index: usize,
        channels: u32,
    }

    impl quickcheck::Arbitrary for GatingBlock {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            use rand::Rng;

            let channels = g.gen_range(1, 16);

            let rate = 48_000;

            let samples_in_100ms = (rate + 5) / 10;
            let (frames_per_block, window) = if g.gen() {
                (4 * samples_in_100ms, 400)
            } else {
                (30 * samples_in_100ms, 3000)
            };

            let mut audio_data_frames = rate * window / 1000;
            if audio_data_frames % samples_in_100ms != 0 {
                // round up to multiple of samples_in_100ms
                audio_data_frames =
                    (audio_data_frames + samples_in_100ms) - (audio_data_frames % samples_in_100ms);
            }

            let mut audio_data = vec![0.0; audio_data_frames * channels as usize];
            for v in &mut audio_data {
                *v = g.gen_range(-1.0, 1.0);
            }

            let audio_data_index = g.gen_range(0, audio_data_frames) * channels as usize;

            GatingBlock {
                frames_per_block,
                audio_data,
                audio_data_index,
                channels,
            }
        }
    }

    fn default_channel_map_c(channels: u32) -> Vec<u32> {
        match channels {
            4 => vec![1, 2, 4, 5],
            5 => vec![1, 2, 3, 4, 5],
            _ => {
                let mut v = vec![0; channels as usize];

                let set_channels = std::cmp::min(channels as usize, 6);
                v[0..set_channels].copy_from_slice(&[1, 2, 3, 0, 4, 5][..set_channels]);

                v
            }
        }
    }

    #[quickcheck]
    fn compare_c_impl_calc_gating_block(block: GatingBlock) {
        let channel_map = crate::ebur128::default_channel_map(block.channels);
        let channel_map_c = default_channel_map_c(block.channels);

        let energy = Filter::calc_gating_block(
            block.frames_per_block,
            &block.audio_data,
            block.audio_data_index,
            &channel_map,
        );
        let energy_c = unsafe {
            calc_gating_block_c(
                block.frames_per_block,
                block.audio_data.as_ptr(),
                block.audio_data.len() / block.channels as usize,
                block.audio_data_index,
                channel_map_c.as_ptr(),
                block.channels as usize,
            )
        };

        assert_float_eq!(energy, energy_c, ulps <= 2);
    }
}
