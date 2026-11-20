// Copyright (c) 2026 Michael Apel
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

//! True-peak scanner for [`Mode::TRUE_PEAK_ACCURATE`](crate::Mode::TRUE_PEAK_ACCURATE).
//!
//! The default true-peak scanner uses the 48-tap example filter from ITU-R BS.1770 Annex 2. Its
//! passband droops near the top of the band (-2.4 dB at 20 kHz for 44.1 kHz input), so content
//! there can read several dB low. This scanner uses a polyphase FIR that is flat to
//! `PASSBAND_EDGE` (20 kHz at 44.1 kHz) and fits a parabola through each oversampled local maximum
//! to catch peaks that fall between the oversampled points. BS.1770 permits any method that gives
//! "similar or superior results" to its guidelines.
//!
//! The filter is defined relative to the sample rate, so its taps per phase don't depend on the
//! oversampling factor; only the factor scales with the rate (see [`TruePeak::new_accurate`]).
//!
//! [`TruePeak::new_accurate`]: crate::true_peak::TruePeak::new_accurate

use std::convert::TryInto;
use std::f64::consts::PI;

/// Taps per polyphase branch. A multiple of `LANES`, so the fused loop has no scalar remainder.
const TAPS: usize = 40;
/// Independent partial sums in the fused filter loop (see `filter_split`).
const LANES: usize = 4;
const _: () = assert!(TAPS % LANES == 0);
/// Passband edge, as a fraction of the sample rate: 20 kHz at 44.1 kHz. The stopband edge mirrors
/// it about Nyquist, so an image of in-band content always lands in the stopband.
const PASSBAND_EDGE: f64 = 0.4535;
/// Kaiser β for 60 dB of stopband attenuation (`0.1102 * (60 - 8.7)`). Images at -60 dB move a
/// reading by under 0.01 dB.
const KAISER_BETA: f64 = 5.653;

#[derive(Debug, Clone)]
struct ChannelState {
    /// `TAPS` samples written twice (at `pos` and `pos + TAPS`) so the newest `TAPS` are always
    /// one contiguous slice, newest first.
    history: [f32; 2 * TAPS],
    pos: usize,
    /// Last two oversampled magnitudes, oldest first, for the parabola.
    tail: [f32; 2],
}

impl ChannelState {
    fn new() -> Self {
        ChannelState {
            history: [0.0; 2 * TAPS],
            pos: 0,
            tail: [0.0; 2],
        }
    }
}

/// Oversampling by `F` with a parabolic fit at each oversampled maximum.
#[derive(Debug)]
pub struct Scanner<const F: usize> {
    /// `coeffs[i][phase]` multiplies the sample `i` steps back. Tap-major, so each input sample
    /// updates all `F` phases with one `F`-wide multiply-add.
    coeffs: Box<[[f32; F]; TAPS]>,
    /// See `scan_with`.
    gate: f32,
    channels: Box<[ChannelState]>,
    /// One channel's samples as `f32`, reused between calls.
    scratch: Vec<f32>,
    /// Whether this CPU has AVX2 and FMA, checked once: picks `scan_avx2` over `scan_generic`.
    /// Always false off x86.
    #[cfg_attr(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        allow(dead_code)
    )]
    avx2: bool,
}

impl<const F: usize> Scanner<F> {
    pub fn new(channels: u32) -> Self {
        Scanner {
            coeffs: design::<F>(),
            // A peak lies at most half an oversampled step from the nearest point, so for in-band
            // content the vertex sits at most 1/cos(π·PASSBAND_EDGE/F) above its middle point.
            gate: ((PI * PASSBAND_EDGE / F as f64).cos() * 0.995) as f32,
            channels: vec![ChannelState::new(); channels as usize].into_boxed_slice(),
            scratch: Vec::new(),
            avx2: has_avx2_fma(),
        }
    }

    pub fn reset(&mut self) {
        self.channels
            .iter_mut()
            .for_each(|c| *c = ChannelState::new());
    }

    pub fn check_true_peak<'a, T: crate::utils::Sample + 'a, S: crate::Samples<'a, T>>(
        &mut self,
        src: S,
        peaks: &mut [f64],
    ) {
        assert!(src.channels() == self.channels.len() && peaks.len() == self.channels.len());
        let mut scratch = std::mem::take(&mut self.scratch);
        for (c, peak) in peaks.iter_mut().enumerate() {
            scratch.clear();
            src.foreach_sample(c, |s| scratch.push(s.to_sample::<f32>()));
            self.scan(c, &scratch, peak);
        }
        self.scratch = scratch;
    }

    fn scan(&mut self, channel: usize, samples: &[f32], peak: &mut f64) {
        let state = &mut self.channels[channel];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if self.avx2 {
            // SAFETY: `avx2` is only true when the CPU reported AVX2 and FMA (`has_avx2_fma`).
            unsafe { scan_avx2(&self.coeffs, self.gate, state, samples, peak) };
            return;
        }
        scan_generic(&self.coeffs, self.gate, state, samples, peak);
    }

    #[cfg(test)]
    fn force_generic(&mut self) {
        self.avx2 = false;
    }
}

/// Whether `mul_add` is a single hardware instruction on every CPU of this target. Elsewhere it
/// can be a slow library call, so the plain split loop is used instead.
const FMA_IN_BASELINE: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// The fused loop only vectorizes at 8x, where each partial sum spans a whole 8-lane vector; at
/// 4x and 2x the compiler leaves it scalar and the plain split loop is faster.
#[inline(always)]
fn scan_generic<const F: usize>(
    coeffs: &[[f32; F]; TAPS],
    gate: f32,
    state: &mut ChannelState,
    samples: &[f32],
    peak: &mut f64,
) {
    if F >= 8 && FMA_IN_BASELINE {
        scan_with(coeffs, gate, state, samples, peak, filter_split_fma);
    } else {
        scan_with(coeffs, gate, state, samples, peak, filter_split);
    }
}

/// The same loops compiled for AVX2+FMA. One second of 96 kHz stereo in release: 3.0 ms at 8x
/// with the fused loop and 3.0 ms at 4x with the plain one, against 5.6 ms and 3.3 ms on SSE2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_avx2<const F: usize>(
    coeffs: &[[f32; F]; TAPS],
    gate: f32,
    state: &mut ChannelState,
    samples: &[f32],
    peak: &mut f64,
) {
    if F >= 8 {
        scan_with(coeffs, gate, state, samples, peak, filter_split_fma);
    } else {
        scan_with(coeffs, gate, state, samples, peak, filter_split);
    }
}

#[inline(always)]
fn scan_with<const F: usize>(
    coeffs: &[[f32; F]; TAPS],
    gate: f32,
    state: &mut ChannelState,
    samples: &[f32],
    peak: &mut f64,
    filter: impl Fn(&[[f32; F]; TAPS], &[f32; TAPS]) -> [f32; F],
) {
    let mut best = *peak as f32;
    let [mut y0, mut y1] = state.tail;
    for &s in samples {
        state.pos = if state.pos == 0 {
            TAPS - 1
        } else {
            state.pos - 1
        };
        state.history[state.pos] = s;
        state.history[state.pos + TAPS] = s;
        let recent: &[f32; TAPS] = state.history[state.pos..state.pos + TAPS]
            .try_into()
            .unwrap();
        for y in filter(coeffs, recent) {
            let y2 = y.abs();
            // The vertex only matters when it could raise the reading, and it can't sit far above
            // its middle point, so most points skip the fit.
            if y1 > best * gate {
                best = best.max(parabola_peak(y0, y1, y2));
            }
            best = best.max(y2);
            y0 = y1;
            y1 = y2;
        }
    }
    state.tail = [y0, y1];
    *peak = best as f64;
}

/// All phases' outputs for one input sample, from `LANES` partial sums over interleaved taps.
/// With one running sum every tap waits on the previous tap's add, and that chain, not the
/// arithmetic, sets the speed.
#[inline(always)]
fn filter_split<const F: usize>(coeffs: &[[f32; F]; TAPS], recent: &[f32; TAPS]) -> [f32; F] {
    let mut part = [[0.0f32; F]; LANES];
    for i in (0..TAPS).step_by(LANES) {
        for (j, p) in part.iter_mut().enumerate() {
            let x = recent[i + j];
            for (a, &t) in p.iter_mut().zip(&coeffs[i + j]) {
                *a += t * x;
            }
        }
    }
    sum_parts(part)
}

/// `filter_split` with fused multiply-adds. Only used where FMA is in hardware.
#[inline(always)]
fn filter_split_fma<const F: usize>(coeffs: &[[f32; F]; TAPS], recent: &[f32; TAPS]) -> [f32; F] {
    let mut part = [[0.0f32; F]; LANES];
    for i in (0..TAPS).step_by(LANES) {
        for (j, p) in part.iter_mut().enumerate() {
            let x = recent[i + j];
            for (a, &t) in p.iter_mut().zip(&coeffs[i + j]) {
                *a = t.mul_add(x, *a);
            }
        }
    }
    sum_parts(part)
}

#[inline(always)]
fn sum_parts<const F: usize>(part: [[f32; F]; LANES]) -> [f32; F] {
    let mut acc = part[0];
    for p in &part[1..] {
        for (a, &v) in acc.iter_mut().zip(p) {
            *a += v;
        }
    }
    acc
}

/// Vertex of the parabola through three equally spaced magnitudes, when the middle one is a local
/// maximum; `0.0` otherwise.
#[inline(always)]
fn parabola_peak(y0: f32, y1: f32, y2: f32) -> f32 {
    let curvature = y0 - 2.0 * y1 + y2;
    if y1 >= y0 && y1 >= y2 && curvature < 0.0 {
        y1 - (y0 - y2) * (y0 - y2) / (8.0 * curvature)
    } else {
        0.0
    }
}

fn has_avx2_fma() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Kaiser-windowed sinc lowpass at Nyquist, at the oversampled rate, split into `F` polyphase
/// branches. Each branch is normalized to unity DC gain, so a constant input reads exactly itself.
fn design<const F: usize>() -> Box<[[f32; F]; TAPS]> {
    let len = F * TAPS;
    let centre = (len - 1) as f64 / 2.0;
    let i0_beta = bessel_i0(KAISER_BETA);
    let proto: Vec<f64> = (0..len)
        .map(|n| {
            let m = n as f64 - centre;
            // Cutoff halfway through the transition band, which is Nyquist.
            let x = m / F as f64;
            let sinc = if x.abs() < 1e-12 {
                1.0
            } else {
                (PI * x).sin() / (PI * x)
            };
            let r = m / centre;
            sinc * bessel_i0(KAISER_BETA * (1.0 - r * r).max(0.0).sqrt()) / i0_beta
        })
        .collect();
    let mut coeffs = Box::new([[0.0f32; F]; TAPS]);
    for phase in 0..F {
        let sum: f64 = (0..TAPS).map(|i| proto[i * F + phase]).sum();
        for (i, taps) in coeffs.iter_mut().enumerate() {
            taps[phase] = (proto[i * F + phase] / sum) as f32;
        }
    }
    coeffs
}

/// Modified Bessel function of the first kind, order 0, by its power series.
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, mut k) = (1.0f64, 1.0f64, 1.0f64);
    while term > 1e-12 * sum {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
        k += 1.0;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EbuR128, Mode};

    /// One period of a 16-sample pattern whose odd harmonics (fs/16 .. 7·fs/16) all peak together
    /// half a sample after index 0.
    const PATTERN: [f32; 16] = [
        1., 1., -1., 1., -1., 1., -1., 1., -1., -1., 1., -1., 1., -1., 1., -1.,
    ];
    /// The pattern's true peak over its sample peak: the sum of the Dirichlet kernel's absolute
    /// values at that half-sample instant.
    const PATTERN_GAIN: f64 = 2.287_016_068_458_563;

    fn db(x: f64) -> f64 {
        20.0 * x.log10()
    }

    /// Steady-state reading of a periodic mono signal through `EbuR128` in `mode`.
    fn read_periodic(period: &[f32], rate: u32, mode: Mode) -> f64 {
        let mut ebu = EbuR128::new(1, rate, mode).unwrap();
        let frames: Vec<f32> = period
            .iter()
            .cycle()
            .take(period.len() * (400 / period.len() + 2))
            .copied()
            .collect();
        ebu.seed_frames_f32(&frames).unwrap();
        let frames: Vec<f32> = period
            .iter()
            .cycle()
            .take(period.len() * 4)
            .copied()
            .collect();
        ebu.add_frames_f32(&frames).unwrap();
        ebu.true_peak(0).unwrap()
    }

    /// The case that motivated this mode: at 0 dBTP the pattern reads 0 dBTP, where the default
    /// mode reads about -1.8 dBTP.
    #[test]
    fn pattern_at_zero_dbtp_reads_zero_dbtp() {
        let period: Vec<f32> = PATTERN
            .iter()
            .map(|&s| (s as f64 / PATTERN_GAIN) as f32)
            .collect();
        for rate in [44_100, 48_000, 96_000] {
            let accurate = db(read_periodic(&period, rate, Mode::TRUE_PEAK_ACCURATE));
            assert!(
                (-0.03..=0.02).contains(&accurate),
                "{} Hz: read {:.4} dBTP for a 0 dBTP signal",
                rate,
                accurate
            );
        }
        let standard = db(read_periodic(&period, 48_000, Mode::TRUE_PEAK));
        assert!(standard < -1.7, "default mode read {:.4} dBTP", standard);
    }

    /// Unit sines up to the passband edge, each at several phases, read 0 dBTP to within a few
    /// hundredths of a dB. Whole cycles per 4096 samples keep each one exactly periodic.
    #[test]
    fn in_band_sines_read_their_true_peak_at_any_phase() {
        let n = 4096usize;
        for (rate, tolerance) in [(48_000, 0.01), (96_000, 0.04)] {
            let mut worst = (0.0f64, 0.0f64);
            for k in (40..=(PASSBAND_EDGE * n as f64) as usize).step_by(61) {
                for p in 0..4 {
                    let phase = p as f64 * 1.2345;
                    let x: Vec<f32> = (0..n)
                        .map(|i| (2.0 * PI * (k * i) as f64 / n as f64 + phase).cos() as f32)
                        .collect();
                    let read = db(read_periodic(&x, rate, Mode::TRUE_PEAK_ACCURATE));
                    worst = (worst.0.min(read), worst.1.max(read));
                }
            }
            assert!(
                worst.0 > -tolerance,
                "{} Hz: worst under-read {:.4} dB",
                rate,
                worst.0
            );
            assert!(
                worst.1 < 0.02,
                "{} Hz: worst over-read {:.4} dB",
                rate,
                worst.1
            );
        }
    }

    /// EBU Tech 3341 (2023) true-peak signals 15-19: within their +0.2/-0.4 dBTP tolerance, and
    /// within 0.02 dB of the exact value.
    #[test]
    fn meets_the_tech_3341_true_peak_signals() {
        for &(case, freq, phase_deg, amp, expected) in &[
            (15, 1.0 / 4.0, 0.0f64, 0.5f64, -6.0f64),
            (16, 1.0 / 4.0, 45.0, 0.5, -6.0),
            (17, 1.0 / 6.0, 60.0, 0.5, -6.0),
            (18, 1.0 / 8.0, 67.5, 0.5, -6.0),
            (19, 1.0 / 4.0, 45.0, 1.41, 3.0),
        ] {
            let period = (1.0 / freq) as usize;
            let x: Vec<f32> = (0..period)
                .map(|i| (amp * (2.0 * PI * freq * i as f64 + phase_deg.to_radians()).sin()) as f32)
                .collect();
            let read = db(read_periodic(&x, 48_000, Mode::TRUE_PEAK_ACCURATE));
            assert!(
                (expected - 0.4..=expected + 0.2).contains(&read),
                "case {}: read {:.3}",
                case,
                read
            );
            assert!(
                (read - db(amp)).abs() < 0.02,
                "case {}: read {:.4}, exact {:.4}",
                case,
                read,
                db(amp)
            );
        }
    }

    /// The AVX2 path and the generic one must agree, so the generic one stays covered on a machine
    /// whose tests would otherwise only run the AVX2 one.
    #[test]
    fn avx2_and_generic_paths_agree() {
        if !has_avx2_fma() {
            return;
        }
        let mut rng = 0x1234_5678u32;
        let x: Vec<f32> = (0..9600)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                (rng as f32 / u32::MAX as f32) * 1.6 - 0.8
            })
            .collect();
        let (mut fast, mut portable) = (Scanner::<8>::new(2), Scanner::<8>::new(2));
        portable.force_generic();
        for chunk in x.chunks(480) {
            let (mut a, mut b) = ([0.0f64; 2], [0.0f64; 2]);
            fast.check_true_peak(crate::Interleaved::new(chunk, 2).unwrap(), &mut a);
            portable.check_true_peak(crate::Interleaved::new(chunk, 2).unwrap(), &mut b);
            for (a, b) in a.iter().zip(&b) {
                assert!(
                    (a - b).abs() <= 1e-5 * a.max(*b),
                    "avx2 {} vs generic {}",
                    a,
                    b
                );
            }
        }
    }

    /// The plain and fused filter loops compute the same outputs. Runs on every target: where FMA
    /// isn't in hardware, `mul_add` is merely slow.
    #[test]
    fn plain_and_fused_filter_loops_agree() {
        let coeffs = design::<8>();
        let mut rng = 0x9e37_79b9u32;
        for _ in 0..200 {
            let mut recent = [0.0f32; TAPS];
            for r in recent.iter_mut() {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                *r = (rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
            }
            let (a, b) = (
                filter_split(&coeffs, &recent),
                filter_split_fma(&coeffs, &recent),
            );
            for (a, b) in a.iter().zip(&b) {
                assert!((a - b).abs() <= 1e-5, "plain {} vs fused {}", a, b);
            }
        }
    }

    /// Channels are measured independently.
    #[test]
    fn channels_are_independent() {
        let x: Vec<f32> = (0..4800)
            .flat_map(|i| {
                let s = (2.0 * PI * 1000.0 * i as f64 / 48_000.0).sin() as f32;
                vec![s * 0.1, s * 0.8]
            })
            .collect();
        let mut ebu = EbuR128::new(2, 48_000, Mode::TRUE_PEAK_ACCURATE).unwrap();
        ebu.add_frames_f32(&x).unwrap();
        assert!((db(ebu.true_peak(0).unwrap()) - db(0.1)).abs() < 0.02);
        assert!((db(ebu.true_peak(1).unwrap()) - db(0.8)).abs() < 0.02);
    }

    /// `reset()` clears the filter history as well as the peaks, as in the default mode.
    #[test]
    fn reset_clears_peaks_and_history() {
        let x: Vec<f32> = (0..4800)
            .map(|i| (2.0 * PI * 1000.0 * i as f64 / 48_000.0).sin() as f32 * 0.5)
            .collect();
        let mut ebu = EbuR128::new(1, 48_000, Mode::TRUE_PEAK_ACCURATE).unwrap();
        ebu.add_frames_f32(&x).unwrap();
        assert!(ebu.true_peak(0).unwrap() > 0.49);
        ebu.reset();
        assert_eq!(ebu.true_peak(0).unwrap(), 0.0);
        ebu.add_frames_f32(&vec![0.0; 480]).unwrap();
        assert_eq!(ebu.true_peak(0).unwrap(), 0.0, "history not cleared");
    }
}
