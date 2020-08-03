use crate::interp::Interp;

#[derive(Debug)]
pub struct TruePeak {
    interp: Interp,
    rate: u32,
    channels: u32,
    buffer_input: Vec<f32>,
    buffer_output: Vec<f32>,
}

impl TruePeak {
    pub fn new(rate: u32, channels: u32) -> Option<Self> {
        let samples_in_100ms = (rate + 5) / 10;

        let (interp, interp_factor) = if rate < 96000 {
            (Interp::new(49, 4, channels), 4)
        } else if rate < 192000 {
            (Interp::new(49, 2, channels), 2)
        } else {
            return None;
        };

        let buffer_input = vec![0.0; 4 * samples_in_100ms as usize * channels as usize];
        let buffer_output = vec![0.0; buffer_input.len() * interp_factor];

        Some(Self {
            interp,
            rate,
            channels,
            buffer_input,
            buffer_output,
        })
    }

    // FIXME: Use f32 for storage
    pub fn check_true_peak<T: AsF32>(&mut self, src: &[T], peaks: &mut [f64]) {
        assert!(src.len() <= self.buffer_input.len());
        assert!(src.len() * self.interp.get_factor() <= self.buffer_output.len());
        assert!(self.buffer_input.len() * self.interp.get_factor() == self.buffer_output.len());
        assert!(peaks.len() == self.channels as usize);

        // Deinterleave and convert to f32 for the resampler
        let frames = src.len() / self.channels as usize;
        for (s, i) in src.chunks_exact(self.channels as usize).enumerate() {
            for (c, i) in i.iter().enumerate() {
                // Safety: self.buffer_input.len() >= src.len() and we just reorder
                *unsafe { self.buffer_input.get_unchecked_mut(frames * c + s) } = i.as_f32();
            }
        }

        let interp_factor = self.interp.get_factor();

        self.interp.process(
            &self.buffer_input[..(src.len())],
            &mut self.buffer_output[..(src.len() * interp_factor)],
        );

        // Find the maximum
        for (c, o) in self.buffer_output[..(frames * self.channels as usize * interp_factor)]
            .chunks_exact(frames * interp_factor)
            .enumerate()
        {
            let mut max = 0.0;
            for v in o {
                max = f32_max(max, v.abs());
            }
            peaks[c] = f64_max(max as f64, peaks[c]);
        }
    }
}

fn f32_max(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

fn f64_max(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}

pub trait AsF32: Copy {
    fn as_f32(self) -> f32;
}

impl AsF32 for i16 {
    fn as_f32(self) -> f32 {
        self as f32 / (-(std::i16::MIN as f32))
    }
}

impl AsF32 for i32 {
    fn as_f32(self) -> f32 {
        self as f32 / (-(std::i32::MIN as f32))
    }
}

impl AsF32 for f32 {
    fn as_f32(self) -> f32 {
        self
    }
}

impl AsF32 for f64 {
    fn as_f32(self) -> f32 {
        self as f32
    }
}

#[no_mangle]
pub unsafe extern "C" fn true_peak_create(rate: u32, channels: u32) -> *mut TruePeak {
    TruePeak::new(rate, channels)
        .map(Box::new)
        .map(Box::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn true_peak_check_short(
    tp: *mut TruePeak,
    frames: usize,
    src: *const i16,
    peaks: *mut f64,
) {
    use std::slice;

    let tp = &mut *tp;
    let src = slice::from_raw_parts(src, tp.channels as usize * frames);
    let peaks = slice::from_raw_parts_mut(peaks, tp.channels as usize);

    tp.check_true_peak(src, peaks);
}

#[no_mangle]
pub unsafe extern "C" fn true_peak_check_int(
    tp: *mut TruePeak,
    frames: usize,
    src: *const i32,
    peaks: *mut f64,
) {
    use std::slice;

    let tp = &mut *tp;
    let src = slice::from_raw_parts(src, tp.channels as usize * frames);
    let peaks = slice::from_raw_parts_mut(peaks, tp.channels as usize);

    tp.check_true_peak(src, peaks);
}

#[no_mangle]
pub unsafe extern "C" fn true_peak_check_float(
    tp: *mut TruePeak,
    frames: usize,
    src: *const f32,
    peaks: *mut f64,
) {
    use std::slice;

    let tp = &mut *tp;
    let src = slice::from_raw_parts(src, tp.channels as usize * frames);
    let peaks = slice::from_raw_parts_mut(peaks, tp.channels as usize);

    tp.check_true_peak(src, peaks);
}

#[no_mangle]
pub unsafe extern "C" fn true_peak_check_double(
    tp: *mut TruePeak,
    frames: usize,
    src: *const f64,
    peaks: *mut f64,
) {
    use std::slice;

    let tp = &mut *tp;
    let src = slice::from_raw_parts(src, tp.channels as usize * frames);
    let peaks = slice::from_raw_parts_mut(peaks, tp.channels as usize);

    tp.check_true_peak(src, peaks);
}

#[no_mangle]
pub unsafe extern "C" fn true_peak_destroy(tp: *mut TruePeak) {
    if tp.is_null() {
        return;
    }

    drop(Box::from_raw(tp));
}

#[cfg(feature = "internal-tests")]
use std::os::raw::c_void;

#[cfg(feature = "internal-tests")]
extern "C" {
    pub fn true_peak_create_c(rate: u32, channels: u32) -> *mut c_void;
    pub fn true_peak_check_short_c(
        tp: *mut c_void,
        frames: usize,
        src: *const i16,
        peaks: *mut f64,
    );
    pub fn true_peak_check_int_c(tp: *mut c_void, frames: usize, src: *const i32, peaks: *mut f64);
    pub fn true_peak_check_float_c(
        tp: *mut c_void,
        frames: usize,
        src: *const f32,
        peaks: *mut f64,
    );
    pub fn true_peak_check_double_c(
        tp: *mut c_void,
        frames: usize,
        src: *const f64,
        peaks: *mut f64,
    );
    pub fn true_peak_destroy_c(tp: *mut c_void);
}

#[cfg(all(test, feature = "internal-tests"))]
mod tests {
    use super::*;

    #[test]
    fn compare_c_impl_i16() {
        use float_cmp::approx_eq;

        let mut data = vec![0i16; 19200 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * std::i16::MAX as f32;
            out[0] = val as i16;
            out[1] = val as i16;
            accumulator += step;
        }

        let mut peaks = vec![0.0f64; 2];
        let mut peaks_c = vec![0.0f64; 2];

        {
            let mut tp = TruePeak::new(48_000, 2).unwrap();
            tp.check_true_peak(&data, &mut peaks);
        }

        unsafe {
            let tp = true_peak_create_c(48_000, 2);
            assert!(!tp.is_null());
            true_peak_check_short_c(tp, 19200, data.as_ptr(), peaks_c.as_mut_ptr());
            true_peak_destroy_c(tp);
        }

        for (i, (r, c)) in peaks.iter().zip(peaks_c.iter()).enumerate() {
            assert!(
                approx_eq!(f64, *r, *c, ulps = 2),
                "Rust and C implementation differ at channel {}: {} != {}",
                i,
                r,
                c
            );
        }
    }

    #[test]
    fn compare_c_impl_i32() {
        use float_cmp::approx_eq;

        let mut data = vec![0i32; 19200 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * std::i32::MAX as f32;
            out[0] = val as i32;
            out[1] = val as i32;
            accumulator += step;
        }

        let mut peaks = vec![0.0f64; 2];
        let mut peaks_c = vec![0.0f64; 2];

        {
            let mut tp = TruePeak::new(48_000, 2).unwrap();
            tp.check_true_peak(&data, &mut peaks);
        }

        unsafe {
            let tp = true_peak_create_c(48_000, 2);
            assert!(!tp.is_null());
            true_peak_check_int_c(tp, 19200, data.as_ptr(), peaks_c.as_mut_ptr());
            true_peak_destroy_c(tp);
        }

        for (i, (r, c)) in peaks.iter().zip(peaks_c.iter()).enumerate() {
            assert!(
                approx_eq!(f64, *r, *c, ulps = 2),
                "Rust and C implementation differ at channel {}: {} != {}",
                i,
                r,
                c
            );
        }
    }

    #[test]
    fn compare_c_impl_f32() {
        use float_cmp::approx_eq;

        let mut data = vec![0.0f32; 19200 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val;
            out[1] = val;
            accumulator += step;
        }

        let mut peaks = vec![0.0f64; 2];
        let mut peaks_c = vec![0.0f64; 2];

        {
            let mut tp = TruePeak::new(48_000, 2).unwrap();
            tp.check_true_peak(&data, &mut peaks);
        }

        unsafe {
            let tp = true_peak_create_c(48_000, 2);
            assert!(!tp.is_null());
            true_peak_check_float_c(tp, 19200, data.as_ptr(), peaks_c.as_mut_ptr());
            true_peak_destroy_c(tp);
        }

        for (i, (r, c)) in peaks.iter().zip(peaks_c.iter()).enumerate() {
            assert!(
                approx_eq!(f64, *r, *c, ulps = 2),
                "Rust and C implementation differ at channel {}: {} != {}",
                i,
                r,
                c
            );
        }
    }

    #[test]
    fn compare_c_impl_f64() {
        use float_cmp::approx_eq;

        let mut data = vec![0.0f64; 19200 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val as f64;
            out[1] = val as f64;
            accumulator += step;
        }

        let mut peaks = vec![0.0f64; 2];
        let mut peaks_c = vec![0.0f64; 2];

        {
            let mut tp = TruePeak::new(48_000, 2).unwrap();
            tp.check_true_peak(&data, &mut peaks);
        }

        unsafe {
            let tp = true_peak_create_c(48_000, 2);
            assert!(!tp.is_null());
            true_peak_check_double_c(tp, 19200, data.as_ptr(), peaks_c.as_mut_ptr());
            true_peak_destroy_c(tp);
        }

        for (i, (r, c)) in peaks.iter().zip(peaks_c.iter()).enumerate() {
            assert!(
                approx_eq!(f64, *r, *c, ulps = 2),
                "Rust and C implementation differ at channel {}: {} != {}",
                i,
                r,
                c
            );
        }
    }
}
