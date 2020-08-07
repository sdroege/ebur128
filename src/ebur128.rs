use crate::ffi;

use bitflags::bitflags;

use std::error;
use std::fmt;
use std::mem;
use std::ptr;

/// Error values for [`EbuR128`](struct.EbuR128.html) functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not enough memory
    NoMem,
    /// Invalid mode selected
    InvalidMode,
    /// Invalid channel index passed
    InvalidChannelIndex,
}

impl Error {
    fn from_ffi<T, F: FnOnce() -> T>(res: ffi::error, func: F) -> Result<T, Error> {
        match res {
            ffi::error_EBUR128_SUCCESS | ffi::error_EBUR128_ERROR_NO_CHANGE => Ok(func()),
            ffi::error_EBUR128_ERROR_NOMEM => Err(Error::NoMem),
            ffi::error_EBUR128_ERROR_INVALID_MODE => Err(Error::InvalidMode),
            ffi::error_EBUR128_ERROR_INVALID_CHANNEL_INDEX => Err(Error::InvalidChannelIndex),
            _ => unreachable!("Unsupported error return {}", res),
        }
    }
}

impl error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::NoMem => write!(f, "NoMem"),
            Error::InvalidMode => write!(f, "Invalid Mode"),
            Error::InvalidChannelIndex => write!(f, "Invalid Channel Index"),
        }
    }
}

bitflags! {
    /// Processing mode.
    ///
    /// Use these values in [`EbuR128::new`](struct.EbuR128.html#method.new). Try to use the lowest
    /// possible modes that suit your needs, as performance will be better.
    pub struct Mode: ffi::mode {
        /// can call [`EbuR128::loudness_momentary`](struct.EbuR128.html#method.loudness_momentary)
        const M = ffi::mode_EBUR128_MODE_M;
        /// can call [`EbuR128::loudness_shortterm`](struct.EbuR128.html#method.loudness_shortterm)
        const S = ffi::mode_EBUR128_MODE_S;
        /// can call [`EbuR128::loudness_global`](struct.EbuR128.html#method.loudness_global) and
        /// [`EbuR128::relative_threshold`](struct.EbuR128.html#method.relative_threshold)
        const I = ffi::mode_EBUR128_MODE_I;
        /// can call [`EbuR128::loudness_range`](struct.EbuR128.html#method.loudness_range)
        const LRA = ffi::mode_EBUR128_MODE_LRA;
        /// can call [`EbuR128::sample_peak`](struct.EbuR128.html#method.sample_peak)
        const SAMPLE_PEAK = ffi::mode_EBUR128_MODE_SAMPLE_PEAK;
        /// can call [`EbuR128::true_peak`](struct.EbuR128.html#method.true_peak)
        const TRUE_PEAK = ffi::mode_EBUR128_MODE_TRUE_PEAK;
        /// uses histogram algorithm to calculate loudness
        const HISTOGRAM = ffi::mode_EBUR128_MODE_HISTOGRAM;
    }
}

/// Channel position.
///
/// Use these values when setting the channel map with
/// [`EbuR128::set_channel`](struct.EbuR128.html#method.set_channel).
/// See definitions in ITU R-REC-BS 1770-4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Channel {
    /// unused channel (for example LFE channel)
    Unused = ffi::channel_EBUR128_UNUSED,
    /// Left or itu M+030
    Left = ffi::channel_EBUR128_LEFT,
    /// Right or itu M-030
    Right = ffi::channel_EBUR128_RIGHT,
    /// Center or itu M+000
    Center = ffi::channel_EBUR128_CENTER,
    /// Left surround or itu M+110
    LeftSurround = ffi::channel_EBUR128_LEFT_SURROUND,
    /// Right surround or itu M-110
    RightSurround = ffi::channel_EBUR128_RIGHT_SURROUND,
    /// a channel that is counted twice
    DualMono = ffi::channel_EBUR128_DUAL_MONO,
    /// itu M+SC
    MpSC = ffi::channel_EBUR128_MpSC,
    /// itu M-SC
    MmSC = ffi::channel_EBUR128_MmSC,
    /// itu M+060
    Mp060 = ffi::channel_EBUR128_Mp060,
    /// itu M-060
    Mm060 = ffi::channel_EBUR128_Mm060,
    /// itu M+090
    Mp090 = ffi::channel_EBUR128_Mp090,
    /// itu M-090
    Mm090 = ffi::channel_EBUR128_Mm090,
    /// itu M+135
    Mp135 = ffi::channel_EBUR128_Mp135,
    /// itu M-135
    Mm135 = ffi::channel_EBUR128_Mm135,
    /// itu M+180
    Mp180 = ffi::channel_EBUR128_Mp180,
    /// itu U+000
    Up000 = ffi::channel_EBUR128_Up000,
    /// itu U+030
    Up030 = ffi::channel_EBUR128_Up030,
    /// itu U-030
    Um030 = ffi::channel_EBUR128_Um030,
    /// itu U+045
    Up045 = ffi::channel_EBUR128_Up045,
    /// itu U-030
    Um045 = ffi::channel_EBUR128_Um045,
    /// itu U+090
    Up090 = ffi::channel_EBUR128_Up090,
    /// itu U-090
    Um090 = ffi::channel_EBUR128_Um090,
    /// itu U+110
    Up110 = ffi::channel_EBUR128_Up110,
    /// itu U-110
    Um110 = ffi::channel_EBUR128_Um110,
    /// itu U+135
    Up135 = ffi::channel_EBUR128_Up135,
    /// itu U-135
    Um135 = ffi::channel_EBUR128_Um135,
    /// itu U+180
    Up180 = ffi::channel_EBUR128_Up180,
    /// itu T+000
    Tp000 = ffi::channel_EBUR128_Tp000,
    /// itu B+000
    Bp000 = ffi::channel_EBUR128_Bp000,
    /// itu B+045
    Bp045 = ffi::channel_EBUR128_Bp045,
    /// itu B-045
    Bm045 = ffi::channel_EBUR128_Bm045,
}

/// EBU R128 loudness analyzer.
#[derive(Debug)]
pub struct EbuR128(ptr::NonNull<ffi::ebur128_state>);

unsafe impl Send for EbuR128 {}

impl EbuR128 {
    /// Create a new instance with the given configuration.
    pub fn new(channels: u32, samplerate: u32, mode: Mode) -> Result<Self, Error> {
        static ONCE: std::sync::Once = std::sync::Once::new();

        ONCE.call_once(|| unsafe { ffi::ebur128_libinit() });

        unsafe {
            let ptr = ffi::ebur128_init(channels, samplerate as _, mode.bits() as i32);
            if ptr.is_null() {
                return Err(Error::NoMem);
            }
            Ok(EbuR128(ptr::NonNull::new_unchecked(ptr)))
        }
    }

    /// Set channel type.
    ///
    /// The default is:
    ///
    /// * 0 \-> `Left`
    /// * 1 \-> `Right`
    /// * 2 \-> `Center`
    /// * 3 \-> `Unused`
    /// * 4 \-> `LeftSurround`
    /// * 5 \-> `RightSurround`
    pub fn set_channel(&mut self, channel_number: u32, value: Channel) -> Result<(), Error> {
        unsafe {
            let res = ffi::ebur128_set_channel(self.0.as_ptr(), channel_number, value as _);
            Error::from_ffi(res as ffi::error, || ())
        }
    }

    /// Change library parameters.
    ///
    /// Note that the channel map will be reset when setting a different number of channels. The
    /// current unfinished block will be lost.
    pub fn change_parameters(&mut self, channels: u32, samplerate: u32) -> Result<(), Error> {
        unsafe {
            let res = ffi::ebur128_change_parameters(self.0.as_ptr(), channels, samplerate as _);
            Error::from_ffi(res as ffi::error, || ())
        }
    }

    /// Set the maximum window duration.
    ///
    /// Set the maximum duration in ms that will be used for
    /// [`EbuR128::loudness_window`](struct.EbuR128.html#method.loudness_window). Note that this
    /// destroys the current content of the audio buffer.
    pub fn set_max_window(&mut self, window: u32) -> Result<(), Error> {
        unsafe {
            let res = ffi::ebur128_set_max_window(self.0.as_ptr(), window as _);
            Error::from_ffi(res as ffi::error, || ())
        }
    }

    /// Set the maximum history.
    ///
    /// Set the maximum history in ms that will be stored for loudness integration. More history
    /// provides more accurate results, but requires more resources.
    ///
    /// Applies to [`EbuR128::loudness_range`](struct.EbuR128.html#method.loudness_range) and
    /// [`EbuR128::loudness_global`](struct.EbuR128.html#method.loudness_global) when
    /// `Mode::HISTOGRAM` is not set.
    ///
    /// Default is `ULONG_MAX` (at least ~50 days). Minimum is 3000ms for `Mode::LRA` and 400ms
    /// for `Mode::M`.
    pub fn set_max_history(&mut self, history: u32) -> Result<(), Error> {
        unsafe {
            let res = ffi::ebur128_set_max_history(self.0.as_ptr(), history as _);
            Error::from_ffi(res as ffi::error, || ())
        }
    }

    /// Add frames to be processed.
    pub fn add_frames_i16(&mut self, frames: &[i16]) -> Result<(), Error> {
        unsafe {
            if self.0.as_ref().channels == 0
                || frames.len() % self.0.as_ref().channels as usize != 0
            {
                return Err(Error::NoMem);
            }

            let res = ffi::ebur128_add_frames_short(
                self.0.as_ptr(),
                frames.as_ptr(),
                frames.len() / self.0.as_ref().channels as usize,
            );
            Error::from_ffi(res as ffi::error, || ())
        }
    }

    /// Add frames to be processed.
    pub fn add_frames_i32(&mut self, frames: &[i32]) -> Result<(), Error> {
        unsafe {
            if self.0.as_ref().channels == 0
                || frames.len() % self.0.as_ref().channels as usize != 0
            {
                return Err(Error::NoMem);
            }

            let res = ffi::ebur128_add_frames_int(
                self.0.as_ptr(),
                frames.as_ptr(),
                frames.len() / self.0.as_ref().channels as usize,
            );
            Error::from_ffi(res as ffi::error, || ())
        }
    }

    /// Add frames to be processed.
    pub fn add_frames_f32(&mut self, frames: &[f32]) -> Result<(), Error> {
        unsafe {
            if self.0.as_ref().channels == 0
                || frames.len() % self.0.as_ref().channels as usize != 0
            {
                return Err(Error::NoMem);
            }

            let res = ffi::ebur128_add_frames_float(
                self.0.as_ptr(),
                frames.as_ptr(),
                frames.len() / self.0.as_ref().channels as usize,
            );
            Error::from_ffi(res as ffi::error, || ())
        }
    }

    /// Add frames to be processed.
    pub fn add_frames_f64(&mut self, frames: &[f64]) -> Result<(), Error> {
        unsafe {
            if self.0.as_ref().channels == 0
                || frames.len() % self.0.as_ref().channels as usize != 0
            {
                return Err(Error::NoMem);
            }
            let res = ffi::ebur128_add_frames_double(
                self.0.as_ptr(),
                frames.as_ptr(),
                frames.len() / self.0.as_ref().channels as usize,
            );
            Error::from_ffi(res as ffi::error, || ())
        }
    }

    /// Get global integrated loudness in LUFS.
    pub fn loudness_global(&self) -> Result<f64, Error> {
        unsafe {
            let mut out = mem::MaybeUninit::uninit();
            let res = ffi::ebur128_loudness_global(self.0.as_ptr(), out.as_mut_ptr());
            Error::from_ffi(res as ffi::error, || out.assume_init())
        }
    }

    /// Get momentary loudness (last 400ms) in LUFS.
    pub fn loudness_momentary(&self) -> Result<f64, Error> {
        unsafe {
            let mut out = mem::MaybeUninit::uninit();
            let res = ffi::ebur128_loudness_momentary(self.0.as_ptr(), out.as_mut_ptr());
            Error::from_ffi(res as ffi::error, || out.assume_init())
        }
    }

    /// Get short-term loudness (last 3s) in LUFS.
    pub fn loudness_shortterm(&self) -> Result<f64, Error> {
        unsafe {
            let mut out = mem::MaybeUninit::uninit();
            let res = ffi::ebur128_loudness_shortterm(self.0.as_ptr(), out.as_mut_ptr());
            Error::from_ffi(res as ffi::error, || out.assume_init())
        }
    }

    /// Get loudness of the specified window in LUFS.
    ///
    /// window must not be larger than the current window. The current window can be changed by
    /// calling [`EbuR128::set_max_window`](struct.EbuR128.html#method.set_max_window).
    pub fn loudness_window(&self, window: u32) -> Result<f64, Error> {
        unsafe {
            let mut out = mem::MaybeUninit::uninit();
            let res = ffi::ebur128_loudness_window(self.0.as_ptr(), window as _, out.as_mut_ptr());
            Error::from_ffi(res as ffi::error, || out.assume_init())
        }
    }

    /// Get loudness range (LRA) of programme in LU.
    ///
    /// Calculates loudness range according to EBU 3342.
    pub fn loudness_range(&self) -> Result<f64, Error> {
        unsafe {
            let mut out = mem::MaybeUninit::uninit();
            let res = ffi::ebur128_loudness_range(self.0.as_ptr(), out.as_mut_ptr());
            Error::from_ffi(res as ffi::error, || out.assume_init())
        }
    }

    /// Get maximum sample peak from all frames that have been processed.
    ///
    /// The equation to convert to dBFS is: 20 * log10(out)
    pub fn sample_peak(&self, channel_number: u32) -> Result<f64, Error> {
        unsafe {
            let mut out = mem::MaybeUninit::uninit();
            let res = ffi::ebur128_sample_peak(self.0.as_ptr(), channel_number, out.as_mut_ptr());
            Error::from_ffi(res as ffi::error, || out.assume_init())
        }
    }

    /// Get maximum sample peak from the last call to
    /// [`EbuR128::add_frames`](struct.EbuR128.html#method.add_frames_i16).
    ///
    /// The equation to convert to dBFS is: 20 * log10(out)
    pub fn prev_sample_peak(&self, channel_number: u32) -> Result<f64, Error> {
        unsafe {
            let mut out = mem::MaybeUninit::uninit();
            let res =
                ffi::ebur128_prev_sample_peak(self.0.as_ptr(), channel_number, out.as_mut_ptr());
            Error::from_ffi(res as ffi::error, || out.assume_init())
        }
    }

    /// Get maximum true peak from all frames that have been processed.
    ///
    /// Uses an implementation defined algorithm to calculate the true peak. Do not try to compare
    /// resulting values across different versions of the library, as the algorithm may change.
    ///
    /// The current implementation uses a custom polyphase FIR interpolator to calculate true peak.
    /// Will oversample 4x for sample rates < 96000 Hz, 2x for sample rates < 192000 Hz and leave
    /// the signal unchanged for 192000 Hz.
    ///
    /// The equation to convert to dBTP is: 20 * log10(out)
    pub fn true_peak(&self, channel_number: u32) -> Result<f64, Error> {
        unsafe {
            let mut out = mem::MaybeUninit::uninit();
            let res = ffi::ebur128_true_peak(self.0.as_ptr(), channel_number, out.as_mut_ptr());
            Error::from_ffi(res as ffi::error, || out.assume_init())
        }
    }

    /// Get maximum true peak from the last call to
    /// [`EbuR128::add_frames`](struct.EbuR128.html#method.add_frames_i16).
    ///
    /// Uses an implementation defined algorithm to calculate the true peak. Do not try to compare
    /// resulting values across different versions of the library, as the algorithm may change.
    ///
    /// The current implementation uses a custom polyphase FIR interpolator to calculate true peak.
    /// Will oversample 4x for sample rates < 96000 Hz, 2x for sample rates < 192000 Hz and leave
    /// the signal unchanged for 192000 Hz.
    ///
    /// The equation to convert to dBTP is: 20 * log10(out)
    pub fn prev_true_peak(&self, channel_number: u32) -> Result<f64, Error> {
        unsafe {
            let mut out = mem::MaybeUninit::uninit();
            let res =
                ffi::ebur128_prev_true_peak(self.0.as_ptr(), channel_number, out.as_mut_ptr());
            Error::from_ffi(res as ffi::error, || out.assume_init())
        }
    }

    /// Get relative threshold in LUFS.
    pub fn relative_threshold(&self) -> Result<f64, Error> {
        unsafe {
            let mut out = mem::MaybeUninit::uninit();
            let res = ffi::ebur128_relative_threshold(self.0.as_ptr(), out.as_mut_ptr());
            Error::from_ffi(res as ffi::error, || out.assume_init())
        }
    }
}

impl Drop for EbuR128 {
    fn drop(&mut self) {
        unsafe {
            let mut state = self.0.as_ptr();
            ffi::ebur128_destroy(&mut state);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::Signal;
    use quickcheck_macros::quickcheck;

    macro_rules! assert_eq_f64(
        ($a:expr, $b:expr) => {
            assert!(
                float_cmp::approx_eq!(f64, $a, $b, ulps = 2),
                "{} != {}",
                $a,
                $b,
            )
        }
    );

    #[test]
    fn sine_stereo_i16() {
        let mut data = vec![0i16; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * (std::i16::MAX - 1) as f32;
            out[0] = val as i16;
            out[1] = val as i16;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all()).unwrap();
        ebu.add_frames_i16(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6500000000000054);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6820309226891973);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6834583474398446);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.875007988101488);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.0);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 0.99993896484375);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 0.99993896484375);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 0.99993896484375);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 0.99993896484375);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0007814168930054);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0007814168930054);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0007814168930054);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0007814168930054);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.650000000000006);
    }

    #[test]
    fn sine_stereo_i32() {
        let mut data = vec![0i32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * (std::i32::MAX - 1) as f32;
            out[0] = val as i32;
            out[1] = val as i32;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all()).unwrap();
        ebu.add_frames_i32(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6500000000000054);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6813325598274425);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6827591715105212);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.8742956620040943);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.0);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 1.0);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0008491277694702);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.650000000000006);
    }

    #[test]
    fn sine_stereo_f32() {
        let mut data = vec![0.0f32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val;
            out[1] = val;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all()).unwrap();
        ebu.add_frames_f32(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6500000000000054);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6813325598268921);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6827591715100236);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.8742956620008693);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.0);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 1.0);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0008491277694702);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.650000000000006);
    }

    #[test]
    fn sine_stereo_f64() {
        let mut data = vec![0.0f64; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val as f64;
            out[1] = val as f64;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all()).unwrap();
        ebu.add_frames_f64(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6500000000000054);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6813325598268921);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6827591715100236);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.8742956620008693);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.0);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 1.0);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0008491277694702);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.650000000000006);
    }

    #[test]
    fn sine_stereo_i16_no_histogram() {
        let mut data = vec![0i16; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * (std::i16::MAX - 1) as f32;
            out[0] = val as i16;
            out[1] = val as i16;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_i16(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.683303243667768);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6820309226891973);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6834583474398446);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.875007988101488);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.00006950793233284625);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 0.99993896484375);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 0.99993896484375);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 0.99993896484375);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 0.99993896484375);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0007814168930054);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0007814168930054);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0007814168930054);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0007814168930054);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.683303243667767);
    }

    #[test]
    fn sine_stereo_i32_no_histogram() {
        let mut data = vec![0i32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * (std::i32::MAX - 1) as f32;
            out[0] = val as i32;
            out[1] = val as i32;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_i32(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6826039914171368);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6813325598274425);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6827591715105212);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.8742956620040943);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.00006921150165073442);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 1.0);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0008491277694702);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.682603991417135);
    }

    #[test]
    fn sine_stereo_f32_no_histogram() {
        let mut data = vec![0.0f32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val;
            out[1] = val;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_f32(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6826039914165554);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6813325598268921);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6827591715100236);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.8742956620008693);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.00006921150169403312);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 1.0);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0008491277694702);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.682603991416554);
    }

    #[test]
    fn sine_stereo_f64_no_histogram() {
        let mut data = vec![0.0f64; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val as f64;
            out[1] = val as f64;
            accumulator += step;
        }

        let mut ebu = EbuR128::new(2, 48_000, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_f64(&data).unwrap();

        assert_eq_f64!(ebu.loudness_global().unwrap(), -0.6826039914165554);
        assert_eq_f64!(ebu.loudness_momentary().unwrap(), -0.6813325598268921);
        assert_eq_f64!(ebu.loudness_shortterm().unwrap(), -0.6827591715100236);
        assert_eq_f64!(ebu.loudness_window(1).unwrap(), -0.8742956620008693);
        assert_eq_f64!(ebu.loudness_range().unwrap(), 0.00006921150169403312);

        assert_eq_f64!(ebu.sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.sample_peak(1).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(0).unwrap(), 1.0);
        assert_eq_f64!(ebu.prev_sample_peak(1).unwrap(), 1.0);

        assert_eq_f64!(ebu.true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.true_peak(1).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(0).unwrap(), 1.0008491277694702);
        assert_eq_f64!(ebu.prev_true_peak(1).unwrap(), 1.0008491277694702);

        assert_eq_f64!(ebu.relative_threshold().unwrap(), -10.682603991416554);
    }

    #[quickcheck]
    fn compare_c_impl_i16(signal: Signal<i16>) {
        let mut ebu = EbuR128::new(signal.channels, signal.rate, Mode::all()).unwrap();
        ebu.add_frames_i16(&signal.data).unwrap();

        let mut ebu_c =
            ebur128_c::EbuR128::new(signal.channels, signal.rate, ebur128_c::Mode::all()).unwrap();
        ebu_c.add_frames_i16(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[quickcheck]
    fn compare_c_impl_i32(signal: Signal<i32>) {
        let mut ebu = EbuR128::new(signal.channels, signal.rate, Mode::all()).unwrap();
        ebu.add_frames_i32(&signal.data).unwrap();

        let mut ebu_c =
            ebur128_c::EbuR128::new(signal.channels, signal.rate, ebur128_c::Mode::all()).unwrap();
        ebu_c.add_frames_i32(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[quickcheck]
    fn compare_c_impl_f32(signal: Signal<f32>) {
        let mut ebu = EbuR128::new(signal.channels, signal.rate, Mode::all()).unwrap();
        ebu.add_frames_f32(&signal.data).unwrap();

        let mut ebu_c =
            ebur128_c::EbuR128::new(signal.channels, signal.rate, ebur128_c::Mode::all()).unwrap();
        ebu_c.add_frames_f32(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[quickcheck]
    fn compare_c_impl_f64(signal: Signal<f64>) {
        let mut ebu = EbuR128::new(signal.channels, signal.rate, Mode::all()).unwrap();
        ebu.add_frames_f64(&signal.data).unwrap();

        let mut ebu_c =
            ebur128_c::EbuR128::new(signal.channels, signal.rate, ebur128_c::Mode::all()).unwrap();
        ebu_c.add_frames_f64(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[quickcheck]
    fn compare_c_impl_i16_no_histogram(signal: Signal<i16>) {
        let mut ebu =
            EbuR128::new(signal.channels, signal.rate, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_i16(&signal.data).unwrap();

        let mut ebu_c = ebur128_c::EbuR128::new(
            signal.channels,
            signal.rate,
            ebur128_c::Mode::all() & !ebur128_c::Mode::HISTOGRAM,
        )
        .unwrap();
        ebu_c.add_frames_i16(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[quickcheck]
    fn compare_c_impl_i32_no_histogram(signal: Signal<i32>) {
        let mut ebu =
            EbuR128::new(signal.channels, signal.rate, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_i32(&signal.data).unwrap();

        let mut ebu_c = ebur128_c::EbuR128::new(
            signal.channels,
            signal.rate,
            ebur128_c::Mode::all() & !ebur128_c::Mode::HISTOGRAM,
        )
        .unwrap();
        ebu_c.add_frames_i32(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[quickcheck]
    fn compare_c_impl_f32_no_histogram(signal: Signal<f32>) {
        let mut ebu =
            EbuR128::new(signal.channels, signal.rate, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_f32(&signal.data).unwrap();

        let mut ebu_c = ebur128_c::EbuR128::new(
            signal.channels,
            signal.rate,
            ebur128_c::Mode::all() & !ebur128_c::Mode::HISTOGRAM,
        )
        .unwrap();
        ebu_c.add_frames_f32(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }

    #[quickcheck]
    fn compare_c_impl_f64_no_histogram(signal: Signal<f64>) {
        let mut ebu =
            EbuR128::new(signal.channels, signal.rate, Mode::all() & !Mode::HISTOGRAM).unwrap();
        ebu.add_frames_f64(&signal.data).unwrap();

        let mut ebu_c = ebur128_c::EbuR128::new(
            signal.channels,
            signal.rate,
            ebur128_c::Mode::all() & !ebur128_c::Mode::HISTOGRAM,
        )
        .unwrap();
        ebu_c.add_frames_f64(&signal.data).unwrap();

        assert_eq_f64!(
            ebu.loudness_global().unwrap(),
            ebu_c.loudness_global().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_momentary().unwrap(),
            ebu_c.loudness_momentary().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_shortterm().unwrap(),
            ebu_c.loudness_shortterm().unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_window(1).unwrap(),
            ebu_c.loudness_window(1).unwrap()
        );
        assert_eq_f64!(
            ebu.loudness_range().unwrap(),
            ebu_c.loudness_range().unwrap()
        );

        for c in 0..signal.channels {
            assert_eq_f64!(ebu.sample_peak(c).unwrap(), ebu_c.sample_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_sample_peak(c).unwrap(),
                ebu_c.prev_sample_peak(c).unwrap()
            );

            assert_eq_f64!(ebu.true_peak(c).unwrap(), ebu_c.true_peak(c).unwrap());
            assert_eq_f64!(
                ebu.prev_true_peak(c).unwrap(),
                ebu_c.prev_true_peak(c).unwrap()
            );
        }

        assert_eq_f64!(
            ebu.relative_threshold().unwrap(),
            ebu_c.relative_threshold().unwrap()
        );
    }
}
