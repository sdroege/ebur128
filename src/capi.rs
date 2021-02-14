use crate::ebur128;

use std::{mem, ptr};

// ABI compatible with ebur128_state
#[repr(C)]
pub struct State {
    mode: i32,
    channels: u32,
    samplerate: std::os::raw::c_ulong,
    internal: *mut ebur128::EbuR128,
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_get_version(major: *mut i32, minor: *mut i32, patch: *mut i32) {
    // We're based on 1.2.6 so let's return that for now
    *major = 1;
    *minor = 2;
    *patch = 6;
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_init(
    channels: u32,
    samplerate: std::os::raw::c_ulong,
    // Same values as our Mode enum
    mode: i32,
) -> *mut State {
    let e = match ebur128::EbuR128::new(
        channels,
        samplerate as u32,
        ebur128::Mode::from_bits_truncate(mode as u8),
    ) {
        Err(_) => return ptr::null_mut(),
        Ok(e) => e,
    };

    let s = State {
        mode,
        channels,
        samplerate,
        internal: Box::into_raw(Box::new(e)),
    };

    Box::into_raw(Box::new(s))
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_destroy(state: *mut *mut State) {
    if state.is_null() || (*state).is_null() {
        return;
    }

    let s = Box::from_raw(*state);
    let e = Box::from_raw(s.internal);
    drop(e);
    drop(s);

    *state = ptr::null_mut();
}

impl From<ebur128::Error> for i32 {
    fn from(v: ebur128::Error) -> i32 {
        match v {
            ebur128::Error::NoMem => 1,
            ebur128::Error::InvalidMode => 2,
            ebur128::Error::InvalidChannelIndex => 3,
        }
    }
}

// Same channel representation
#[no_mangle]
pub unsafe extern "C" fn ebur128_set_channel(
    state: *mut State,
    channel_number: u32,
    value: i32,
) -> i32 {
    let s = &mut *state;
    let e = &mut *s.internal;

    match e.set_channel(channel_number, mem::transmute(value)) {
        Err(err) => err.into(),
        Ok(_) => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_change_parameters(
    state: *mut State,
    channels: u32,
    samplerate: std::os::raw::c_ulong,
) -> i32 {
    let s = &mut *state;

    if s.channels == channels && s.samplerate == samplerate {
        return 4; // EBUR128_ERROR_NO_CHANGE
    }

    let e = &mut *s.internal;

    match e.change_parameters(channels, samplerate as u32) {
        Err(err) => err.into(),
        Ok(_) => {
            s.channels = channels;
            s.samplerate = samplerate;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_set_max_window(
    state: *mut State,
    window: std::os::raw::c_ulong,
) -> i32 {
    if window > u32::MAX as std::os::raw::c_ulong {
        return ebur128::Error::NoMem.into();
    }

    let s = &mut *state;
    let e = &mut *s.internal;

    if e.max_window() == window as usize {
        return 4; // EBUR128_ERROR_NO_CHANGE
    }

    match e.set_max_window(window as u32) {
        Err(err) => err.into(),
        Ok(_) => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_set_max_history(
    state: *mut State,
    history: std::os::raw::c_ulong,
) -> i32 {
    if history > u32::MAX as std::os::raw::c_ulong {
        return ebur128::Error::NoMem.into();
    }

    let s = &mut *state;
    let e = &mut *s.internal;

    if e.max_history() == history as usize {
        return 4; // EBUR128_ERROR_NO_CHANGE
    }

    match e.set_max_history(history as u32) {
        Err(err) => err.into(),
        Ok(_) => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_add_frames_short(
    state: *mut State,
    src: *const i16,
    frames: usize,
) -> i32 {
    use std::slice;

    let s = &mut *state;
    let e = &mut *s.internal;

    let samples = match frames.checked_mul(s.channels as usize) {
        None => return crate::ebur128::Error::NoMem.into(),
        Some(samples) => samples,
    };

    match e.add_frames_i16(slice::from_raw_parts(src, samples)) {
        Err(err) => err.into(),
        Ok(_) => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_add_frames_int(
    state: *mut State,
    src: *const i32,
    frames: usize,
) -> i32 {
    use std::slice;

    let s = &mut *state;
    let e = &mut *s.internal;

    let samples = match frames.checked_mul(s.channels as usize) {
        None => return crate::ebur128::Error::NoMem.into(),
        Some(samples) => samples,
    };

    match e.add_frames_i32(slice::from_raw_parts(src, samples)) {
        Err(err) => err.into(),
        Ok(_) => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_add_frames_float(
    state: *mut State,
    src: *const f32,
    frames: usize,
) -> i32 {
    use std::slice;

    let s = &mut *state;
    let e = &mut *s.internal;

    let samples = match frames.checked_mul(s.channels as usize) {
        None => return crate::ebur128::Error::NoMem.into(),
        Some(samples) => samples,
    };

    match e.add_frames_f32(slice::from_raw_parts(src, samples)) {
        Err(err) => err.into(),
        Ok(_) => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_add_frames_double(
    state: *mut State,
    src: *const f64,
    frames: usize,
) -> i32 {
    use std::slice;

    let s = &mut *state;
    let e = &mut *s.internal;

    let samples = match frames.checked_mul(s.channels as usize) {
        None => return crate::ebur128::Error::NoMem.into(),
        Some(samples) => samples,
    };

    match e.add_frames_f64(slice::from_raw_parts(src, samples)) {
        Err(err) => err.into(),
        Ok(_) => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_loudness_global(state: *mut State, out: *mut f64) -> i32 {
    let s = &*state;
    let e = &*s.internal;

    match e.loudness_global() {
        Err(err) => err.into(),
        Ok(val) => {
            *out = val;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_loudness_global_multiple(
    state: *mut *mut State,
    size: usize,
    out: *mut f64,
) -> i32 {
    use std::slice;

    let s = slice::from_raw_parts(state, size);
    let iter = s.iter().copied().map(|s: *mut State| &*(*s).internal);

    match ebur128::EbuR128::loudness_global_multiple(iter) {
        Err(err) => err.into(),
        Ok(val) => {
            *out = val;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_loudness_momentary(state: *mut State, out: *mut f64) -> i32 {
    let s = &*state;
    let e = &*s.internal;

    match e.loudness_momentary() {
        Err(err) => err.into(),
        Ok(val) => {
            *out = val;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_loudness_shortterm(state: *mut State, out: *mut f64) -> i32 {
    let s = &*state;
    let e = &*s.internal;

    match e.loudness_shortterm() {
        Err(err) => err.into(),
        Ok(val) => {
            *out = val;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_loudness_window(
    state: *mut State,
    window: std::os::raw::c_ulong,
    out: *mut f64,
) -> i32 {
    let s = &*state;
    let e = &*s.internal;

    if window > u32::MAX as std::os::raw::c_ulong {
        return ebur128::Error::NoMem.into();
    }

    match e.loudness_window(window as u32) {
        Err(err) => err.into(),
        Ok(val) => {
            *out = val;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_loudness_range(state: *mut State, out: *mut f64) -> i32 {
    let s = &*state;
    let e = &*s.internal;

    match e.loudness_range() {
        Err(err) => err.into(),
        Ok(val) => {
            *out = val;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_loudness_range_multiple(
    state: *mut *mut State,
    size: usize,
    out: *mut f64,
) -> i32 {
    use std::slice;

    let s = slice::from_raw_parts(state, size);
    let iter = s.iter().copied().map(|s: *mut State| &*(*s).internal);

    match ebur128::EbuR128::loudness_range_multiple(iter) {
        Err(err) => err.into(),
        Ok(val) => {
            *out = val;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_sample_peak(
    state: *mut State,
    channel_number: u32,
    out: *mut f64,
) -> i32 {
    let s = &*state;
    let e = &*s.internal;

    match e.sample_peak(channel_number) {
        Err(err) => err.into(),
        Ok(val) => {
            *out = val;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_prev_sample_peak(
    state: *mut State,
    channel_number: u32,
    out: *mut f64,
) -> i32 {
    let s = &*state;
    let e = &*s.internal;

    match e.prev_sample_peak(channel_number) {
        Err(err) => err.into(),
        Ok(val) => {
            *out = val;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_true_peak(
    state: *mut State,
    channel_number: u32,
    out: *mut f64,
) -> i32 {
    let s = &*state;
    let e = &*s.internal;

    match e.true_peak(channel_number) {
        Err(err) => err.into(),
        Ok(val) => {
            *out = val;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_prev_true_peak(
    state: *mut State,
    channel_number: u32,
    out: *mut f64,
) -> i32 {
    let s = &*state;
    let e = &*s.internal;

    match e.prev_true_peak(channel_number) {
        Err(err) => err.into(),
        Ok(val) => {
            *out = val;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ebur128_relative_threshold(state: *mut State, out: *mut f64) -> i32 {
    let s = &*state;
    let e = &*s.internal;

    match e.relative_threshold() {
        Err(err) => err.into(),
        Ok(val) => {
            *out = val;
            0
        }
    }
}
