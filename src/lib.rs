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

#[allow(unused, non_camel_case_types, non_upper_case_globals)]
mod ffi;

mod ebur128;
pub use self::ebur128::*;

#[cfg(feature = "internal-tests")]
pub mod interp;
#[cfg(not(feature = "internal-tests"))]
mod interp;
