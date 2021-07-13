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
//!   * Implements M, S and I modes ([EBU - TECH 3341](https://tech.ebu.ch/docs/tech/tech3341.pdf))
//!   * Implements loudness range measurement ([EBU - TECH 3342](https://tech.ebu.ch/docs/tech/tech3342.pdf))
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

#[allow(clippy::excessive_precision)]
mod histogram_bins;

#[cfg(feature = "internal-tests")]
pub mod filter;
#[cfg(not(feature = "internal-tests"))]
pub(crate) mod filter;

#[cfg(feature = "internal-tests")]
pub mod utils;
#[cfg(not(feature = "internal-tests"))]
pub(crate) mod utils;

#[cfg(feature = "internal-tests")]
pub use utils::{energy_to_loudness, Interleaved, Planar, Samples};
#[cfg(not(feature = "internal-tests"))]
pub(crate) use utils::{energy_to_loudness, Interleaved, Planar, Samples};

#[cfg(test)]
pub mod tests {
    pub use super::utils::tests::Signal;
}

#[cfg(feature = "capi")]
#[allow(clippy::missing_safety_doc)]
pub mod capi;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub mod wasm;
