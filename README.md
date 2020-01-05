# ebur128 [![crates.io](https://img.shields.io/crates/v/ebur128.svg)](https://crates.io/crates/ebur128) [![Build Status](https://travis-ci.org/sdroege/ebur128.svg?branch=master)](https://travis-ci.org/sdroege/ebur128) [![docs.rs](https://docs.rs/ebur128/badge.svg)](https://docs.rs/ebur128)

Implementation of the [EBU R128 loudness standard](https://tech.ebu.ch/docs/r/r128.pdf).

The European Broadcasting Union Loudness Recommendation (EBU R128) informs broadcasters how
they can analyze and normalize audio so that each piece of audio sounds roughly the same
volume to the human ear.

This crate provides an API which analyzes audio and outputs perceived loudness. The results
can then be used to normalize volume during playback.

Features:
 * Implements M, S and I modes
 * Implements loudness range measurement (EBU - TECH 3342)
 * True peak scanning
 * Supports all samplerates by recalculation of the filter coefficients

This crate is based on the [libebur128](https://github.com/jiixyj/libebur128) C
library.

## LICENSE

ebur128 is licensed under the MIT license ([LICENSE](LICENSE) or
http://opensource.org/licenses/MIT).

## Contribution

Any kinds of contributions are welcome as a pull request.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in send-cell by you shall be licensed under the MIT license as above,
without any additional terms or conditions.
