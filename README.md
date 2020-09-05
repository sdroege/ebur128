# ebur128 [![crates.io](https://img.shields.io/crates/v/ebur128.svg)](https://crates.io/crates/ebur128) [![Actions Status](https://github.com/sdroege/ebur128/workflows/ebur128/badge.svg)](https://github.com/sdroege/ebur128/actions) [![docs.rs](https://docs.rs/ebur128/badge.svg)](https://docs.rs/ebur128)

Implementation of the [EBU R128 loudness standard](https://tech.ebu.ch/docs/r/r128.pdf).

The European Broadcasting Union Loudness Recommendation (EBU R128) informs broadcasters how
they can analyze and normalize audio so that each piece of audio sounds roughly the same
volume to the human ear.

This crate provides an API which analyzes audio and outputs perceived loudness. The results
can then be used to normalize volume during playback.

Features:
 * Implements M, S and I modes ([EBU - TECH 3341](https://tech.ebu.ch/docs/tech/tech3341.pdf))
 * Implements loudness range measurement ([EBU - TECH 3342](https://tech.ebu.ch/docs/tech/tech3342.pdf))
 * True peak scanning
 * Supports all samplerates by recalculation of the filter coefficients

This crate is a Rust port of the [libebur128](https://github.com/jiixyj/libebur128) C library, produces the
same results as the C library and has comparable performance.

## EBU TECH 3341/3342 Compliance

Currently, the implementation passes all tests defined in [EBU - TECH 3341](https://tech.ebu.ch/docs/tech/tech3341.pdf)
and [EBU - TECH 3342](https://tech.ebu.ch/docs/tech/tech3342.pdf).

## C API

ebur128 optionally provides a C API that is API/ABI-compatible with
libebur128. It can be built and installed via [`cargo-c`](https://crates.io/crates/cargo-c):

```sh
# If cargo-c was not installed yet
$ cargo install cargo-c
# Change the prefix to the place where it should be installed
$ cargo cbuild --prefix /usr/local
$ cargo cinstall --prefix /usr/local
```

This installs a shared library, static library, C header and [`pkg-config`](https://www.freedesktop.org/wiki/Software/pkg-config/)
file that is compatible with libebur128.

## LICENSE

ebur128 is licensed under the MIT license ([LICENSE](LICENSE) or
http://opensource.org/licenses/MIT).

## Contribution

Any kinds of contributions are welcome as a pull request.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in ebur128 by you shall be licensed under the MIT
license as above, without any additional terms or conditions.
