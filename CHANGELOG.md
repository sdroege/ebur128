# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html),
specifically the [variant used by Rust](http://doc.crates.io/manifest.html#the-version-field).

## [Unreleased] - TBD

## [0.1.6] - 2021-07-13
### Fixed
- Flush-to-zero implementation was refactored to prevent the possibility to
  pass it to `mem::forget()` and thus leading to UB. This was not a problem in
  this codebase.

### Added
- Specialized, optimized implementation for common interpolations. This speeds
  up the resampling/interpolation used by the true peak detection for mono,
  stereo and four channels by a factor of 2-5x.
- Buffer-less true peak scanning instead of the previous implementation that
  goes via a temporary buffer. This speeds up true peak detection even more.
- Add seeding API for supporting chunked analysis, i.e. analyzing a bigger
  input split into separate chunks in parallel.

### Changed
- Various refactoring.
- `unsafe` code was removed from the histogram handling without a measurable
  impact on the performance.
- Use the `dasp` crates instead of the homegrown sample/frame abstraction.

## [0.1.5] - 2020-09-07
### Fixed
- Allow only a single channel when setting `DualMono` also in
  `EbuR128::set_channel_map()`. The same constraint was already checked in
  `set_channel()`.
- Chunk size in some of the reference tests was changed as it was too big.
  This fixed the remaining two reference tests that were failing before.

### Added
- `EbuR128::reset()` to reset all state without reallocation.
- Support for planar/non-interleaved inputs.

### Changed
- Sample peak measurement was changed slightly and is now faster than the C
  implementation while still giving the same results.
- Use `SmallVec` for temporary vecs in `EbuR128::loudness_global_multiple()`
  and `loudness_range_multiple()` to prevent heap allocations, and also use it
  in the filter of the interpolator, which slightly speeds it up.
- CI was switched from Travis to GitHub actions and greatly improved.

## [0.1.4] - 2020-08-31
### Fixed
- Fix compiler warning about unused use statement in benchmarks
- Fix various spelling errors in code comments
- Protect various calculations from overflows when creating a new instance or
  changing settings so that instead of panicking we return a proper error
- Use `std::f64::INFINITY` and similar instead of `f64::INFINITY` to fix
  compilation with older Rust versions. Version up to Rust 1.32.0 are
  supported now.

### Changed
- Replaced various `Vec<T>` with `Box<[T]>` to make it clearer that these are
  never changing their size after the initial allocation, and to also get rid
  of the additional unused storage of the capacity.

### Added
- More documentation for internal types

## [0.1.3] - 2020-08-28
### Fixed
- Return `-f64::INFINITY` instead of `f64::MIN` for values below the
  thresholds from various functions. This matches the behaviour of the C
  library and generally makes more sense.

## [0.1.2] - 2020-08-28
### Changed
- Port libebur128 C code to Rust. This is a pure Rust implementation of it now
  and the only remaining C code is in the tests for comparing the two
  implementations.

## [0.1.1] - 2020-04-14
### Fixed
- Fix build with the MSVC toolchain.

## 0.1.0 - 2020-01-06
- Initial release of ebur128.

[Unreleased]: https://github.com/sdroege/ebur128/compare/0.1.6...HEAD
[0.1.6]: https://github.com/sdroege/ebur128/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/sdroege/ebur128/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/sdroege/ebur128/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/sdroege/ebur128/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/sdroege/ebur128/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/sdroege/ebur128/compare/0.1.0...0.1.1
