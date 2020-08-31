# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html),
specifically the [variant used by Rust](http://doc.crates.io/manifest.html#the-version-field).

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

[Unreleased]: https://github.com/sdroege/rust-muldiv/compare/0.1.4...HEAD
[0.1.4]: https://github.com/sdroege/ebur128/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/sdroege/ebur128/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/sdroege/ebur128/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/sdroege/ebur128/compare/0.1.0...0.1.1
