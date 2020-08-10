use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "c-tests")]
fn ebur128_i16_c(channels: u32, rate: u32, mode: ebur128_c::Mode, data: &[i16]) {
    use ebur128_c::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
    ebu.add_frames_i16(&data).unwrap();

    if mode.contains(ebur128_c::Mode::I) {
        black_box(ebu.loudness_global().unwrap());
    }
    if mode.contains(ebur128_c::Mode::M) {
        black_box(ebu.loudness_momentary().unwrap());
    }
    if mode.contains(ebur128_c::Mode::S) {
        black_box(ebu.loudness_shortterm().unwrap());
    }
    black_box(ebu.loudness_window(1).unwrap());

    if mode.contains(ebur128_c::Mode::LRA) {
        black_box(ebu.loudness_range().unwrap());
    }

    if mode.contains(ebur128_c::Mode::SAMPLE_PEAK) {
        black_box(ebu.sample_peak(0).unwrap());
        black_box(ebu.sample_peak(1).unwrap());
        black_box(ebu.prev_sample_peak(0).unwrap());
        black_box(ebu.prev_sample_peak(1).unwrap());
    }

    if mode.contains(ebur128_c::Mode::TRUE_PEAK) {
        black_box(ebu.true_peak(0).unwrap());
        black_box(ebu.true_peak(1).unwrap());
        black_box(ebu.prev_true_peak(0).unwrap());
        black_box(ebu.prev_true_peak(1).unwrap());
    }

    if mode.contains(ebur128_c::Mode::I) {
        black_box(ebu.relative_threshold().unwrap());
    }
}

fn ebur128_i16(channels: u32, rate: u32, mode: ebur128::Mode, data: &[i16]) {
    use ebur128::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
    ebu.add_frames_i16(&data).unwrap();

    if mode.contains(ebur128::Mode::I) {
        black_box(ebu.loudness_global().unwrap());
    }
    if mode.contains(ebur128::Mode::M) {
        black_box(ebu.loudness_momentary().unwrap());
    }
    if mode.contains(ebur128::Mode::S) {
        black_box(ebu.loudness_shortterm().unwrap());
    }
    black_box(ebu.loudness_window(1).unwrap());

    if mode.contains(ebur128::Mode::LRA) {
        black_box(ebu.loudness_range().unwrap());
    }

    if mode.contains(ebur128::Mode::SAMPLE_PEAK) {
        black_box(ebu.sample_peak(0).unwrap());
        black_box(ebu.sample_peak(1).unwrap());
        black_box(ebu.prev_sample_peak(0).unwrap());
        black_box(ebu.prev_sample_peak(1).unwrap());
    }

    if mode.contains(ebur128::Mode::TRUE_PEAK) {
        black_box(ebu.true_peak(0).unwrap());
        black_box(ebu.true_peak(1).unwrap());
        black_box(ebu.prev_true_peak(0).unwrap());
        black_box(ebu.prev_true_peak(1).unwrap());
    }

    if mode.contains(ebur128::Mode::I) {
        black_box(ebu.relative_threshold().unwrap());
    }
}

#[cfg(feature = "c-tests")]
fn ebur128_i32_c(channels: u32, rate: u32, mode: ebur128_c::Mode, data: &[i32]) {
    use ebur128_c::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
    ebu.add_frames_i32(&data).unwrap();

    if mode.contains(ebur128_c::Mode::I) {
        black_box(ebu.loudness_global().unwrap());
    }
    if mode.contains(ebur128_c::Mode::M) {
        black_box(ebu.loudness_momentary().unwrap());
    }
    if mode.contains(ebur128_c::Mode::S) {
        black_box(ebu.loudness_shortterm().unwrap());
    }
    black_box(ebu.loudness_window(1).unwrap());

    if mode.contains(ebur128_c::Mode::LRA) {
        black_box(ebu.loudness_range().unwrap());
    }

    if mode.contains(ebur128_c::Mode::SAMPLE_PEAK) {
        black_box(ebu.sample_peak(0).unwrap());
        black_box(ebu.sample_peak(1).unwrap());
        black_box(ebu.prev_sample_peak(0).unwrap());
        black_box(ebu.prev_sample_peak(1).unwrap());
    }

    if mode.contains(ebur128_c::Mode::TRUE_PEAK) {
        black_box(ebu.true_peak(0).unwrap());
        black_box(ebu.true_peak(1).unwrap());
        black_box(ebu.prev_true_peak(0).unwrap());
        black_box(ebu.prev_true_peak(1).unwrap());
    }

    if mode.contains(ebur128_c::Mode::I) {
        black_box(ebu.relative_threshold().unwrap());
    }
}

fn ebur128_i32(channels: u32, rate: u32, mode: ebur128::Mode, data: &[i32]) {
    use ebur128::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
    ebu.add_frames_i32(&data).unwrap();

    if mode.contains(ebur128::Mode::I) {
        black_box(ebu.loudness_global().unwrap());
    }
    if mode.contains(ebur128::Mode::M) {
        black_box(ebu.loudness_momentary().unwrap());
    }
    if mode.contains(ebur128::Mode::S) {
        black_box(ebu.loudness_shortterm().unwrap());
    }
    black_box(ebu.loudness_window(1).unwrap());

    if mode.contains(ebur128::Mode::LRA) {
        black_box(ebu.loudness_range().unwrap());
    }

    if mode.contains(ebur128::Mode::SAMPLE_PEAK) {
        black_box(ebu.sample_peak(0).unwrap());
        black_box(ebu.sample_peak(1).unwrap());
        black_box(ebu.prev_sample_peak(0).unwrap());
        black_box(ebu.prev_sample_peak(1).unwrap());
    }

    if mode.contains(ebur128::Mode::TRUE_PEAK) {
        black_box(ebu.true_peak(0).unwrap());
        black_box(ebu.true_peak(1).unwrap());
        black_box(ebu.prev_true_peak(0).unwrap());
        black_box(ebu.prev_true_peak(1).unwrap());
    }

    if mode.contains(ebur128::Mode::I) {
        black_box(ebu.relative_threshold().unwrap());
    }
}

#[cfg(feature = "c-tests")]
fn ebur128_f32_c(channels: u32, rate: u32, mode: ebur128_c::Mode, data: &[f32]) {
    use ebur128_c::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
    ebu.add_frames_f32(&data).unwrap();

    if mode.contains(ebur128_c::Mode::I) {
        black_box(ebu.loudness_global().unwrap());
    }
    if mode.contains(ebur128_c::Mode::M) {
        black_box(ebu.loudness_momentary().unwrap());
    }
    if mode.contains(ebur128_c::Mode::S) {
        black_box(ebu.loudness_shortterm().unwrap());
    }
    black_box(ebu.loudness_window(1).unwrap());

    if mode.contains(ebur128_c::Mode::LRA) {
        black_box(ebu.loudness_range().unwrap());
    }

    if mode.contains(ebur128_c::Mode::SAMPLE_PEAK) {
        black_box(ebu.sample_peak(0).unwrap());
        black_box(ebu.sample_peak(1).unwrap());
        black_box(ebu.prev_sample_peak(0).unwrap());
        black_box(ebu.prev_sample_peak(1).unwrap());
    }

    if mode.contains(ebur128_c::Mode::TRUE_PEAK) {
        black_box(ebu.true_peak(0).unwrap());
        black_box(ebu.true_peak(1).unwrap());
        black_box(ebu.prev_true_peak(0).unwrap());
        black_box(ebu.prev_true_peak(1).unwrap());
    }

    if mode.contains(ebur128_c::Mode::I) {
        black_box(ebu.relative_threshold().unwrap());
    }
}

fn ebur128_f32(channels: u32, rate: u32, mode: ebur128::Mode, data: &[f32]) {
    use ebur128::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
    ebu.add_frames_f32(&data).unwrap();

    if mode.contains(ebur128::Mode::I) {
        black_box(ebu.loudness_global().unwrap());
    }
    if mode.contains(ebur128::Mode::M) {
        black_box(ebu.loudness_momentary().unwrap());
    }
    if mode.contains(ebur128::Mode::S) {
        black_box(ebu.loudness_shortterm().unwrap());
    }
    black_box(ebu.loudness_window(1).unwrap());

    if mode.contains(ebur128::Mode::LRA) {
        black_box(ebu.loudness_range().unwrap());
    }

    if mode.contains(ebur128::Mode::SAMPLE_PEAK) {
        black_box(ebu.sample_peak(0).unwrap());
        black_box(ebu.sample_peak(1).unwrap());
        black_box(ebu.prev_sample_peak(0).unwrap());
        black_box(ebu.prev_sample_peak(1).unwrap());
    }

    if mode.contains(ebur128::Mode::TRUE_PEAK) {
        black_box(ebu.true_peak(0).unwrap());
        black_box(ebu.true_peak(1).unwrap());
        black_box(ebu.prev_true_peak(0).unwrap());
        black_box(ebu.prev_true_peak(1).unwrap());
    }

    if mode.contains(ebur128::Mode::I) {
        black_box(ebu.relative_threshold().unwrap());
    }
}

#[cfg(feature = "c-tests")]
fn ebur128_f64_c(channels: u32, rate: u32, mode: ebur128_c::Mode, data: &[f64]) {
    use ebur128_c::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
    ebu.add_frames_f64(&data).unwrap();

    if mode.contains(ebur128_c::Mode::I) {
        black_box(ebu.loudness_global().unwrap());
    }
    if mode.contains(ebur128_c::Mode::M) {
        black_box(ebu.loudness_momentary().unwrap());
    }
    if mode.contains(ebur128_c::Mode::S) {
        black_box(ebu.loudness_shortterm().unwrap());
    }
    black_box(ebu.loudness_window(1).unwrap());

    if mode.contains(ebur128_c::Mode::LRA) {
        black_box(ebu.loudness_range().unwrap());
    }

    if mode.contains(ebur128_c::Mode::SAMPLE_PEAK) {
        black_box(ebu.sample_peak(0).unwrap());
        black_box(ebu.sample_peak(1).unwrap());
        black_box(ebu.prev_sample_peak(0).unwrap());
        black_box(ebu.prev_sample_peak(1).unwrap());
    }

    if mode.contains(ebur128_c::Mode::TRUE_PEAK) {
        black_box(ebu.true_peak(0).unwrap());
        black_box(ebu.true_peak(1).unwrap());
        black_box(ebu.prev_true_peak(0).unwrap());
        black_box(ebu.prev_true_peak(1).unwrap());
    }

    if mode.contains(ebur128_c::Mode::I) {
        black_box(ebu.relative_threshold().unwrap());
    }
}

fn ebur128_f64(channels: u32, rate: u32, mode: ebur128::Mode, data: &[f64]) {
    use ebur128::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
    ebu.add_frames_f64(&data).unwrap();

    if mode.contains(ebur128::Mode::I) {
        black_box(ebu.loudness_global().unwrap());
    }
    if mode.contains(ebur128::Mode::M) {
        black_box(ebu.loudness_momentary().unwrap());
    }
    if mode.contains(ebur128::Mode::S) {
        black_box(ebu.loudness_shortterm().unwrap());
    }
    black_box(ebu.loudness_window(1).unwrap());

    if mode.contains(ebur128::Mode::LRA) {
        black_box(ebu.loudness_range().unwrap());
    }

    if mode.contains(ebur128::Mode::SAMPLE_PEAK) {
        black_box(ebu.sample_peak(0).unwrap());
        black_box(ebu.sample_peak(1).unwrap());
        black_box(ebu.prev_sample_peak(0).unwrap());
        black_box(ebu.prev_sample_peak(1).unwrap());
    }

    if mode.contains(ebur128::Mode::TRUE_PEAK) {
        black_box(ebu.true_peak(0).unwrap());
        black_box(ebu.true_peak(1).unwrap());
        black_box(ebu.prev_true_peak(0).unwrap());
        black_box(ebu.prev_true_peak(1).unwrap());
    }

    if mode.contains(ebur128::Mode::I) {
        black_box(ebu.relative_threshold().unwrap());
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let modes = [
        ("M", ebur128::Mode::M, ebur128_c::Mode::M),
        ("S", ebur128::Mode::S, ebur128_c::Mode::S),
        ("I", ebur128::Mode::I, ebur128_c::Mode::I),
        ("LRA", ebur128::Mode::LRA, ebur128_c::Mode::LRA),
        (
            "SAMPLE_PEAK",
            ebur128::Mode::SAMPLE_PEAK,
            ebur128_c::Mode::SAMPLE_PEAK,
        ),
        (
            "TRUE_PEAK",
            ebur128::Mode::TRUE_PEAK,
            ebur128_c::Mode::TRUE_PEAK,
        ),
        (
            "M histogram",
            ebur128::Mode::M | ebur128::Mode::HISTOGRAM,
            ebur128_c::Mode::M | ebur128_c::Mode::HISTOGRAM,
        ),
        (
            "S histogram",
            ebur128::Mode::S | ebur128::Mode::HISTOGRAM,
            ebur128_c::Mode::S | ebur128_c::Mode::HISTOGRAM,
        ),
        (
            "I histogram",
            ebur128::Mode::I | ebur128::Mode::HISTOGRAM,
            ebur128_c::Mode::I | ebur128_c::Mode::HISTOGRAM,
        ),
        (
            "LRA histogram",
            ebur128::Mode::LRA | ebur128::Mode::HISTOGRAM,
            ebur128_c::Mode::LRA | ebur128_c::Mode::HISTOGRAM,
        ),
        (
            "SAMPLE_PEAK histogram",
            ebur128::Mode::SAMPLE_PEAK | ebur128::Mode::HISTOGRAM,
            ebur128_c::Mode::SAMPLE_PEAK | ebur128_c::Mode::HISTOGRAM,
        ),
        (
            "TRUE_PEAK histogram",
            ebur128::Mode::TRUE_PEAK | ebur128::Mode::HISTOGRAM,
            ebur128_c::Mode::TRUE_PEAK | ebur128_c::Mode::HISTOGRAM,
        ),
        (
            "all",
            ebur128::Mode::all() & !ebur128::Mode::HISTOGRAM,
            ebur128_c::Mode::all() & !ebur128_c::Mode::HISTOGRAM,
        ),
        (
            "all histogram",
            ebur128::Mode::all(),
            ebur128_c::Mode::all(),
        ),
    ];

    #[allow(unused_variables)]
    for (name, mode, mode_c) in &modes {
        let mode = *mode;
        #[cfg(feature = "c-tests")]
        let mode_c = *mode_c;

        let mut data = vec![0i16; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * (std::i16::MAX - 1) as f32;
            out[0] = val as i16;
            out[1] = val as i16;
            accumulator += step;
        }

        let mut group = c.benchmark_group(format!("ebur128: 48kHz i16 2ch {}", name));

        #[cfg(feature = "c-tests")]
        {
            group.bench_function("C", |b| {
                b.iter(|| {
                    ebur128_i16_c(
                        black_box(2),
                        black_box(48_000),
                        black_box(mode_c),
                        black_box(&data),
                    )
                })
            });
        }
        group.bench_function("Rust", |b| {
            b.iter(|| {
                ebur128_i16(
                    black_box(2),
                    black_box(48_000),
                    black_box(mode),
                    black_box(&data),
                )
            })
        });

        group.finish();

        let mut data = vec![0i32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * (std::i32::MAX - 1) as f32;
            out[0] = val as i32;
            out[1] = val as i32;
            accumulator += step;
        }

        let mut group = c.benchmark_group(format!("ebur128: 48kHz i32 2ch {}", name));

        #[cfg(feature = "c-tests")]
        {
            group.bench_function("C", |b| {
                b.iter(|| {
                    ebur128_i32_c(
                        black_box(2),
                        black_box(48_000),
                        black_box(mode_c),
                        black_box(&data),
                    )
                })
            });
        }
        group.bench_function("Rust", |b| {
            b.iter(|| {
                ebur128_i32(
                    black_box(2),
                    black_box(48_000),
                    black_box(mode),
                    black_box(&data),
                )
            })
        });

        group.finish();

        let mut data = vec![0.0f32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val;
            out[1] = val;
            accumulator += step;
        }

        let mut group = c.benchmark_group(format!("ebur128: 48kHz f32 2ch {}", name));

        #[cfg(feature = "c-tests")]
        {
            group.bench_function("C", |b| {
                b.iter(|| {
                    ebur128_f32_c(
                        black_box(2),
                        black_box(48_000),
                        black_box(mode_c),
                        black_box(&data),
                    )
                })
            });
        }
        group.bench_function("Rust", |b| {
            b.iter(|| {
                ebur128_f32(
                    black_box(2),
                    black_box(48_000),
                    black_box(mode),
                    black_box(&data),
                )
            })
        });

        group.finish();

        let mut data = vec![0.0f64; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val as f64;
            out[1] = val as f64;
            accumulator += step;
        }

        let mut group = c.benchmark_group(format!("ebur128: 48kHz f64 2ch {}", name));

        #[cfg(feature = "c-tests")]
        {
            group.bench_function("C", |b| {
                b.iter(|| {
                    ebur128_f64_c(
                        black_box(2),
                        black_box(48_000),
                        black_box(mode_c),
                        black_box(&data),
                    )
                })
            });
        }
        group.bench_function("Rust", |b| {
            b.iter(|| {
                ebur128_f64(
                    black_box(2),
                    black_box(48_000),
                    black_box(mode),
                    black_box(&data),
                )
            })
        });

        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
