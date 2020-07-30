use criterion::{black_box, criterion_group, criterion_main, Criterion};

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

fn ebur128_i16_c(channels: u32, rate: u32, mode: ebur128_c::Mode, data: &[i16]) {
    use ebur128_c::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
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

fn ebur128_i16(channels: u32, rate: u32, mode: ebur128::Mode, data: &[i16]) {
    use ebur128::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
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

fn ebur128_i32_c(channels: u32, rate: u32, mode: ebur128_c::Mode, data: &[i32]) {
    use ebur128_c::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
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

fn ebur128_i32(channels: u32, rate: u32, mode: ebur128::Mode, data: &[i32]) {
    use ebur128::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
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

fn ebur128_f32_c(channels: u32, rate: u32, mode: ebur128_c::Mode, data: &[f32]) {
    use ebur128_c::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
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

fn ebur128_f32(channels: u32, rate: u32, mode: ebur128::Mode, data: &[f32]) {
    use ebur128::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
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

fn ebur128_f64_c(channels: u32, rate: u32, mode: ebur128_c::Mode, data: &[f64]) {
    use ebur128_c::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
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

fn ebur128_f64(channels: u32, rate: u32, mode: ebur128::Mode, data: &[f64]) {
    use ebur128::EbuR128;

    let mut ebu = EbuR128::new(channels, rate, mode).unwrap();
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

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut data = vec![0i16; 48_000 * 5 * 2];
    let mut accumulator = 0.0;
    let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
    for out in data.chunks_exact_mut(2) {
        let val = f32::sin(accumulator) * (std::i16::MAX - 1) as f32;
        out[0] = val as i16;
        out[1] = val as i16;
        accumulator += step;
    }

    let mut group = c.benchmark_group("ebur128: 48kHz i16 2ch all");

    group.bench_function("C", |b| {
        b.iter(|| {
            ebur128_i16_c(
                black_box(2),
                black_box(48_000),
                black_box(ebur128_c::Mode::all()),
                black_box(&data),
            )
        })
    });
    group.bench_function("Rust", |b| {
        b.iter(|| {
            ebur128_i16(
                black_box(2),
                black_box(48_000),
                black_box(ebur128::Mode::all()),
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

    let mut group = c.benchmark_group("ebur128: 48kHz i32 2ch all");

    group.bench_function("C", |b| {
        b.iter(|| {
            ebur128_i32_c(
                black_box(2),
                black_box(48_000),
                black_box(ebur128_c::Mode::all()),
                black_box(&data),
            )
        })
    });
    group.bench_function("Rust", |b| {
        b.iter(|| {
            ebur128_i32(
                black_box(2),
                black_box(48_000),
                black_box(ebur128::Mode::all()),
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

    let mut group = c.benchmark_group("ebur128: 48kHz f32 2ch all");

    group.bench_function("C", |b| {
        b.iter(|| {
            ebur128_f32_c(
                black_box(2),
                black_box(48_000),
                black_box(ebur128_c::Mode::all()),
                black_box(&data),
            )
        })
    });
    group.bench_function("Rust", |b| {
        b.iter(|| {
            ebur128_f32(
                black_box(2),
                black_box(48_000),
                black_box(ebur128::Mode::all()),
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

    let mut group = c.benchmark_group("ebur128: 48kHz f64 2ch all");

    group.bench_function("C", |b| {
        b.iter(|| {
            ebur128_f64_c(
                black_box(2),
                black_box(48_000),
                black_box(ebur128_c::Mode::all()),
                black_box(&data),
            )
        })
    });
    group.bench_function("Rust", |b| {
        b.iter(|| {
            ebur128_f64(
                black_box(2),
                black_box(48_000),
                black_box(ebur128::Mode::all()),
                black_box(&data),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
