use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ebur128::true_peak;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut data = vec![0i16; 19200 * 2];
    let mut data_planar = vec![0i16; 19200 * 2];
    let (fst, snd) = data_planar.split_at_mut(19200);
    let mut accumulator = 0.0;
    let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
    for (out, (fst, snd)) in Iterator::zip(
        data.chunks_exact_mut(2),
        Iterator::zip(fst.iter_mut(), snd.iter_mut()),
    ) {
        let val = f32::sin(accumulator) * std::i16::MAX as f32;
        out[0] = val as i16;
        out[1] = val as i16;
        *fst = val as i16;
        *snd = val as i16;
        accumulator += step;
    }

    let mut peaks = vec![0.0f64; 2];

    let mut group = c.benchmark_group("true_peak: 48kHz 2ch i16");

    #[cfg(feature = "c-tests")]
    unsafe {
        let tp = true_peak::true_peak_create_c(black_box(48_000), black_box(2));

        group.bench_function("C", |b| {
            b.iter(|| {
                true_peak::true_peak_check_short_c(
                    black_box(tp),
                    black_box(data.len() / 2usize),
                    black_box(data.as_ptr()),
                    black_box(peaks.as_mut_ptr()),
                );
            })
        });

        true_peak::true_peak_destroy_c(tp);
    }

    {
        let mut tp = true_peak::TruePeak::new(black_box(48_000), black_box(2)).unwrap();
        group.bench_function("Rust/Interleaved", |b| {
            b.iter(|| {
                tp.check_true_peak(
                    black_box(ebur128::Interleaved::new(&data, 2).unwrap()),
                    black_box(&mut peaks),
                );
            })
        });

        let mut tp = true_peak::TruePeak::new(black_box(48_000), black_box(2)).unwrap();
        group.bench_function("Rust/Planar", |b| {
            b.iter(|| {
                tp.check_true_peak(
                    black_box(ebur128::Planar::new(&[fst, snd]).unwrap()),
                    black_box(&mut peaks),
                );
            })
        });
    }

    group.finish();

    let mut data = vec![0i32; 19200 * 2];
    let mut data_planar = vec![0i32; 19200 * 2];
    let (fst, snd) = data_planar.split_at_mut(19200);
    let mut accumulator = 0.0;
    let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
    for (out, (fst, snd)) in Iterator::zip(
        data.chunks_exact_mut(2),
        Iterator::zip(fst.iter_mut(), snd.iter_mut()),
    ) {
        let val = f32::sin(accumulator) * std::i32::MAX as f32;
        out[0] = val as i32;
        out[1] = val as i32;
        *fst = val as i32;
        *snd = val as i32;
        accumulator += step;
    }

    let mut group = c.benchmark_group("true_peak: 48kHz 2ch i32");

    #[cfg(feature = "c-tests")]
    unsafe {
        let tp = true_peak::true_peak_create_c(black_box(48_000), black_box(2));

        group.bench_function("C", |b| {
            b.iter(|| {
                true_peak::true_peak_check_int_c(
                    black_box(tp),
                    black_box(data.len() / 2usize),
                    black_box(data.as_ptr()),
                    black_box(peaks.as_mut_ptr()),
                );
            })
        });

        true_peak::true_peak_destroy_c(tp);
    }

    {
        let mut tp = true_peak::TruePeak::new(black_box(48_000), black_box(2)).unwrap();
        group.bench_function("Rust/Interleaved", |b| {
            b.iter(|| {
                tp.check_true_peak(
                    black_box(ebur128::Interleaved::new(&data, 2).unwrap()),
                    black_box(&mut peaks),
                );
            })
        });

        let mut tp = true_peak::TruePeak::new(black_box(48_000), black_box(2)).unwrap();
        group.bench_function("Rust/Planar", |b| {
            b.iter(|| {
                tp.check_true_peak(
                    black_box(ebur128::Planar::new(&[fst, snd]).unwrap()),
                    black_box(&mut peaks),
                );
            })
        });
    }

    group.finish();

    let mut data = vec![0.0f32; 19200 * 2];
    let mut data_planar = vec![0.0f32; 19200 * 2];
    let (fst, snd) = data_planar.split_at_mut(19200);
    let mut accumulator = 0.0;
    let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
    for (out, (fst, snd)) in Iterator::zip(
        data.chunks_exact_mut(2),
        Iterator::zip(fst.iter_mut(), snd.iter_mut()),
    ) {
        let val = f32::sin(accumulator);
        out[0] = val;
        out[1] = val;
        *fst = val;
        *snd = val;
        accumulator += step;
    }

    let mut group = c.benchmark_group("true_peak: 48kHz 2ch f32");

    #[cfg(feature = "c-tests")]
    unsafe {
        let tp = true_peak::true_peak_create_c(black_box(48_000), black_box(2));

        group.bench_function("C", |b| {
            b.iter(|| {
                true_peak::true_peak_check_float_c(
                    black_box(tp),
                    black_box(data.len() / 2usize),
                    black_box(data.as_ptr()),
                    black_box(peaks.as_mut_ptr()),
                );
            })
        });

        true_peak::true_peak_destroy_c(tp);
    }

    {
        let mut tp = true_peak::TruePeak::new(black_box(48_000), black_box(2)).unwrap();
        group.bench_function("Rust/Interleaved", |b| {
            b.iter(|| {
                tp.check_true_peak(
                    black_box(ebur128::Interleaved::new(&data, 2).unwrap()),
                    black_box(&mut peaks),
                );
            })
        });

        let mut tp = true_peak::TruePeak::new(black_box(48_000), black_box(2)).unwrap();
        group.bench_function("Rust/Planar", |b| {
            b.iter(|| {
                tp.check_true_peak(
                    black_box(ebur128::Planar::new(&[fst, snd]).unwrap()),
                    black_box(&mut peaks),
                );
            })
        });
    }

    group.finish();

    let mut data = vec![0.0f64; 19200 * 2];
    let mut data_planar = vec![0.0f64; 19200 * 2];
    let (fst, snd) = data_planar.split_at_mut(19200);
    let mut accumulator = 0.0;
    let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
    for (out, (fst, snd)) in Iterator::zip(
        data.chunks_exact_mut(2),
        Iterator::zip(fst.iter_mut(), snd.iter_mut()),
    ) {
        let val = f32::sin(accumulator);
        out[0] = val as f64;
        out[1] = val as f64;
        *fst = val as f64;
        *snd = val as f64;
        accumulator += step;
    }

    let mut group = c.benchmark_group("true_peak: 48kHz 2ch f64");

    #[cfg(feature = "c-tests")]
    unsafe {
        let tp = true_peak::true_peak_create_c(black_box(48_000), black_box(2));

        group.bench_function("C", |b| {
            b.iter(|| {
                true_peak::true_peak_check_double_c(
                    black_box(tp),
                    black_box(data.len() / 2usize),
                    black_box(data.as_ptr()),
                    black_box(peaks.as_mut_ptr()),
                );
            })
        });

        true_peak::true_peak_destroy_c(tp);
    }

    {
        let mut tp = true_peak::TruePeak::new(black_box(48_000), black_box(2)).unwrap();
        group.bench_function("Rust/Interleaved", |b| {
            b.iter(|| {
                tp.check_true_peak(
                    black_box(ebur128::Interleaved::new(&data, 2).unwrap()),
                    black_box(&mut peaks),
                );
            })
        });

        let mut tp = true_peak::TruePeak::new(black_box(48_000), black_box(2)).unwrap();
        group.bench_function("Rust/Planar", |b| {
            b.iter(|| {
                tp.check_true_peak(
                    black_box(ebur128::Planar::new(&[fst, snd]).unwrap()),
                    black_box(&mut peaks),
                );
            })
        });
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
