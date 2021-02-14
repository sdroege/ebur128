use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ebur128::{EbuR128, Mode};

fn get_results(ebu: &EbuR128, mode: Mode) {
    if mode.contains(Mode::I) {
        black_box(ebu.loudness_global().unwrap());
    }
    if mode.contains(Mode::M) {
        black_box(ebu.loudness_momentary().unwrap());
    }
    if mode.contains(Mode::S) {
        black_box(ebu.loudness_shortterm().unwrap());
    }
    black_box(ebu.loudness_window(1).unwrap());

    if mode.contains(Mode::LRA) {
        black_box(ebu.loudness_range().unwrap());
    }

    if mode.contains(Mode::SAMPLE_PEAK) {
        black_box(ebu.sample_peak(0).unwrap());
        black_box(ebu.sample_peak(1).unwrap());
        black_box(ebu.prev_sample_peak(0).unwrap());
        black_box(ebu.prev_sample_peak(1).unwrap());
    }

    if mode.contains(Mode::TRUE_PEAK) {
        black_box(ebu.true_peak(0).unwrap());
        black_box(ebu.true_peak(1).unwrap());
        black_box(ebu.prev_true_peak(0).unwrap());
        black_box(ebu.prev_true_peak(1).unwrap());
    }

    if mode.contains(Mode::I) {
        black_box(ebu.relative_threshold().unwrap());
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let modes = [
        ("M", Mode::M),
        ("S", Mode::S),
        ("I", Mode::I),
        ("LRA", Mode::LRA),
        ("SAMPLE_PEAK", Mode::SAMPLE_PEAK),
        ("TRUE_PEAK", Mode::TRUE_PEAK),
        ("I histogram", Mode::I | Mode::HISTOGRAM),
        ("LRA histogram", Mode::LRA | Mode::HISTOGRAM),
        ("all", Mode::all() & !Mode::HISTOGRAM),
        ("all histogram", Mode::all()),
    ];

    #[allow(unused_variables)]
    for (name, mode) in &modes {
        let mode = *mode;

        let mut group = c.benchmark_group(format!("ebur128 create: 48kHz 2ch {}", name));

        group.bench_function("Rust", |b| {
            b.iter(|| {
                let ebu = EbuR128::new(black_box(2), black_box(48_000), black_box(mode)).unwrap();
                drop(black_box(ebu));
            })
        });

        group.finish();

        let mut data = vec![0i16; 48_000 * 5 * 2];
        let mut data_planar = vec![0i16; 48_000 * 5 * 2];
        let (fst, snd) = data_planar.split_at_mut(48_000 * 5);
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

        let mut group = c.benchmark_group(format!("ebur128 process: 48kHz i16 2ch {}", name));

        group.bench_function("Rust/Interleaved", |b| {
            b.iter(|| {
                let mut ebu =
                    EbuR128::new(black_box(2), black_box(48_000), black_box(mode)).unwrap();
                ebu.add_frames_i16(&data).unwrap();

                get_results(&ebu, black_box(mode));
            })
        });
        group.bench_function("Rust/Planar", |b| {
            b.iter(|| {
                let mut ebu =
                    EbuR128::new(black_box(2), black_box(48_000), black_box(mode)).unwrap();
                ebu.add_frames_planar_i16(&[fst, snd]).unwrap();

                get_results(&ebu, black_box(mode));
            })
        });

        group.finish();

        let mut data = vec![0i32; 48_000 * 5 * 2];
        let mut data_planar = vec![0i32; 48_000 * 5 * 2];
        let (fst, snd) = data_planar.split_at_mut(48_000 * 5);
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

        let mut group = c.benchmark_group(format!("ebur128 process: 48kHz i32 2ch {}", name));

        group.bench_function("Rust/Interleaved", |b| {
            b.iter(|| {
                let mut ebu =
                    EbuR128::new(black_box(2), black_box(48_000), black_box(mode)).unwrap();
                ebu.add_frames_i32(&data).unwrap();

                get_results(&ebu, black_box(mode));
            })
        });
        group.bench_function("Rust/Planar", |b| {
            b.iter(|| {
                let mut ebu =
                    EbuR128::new(black_box(2), black_box(48_000), black_box(mode)).unwrap();
                ebu.add_frames_planar_i32(&[fst, snd]).unwrap();

                get_results(&ebu, black_box(mode));
            })
        });

        group.finish();

        let mut data = vec![0.0f32; 48_000 * 5 * 2];
        let mut data_planar = vec![0.0f32; 48_000 * 5 * 2];
        let (fst, snd) = data_planar.split_at_mut(48_000 * 5);
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

        let mut group = c.benchmark_group(format!("ebur128 process: 48kHz f32 2ch {}", name));

        group.bench_function("Rust/Interleaved", |b| {
            b.iter(|| {
                let mut ebu =
                    EbuR128::new(black_box(2), black_box(48_000), black_box(mode)).unwrap();
                ebu.add_frames_f32(&data).unwrap();

                get_results(&ebu, black_box(mode));
            })
        });
        group.bench_function("Rust/Planar", |b| {
            b.iter(|| {
                let mut ebu =
                    EbuR128::new(black_box(2), black_box(48_000), black_box(mode)).unwrap();
                ebu.add_frames_planar_f32(&[fst, snd]).unwrap();

                get_results(&ebu, black_box(mode));
            })
        });

        group.finish();

        let mut data = vec![0.0f64; 48_000 * 5 * 2];
        let mut data_planar = vec![0.0f64; 48_000 * 5 * 2];
        let (fst, snd) = data_planar.split_at_mut(48_000 * 5);
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

        let mut group = c.benchmark_group(format!("ebur128 process: 48kHz f64 2ch {}", name));

        group.bench_function("Rust/Interleaved", |b| {
            b.iter(|| {
                let mut ebu =
                    EbuR128::new(black_box(2), black_box(48_000), black_box(mode)).unwrap();
                ebu.add_frames_f64(&data).unwrap();

                get_results(&ebu, black_box(mode));
            })
        });
        group.bench_function("Rust/Planar", |b| {
            b.iter(|| {
                let mut ebu =
                    EbuR128::new(black_box(2), black_box(48_000), black_box(mode)).unwrap();
                ebu.add_frames_planar_f64(&[fst, snd]).unwrap();

                get_results(&ebu, black_box(mode));
            })
        });

        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
