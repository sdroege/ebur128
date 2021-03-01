use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ebur128::filter;

// Run filter benchmarks on the same filter instance to not measure the setup time
// and measure once with and another time without calculating the sample peak.
//
// We don't calculate the true peak because that has its own benchmark.

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter create: 48kHz 2ch");

    #[cfg(feature = "c-tests")]
    group.bench_function("C", |b| {
        b.iter(|| unsafe {
            let f = filter::filter_create_c(
                black_box(48_000),
                black_box(2),
                black_box(0),
                black_box(0),
            );
            filter::filter_destroy_c(f);
        })
    });

    group.bench_function("Rust", |b| {
        b.iter(|| {
            let f = filter::Filter::new(
                black_box(48_000),
                black_box(2),
                black_box(false),
                black_box(false),
            );
            drop(black_box(f));
        })
    });

    group.finish();

    for (sample_peak, name) in &[(true, " with sample peak"), (false, "")] {
        #[cfg(feature = "c-tests")]
        let channel_map_c = [1; 2];
        let channel_map = [ebur128::Channel::Left; 2];
        let mut data_out = vec![0.0f64; 19200 * 2];
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

        let mut group = c.benchmark_group(format!("filter process: 48kHz 2ch i16{}", name));

        #[cfg(feature = "c-tests")]
        unsafe {
            let f = filter::filter_create_c(48_000, 2, if *sample_peak { 1 } else { 0 }, 0);
            group.bench_function("C", |b| {
                b.iter(|| {
                    filter::filter_process_short_c(
                        black_box(f),
                        black_box(data.len() / 2),
                        black_box(data.as_ptr()),
                        black_box(data_out.as_mut_ptr()),
                        black_box(channel_map_c.as_ptr()),
                    )
                })
            });
            filter::filter_destroy_c(f);
        }

        {
            let mut f = filter::Filter::new(48_000, 2, *sample_peak, false);
            group.bench_function("Rust/Interleaved", |b| {
                b.iter(|| {
                    f.process(
                        black_box(ebur128::Interleaved::new(&*data, 2).unwrap()),
                        black_box(&mut data_out),
                        black_box(0),
                        black_box(&channel_map),
                    );
                })
            });

            let mut f = filter::Filter::new(48_000, 2, *sample_peak, false);
            group.bench_function("Rust/Planar", |b| {
                b.iter(|| {
                    f.process(
                        black_box(ebur128::Planar::new(&[fst, snd]).unwrap()),
                        black_box(&mut data_out),
                        black_box(0),
                        black_box(&channel_map),
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

        let mut group = c.benchmark_group(format!("filter process: 48kHz 2ch i32{}", name));

        #[cfg(feature = "c-tests")]
        unsafe {
            let f = filter::filter_create_c(48_000, 2, if *sample_peak { 1 } else { 0 }, 0);
            group.bench_function("C", |b| {
                b.iter(|| {
                    filter::filter_process_int_c(
                        black_box(f),
                        black_box(data.len() / 2),
                        black_box(data.as_ptr()),
                        black_box(data_out.as_mut_ptr()),
                        black_box(channel_map_c.as_ptr()),
                    )
                })
            });
            filter::filter_destroy_c(f);
        }

        {
            let mut f = filter::Filter::new(48_000, 2, *sample_peak, false);
            group.bench_function("Rust/Interleaved", |b| {
                b.iter(|| {
                    f.process(
                        black_box(ebur128::Interleaved::new(&*data, 2).unwrap()),
                        black_box(&mut data_out),
                        black_box(0),
                        black_box(&channel_map),
                    );
                })
            });

            let mut f = filter::Filter::new(48_000, 2, *sample_peak, false);
            group.bench_function("Rust/Planar", |b| {
                b.iter(|| {
                    f.process(
                        black_box(ebur128::Planar::new(&[fst, snd]).unwrap()),
                        black_box(&mut data_out),
                        black_box(0),
                        black_box(&channel_map),
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

        let mut group = c.benchmark_group(format!("filter process: 48kHz 2ch f32{}", name));

        #[cfg(feature = "c-tests")]
        unsafe {
            let f = filter::filter_create_c(48_000, 2, if *sample_peak { 1 } else { 0 }, 0);
            group.bench_function("C", |b| {
                b.iter(|| {
                    filter::filter_process_float_c(
                        black_box(f),
                        black_box(data.len() / 2),
                        black_box(data.as_ptr()),
                        black_box(data_out.as_mut_ptr()),
                        black_box(channel_map_c.as_ptr()),
                    )
                })
            });
            filter::filter_destroy_c(f);
        }

        {
            let mut f = filter::Filter::new(48_000, 2, *sample_peak, false);
            group.bench_function("Rust/Interleaved", |b| {
                b.iter(|| {
                    f.process(
                        black_box(ebur128::Interleaved::new(&*data, 2).unwrap()),
                        black_box(&mut data_out),
                        black_box(0),
                        black_box(&channel_map),
                    );
                })
            });

            let mut f = filter::Filter::new(48_000, 2, *sample_peak, false);
            group.bench_function("Rust/Planar", |b| {
                b.iter(|| {
                    f.process(
                        black_box(ebur128::Planar::new(&[fst, snd]).unwrap()),
                        black_box(&mut data_out),
                        black_box(0),
                        black_box(&channel_map),
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

        let mut group = c.benchmark_group(format!("filter process: 48kHz 2ch f64{}", name));

        #[cfg(feature = "c-tests")]
        unsafe {
            let f = filter::filter_create_c(48_000, 2, if *sample_peak { 1 } else { 0 }, 0);
            group.bench_function("C", |b| {
                b.iter(|| {
                    filter::filter_process_double_c(
                        black_box(f),
                        black_box(data.len() / 2),
                        black_box(data.as_ptr()),
                        black_box(data_out.as_mut_ptr()),
                        black_box(channel_map_c.as_ptr()),
                    )
                })
            });
            filter::filter_destroy_c(f);
        }

        {
            let mut f = filter::Filter::new(48_000, 2, *sample_peak, false);
            group.bench_function("Rust/Interleaved", |b| {
                b.iter(|| {
                    f.process(
                        black_box(ebur128::Interleaved::new(&*data, 2).unwrap()),
                        black_box(&mut data_out),
                        black_box(0),
                        black_box(&channel_map),
                    );
                })
            });

            let mut f = filter::Filter::new(48_000, 2, *sample_peak, false);
            group.bench_function("Rust/Planar", |b| {
                b.iter(|| {
                    f.process(
                        black_box(ebur128::Planar::new(&[fst, snd]).unwrap()),
                        black_box(&mut data_out),
                        black_box(0),
                        black_box(&channel_map),
                    );
                })
            });
        }

        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
