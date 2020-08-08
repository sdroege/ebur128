use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Run all benchmarks without true peak calculation as that's just another function call
// and we measure that one in the true peak benchmarks already.

#[cfg(feature = "internal-tests")]
fn filter_i16_c(rate: u32, channels: u32, src: &[i16], dest: &mut [f64], channel_map: &[u32]) {
    use ebur128::filter;

    unsafe {
        let f = filter::filter_create_c(rate, channels, 1, 0);
        filter::filter_process_short_c(
            f,
            src.len() / channels as usize,
            src.as_ptr(),
            dest.as_mut_ptr(),
            channel_map.as_ptr(),
        );
        filter::filter_destroy_c(f);
    }
}

#[cfg(feature = "internal-tests")]
fn filter_i16(rate: u32, channels: u32, src: &[i16], dest: &mut [f64], channel_map: &[u32]) {
    use ebur128::filter;

    let mut f = filter::Filter::new(rate, channels, true, false);
    f.process(src, dest, channel_map);
}

#[cfg(feature = "internal-tests")]
fn filter_i32_c(rate: u32, channels: u32, src: &[i32], dest: &mut [f64], channel_map: &[u32]) {
    use ebur128::filter;

    unsafe {
        let f = filter::filter_create_c(rate, channels, 1, 0);
        filter::filter_process_int_c(
            f,
            src.len() / channels as usize,
            src.as_ptr(),
            dest.as_mut_ptr(),
            channel_map.as_ptr(),
        );
        filter::filter_destroy_c(f);
    }
}

#[cfg(feature = "internal-tests")]
fn filter_i32(rate: u32, channels: u32, src: &[i32], dest: &mut [f64], channel_map: &[u32]) {
    use ebur128::filter;

    let mut f = filter::Filter::new(rate, channels, true, false);
    f.process(src, dest, channel_map);
}

#[cfg(feature = "internal-tests")]
fn filter_f32_c(rate: u32, channels: u32, src: &[f32], dest: &mut [f64], channel_map: &[u32]) {
    use ebur128::filter;

    unsafe {
        let f = filter::filter_create_c(rate, channels, 1, 0);
        filter::filter_process_float_c(
            f,
            src.len() / channels as usize,
            src.as_ptr(),
            dest.as_mut_ptr(),
            channel_map.as_ptr(),
        );
        filter::filter_destroy_c(f);
    }
}

#[cfg(feature = "internal-tests")]
fn filter_f32(rate: u32, channels: u32, src: &[f32], dest: &mut [f64], channel_map: &[u32]) {
    use ebur128::filter;

    let mut f = filter::Filter::new(rate, channels, true, false);
    f.process(src, dest, channel_map);
}

#[cfg(feature = "internal-tests")]
fn filter_f64_c(rate: u32, channels: u32, src: &[f64], dest: &mut [f64], channel_map: &[u32]) {
    use ebur128::filter;

    unsafe {
        let f = filter::filter_create_c(rate, channels, 1, 0);
        filter::filter_process_double_c(
            f,
            src.len() / channels as usize,
            src.as_ptr(),
            dest.as_mut_ptr(),
            channel_map.as_ptr(),
        );
        filter::filter_destroy_c(f);
    }
}

#[cfg(feature = "internal-tests")]
fn filter_f64(rate: u32, channels: u32, src: &[f64], dest: &mut [f64], channel_map: &[u32]) {
    use ebur128::filter;

    let mut f = filter::Filter::new(rate, channels, true, false);
    f.process(src, dest, channel_map);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    #[cfg(feature = "internal-tests")]
    {
        let channel_map = [1; 2];
        let mut data_out = vec![0.0f64; 19200 * 2];
        let mut data = vec![0i16; 19200 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * std::i16::MAX as f32;
            out[0] = val as i16;
            out[1] = val as i16;
            accumulator += step;
        }

        let mut group = c.benchmark_group("filter: 48kHz 2ch i16");

        group.bench_function("C", |b| {
            b.iter(|| {
                filter_i16_c(
                    black_box(48_000),
                    black_box(2),
                    black_box(&data),
                    black_box(&mut data_out),
                    black_box(&channel_map),
                )
            })
        });

        group.bench_function("Rust", |b| {
            b.iter(|| {
                filter_i16(
                    black_box(48_000),
                    black_box(2),
                    black_box(&data),
                    black_box(&mut data_out),
                    black_box(&channel_map),
                )
            })
        });

        group.finish();

        let mut data = vec![0i32; 19200 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * std::i32::MAX as f32;
            out[0] = val as i32;
            out[1] = val as i32;
            accumulator += step;
        }

        let mut group = c.benchmark_group("filter: 48kHz 2ch i32");

        group.bench_function("C", |b| {
            b.iter(|| {
                filter_i32_c(
                    black_box(48_000),
                    black_box(2),
                    black_box(&data),
                    black_box(&mut data_out),
                    black_box(&channel_map),
                )
            })
        });

        group.bench_function("Rust", |b| {
            b.iter(|| {
                filter_i32(
                    black_box(48_000),
                    black_box(2),
                    black_box(&data),
                    black_box(&mut data_out),
                    black_box(&channel_map),
                )
            })
        });

        group.finish();

        let mut data = vec![0.0f32; 19200 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val;
            out[1] = val;
            accumulator += step;
        }

        let mut group = c.benchmark_group("filter: 48kHz 2ch f32");

        group.bench_function("C", |b| {
            b.iter(|| {
                filter_f32_c(
                    black_box(48_000),
                    black_box(2),
                    black_box(&data),
                    black_box(&mut data_out),
                    black_box(&channel_map),
                )
            })
        });

        group.bench_function("Rust", |b| {
            b.iter(|| {
                filter_f32(
                    black_box(48_000),
                    black_box(2),
                    black_box(&data),
                    black_box(&mut data_out),
                    black_box(&channel_map),
                )
            })
        });

        group.finish();

        let mut data = vec![0.0f64; 19200 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val as f64;
            out[1] = val as f64;
            accumulator += step;
        }

        let mut group = c.benchmark_group("filter: 48kHz 2ch f64");

        group.bench_function("C", |b| {
            b.iter(|| {
                filter_f64_c(
                    black_box(48_000),
                    black_box(2),
                    black_box(&data),
                    black_box(&mut data_out),
                    black_box(&channel_map),
                )
            })
        });

        group.bench_function("Rust", |b| {
            b.iter(|| {
                filter_f64(
                    black_box(48_000),
                    black_box(2),
                    black_box(&data),
                    black_box(&mut data_out),
                    black_box(&channel_map),
                )
            })
        });

        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
