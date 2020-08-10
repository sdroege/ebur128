use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "c-tests")]
fn true_peak_i16_c(rate: u32, channels: u32, data: &[i16], peaks: &mut [f64]) {
    use ebur128::true_peak;

    unsafe {
        let tp = true_peak::true_peak_create_c(rate, channels);
        true_peak::true_peak_check_short_c(
            tp,
            data.len() / channels as usize,
            data.as_ptr(),
            peaks.as_mut_ptr(),
        );
        true_peak::true_peak_destroy_c(tp);
    }
}

#[cfg(feature = "internal-tests")]
fn true_peak_i16(rate: u32, channels: u32, data: &[i16], peaks: &mut [f64]) {
    use ebur128::true_peak;

    let mut tp = true_peak::TruePeak::new(rate, channels).unwrap();
    tp.check_true_peak(data, peaks);
}

#[cfg(feature = "c-tests")]
fn true_peak_i32_c(rate: u32, channels: u32, data: &[i32], peaks: &mut [f64]) {
    use ebur128::true_peak;

    unsafe {
        let tp = true_peak::true_peak_create_c(rate, channels);
        true_peak::true_peak_check_int_c(
            tp,
            data.len() / channels as usize,
            data.as_ptr(),
            peaks.as_mut_ptr(),
        );
        true_peak::true_peak_destroy_c(tp);
    }
}

#[cfg(feature = "internal-tests")]
fn true_peak_i32(rate: u32, channels: u32, data: &[i32], peaks: &mut [f64]) {
    use ebur128::true_peak;

    let mut tp = true_peak::TruePeak::new(rate, channels).unwrap();
    tp.check_true_peak(data, peaks);
}

#[cfg(feature = "c-tests")]
fn true_peak_f32_c(rate: u32, channels: u32, data: &[f32], peaks: &mut [f64]) {
    use ebur128::true_peak;

    unsafe {
        let tp = true_peak::true_peak_create_c(rate, channels);
        true_peak::true_peak_check_float_c(
            tp,
            data.len() / channels as usize,
            data.as_ptr(),
            peaks.as_mut_ptr(),
        );
        true_peak::true_peak_destroy_c(tp);
    }
}

#[cfg(feature = "internal-tests")]
fn true_peak_f32(rate: u32, channels: u32, data: &[f32], peaks: &mut [f64]) {
    use ebur128::true_peak;

    let mut tp = true_peak::TruePeak::new(rate, channels).unwrap();
    tp.check_true_peak(data, peaks);
}

#[cfg(feature = "c-tests")]
fn true_peak_f64_c(rate: u32, channels: u32, data: &[f64], peaks: &mut [f64]) {
    use ebur128::true_peak;

    unsafe {
        let tp = true_peak::true_peak_create_c(rate, channels);
        true_peak::true_peak_check_double_c(
            tp,
            data.len() / channels as usize,
            data.as_ptr(),
            peaks.as_mut_ptr(),
        );
        true_peak::true_peak_destroy_c(tp);
    }
}

#[cfg(feature = "internal-tests")]
fn true_peak_f64(rate: u32, channels: u32, data: &[f64], peaks: &mut [f64]) {
    use ebur128::true_peak;

    let mut tp = true_peak::TruePeak::new(rate, channels).unwrap();
    tp.check_true_peak(data, peaks);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    #[cfg(feature = "internal-tests")]
    {
        let mut data = vec![0i16; 19200 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator) * std::i16::MAX as f32;
            out[0] = val as i16;
            out[1] = val as i16;
            accumulator += step;
        }

        let mut peaks = vec![0.0f64; 2];

        let mut group = c.benchmark_group("true_peak: 48kHz 2ch i16");

        #[cfg(feature = "c-tests")]
        {
            group.bench_function("C", |b| {
                b.iter(|| {
                    true_peak_i16_c(
                        black_box(48_000),
                        black_box(2),
                        black_box(&data),
                        black_box(&mut peaks),
                    )
                })
            });
        }

        group.bench_function("Rust", |b| {
            b.iter(|| {
                true_peak_i16(
                    black_box(48_000),
                    black_box(2),
                    black_box(&data),
                    black_box(&mut peaks),
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

        let mut group = c.benchmark_group("true_peak: 48kHz 2ch i32");

        #[cfg(feature = "c-tests")]
        {
            group.bench_function("C", |b| {
                b.iter(|| {
                    true_peak_i32_c(
                        black_box(48_000),
                        black_box(2),
                        black_box(&data),
                        black_box(&mut peaks),
                    )
                })
            });
        }

        group.bench_function("Rust", |b| {
            b.iter(|| {
                true_peak_i32(
                    black_box(48_000),
                    black_box(2),
                    black_box(&data),
                    black_box(&mut peaks),
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

        let mut group = c.benchmark_group("true_peak: 48kHz 2ch f32");

        #[cfg(feature = "c-tests")]
        {
            group.bench_function("C", |b| {
                b.iter(|| {
                    true_peak_f32_c(
                        black_box(48_000),
                        black_box(2),
                        black_box(&data),
                        black_box(&mut peaks),
                    )
                })
            });
        }

        group.bench_function("Rust", |b| {
            b.iter(|| {
                true_peak_f32(
                    black_box(48_000),
                    black_box(2),
                    black_box(&data),
                    black_box(&mut peaks),
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

        let mut group = c.benchmark_group("true_peak: 48kHz 2ch f64");

        #[cfg(feature = "c-tests")]
        {
            group.bench_function("C", |b| {
                b.iter(|| {
                    true_peak_f64_c(
                        black_box(48_000),
                        black_box(2),
                        black_box(&data),
                        black_box(&mut peaks),
                    )
                })
            });
        }

        group.bench_function("Rust", |b| {
            b.iter(|| {
                true_peak_f64(
                    black_box(48_000),
                    black_box(2),
                    black_box(&data),
                    black_box(&mut peaks),
                )
            })
        });

        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
