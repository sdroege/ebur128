use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "c-tests")]
fn interp_c(taps: u32, channels: u32, data: &[f32], data_out: &mut [f32]) {
    use ebur128::interp;

    unsafe {
        let interp = interp::interp_create_c(taps, (data_out.len() / data.len()) as u32, channels);
        interp::interp_process_c(interp, 48_000 * 5, data.as_ptr(), data_out.as_mut_ptr());
        interp::interp_destroy_c(interp);
    }
}

#[cfg(feature = "internal-tests")]
fn interp(taps: u32, channels: u32, data: &[f32], data_out: &mut [f32]) {
    use ebur128::interp;

    let mut interp = interp::Interp::new(taps as usize, data_out.len() / data.len(), channels);
    interp.process(data, data_out);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    #[cfg(feature = "internal-tests")]
    {
        let mut data = vec![0f32; 48_000 * 5 * 2];
        let mut accumulator = 0.0;
        let step = 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
        for out in data.chunks_exact_mut(2) {
            let val = f32::sin(accumulator);
            out[0] = val;
            out[1] = val;
            accumulator += step;
        }
        let mut data_out = vec![0.0f32; 48_000 * 5 * 2 * 2];

        let mut group = c.benchmark_group("interp: 49 taps 2 factors 2ch");

        #[cfg(feature = "c-tests")]
        {
            group.bench_function("C", |b| {
                b.iter(|| {
                    interp_c(
                        black_box(49),
                        black_box(2),
                        black_box(&data),
                        black_box(&mut data_out),
                    )
                })
            });
        }
        group.bench_function("Rust", |b| {
            b.iter(|| {
                interp(
                    black_box(49),
                    black_box(2),
                    black_box(&data),
                    black_box(&mut data_out),
                )
            })
        });

        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
