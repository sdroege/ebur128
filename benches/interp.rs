use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ebur128::interp;

pub fn criterion_benchmark(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("interp create: 49 taps 2 factors 2ch");

    #[cfg(feature = "c-tests")]
    unsafe {
        group.bench_function("C", |b| {
            b.iter(|| {
                let interp = interp::interp_create_c(
                    black_box(49),
                    black_box(data_out.len() / data.len()) as u32,
                    black_box(2),
                );
                interp::interp_destroy_c(interp);
            })
        });
    }
    group.bench_function("Rust", |b| {
        b.iter(|| {
            let interp = interp::Interp::new(
                black_box(49),
                black_box(data_out.len() / data.len()),
                black_box(2),
            );
            drop(black_box(interp));
        })
    });

    group.finish();

    let mut group = c.benchmark_group("interp process: 49 taps 2 factors 2ch");

    #[cfg(feature = "c-tests")]
    unsafe {
        let interp = interp::interp_create_c(
            black_box(49),
            black_box(data_out.len() / data.len()) as u32,
            black_box(2),
        );
        group.bench_function("C", |b| {
            b.iter(|| {
                interp::interp_process_c(interp, 48_000 * 5, data.as_ptr(), data_out.as_mut_ptr());
            })
        });
        interp::interp_destroy_c(interp);
    }
    {
        let mut interp = interp::Interp::new(
            black_box(49),
            black_box(data_out.len() / data.len()),
            black_box(2),
        );
        group.bench_function("Rust", |b| {
            b.iter(|| {
                interp.process(&data, &mut data_out);
            })
        });
    }

    group.finish();

    let mut group = c.benchmark_group("interp process: 49 taps 4 factors 2ch");
    let mut data_out = vec![0.0f32; 48_000 * 5 * 2 * 4];

    #[cfg(feature = "c-tests")]
    unsafe {
        let interp = interp::interp_create_c(
            black_box(49),
            black_box(data_out.len() / data.len()) as u32,
            black_box(2),
        );
        group.bench_function("C", |b| {
            b.iter(|| {
                interp::interp_process_c(interp, 48_000 * 5, data.as_ptr(), data_out.as_mut_ptr());
            })
        });
        interp::interp_destroy_c(interp);
    }
    {
        let mut interp = interp::Interp::new(
            black_box(49),
            black_box(data_out.len() / data.len()),
            black_box(2),
        );
        group.bench_function("Rust", |b| {
            b.iter(|| {
                interp.process(&data, &mut data_out);
            })
        });
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
