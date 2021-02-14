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

    group.bench_function("Rust", |b| {
        b.iter(|| {
            let interp = interp::Interp2F::<[f32; 4]>::new();
            drop(black_box(interp));
        })
    });

    group.finish();

    let mut group = c.benchmark_group("interp process: 49 taps 2 factors 2ch");

    {
        let mut interp = interp::Interp2F::new();
        let (_, data, _) = unsafe { data.align_to::<[f32; 2]>() };
        let (_, data_out, _) = unsafe { data_out.align_to_mut::<[f32; 2]>() };
        group.bench_function("Rust", |b| {
            b.iter(|| {
                for (input_frame, output_frames) in
                    Iterator::zip(data.iter(), data_out.chunks_exact_mut(2))
                {
                    output_frames.copy_from_slice(&interp.interpolate(*input_frame));
                }
            })
        });
    }

    group.finish();

    let mut group = c.benchmark_group("interp process: 49 taps 4 factors 2ch");
    let mut data_out = vec![0.0f32; 48_000 * 5 * 2 * 4];

    {
        let mut interp = interp::Interp4F::new();
        let (_, data, _) = unsafe { data.align_to::<[f32; 2]>() };
        let (_, data_out, _) = unsafe { data_out.align_to_mut::<[f32; 2]>() };
        group.bench_function("Rust", |b| {
            b.iter(|| {
                for (input_frame, output_frames) in
                    Iterator::zip(data.iter(), data_out.chunks_exact_mut(4))
                {
                    output_frames.copy_from_slice(&interp.interpolate(*input_frame));
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
