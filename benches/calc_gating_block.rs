use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn calc_gating_block(
    frames_per_block: usize,
    audio_data: &[f64],
    audio_data_index: usize,
    channel_map: &[ebur128::Channel],
) -> f64 {
    ebur128::filter::Filter::calc_gating_block(
        frames_per_block,
        audio_data,
        audio_data_index,
        channel_map,
    )
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut data = vec![0f64; 48_000 * 3 * 2];
    let mut accumulator = 0.0;
    let step = 2.0 * std::f64::consts::PI * 440.0 / 48_000.0;
    for out in data.chunks_exact_mut(2) {
        let val = f64::sin(accumulator);
        out[0] = val;
        out[1] = val;
        accumulator += step;
    }

    let channel_map = [ebur128::Channel::Left; 2];

    let frames_per_block = 144_000;

    let mut group = c.benchmark_group("calc gating block: 48kHz 2ch");

    group.bench_function("Rust", |b| {
        b.iter(|| {
            calc_gating_block(
                black_box(frames_per_block),
                black_box(&data),
                black_box(0),
                black_box(&channel_map),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
