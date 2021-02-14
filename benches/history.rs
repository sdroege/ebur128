use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ebur128::history;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut energies = vec![0.0; 1_000_000];
    for (i, e) in energies.iter_mut().enumerate() {
        *e = f64::powf(10.0, ((i % 1000) as f64 / 10.0 - 69.95 + 0.691) / 10.0);
    }

    // Initialize histogram state first
    drop(black_box(history::History::new(true, 0)));

    for (histogram, name) in &[(true, "Histogram"), (false, "Queue")] {
        let mut group = c.benchmark_group(format!("history add: 1M {}", name));
        group.bench_function("Rust", |b| {
            b.iter(|| {
                let mut hist = history::History::new(*histogram, 100_000);
                for e in black_box(&energies) {
                    hist.add(*e);
                }
            })
        });
        group.finish();

        let mut group = c.benchmark_group(format!("history gated loudness: 1M {}", name));
        {
            let mut hist = history::History::new(*histogram, 100_000);

            for e in black_box(&energies) {
                hist.add(*e);
            }

            group.bench_function("Rust", |b| {
                b.iter(|| {
                    black_box(hist.gated_loudness());
                })
            });
        }
        group.finish();

        let mut group = c.benchmark_group(format!("history relative threshold: 1M {}", name));
        {
            let mut hist = history::History::new(*histogram, 100_000);

            for e in black_box(&energies) {
                hist.add(*e);
            }

            group.bench_function("Rust", |b| {
                b.iter(|| {
                    black_box(hist.relative_threshold());
                })
            });
        }
        group.finish();

        let mut group = c.benchmark_group(format!("history loudness range: 1M {}", name));
        {
            let mut hist = history::History::new(*histogram, 100_000);

            for e in black_box(&energies) {
                hist.add(*e);
            }

            group.bench_function("Rust", |b| {
                b.iter(|| {
                    black_box(hist.loudness_range());
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
