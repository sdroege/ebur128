use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    #[cfg(feature = "internal-tests")]
    {
        let mut energies = vec![0.0; 1_000_000];
        for (i, e) in energies.iter_mut().enumerate() {
            *e = f64::powf(10.0, ((i % 1000) as f64 / 10.0 - 69.95 + 0.691) / 10.0);
        }

        let mut group = c.benchmark_group("history add: 1M Histogram");
        group.bench_function("C", |b| {
            b.iter(|| {
                use ebur128::history;

                unsafe {
                    let hist = history::history_create_c(1, 100_000);

                    for e in black_box(&energies) {
                        history::history_add_c(hist, *e);
                    }

                    history::history_destroy_c(hist);
                }
            })
        });
        group.bench_function("Rust", |b| {
            b.iter(|| {
                use ebur128::history;

                let mut hist = history::History::new(true, 100_000);
                for e in black_box(&energies) {
                    hist.add(*e);
                }
            })
        });
        group.finish();

        let mut group = c.benchmark_group("history add: 1M Queue");
        group.bench_function("C", |b| {
            b.iter(|| {
                use ebur128::history;

                unsafe {
                    let hist = history::history_create_c(0, 100_000);

                    for e in black_box(&energies) {
                        history::history_add_c(hist, *e);
                    }

                    history::history_destroy_c(hist);
                }
            })
        });
        group.bench_function("Rust", |b| {
            b.iter(|| {
                use ebur128::history;

                let mut hist = history::History::new(false, 100_000);
                for e in black_box(&energies) {
                    hist.add(*e);
                }
            })
        });
        group.finish();

        let mut group = c.benchmark_group("history gated loudness: 1M Histogram");
        unsafe {
            use ebur128::history;

            let hist = history::history_create_c(1, 100_000);

            for e in black_box(&energies) {
                history::history_add_c(hist, *e);
            }

            group.bench_function("C", |b| {
                b.iter(|| {
                    black_box(history::history_gated_loudness_c(hist));
                })
            });

            history::history_destroy_c(hist);
        }
        {
            use ebur128::history;
            let mut hist = history::History::new(true, 100_000);

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

        let mut group = c.benchmark_group("history gated loudness: 1M Queue");
        unsafe {
            use ebur128::history;

            let hist = history::history_create_c(0, 100_000);

            for e in black_box(&energies) {
                history::history_add_c(hist, *e);
            }

            group.bench_function("C", |b| {
                b.iter(|| {
                    black_box(history::history_gated_loudness_c(hist));
                })
            });

            history::history_destroy_c(hist);
        }
        {
            use ebur128::history;
            let mut hist = history::History::new(false, 100_000);

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

        let mut group = c.benchmark_group("history relative threshold: 1M Histogram");
        unsafe {
            use ebur128::history;

            let hist = history::history_create_c(1, 100_000);

            for e in black_box(&energies) {
                history::history_add_c(hist, *e);
            }

            group.bench_function("C", |b| {
                b.iter(|| {
                    black_box(history::history_relative_threshold_c(hist));
                })
            });

            history::history_destroy_c(hist);
        }
        {
            use ebur128::history;
            let mut hist = history::History::new(true, 100_000);

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

        let mut group = c.benchmark_group("history relative threshold: 1M Queue");
        unsafe {
            use ebur128::history;

            let hist = history::history_create_c(0, 100_000);

            for e in black_box(&energies) {
                history::history_add_c(hist, *e);
            }

            group.bench_function("C", |b| {
                b.iter(|| {
                    black_box(history::history_relative_threshold_c(hist));
                })
            });

            history::history_destroy_c(hist);
        }
        {
            use ebur128::history;
            let mut hist = history::History::new(false, 100_000);

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

        let mut group = c.benchmark_group("history loudness range: 1M Histogram");
        unsafe {
            use ebur128::history;

            let hist = history::history_create_c(1, 100_000);

            for e in black_box(&energies) {
                history::history_add_c(hist, *e);
            }

            group.bench_function("C", |b| {
                b.iter(|| {
                    black_box(history::history_loudness_range_c(hist));
                })
            });

            history::history_destroy_c(hist);
        }
        {
            use ebur128::history;
            let mut hist = history::History::new(true, 100_000);

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

        let mut group = c.benchmark_group("history loudness range: 1M Queue");
        unsafe {
            use ebur128::history;

            let hist = history::history_create_c(0, 100_000);

            for e in black_box(&energies) {
                history::history_add_c(hist, *e);
            }

            group.bench_function("C", |b| {
                b.iter(|| {
                    black_box(history::history_loudness_range_c(hist));
                })
            });

            history::history_destroy_c(hist);
        }
        {
            use ebur128::history;
            let mut hist = history::History::new(false, 100_000);

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
