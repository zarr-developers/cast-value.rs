//! Benchmarks for slice conversion functions.
//!
//! Covers the four conversion paths (float->int, int->int, float->float,
//! int->float) with varying configurations (clamp, wrap, scalar map) and
//! array sizes.
//!
//! Benchmark IDs include the full type path (e.g. "f64_to_u8/clamp") so
//! that CodSpeed reports are unambiguous without the group name.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zarr_cast_value::{
    convert_slice_float_to_float, convert_slice_float_to_int, convert_slice_int_to_float,
    convert_slice_int_to_int, FloatToFloatConfig, FloatToIntConfig, IntToFloatConfig,
    IntToIntConfig, MapEntry, OutOfRangeMode, RoundingMode,
};

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

fn f64_in_range(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i % 256) as f64 + 0.3).collect()
}

fn f64_mixed(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| match i % 20 {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => -10.0,
            3 => 300.0,
            _ => (i % 256) as f64 + 0.7,
        })
        .collect()
}

fn i32_in_range(n: usize) -> Vec<i32> {
    (0..n).map(|i| (i % 256) as i32).collect()
}

fn i32_mixed(n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| match i % 10 {
            0 => -100,
            1 => 500,
            _ => (i % 256) as i32,
        })
        .collect()
}

fn f64_for_narrowing(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.123_456_789).collect()
}

fn i64_for_float(n: usize) -> Vec<i64> {
    (0..n)
        .map(|i| {
            if i % 4 == 0 {
                (1_i64 << 24) + i as i64
            } else {
                i as i64
            }
        })
        .collect()
}

const SIZES: &[usize] = &[1_024, 64_000, 1_000_000];

// ---------------------------------------------------------------------------
// float64 -> uint8
// ---------------------------------------------------------------------------

fn bench_f64_to_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversions");

    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        let src = f64_in_range(n);
        let config = FloatToIntConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        group.bench_with_input(BenchmarkId::new("f64_to_u8/clamp", n), &n, |b, &n| {
            let mut dst = vec![0u8; n];
            b.iter(|| {
                convert_slice_float_to_int(
                    black_box(&src),
                    black_box(&mut dst),
                    black_box(&config),
                )
                .unwrap();
            });
        });

        let src_mixed = f64_mixed(n);
        let config_map = FloatToIntConfig {
            map_entries: vec![
                MapEntry {
                    src: f64::NAN,
                    tgt: 0u8,
                },
                MapEntry {
                    src: f64::INFINITY,
                    tgt: 255u8,
                },
            ],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        group.bench_with_input(
            BenchmarkId::new("f64_to_u8/clamp+scalar_map", n),
            &n,
            |b, &n| {
                let mut dst = vec![0u8; n];
                b.iter(|| {
                    convert_slice_float_to_int(
                        black_box(&src_mixed),
                        black_box(&mut dst),
                        black_box(&config_map),
                    )
                    .unwrap();
                });
            },
        );

        let config_wrap = FloatToIntConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Wrap),
        };
        group.bench_with_input(BenchmarkId::new("f64_to_u8/wrap", n), &n, |b, &n| {
            let mut dst = vec![0u8; n];
            b.iter(|| {
                convert_slice_float_to_int(
                    black_box(&src),
                    black_box(&mut dst),
                    black_box(&config_wrap),
                )
                .unwrap();
            });
        });

        for mode in [
            RoundingMode::TowardsZero,
            RoundingMode::TowardsPositive,
            RoundingMode::TowardsNegative,
        ] {
            let config_round = FloatToIntConfig {
                map_entries: vec![],
                rounding: mode,
                out_of_range: Some(OutOfRangeMode::Clamp),
            };
            let name = format!("f64_to_u8/clamp+{mode:?}");
            group.bench_with_input(BenchmarkId::new(&name, n), &n, |b, &n| {
                let mut dst = vec![0u8; n];
                b.iter(|| {
                    convert_slice_float_to_int(
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(&config_round),
                    )
                    .unwrap();
                });
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// float64 -> int32
// ---------------------------------------------------------------------------

fn bench_f64_to_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversions");

    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        let src = f64_in_range(n);
        let config = FloatToIntConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        group.bench_with_input(BenchmarkId::new("f64_to_i32/clamp", n), &n, |b, &n| {
            let mut dst = vec![0i32; n];
            b.iter(|| {
                convert_slice_float_to_int(
                    black_box(&src),
                    black_box(&mut dst),
                    black_box(&config),
                )
                .unwrap();
            });
        });

        // No out_of_range (error on overflow) — all values in range
        let config_none = FloatToIntConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        group.bench_with_input(BenchmarkId::new("f64_to_i32/no_oor", n), &n, |b, &n| {
            let mut dst = vec![0i32; n];
            b.iter(|| {
                convert_slice_float_to_int(
                    black_box(&src),
                    black_box(&mut dst),
                    black_box(&config_none),
                )
                .unwrap();
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// int32 -> uint8
// ---------------------------------------------------------------------------

fn bench_i32_to_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversions");

    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        let src = i32_in_range(n);
        let config = IntToIntConfig {
            map_entries: vec![],
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        group.bench_with_input(BenchmarkId::new("i32_to_u8/clamp", n), &n, |b, &n| {
            let mut dst = vec![0u8; n];
            b.iter(|| {
                convert_slice_int_to_int(black_box(&src), black_box(&mut dst), black_box(&config))
                    .unwrap();
            });
        });

        let config_wrap = IntToIntConfig {
            map_entries: vec![],
            out_of_range: Some(OutOfRangeMode::Wrap),
        };
        group.bench_with_input(BenchmarkId::new("i32_to_u8/wrap", n), &n, |b, &n| {
            let mut dst = vec![0u8; n];
            b.iter(|| {
                convert_slice_int_to_int(
                    black_box(&src),
                    black_box(&mut dst),
                    black_box(&config_wrap),
                )
                .unwrap();
            });
        });

        let src_mixed = i32_mixed(n);
        let config_map = IntToIntConfig {
            map_entries: vec![MapEntry {
                src: -100i32,
                tgt: 0u8,
            }],
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        group.bench_with_input(
            BenchmarkId::new("i32_to_u8/clamp+scalar_map", n),
            &n,
            |b, &n| {
                let mut dst = vec![0u8; n];
                b.iter(|| {
                    convert_slice_int_to_int(
                        black_box(&src_mixed),
                        black_box(&mut dst),
                        black_box(&config_map),
                    )
                    .unwrap();
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// float64 -> float32
// ---------------------------------------------------------------------------

fn bench_f64_to_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversions");

    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        let src = f64_for_narrowing(n);

        let config = FloatToFloatConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        group.bench_with_input(
            BenchmarkId::new("f64_to_f32/nearest_even", n),
            &n,
            |b, &n| {
                let mut dst = vec![0f32; n];
                b.iter(|| {
                    convert_slice_float_to_float(
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(&config),
                    )
                    .unwrap();
                });
            },
        );

        let config_clamp = FloatToFloatConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        group.bench_with_input(BenchmarkId::new("f64_to_f32/clamp", n), &n, |b, &n| {
            let mut dst = vec![0f32; n];
            b.iter(|| {
                convert_slice_float_to_float(
                    black_box(&src),
                    black_box(&mut dst),
                    black_box(&config_clamp),
                )
                .unwrap();
            });
        });

        // Towards-zero rounding, no out_of_range (scalar fallback path)
        let config_tz = FloatToFloatConfig {
            map_entries: vec![],
            rounding: RoundingMode::TowardsZero,
            out_of_range: None,
        };
        group.bench_with_input(
            BenchmarkId::new("f64_to_f32/towards_zero", n),
            &n,
            |b, &n| {
                let mut dst = vec![0f32; n];
                b.iter(|| {
                    convert_slice_float_to_float(
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(&config_tz),
                    )
                    .unwrap();
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// int64 -> float32
// ---------------------------------------------------------------------------

fn bench_i64_to_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversions");

    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        let src = i64_for_float(n);

        let config = IntToFloatConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
        };
        group.bench_with_input(
            BenchmarkId::new("i64_to_f32/nearest_even", n),
            &n,
            |b, &n| {
                let mut dst = vec![0f32; n];
                b.iter(|| {
                    convert_slice_int_to_float(
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(&config),
                    )
                    .unwrap();
                });
            },
        );

        let config_tz = IntToFloatConfig {
            map_entries: vec![],
            rounding: RoundingMode::TowardsZero,
        };
        group.bench_with_input(
            BenchmarkId::new("i64_to_f32/towards_zero", n),
            &n,
            |b, &n| {
                let mut dst = vec![0f32; n];
                b.iter(|| {
                    convert_slice_int_to_float(
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(&config_tz),
                    )
                    .unwrap();
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// float32 -> uint8
// ---------------------------------------------------------------------------

fn bench_f32_to_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversions");

    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        let src: Vec<f32> = (0..n).map(|i| (i % 256) as f32 + 0.3).collect();
        let config = FloatToIntConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        group.bench_with_input(BenchmarkId::new("f32_to_u8/clamp", n), &n, |b, &n| {
            let mut dst = vec![0u8; n];
            b.iter(|| {
                convert_slice_float_to_int(
                    black_box(&src),
                    black_box(&mut dst),
                    black_box(&config),
                )
                .unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_f64_to_u8,
    bench_f64_to_i32,
    bench_i32_to_u8,
    bench_f64_to_f32,
    bench_i64_to_f32,
    bench_f32_to_u8,
);
criterion_main!(benches);
