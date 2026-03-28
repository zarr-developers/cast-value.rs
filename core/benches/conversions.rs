//! Benchmarks for slice conversion functions.
//!
//! Covers the four conversion paths (float->int, int->int, float->float,
//! int->float) with varying configurations (clamp, wrap, scalar map) and
//! array sizes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zarr_cast_value::{
    convert_slice_float_to_float, convert_slice_float_to_int, convert_slice_int_to_float,
    convert_slice_int_to_int, FloatToFloatConfig, FloatToIntConfig, IntToFloatConfig,
    IntToIntConfig, MapEntry, OutOfRangeMode, RoundingMode,
};

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

/// Generate f64 values in [0, 255] — all in range for u8, no NaN/Inf.
fn f64_in_range(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i % 256) as f64 + 0.3).collect()
}

/// Generate f64 values with some out-of-range, NaN, and Inf mixed in.
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

/// Generate i32 values in [0, 255] — all in range for u8.
fn i32_in_range(n: usize) -> Vec<i32> {
    (0..n).map(|i| (i % 256) as i32).collect()
}

/// Generate i32 values with some out-of-range values.
fn i32_mixed(n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| match i % 10 {
            0 => -100,
            1 => 500,
            _ => (i % 256) as i32,
        })
        .collect()
}

/// Generate f64 values for float->float narrowing (f64->f32).
fn f64_for_narrowing(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.123_456_789).collect()
}

/// Generate i64 values for int->float conversion (some > 2^24 to test
/// precision loss in f32).
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

// ---------------------------------------------------------------------------
// Array sizes: small (L1 cache), medium (L2), large (L3+)
// ---------------------------------------------------------------------------

const SIZES: &[usize] = &[1_024, 64_000, 1_000_000];

// ---------------------------------------------------------------------------
// float64 -> uint8 benchmarks (the most common real-world case)
// ---------------------------------------------------------------------------

fn bench_f64_to_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_to_u8");

    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        // Clamp, no scalar map — the SIMD fast path candidate
        let src = f64_in_range(n);
        let config = FloatToIntConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        group.bench_with_input(BenchmarkId::new("clamp", n), &n, |b, &n| {
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

        // Clamp with scalar map (NaN->0, Inf->255)
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
        group.bench_with_input(BenchmarkId::new("clamp+scalar_map", n), &n, |b, &n| {
            let mut dst = vec![0u8; n];
            b.iter(|| {
                convert_slice_float_to_int(
                    black_box(&src_mixed),
                    black_box(&mut dst),
                    black_box(&config_map),
                )
                .unwrap();
            });
        });

        // Wrap mode, no scalar map
        let config_wrap = FloatToIntConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Wrap),
        };
        group.bench_with_input(BenchmarkId::new("wrap", n), &n, |b, &n| {
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

        // Different rounding modes (clamp, no scalar map)
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
            let mode_name = format!("clamp+{mode:?}");
            group.bench_with_input(BenchmarkId::new(&mode_name, n), &n, |b, &n| {
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
// float64 -> int32 (wider target — different clamp bounds)
// ---------------------------------------------------------------------------

fn bench_f64_to_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_to_i32");

    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        let src = f64_in_range(n);
        let config = FloatToIntConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        group.bench_with_input(BenchmarkId::new("clamp", n), &n, |b, &n| {
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
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// int32 -> uint8 benchmarks
// ---------------------------------------------------------------------------

fn bench_i32_to_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("i32_to_u8");

    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        // Clamp, no scalar map
        let src = i32_in_range(n);
        let config = IntToIntConfig {
            map_entries: vec![],
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        group.bench_with_input(BenchmarkId::new("clamp", n), &n, |b, &n| {
            let mut dst = vec![0u8; n];
            b.iter(|| {
                convert_slice_int_to_int(black_box(&src), black_box(&mut dst), black_box(&config))
                    .unwrap();
            });
        });

        // Wrap mode
        let config_wrap = IntToIntConfig {
            map_entries: vec![],
            out_of_range: Some(OutOfRangeMode::Wrap),
        };
        group.bench_with_input(BenchmarkId::new("wrap", n), &n, |b, &n| {
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

        // Clamp with scalar map
        let src_mixed = i32_mixed(n);
        let config_map = IntToIntConfig {
            map_entries: vec![MapEntry {
                src: -100i32,
                tgt: 0u8,
            }],
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        group.bench_with_input(BenchmarkId::new("clamp+scalar_map", n), &n, |b, &n| {
            let mut dst = vec![0u8; n];
            b.iter(|| {
                convert_slice_int_to_int(
                    black_box(&src_mixed),
                    black_box(&mut dst),
                    black_box(&config_map),
                )
                .unwrap();
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// float64 -> float32 benchmarks
// ---------------------------------------------------------------------------

fn bench_f64_to_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_to_f32");

    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        let src = f64_for_narrowing(n);

        // Nearest-even, no overflow handling (common default)
        let config = FloatToFloatConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        group.bench_with_input(BenchmarkId::new("nearest_even", n), &n, |b, &n| {
            let mut dst = vec![0f32; n];
            b.iter(|| {
                convert_slice_float_to_float(
                    black_box(&src),
                    black_box(&mut dst),
                    black_box(&config),
                )
                .unwrap();
            });
        });

        // With overflow clamping
        let config_clamp = FloatToFloatConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        group.bench_with_input(BenchmarkId::new("clamp", n), &n, |b, &n| {
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
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// int64 -> float32 benchmarks (precision loss path)
// ---------------------------------------------------------------------------

fn bench_i64_to_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("i64_to_f32");

    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        let src = i64_for_float(n);

        let config = IntToFloatConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
        };
        group.bench_with_input(BenchmarkId::new("nearest_even", n), &n, |b, &n| {
            let mut dst = vec![0f32; n];
            b.iter(|| {
                convert_slice_int_to_float(
                    black_box(&src),
                    black_box(&mut dst),
                    black_box(&config),
                )
                .unwrap();
            });
        });

        // Towards-zero rounding (requires adjustment path)
        let config_tz = IntToFloatConfig {
            map_entries: vec![],
            rounding: RoundingMode::TowardsZero,
        };
        group.bench_with_input(BenchmarkId::new("towards_zero", n), &n, |b, &n| {
            let mut dst = vec![0f32; n];
            b.iter(|| {
                convert_slice_int_to_float(
                    black_box(&src),
                    black_box(&mut dst),
                    black_box(&config_tz),
                )
                .unwrap();
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// float32 -> uint8 (f32 source — tests a different monomorphization)
// ---------------------------------------------------------------------------

fn bench_f32_to_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_to_u8");

    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        let src: Vec<f32> = (0..n).map(|i| (i % 256) as f32 + 0.3).collect();
        let config = FloatToIntConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        group.bench_with_input(BenchmarkId::new("clamp", n), &n, |b, &n| {
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

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

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
