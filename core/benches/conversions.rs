//! Benchmarks for slice conversion functions.
//!
//! Covers the main conversion paths with realistic array sizes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zarr_cast_value::{
    convert_slice_float_to_float, convert_slice_float_to_int, convert_slice_int_to_float,
    convert_slice_int_to_int, FloatToFloatConfig, FloatToIntConfig, IntToFloatConfig,
    IntToIntConfig, MapEntry, OutOfRangeMode, RoundingMode,
};

const SIZES: &[usize] = &[1_024, 64_000, 1_000_000];

// ---------------------------------------------------------------------------
// Data generators
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

// ---------------------------------------------------------------------------
// f64 -> u8
// ---------------------------------------------------------------------------

fn bench_f64_to_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_to_u8");
    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

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
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// i32 -> u8
// ---------------------------------------------------------------------------

fn bench_i32_to_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("i32_to_u8");
    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

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
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// f64 -> f32
// ---------------------------------------------------------------------------

fn bench_f64_to_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_to_f32");
    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));

        let src = f64_for_narrowing(n);
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
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// i64 -> f32
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
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// f32 -> u8
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

criterion_group!(
    benches,
    bench_f64_to_u8,
    bench_i32_to_u8,
    bench_f64_to_f32,
    bench_i64_to_f32,
    bench_f32_to_u8,
);
criterion_main!(benches);
