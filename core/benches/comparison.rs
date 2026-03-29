//! Quick benchmark comparing SIMD (generic fallback on this CPU) vs pure scalar.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use zarr_cast_value::{FloatToIntConfig, OutOfRangeMode, RoundingMode};

fn bench_simd_vs_scalar(c: &mut Criterion) {
    let sizes = [1024, 64_000, 1_000_000];

    for &n in &sizes {
        let mut group = c.benchmark_group(format!("f64_to_u8_{}", n));
        group.throughput(Throughput::Elements(n as u64));

        let src: Vec<f64> = (0..n).map(|i| (i % 256) as f64 + 0.3).collect();
        let config = FloatToIntConfig {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };

        // Current path (uses generic SIMD on this AMD FX-8350)
        group.bench_function("with_simd", |b| {
            let mut dst = vec![0u8; n];
            b.iter(|| {
                zarr_cast_value::convert_slice_float_to_int(
                    black_box(&src),
                    black_box(&mut dst),
                    black_box(&config),
                )
                .unwrap();
            });
        });

        // Pure scalar path (simulate by using scalar map)
        let config_scalar = FloatToIntConfig {
            map_entries: vec![zarr_cast_value::MapEntry {
                src: -999.0,
                tgt: 0u8,
            }],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };

        group.bench_function("pure_scalar", |b| {
            let mut dst = vec![0u8; n];
            b.iter(|| {
                zarr_cast_value::convert_slice_float_to_int(
                    black_box(&src),
                    black_box(&mut dst),
                    black_box(&config_scalar),
                )
                .unwrap();
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_simd_vs_scalar);
criterion_main!(benches);
