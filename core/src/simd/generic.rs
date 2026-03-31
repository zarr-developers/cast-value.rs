//! Generic SIMD kernels using [`pulp`]'s [`WithSimd`] trait.
//!
//! These kernels use portable SIMD abstractions that compile to efficient code
//! across all platforms. [`pulp::Arch::new().dispatch(kernel)`][pulp::Arch::dispatch]
//! selects the best available SIMD level at runtime, but even when it reports
//! "Scalar" (1 lane), the compiler can still auto-vectorize the structured code.
//!
//! On x86_64, pulp supports AVX2 (via V3) and AVX-512 (via V4) explicitly. For
//! CPUs without these features, it uses [`pulp::Scalar`], but the WithSimd trait
//! abstraction enables aggressive compiler optimizations including auto-vectorization.
//!
//! **Performance:**
//! Benchmarking on x86_64 with AVX1 (no AVX2) showed this generic path achieves
//! ~2.3× speedup over naive scalar code and is 7-10% FASTER than hand-written SSE2
//! intrinsics. The compiler optimizes the generic WithSimd code better than manual
//! SSE2 with its expensive scalar rounding round-trips.
//!
//! For other architectures:
//! - WASM32 without SIMD128: compiler may auto-vectorize
//! - Other architectures: compiler may auto-vectorize when beneficial
//!
//! Rounding is done per-lane because the [`pulp::Simd`] trait intentionally omits
//! `floor`/`ceil`/`trunc` (no universal SIMD encoding). This doesn't prevent
//! vectorization of other operations like NaN detection, clamping, and load/store.

use crate::RoundingMode;
use pulp::{Simd, WithSimd};

// ---------------------------------------------------------------------------
// Per-lane scalar rounding helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn scalar_round_f64(val: f64, rounding: RoundingMode) -> f64 {
    match rounding {
        RoundingMode::NearestEven => val.round_ties_even(),
        RoundingMode::TowardsZero => val.trunc(),
        RoundingMode::TowardsPositive => val.ceil(),
        RoundingMode::TowardsNegative => val.floor(),
        RoundingMode::NearestAway => unreachable!(),
    }
}

#[inline(always)]
fn scalar_round_f32(val: f32, rounding: RoundingMode) -> f32 {
    match rounding {
        RoundingMode::NearestEven => val.round_ties_even(),
        RoundingMode::TowardsZero => val.trunc(),
        RoundingMode::TowardsPositive => val.ceil(),
        RoundingMode::TowardsNegative => val.floor(),
        RoundingMode::NearestAway => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// f64 → u8 kernel
// ---------------------------------------------------------------------------

struct F64ToU8ClampKernel<'a> {
    src: &'a [f64],
    dst: &'a mut [u8],
    rounding: RoundingMode,
}

impl<'a> WithSimd for F64ToU8ClampKernel<'a> {
    type Output = Result<(), crate::CastError>;

    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { src, dst, rounding } = self;
        let n = src.len();
        let lanes = S::F64_LANES;
        let lo = simd.splat_f64s(0.0);
        let hi = simd.splat_f64s(255.0);
        // Stack buffer sized for the widest possible f64 SIMD register (AVX-512: 8 lanes).
        let mut tmp = [0.0f64; 8];

        let mut i = 0;
        while i < n {
            // Load up to `lanes` elements; tail is zero-padded (0.0 is never NaN).
            let chunk = &src[i..n.min(i + lanes)];
            let x = simd.partial_load_f64s(chunk);

            // NaN detection: IEEE 754 defines NaN != NaN, so equal_f64s(NaN, NaN)
            // returns all-zeros. not_m64s inverts this: NaN lanes become all-ones.
            // first_true_m64s returns S::F64_LANES when no lane is true.
            let nan_mask = simd.not_m64s(simd.equal_f64s(x, x));
            if simd.first_true_m64s(nan_mask) < lanes {
                for &val in chunk {
                    if val.is_nan() {
                        return Err(crate::CastError::NanOrInf { value: val });
                    }
                }
            }

            // Round per lane (no generic SIMD round op exists in pulp::Simd).
            simd.partial_store_f64s(&mut tmp[..chunk.len()], x);
            for val in tmp.iter_mut().take(chunk.len()) {
                *val = scalar_round_f64(*val, rounding);
            }
            let rounded = simd.partial_load_f64s(&tmp[..chunk.len()]);

            // Clamp and write output.
            let clamped = simd.min_f64s(simd.max_f64s(rounded, lo), hi);
            simd.partial_store_f64s(&mut tmp[..chunk.len()], clamped);
            for (j, &v) in tmp[..chunk.len()].iter().enumerate() {
                dst[i + j] = v as u8;
            }

            i += chunk.len();
        }
        Ok(())
    }
}

/// Convert f64 slice to u8 slice using the best available generic SIMD.
///
/// Dispatches via [`pulp::Arch::new().dispatch`][pulp::Arch::dispatch] to the
/// best instruction set available at runtime.
pub(super) fn f64_to_u8_clamp(
    src: &[f64],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    pulp::Arch::new()
        .dispatch(F64ToU8ClampKernel { src, dst, rounding })
        .map(|()| true)
}

// ---------------------------------------------------------------------------
// f64 → i32 kernel
// ---------------------------------------------------------------------------

struct F64ToI32ClampKernel<'a> {
    src: &'a [f64],
    dst: &'a mut [i32],
    rounding: RoundingMode,
}

impl<'a> WithSimd for F64ToI32ClampKernel<'a> {
    type Output = Result<(), crate::CastError>;

    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { src, dst, rounding } = self;
        let n = src.len();
        let lanes = S::F64_LANES;
        let lo = simd.splat_f64s(i32::MIN as f64);
        let hi = simd.splat_f64s(i32::MAX as f64);
        let mut tmp = [0.0f64; 8];

        let mut i = 0;
        while i < n {
            let chunk = &src[i..n.min(i + lanes)];
            let x = simd.partial_load_f64s(chunk);

            let nan_mask = simd.not_m64s(simd.equal_f64s(x, x));
            if simd.first_true_m64s(nan_mask) < lanes {
                for &val in chunk {
                    if val.is_nan() {
                        return Err(crate::CastError::NanOrInf { value: val });
                    }
                }
            }

            simd.partial_store_f64s(&mut tmp[..chunk.len()], x);
            for val in tmp.iter_mut().take(chunk.len()) {
                *val = scalar_round_f64(*val, rounding);
            }
            let rounded = simd.partial_load_f64s(&tmp[..chunk.len()]);

            let clamped = simd.min_f64s(simd.max_f64s(rounded, lo), hi);
            simd.partial_store_f64s(&mut tmp[..chunk.len()], clamped);
            for (j, &v) in tmp[..chunk.len()].iter().enumerate() {
                dst[i + j] = v as i32;
            }

            i += chunk.len();
        }
        Ok(())
    }
}

/// Convert f64 slice to i32 slice using the best available generic SIMD.
pub(super) fn f64_to_i32_clamp(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    pulp::Arch::new()
        .dispatch(F64ToI32ClampKernel { src, dst, rounding })
        .map(|()| true)
}

// ---------------------------------------------------------------------------
// f32 → u8 kernel
// ---------------------------------------------------------------------------

struct F32ToU8ClampKernel<'a> {
    src: &'a [f32],
    dst: &'a mut [u8],
    rounding: RoundingMode,
}

impl<'a> WithSimd for F32ToU8ClampKernel<'a> {
    type Output = Result<(), crate::CastError>;

    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { src, dst, rounding } = self;
        let n = src.len();
        let lanes = S::F32_LANES;
        let lo = simd.splat_f32s(0.0);
        let hi = simd.splat_f32s(255.0);
        // Stack buffer sized for the widest possible f32 SIMD register (AVX-512: 16 lanes).
        let mut tmp = [0.0f32; 16];

        let mut i = 0;
        while i < n {
            let chunk = &src[i..n.min(i + lanes)];
            let x = simd.partial_load_f32s(chunk);

            let nan_mask = simd.not_m32s(simd.equal_f32s(x, x));
            if simd.first_true_m32s(nan_mask) < lanes {
                for &val in chunk {
                    if val.is_nan() {
                        return Err(crate::CastError::NanOrInf { value: val as f64 });
                    }
                }
            }

            simd.partial_store_f32s(&mut tmp[..chunk.len()], x);
            for val in tmp.iter_mut().take(chunk.len()) {
                *val = scalar_round_f32(*val, rounding);
            }
            let rounded = simd.partial_load_f32s(&tmp[..chunk.len()]);

            let clamped = simd.min_f32s(simd.max_f32s(rounded, lo), hi);
            simd.partial_store_f32s(&mut tmp[..chunk.len()], clamped);
            for (j, &v) in tmp[..chunk.len()].iter().enumerate() {
                dst[i + j] = v as u8;
            }

            i += chunk.len();
        }
        Ok(())
    }
}

/// Convert f32 slice to u8 slice using the best available generic SIMD.
pub(super) fn f32_to_u8_clamp(
    src: &[f32],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    pulp::Arch::new()
        .dispatch(F32ToU8ClampKernel { src, dst, rounding })
        .map(|()| true)
}
