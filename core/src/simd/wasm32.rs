//! WebAssembly SIMD128 kernels.
//!
//! Uses 128-bit SIMD operations available in wasm32-simd128:
//! - 2×f64 or 4×f32 per vector
//! - Available in modern browsers (Chrome 91+, Firefox 89+, Safari 16.4+)
//!
//! All functions are `pub(super) unsafe` — callers must ensure SIMD128 feature
//! is available (enforced by feature detection in the parent module).

use crate::RoundingMode;
use core::arch::wasm32::*;

// ---------------------------------------------------------------------------
// Rounding helpers
//
// WASM SIMD provides dedicated rounding instructions for each IEEE 754 mode.
// ---------------------------------------------------------------------------

/// Round a `v128` f64x2 vector according to the given rounding mode.
#[inline(always)]
unsafe fn round_f64x2(v: v128, mode: RoundingMode) -> v128 {
    match mode {
        RoundingMode::NearestEven => f64x2_nearest(v),
        RoundingMode::TowardsZero => f64x2_trunc(v),
        RoundingMode::TowardsPositive => f64x2_ceil(v),
        RoundingMode::TowardsNegative => f64x2_floor(v),
        RoundingMode::NearestAway => unreachable!(),
    }
}

/// Round a `v128` f32x4 vector according to the given rounding mode.
#[inline(always)]
unsafe fn round_f32x4(v: v128, mode: RoundingMode) -> v128 {
    match mode {
        RoundingMode::NearestEven => f32x4_nearest(v),
        RoundingMode::TowardsZero => f32x4_trunc(v),
        RoundingMode::TowardsPositive => f32x4_ceil(v),
        RoundingMode::TowardsNegative => f32x4_floor(v),
        RoundingMode::NearestAway => unreachable!(),
    }
}

/// Return `true` if any lane in a f64x2 vector is NaN.
#[inline(always)]
unsafe fn any_nan_f64x2(v: v128) -> bool {
    // f64x2_eq returns all-ones for non-NaN, all-zeros for NaN
    let eq = f64x2_eq(v, v);
    // Extract lanes and check if any is zero (NaN)
    u64x2_extract_lane::<0>(eq) == 0 || u64x2_extract_lane::<1>(eq) == 0
}

/// Return `true` if any lane in a f32x4 vector is NaN.
#[inline(always)]
unsafe fn any_nan_f32x4(v: v128) -> bool {
    let eq = f32x4_eq(v, v);
    u32x4_extract_lane::<0>(eq) == 0
        || u32x4_extract_lane::<1>(eq) == 0
        || u32x4_extract_lane::<2>(eq) == 0
        || u32x4_extract_lane::<3>(eq) == 0
}

// ---------------------------------------------------------------------------
// f64 → u8 with clamping
// ---------------------------------------------------------------------------

/// Convert `f64` slice to `u8` slice with clamping, using WASM SIMD128.
///
/// # Safety
///
/// Caller must ensure WASM SIMD128 feature is available.
pub(super) unsafe fn f64_to_u8_clamp(
    src: &[f64],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {


    let mut i = 0;
    let len = src.len();

    // Process 2 f64 at a time using f64x2
    while i + 2 <= len {
        let v = v128_load(src.as_ptr().add(i) as *const v128);

        if any_nan_f64x2(v) {
            // Find which value is NaN for error reporting
            for &val in &src[i..i + 2] {
                if val.is_nan() {
                    return Err(crate::CastError::NanOrInf { value: val });
                }
            }
        }

        // Round according to mode
        let rounded = round_f64x2(v, rounding);

        // Clamp to [0.0, 255.0]
        let zero = f64x2_splat(0.0);
        let max = f64x2_splat(255.0);
        let clamped = f64x2_pmin(f64x2_pmax(rounded, zero), max);

        // Convert to i32 (saturating)
        let i32_vals = i32x4_trunc_sat_f64x2_zero(clamped);

        // Extract and store as u8
        dst[i] = i32x4_extract_lane::<0>(i32_vals) as u8;
        dst[i + 1] = i32x4_extract_lane::<1>(i32_vals) as u8;

        i += 2;
    }

    // Scalar tail
    for idx in i..len {
        let val = src[idx];
        if val.is_nan() {
            return Err(crate::CastError::NanOrInf { value: val });
        }
        let rounded = match rounding {
            RoundingMode::NearestEven => val.round(),
            RoundingMode::TowardsZero => val.trunc(),
            RoundingMode::TowardsPositive => val.ceil(),
            RoundingMode::TowardsNegative => val.floor(),
            RoundingMode::NearestAway => unreachable!(),
        };
        dst[idx] = rounded.clamp(0.0, 255.0) as u8;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// f64 → i32 with clamping
// ---------------------------------------------------------------------------

/// Convert `f64` slice to `i32` slice with clamping, using WASM SIMD128.
///
/// # Safety
///
/// Caller must ensure WASM SIMD128 feature is available.
pub(super) unsafe fn f64_to_i32_clamp(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {


    let mut i = 0;
    let len = src.len();

    // Process 2 f64 at a time
    while i + 2 <= len {
        let v = v128_load(src.as_ptr().add(i) as *const v128);

        if any_nan_f64x2(v) {
            // Find which value is NaN for error reporting
            for &val in &src[i..i + 2] {
                if val.is_nan() {
                    return Err(crate::CastError::NanOrInf { value: val });
                }
            }
        }

        // Round according to mode
        let rounded = round_f64x2(v, rounding);

        // Clamp to [i32::MIN, i32::MAX]
        let min = f64x2_splat(i32::MIN as f64);
        let max = f64x2_splat(i32::MAX as f64);
        let clamped = f64x2_pmin(f64x2_pmax(rounded, min), max);

        // Convert to i32 (saturating) - result is in lower 2 lanes of i32x4
        let i32_vals = i32x4_trunc_sat_f64x2_zero(clamped);

        // Extract and store
        dst[i] = i32x4_extract_lane::<0>(i32_vals);
        dst[i + 1] = i32x4_extract_lane::<1>(i32_vals);

        i += 2;
    }

    // Scalar tail
    for idx in i..len {
        let val = src[idx];
        if val.is_nan() {
            return Err(crate::CastError::NanOrInf { value: val });
        }
        let rounded = match rounding {
            RoundingMode::NearestEven => val.round(),
            RoundingMode::TowardsZero => val.trunc(),
            RoundingMode::TowardsPositive => val.ceil(),
            RoundingMode::TowardsNegative => val.floor(),
            RoundingMode::NearestAway => unreachable!(),
        };
        dst[idx] = rounded.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// f32 → u8 with clamping
// ---------------------------------------------------------------------------

/// Convert `f32` slice to `u8` slice with clamping, using WASM SIMD128.
///
/// # Safety
///
/// Caller must ensure WASM SIMD128 feature is available.
pub(super) unsafe fn f32_to_u8_clamp(
    src: &[f32],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {


    let mut i = 0;
    let len = src.len();

    // Process 4 f32 at a time using f32x4
    while i + 4 <= len {
        let v = v128_load(src.as_ptr().add(i) as *const v128);

        if any_nan_f32x4(v) {
            // Find which value is NaN for error reporting
            for &val in &src[i..i + 4] {
                if val.is_nan() {
                    return Err(crate::CastError::NanOrInf { value: val as f64 });
                }
            }
        }

        // Round according to mode
        let rounded = round_f32x4(v, rounding);

        // Clamp to [0.0, 255.0]
        let zero = f32x4_splat(0.0);
        let max = f32x4_splat(255.0);
        let clamped = f32x4_pmin(f32x4_pmax(rounded, zero), max);

        // Convert to i32 (saturating)
        let i32_vals = i32x4_trunc_sat_f32x4(clamped);

        // Extract and store as u8
        dst[i] = i32x4_extract_lane::<0>(i32_vals) as u8;
        dst[i + 1] = i32x4_extract_lane::<1>(i32_vals) as u8;
        dst[i + 2] = i32x4_extract_lane::<2>(i32_vals) as u8;
        dst[i + 3] = i32x4_extract_lane::<3>(i32_vals) as u8;

        i += 4;
    }

    // Scalar tail
    for idx in i..len {
        let val = src[idx];
        if val.is_nan() {
            return Err(crate::CastError::NanOrInf { value: val as f64 });
        }
        let rounded = match rounding {
            RoundingMode::NearestEven => val.round(),
            RoundingMode::TowardsZero => val.trunc(),
            RoundingMode::TowardsPositive => val.ceil(),
            RoundingMode::TowardsNegative => val.floor(),
            RoundingMode::NearestAway => unreachable!(),
        };
        dst[idx] = rounded.clamp(0.0, 255.0) as u8;
    }

    Ok(())
}
