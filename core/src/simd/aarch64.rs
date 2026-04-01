//! AArch64 NEON SIMD kernels.
//!
//! NEON is always available on AArch64 targets, so no runtime detection is
//! required. All functions are `pub(super) unsafe` — callers must ensure they
//! run on an AArch64 target (enforced by the `#[cfg(target_arch = "aarch64")]`
//! gate in the parent module).

use crate::RoundingMode;
use core::arch::aarch64::*;

// ---------------------------------------------------------------------------
// Rounding helpers
//
// NEON provides dedicated rounding instructions for each IEEE 754 mode.
// ---------------------------------------------------------------------------

/// Round a `float64x2_t` vector according to the given rounding mode.
#[inline(always)]
unsafe fn round_f64x2(v: float64x2_t, mode: RoundingMode) -> float64x2_t {
    match mode {
        // vrndnq: round to nearest, ties to even
        RoundingMode::NearestEven => vrndnq_f64(v),
        // vrndq: round towards zero (truncate)
        RoundingMode::TowardsZero => vrndq_f64(v),
        // vrndpq: round towards +inf (ceil)
        RoundingMode::TowardsPositive => vrndpq_f64(v),
        // vrndmq: round towards -inf (floor)
        RoundingMode::TowardsNegative => vrndmq_f64(v),
        RoundingMode::NearestAway => unreachable!(),
    }
}

/// Round a `float32x4_t` vector according to the given rounding mode.
#[inline(always)]
unsafe fn round_f32x4(v: float32x4_t, mode: RoundingMode) -> float32x4_t {
    match mode {
        RoundingMode::NearestEven => vrndnq_f32(v),
        RoundingMode::TowardsZero => vrndq_f32(v),
        RoundingMode::TowardsPositive => vrndpq_f32(v),
        RoundingMode::TowardsNegative => vrndmq_f32(v),
        RoundingMode::NearestAway => unreachable!(),
    }
}

/// Return `true` if any lane in a `float64x2_t` is NaN.
#[inline(always)]
unsafe fn any_nan_f64x2(v: float64x2_t) -> bool {
    // vceqq_f64(v, v) yields all-ones for non-NaN, all-zeros for NaN.
    // If the minimum byte across all lanes is 0, at least one lane was NaN.
    let eq = vceqq_f64(v, v);
    let eq_bytes: uint8x16_t = vreinterpretq_u8_u64(eq);
    vminvq_u8(eq_bytes) == 0
}

/// Return `true` if any lane in a `float32x4_t` is NaN.
#[inline(always)]
unsafe fn any_nan_f32x4(v: float32x4_t) -> bool {
    let eq = vceqq_f32(v, v);
    let not_eq: uint8x16_t = vreinterpretq_u8_u32(vmvnq_u32(eq));
    vmaxvq_u8(not_eq) != 0
}

/// Convert f64 slice to u8 slice with rounding and clamping.
///
/// NEON processes 2 f64 lanes at a time (`float64x2`). We accumulate
/// 8 clamped values (from 4 × `float64x2`) then narrow to u8.
///
/// Pipeline per 8 elements:
/// 1. Load 4 × `float64x2`
/// 2. NaN check (batch)
/// 3. Round each
/// 4. Clamp to [0.0, 255.0]
/// 5. Convert `f64x2` → `i32x2` (`vcvtq_s64_f64` + `vmovn_s64`)
/// 6. Combine 4 × `i32x2` → `i32x4` + `i32x4` → `i16x8`
/// 7. Narrow `i16x8` → `u8x8`
///
/// # Safety
///
/// Caller must ensure this runs on an AArch64 target (NEON is always
/// available on AArch64).
pub(super) unsafe fn f64_to_u8_clamp(
    src: &[f64],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {
    let n = src.len();
    let simd_len = n / 8 * 8;

    let lo = vdupq_n_f64(0.0);
    let hi = vdupq_n_f64(255.0);

    for i in (0..simd_len).step_by(8) {
        let ptr = src.as_ptr().add(i);

        // Load 4 x f64x2 (8 f64 values)
        let v0 = vld1q_f64(ptr);
        let v1 = vld1q_f64(ptr.add(2));
        let v2 = vld1q_f64(ptr.add(4));
        let v3 = vld1q_f64(ptr.add(6));

        // Batch NaN check
        if any_nan_f64x2(v0) || any_nan_f64x2(v1) || any_nan_f64x2(v2) || any_nan_f64x2(v3) {
            for &val in &src[i..std::cmp::min(i + 8, n)] {
                if val.is_nan() {
                    return Err(crate::CastError::NanOrInf { value: val });
                }
            }
        }

        // Round
        let r0 = round_f64x2(v0, rounding);
        let r1 = round_f64x2(v1, rounding);
        let r2 = round_f64x2(v2, rounding);
        let r3 = round_f64x2(v3, rounding);

        // Clamp to [0, 255]
        let c0 = vminq_f64(vmaxq_f64(r0, lo), hi);
        let c1 = vminq_f64(vmaxq_f64(r1, lo), hi);
        let c2 = vminq_f64(vmaxq_f64(r2, lo), hi);
        let c3 = vminq_f64(vmaxq_f64(r3, lo), hi);

        // Convert f64x2 → i32x2 (truncation after rounding is correct)
        let i0 = vmovn_s64(vcvtq_s64_f64(c0));
        let i1 = vmovn_s64(vcvtq_s64_f64(c1));
        let i2 = vmovn_s64(vcvtq_s64_f64(c2));
        let i3 = vmovn_s64(vcvtq_s64_f64(c3));

        // Combine i32x2 pairs → i32x4
        let i32_01 = vcombine_s32(i0, i1);
        let i32_23 = vcombine_s32(i2, i3);

        // Narrow i32x4 → i16x4 (saturating), then combine → i16x8
        let i16_01 = vqmovn_s32(i32_01);
        let i16_23 = vqmovn_s32(i32_23);
        let i16_all = vcombine_s16(i16_01, i16_23);

        // Narrow i16x8 → u8x8 (saturating unsigned)
        let u8_all = vqmovun_s16(i16_all);

        // Store 8 bytes
        vst1_u8(dst.as_mut_ptr().add(i), u8_all);
    }

    // Scalar tail (at most 7 elements)
    scalar_tail_f64_to_u8(src, dst, simd_len, rounding)?;

    Ok(())
}

/// Convert f64 slice to i32 slice with rounding and clamping.
///
/// Processes 2 f64 values at a time via `float64x2`.
///
/// # Safety
///
/// Caller must ensure this runs on an AArch64 target.
pub(super) unsafe fn f64_to_i32_clamp(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {
    let n = src.len();
    let simd_len = n / 2 * 2;

    let lo = vdupq_n_f64(i32::MIN as f64);
    let hi = vdupq_n_f64(i32::MAX as f64);

    for i in (0..simd_len).step_by(2) {
        let v = vld1q_f64(src.as_ptr().add(i));

        if any_nan_f64x2(v) {
            for &val in &src[i..std::cmp::min(i + 2, n)] {
                if val.is_nan() {
                    return Err(crate::CastError::NanOrInf { value: val });
                }
            }
        }

        let r = round_f64x2(v, rounding);
        let c = vminq_f64(vmaxq_f64(r, lo), hi);

        // f64x2 → i64x2 → i32x2 (narrow)
        let i32_val = vmovn_s64(vcvtq_s64_f64(c));

        // Store 2 i32 values
        vst1_s32(dst.as_mut_ptr().add(i), i32_val);
    }

    // Scalar tail (at most 1 element)
    for i in simd_len..n {
        let val = src[i];
        if val.is_nan() {
            return Err(crate::CastError::NanOrInf { value: val });
        }
        let rounded = scalar_round_f64(val, rounding);
        dst[i] = rounded.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
    }

    Ok(())
}

/// Convert f32 slice to u8 slice with rounding and clamping.
///
/// NEON processes 4 f32 lanes at a time (`float32x4`). We process
/// 8 elements per iteration (2 × `float32x4`) to produce a `u8x8`.
///
/// Pipeline per 8 elements:
/// 1. Load 2 × `float32x4`
/// 2. NaN check
/// 3. Round, clamp to [0, 255]
/// 4. Convert `f32x4` → `i32x4`
/// 5. Narrow `i32x4` → `i16x4`, combine → `i16x8`
/// 6. Narrow `i16x8` → `u8x8`
///
/// # Safety
///
/// Caller must ensure this runs on an AArch64 target.
pub(super) unsafe fn f32_to_u8_clamp(
    src: &[f32],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {
    let n = src.len();
    let simd_len = n / 8 * 8;

    let lo = vdupq_n_f32(0.0);
    let hi = vdupq_n_f32(255.0);

    for i in (0..simd_len).step_by(8) {
        let ptr = src.as_ptr().add(i);

        let v0 = vld1q_f32(ptr);
        let v1 = vld1q_f32(ptr.add(4));

        // NaN check
        if any_nan_f32x4(v0) || any_nan_f32x4(v1) {
            for &val in &src[i..std::cmp::min(i + 8, n)] {
                if val.is_nan() {
                    return Err(crate::CastError::NanOrInf { value: val as f64 });
                }
            }
        }

        // Round
        let r0 = round_f32x4(v0, rounding);
        let r1 = round_f32x4(v1, rounding);

        // Clamp
        let c0 = vminq_f32(vmaxq_f32(r0, lo), hi);
        let c1 = vminq_f32(vmaxq_f32(r1, lo), hi);

        // Convert f32x4 → i32x4
        let i0 = vcvtq_s32_f32(c0);
        let i1 = vcvtq_s32_f32(c1);

        // Narrow i32x4 → i16x4 (saturating), combine → i16x8
        let i16_0 = vqmovn_s32(i0);
        let i16_1 = vqmovn_s32(i1);
        let i16_all = vcombine_s16(i16_0, i16_1);

        // Narrow i16x8 → u8x8 (saturating unsigned)
        let u8_all = vqmovun_s16(i16_all);

        // Store 8 bytes
        vst1_u8(dst.as_mut_ptr().add(i), u8_all);
    }

    // Scalar tail (at most 7 elements)
    for i in simd_len..n {
        let val = src[i];
        if val.is_nan() {
            return Err(crate::CastError::NanOrInf { value: val as f64 });
        }
        let rounded = match rounding {
            RoundingMode::NearestEven => val.round_ties_even(),
            RoundingMode::TowardsZero => val.trunc(),
            RoundingMode::TowardsPositive => val.ceil(),
            RoundingMode::TowardsNegative => val.floor(),
            RoundingMode::NearestAway => unreachable!(),
        };
        dst[i] = rounded.clamp(0.0, 255.0) as u8;
    }

    Ok(())
}

/// Convert f64 slice to f32 slice using nearest-even rounding.
///
/// NEON's `vcvt_f32_f64` performs the narrowing with nearest-even rounding
/// (the default IEEE 754 mode), processing 2 f64 → 2 f32 per instruction.
///
/// Two-pass approach:
/// 1. Fast convert pass: just `vcvt_f32_f64` + store (no branching).
/// 2. If `error_on_overflow`, a second pass checks for finite→infinite overflow.
///
/// This keeps the hot convert loop branch-free for maximum throughput.
///
/// # Safety
///
/// Caller must ensure this runs on an AArch64 target.
pub(super) unsafe fn f64_to_f32_nearest(
    src: &[f64],
    dst: &mut [f32],
    error_on_overflow: bool,
) -> Result<(), crate::CastError> {
    let n = src.len();
    let simd_len = n / 2 * 2;

    // Pass 1: branch-free narrowing conversion
    for i in (0..simd_len).step_by(2) {
        let v = vld1q_f64(src.as_ptr().add(i));
        let narrowed = vcvt_f32_f64(v);
        vst1_f32(dst.as_mut_ptr().add(i), narrowed);
    }
    // Scalar tail
    for i in simd_len..n {
        dst[i] = src[i] as f32;
    }

    // Pass 2: overflow check (only when out_of_range is None)
    if error_on_overflow {
        let inf_f32 = vdup_n_f32(f32::INFINITY);
        for i in (0..simd_len).step_by(2) {
            // Check if result is ±Inf
            let result = vld1_f32(dst.as_ptr().add(i));
            let abs_result = vabs_f32(result);
            let result_is_inf = vceq_f32(abs_result, inf_f32);
            // Quick reject: if no Inf in result, no overflow possible
            let inf_bytes: uint8x8_t = vreinterpret_u8_u32(result_is_inf);
            if vmaxv_u8(inf_bytes) != 0 {
                // At least one result is Inf — check if source was finite
                for (&sv, &dv) in src[i..].iter().zip(dst[i..].iter()).take(2) {
                    if sv.is_finite() && dv.is_infinite() {
                        return Err(crate::CastError::OutOfRange {
                            value: sv,
                            lo: f32::MIN as f64,
                            hi: f32::MAX as f64,
                        });
                    }
                }
            }
        }
        // Tail check
        for i in simd_len..n {
            if src[i].is_finite() && dst[i].is_infinite() {
                return Err(crate::CastError::OutOfRange {
                    value: src[i],
                    lo: f32::MIN as f64,
                    hi: f32::MAX as f64,
                });
            }
        }
    }

    Ok(())
}

/// Convert f64 slice to i32 slice with rounding, returning an error if any
/// value is out of range (no clamping).
///
/// Same pipeline as `f64_to_i32_clamp` but instead of clamping, we
/// batch-check that all rounded values fall within [i32::MIN, i32::MAX]
/// and error if not.
///
/// # Safety
///
/// Caller must ensure this runs on an AArch64 target.
pub(super) unsafe fn f64_to_i32_check(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {
    let n = src.len();
    let simd_len = n / 2 * 2;

    let lo = vdupq_n_f64(i32::MIN as f64);
    let hi = vdupq_n_f64(i32::MAX as f64);

    for i in (0..simd_len).step_by(2) {
        let v = vld1q_f64(src.as_ptr().add(i));

        // NaN check
        if any_nan_f64x2(v) {
            for &val in &src[i..std::cmp::min(i + 2, n)] {
                if val.is_nan() {
                    return Err(crate::CastError::NanOrInf { value: val });
                }
            }
        }

        let r = round_f64x2(v, rounding);

        // Range check: error if any value < lo or > hi
        // vcltq_f64 returns all-ones for true, all-zeros for false
        let below = vcltq_f64(r, lo);
        let above = vcgtq_f64(r, hi);
        let out_of_range = vorrq_u64(below, above);
        let oor_bytes: uint8x16_t = vreinterpretq_u8_u64(out_of_range);
        if vmaxvq_u8(oor_bytes) != 0 {
            // Find exact offending element
            for &val in src[i..].iter().take(2) {
                let rounded = scalar_round_f64(val, rounding);
                if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
                    return Err(crate::CastError::OutOfRange {
                        value: val,
                        lo: i32::MIN as f64,
                        hi: i32::MAX as f64,
                    });
                }
            }
        }

        // Convert (values are in range, truncation after rounding is correct)
        let i32_val = vmovn_s64(vcvtq_s64_f64(r));
        vst1_s32(dst.as_mut_ptr().add(i), i32_val);
    }

    // Scalar tail
    for i in simd_len..n {
        let val = src[i];
        if val.is_nan() {
            return Err(crate::CastError::NanOrInf { value: val });
        }
        let rounded = scalar_round_f64(val, rounding);
        if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
            return Err(crate::CastError::OutOfRange {
                value: val,
                lo: i32::MIN as f64,
                hi: i32::MAX as f64,
            });
        }
        dst[i] = rounded as i32;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar tail helpers (shared across kernels)
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn scalar_round_f64(val: f64, rounding: RoundingMode) -> f64 {
    match rounding {
        RoundingMode::NearestEven => val.round_ties_even(),
        RoundingMode::TowardsZero => val.trunc(),
        RoundingMode::TowardsPositive => val.ceil(),
        RoundingMode::TowardsNegative => val.floor(),
        RoundingMode::NearestAway => unreachable!(),
    }
}

#[inline(always)]
unsafe fn scalar_tail_f64_to_u8(
    src: &[f64],
    dst: &mut [u8],
    start: usize,
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {
    for i in start..src.len() {
        let val = src[i];
        if val.is_nan() {
            return Err(crate::CastError::NanOrInf { value: val });
        }
        let rounded = scalar_round_f64(val, rounding);
        dst[i] = rounded.clamp(0.0, 255.0) as u8;
    }
    Ok(())
}
