//! AVX2 SIMD kernels for x86_64.
//!
//! All functions require AVX2 + SSE4.1, guaranteed by [`pulp::x86::V3`].
//! Each function processes the bulk of the slice with SIMD and finishes
//! any remainder with a scalar tail loop.

use crate::RoundingMode;
use core::arch::x86_64::*;
use pulp::x86::V3;

/// Select the AVX2 rounding mode constant for `_mm256_round_pd`/`_mm256_round_ps`.
fn avx2_round_mode(rounding: RoundingMode) -> i32 {
    match rounding {
        // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
        RoundingMode::NearestEven => 0x08,
        // _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC
        RoundingMode::TowardsZero => 0x0B,
        // _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC
        RoundingMode::TowardsPositive => 0x0A,
        // _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC
        RoundingMode::TowardsNegative => 0x09,
        // NearestAway is excluded at the call site
        RoundingMode::NearestAway => unreachable!(),
    }
}

/// Convert f64 slice to u8 slice with rounding and clamping.
///
/// Pipeline per 16 elements:
/// 1. Load 4 x f64x4
/// 2. Round each (mode-dependent)
/// 3. Clamp to [0.0, 255.0]
/// 4. Convert f64x4 → i32x4 (4 x __m128i)
/// 5. Pack i32 → i16 → u8 (yields 16 x u8)
///
/// # Safety
///
/// Caller must ensure AVX2 + SSE4.1 are available (guaranteed by `pulp::x86::V3`).
#[target_feature(enable = "avx2")]
pub(super) unsafe fn f64_to_u8_clamp(
    simd: V3,
    src: &[f64],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {
    let _ = simd;
    let round_mode = avx2_round_mode(rounding);
    let n = src.len();
    let simd_len = n / 16 * 16;

    let lo = _mm256_set1_pd(0.0);
    let hi = _mm256_set1_pd(255.0);

    for i in (0..simd_len).step_by(16) {
        let ptr = src.as_ptr().add(i);

        // Load 4 x f64x4 (16 f64 values)
        let v0 = _mm256_loadu_pd(ptr);
        let v1 = _mm256_loadu_pd(ptr.add(4));
        let v2 = _mm256_loadu_pd(ptr.add(8));
        let v3 = _mm256_loadu_pd(ptr.add(12));

        // Inline NaN check: OR all unordered comparison masks
        let nan0 = _mm256_cmp_pd(v0, v0, _CMP_UNORD_Q);
        let nan1 = _mm256_cmp_pd(v1, v1, _CMP_UNORD_Q);
        let nan2 = _mm256_cmp_pd(v2, v2, _CMP_UNORD_Q);
        let nan3 = _mm256_cmp_pd(v3, v3, _CMP_UNORD_Q);
        let nan_any = _mm256_or_pd(_mm256_or_pd(nan0, nan1), _mm256_or_pd(nan2, nan3));
        if _mm256_movemask_pd(nan_any) != 0 {
            // Find the exact NaN for the error message
            for &val in &src[i..std::cmp::min(i + 16, n)] {
                if val.is_nan() {
                    return Err(crate::CastError::NanOrInf { value: val });
                }
            }
        }

        // Round
        let (r0, r1, r2, r3) = round_4x_f64(v0, v1, v2, v3, round_mode);

        // Clamp to [0, 255]
        let c0 = _mm256_min_pd(_mm256_max_pd(r0, lo), hi);
        let c1 = _mm256_min_pd(_mm256_max_pd(r1, lo), hi);
        let c2 = _mm256_min_pd(_mm256_max_pd(r2, lo), hi);
        let c3 = _mm256_min_pd(_mm256_max_pd(r3, lo), hi);

        // Convert f64x4 → i32x4 (each yields 128-bit with 4 i32s)
        let i0 = _mm256_cvtpd_epi32(c0);
        let i1 = _mm256_cvtpd_epi32(c1);
        let i2 = _mm256_cvtpd_epi32(c2);
        let i3 = _mm256_cvtpd_epi32(c3);

        // Pack i32x4 → u16x8 (two at a time)
        let u16_01 = _mm_packus_epi32(i0, i1); // 8 x u16
        let u16_23 = _mm_packus_epi32(i2, i3); // 8 x u16

        // Pack u16x8 → u8x16
        let u8_all = _mm_packus_epi16(u16_01, u16_23); // 16 x u8

        // Store 16 bytes
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, u8_all);
    }

    // Scalar tail (at most 15 elements)
    for i in simd_len..n {
        let val = src[i];
        if val.is_nan() {
            return Err(crate::CastError::NanOrInf { value: val });
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

/// Convert f64 slice to i32 slice with rounding and clamping.
///
/// # Safety
///
/// Caller must ensure AVX2 + SSE4.1 are available (guaranteed by `pulp::x86::V3`).
#[target_feature(enable = "avx2")]
pub(super) unsafe fn f64_to_i32_clamp(
    simd: V3,
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {
    let _ = simd;
    let round_mode = avx2_round_mode(rounding);
    let n = src.len();
    let simd_len = n / 4 * 4;

    let lo = _mm256_set1_pd(i32::MIN as f64);
    let hi = _mm256_set1_pd(i32::MAX as f64);

    for i in (0..simd_len).step_by(4) {
        let v = _mm256_loadu_pd(src.as_ptr().add(i));

        // Inline NaN check
        let nan_mask = _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
        if _mm256_movemask_pd(nan_mask) != 0 {
            for &val in &src[i..std::cmp::min(i + 4, n)] {
                if val.is_nan() {
                    return Err(crate::CastError::NanOrInf { value: val });
                }
            }
        }

        let r = round_f64(v, round_mode);
        let c = _mm256_min_pd(_mm256_max_pd(r, lo), hi);
        let converted = _mm256_cvtpd_epi32(c);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, converted);
    }

    // Scalar tail
    for i in simd_len..n {
        let val = src[i];
        if val.is_nan() {
            return Err(crate::CastError::NanOrInf { value: val });
        }
        let rounded = match rounding {
            RoundingMode::NearestEven => val.round_ties_even(),
            RoundingMode::TowardsZero => val.trunc(),
            RoundingMode::TowardsPositive => val.ceil(),
            RoundingMode::TowardsNegative => val.floor(),
            RoundingMode::NearestAway => unreachable!(),
        };
        dst[i] = rounded.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
    }

    Ok(())
}

/// Convert f32 slice to u8 slice with rounding and clamping.
///
/// Processes 16 f32s at a time (2 x f32x8):
/// 1. Round, clamp to [0, 255]
/// 2. Convert f32x8 → i32x8
/// 3. Pack i32x8 → u16 → u8
///
/// # Safety
///
/// Caller must ensure AVX2 + SSE4.1 are available (guaranteed by `pulp::x86::V3`).
#[target_feature(enable = "avx2")]
pub(super) unsafe fn f32_to_u8_clamp(
    simd: V3,
    src: &[f32],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {
    let _ = simd;
    let round_mode = avx2_round_mode(rounding);
    let n = src.len();
    let simd_len = n / 16 * 16;

    let lo = _mm256_set1_ps(0.0);
    let hi = _mm256_set1_ps(255.0);

    for i in (0..simd_len).step_by(16) {
        let ptr = src.as_ptr().add(i);

        let v0 = _mm256_loadu_ps(ptr);
        let v1 = _mm256_loadu_ps(ptr.add(8));

        // Inline NaN check
        let nan0 = _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q);
        let nan1 = _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q);
        if (_mm256_movemask_ps(nan0) | _mm256_movemask_ps(nan1)) != 0 {
            for &val in &src[i..std::cmp::min(i + 16, n)] {
                if val.is_nan() {
                    return Err(crate::CastError::NanOrInf { value: val as f64 });
                }
            }
        }

        let (r0, r1) = round_2x_f32(v0, v1, round_mode);

        let c0 = _mm256_min_ps(_mm256_max_ps(r0, lo), hi);
        let c1 = _mm256_min_ps(_mm256_max_ps(r1, lo), hi);

        // Convert f32x8 → i32x8
        let i0 = _mm256_cvtps_epi32(c0);
        let i1 = _mm256_cvtps_epi32(c1);

        // Pack i32x8 → u16x16 (256-bit)
        // _mm256_packus_epi32 interleaves lanes, so we need to
        // fix the order with a permute.
        let u16_raw = _mm256_packus_epi32(i0, i1);
        let u16_ordered = _mm256_permute4x64_epi64(u16_raw, 0b11_01_10_00);

        // Pack u16x16 → u8x16 (take lower 128 bits)
        // We need both halves to pack into u8.
        let u16_lo = _mm256_castsi256_si128(u16_ordered);
        let u16_hi = _mm256_extracti128_si256(u16_ordered, 1);
        let u8_all = _mm_packus_epi16(u16_lo, u16_hi);

        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, u8_all);
    }

    // Scalar tail
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

// ---------------------------------------------------------------------------
// Rounding helpers
// ---------------------------------------------------------------------------

// `_mm256_round_pd`/`_mm256_round_ps` require a compile-time constant for the
// rounding mode parameter, so we must use a match to dispatch to the right
// intrinsic call with a literal constant.

#[inline(always)]
unsafe fn round_f64(v: __m256d, mode: i32) -> __m256d {
    match mode {
        0x08 => _mm256_round_pd(v, 0x08),
        0x09 => _mm256_round_pd(v, 0x09),
        0x0A => _mm256_round_pd(v, 0x0A),
        0x0B => _mm256_round_pd(v, 0x0B),
        _ => unreachable!(),
    }
}

#[inline(always)]
unsafe fn round_4x_f64(
    v0: __m256d,
    v1: __m256d,
    v2: __m256d,
    v3: __m256d,
    mode: i32,
) -> (__m256d, __m256d, __m256d, __m256d) {
    (
        round_f64(v0, mode),
        round_f64(v1, mode),
        round_f64(v2, mode),
        round_f64(v3, mode),
    )
}

#[inline(always)]
unsafe fn round_f32(v: __m256, mode: i32) -> __m256 {
    match mode {
        0x08 => _mm256_round_ps(v, 0x08),
        0x09 => _mm256_round_ps(v, 0x09),
        0x0A => _mm256_round_ps(v, 0x0A),
        0x0B => _mm256_round_ps(v, 0x0B),
        _ => unreachable!(),
    }
}

#[inline(always)]
unsafe fn round_2x_f32(v0: __m256, v1: __m256, mode: i32) -> (__m256, __m256) {
    (round_f32(v0, mode), round_f32(v1, mode))
}
