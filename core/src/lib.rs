//! zarr-cast-value: Pure Rust implementation of the cast_value codec's
//! per-element conversion logic.
//!
//! This crate is independent of Python/PyO3 and can be used by any Rust consumer
//! (e.g. zarrs, zarr-python bindings).

use num_traits::{Float, One, ToPrimitive};
use std::ops::{Add, Sub};

mod simd;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during element conversion.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CastError {
    /// A NaN or Infinity value was encountered when casting to an integer type.
    #[error("Cannot cast {value} to integer type without scalar_map")]
    NanOrInf { value: f64 },
    /// A value is out of range for the target type and no out_of_range mode was set.
    #[error(
        "Value {value} out of range [{lo}, {hi}]. \
         Set out_of_range='clamp' or out_of_range='wrap' to handle this."
    )]
    OutOfRange { value: f64, lo: f64, hi: f64 },
}

// ---------------------------------------------------------------------------
// Rounding modes
// ---------------------------------------------------------------------------

/// How to round floating-point values when casting to integer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RoundingMode {
    NearestEven,
    TowardsZero,
    TowardsPositive,
    TowardsNegative,
    NearestAway,
}

impl std::str::FromStr for RoundingMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "nearest-even" => Ok(Self::NearestEven),
            "towards-zero" => Ok(Self::TowardsZero),
            "towards-positive" => Ok(Self::TowardsPositive),
            "towards-negative" => Ok(Self::TowardsNegative),
            "nearest-away" => Ok(Self::NearestAway),
            _ => Err(format!("Unknown rounding mode: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Out-of-range modes
// ---------------------------------------------------------------------------

/// How to handle values outside the target integer type's range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutOfRangeMode {
    Clamp,
    Wrap,
}

impl std::str::FromStr for OutOfRangeMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "clamp" => Ok(Self::Clamp),
            "wrap" => Ok(Self::Wrap),
            _ => Err(format!("Unknown out_of_range mode: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar map entry
// ---------------------------------------------------------------------------

/// A pre-parsed scalar map entry, typed on Src and Dst.
#[derive(Debug, Clone, Copy)]
pub struct MapEntry<Src, Dst> {
    pub src: Src,
    pub tgt: Dst,
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Common base for all numeric types that participate in cast_value conversions.
/// Provides only the operations shared by both integers and floats.
///
/// Uses `num_traits::ToPrimitive` for f64 conversion (used in error reporting).
pub trait CastNum: Copy + PartialEq + PartialOrd + std::fmt::Debug + ToPrimitive {}

/// An integer numeric type (i8..i64, u8..u64).
/// Marker trait — integers need no special operations beyond `CastNum`.
pub trait CastInt: CastNum {}

/// A floating-point numeric type (f32, f64).
/// Carries float-specific operations not covered by `num_traits::Float`:
/// mode-aware rounding, Euclidean remainder, and IEEE 754 nextUp/nextDown.
///
/// `num_traits::Float` provides: `is_nan`, `is_infinite`, `is_finite`, `abs`.
/// `num_traits::One` provides: `one()`.
pub trait CastFloat: CastNum + Float + One + Add<Output = Self> + Sub<Output = Self> {
    /// Round to integer according to the given rounding mode.
    fn round_with_mode(self, mode: RoundingMode) -> Self;
    /// Euclidean remainder (used in float→int wrap mode).
    fn rem_euclid(self, rhs: Self) -> Self;
    /// The smallest representable value greater than self (IEEE 754 nextUp).
    fn next_up(self) -> Self;
    /// The largest representable value less than self (IEEE 754 nextDown).
    fn next_down(self) -> Self;
}

/// Conversion from Src to Dst. Implemented per (Src, Dst) pair.
pub trait CastInto<Dst: CastNum>: CastNum {
    /// The minimum Dst value, represented in Src's type.
    fn dst_min() -> Self;
    /// The maximum Dst value, represented in Src's type.
    fn dst_max() -> Self;
    /// Convert self to Dst. Caller must ensure value is in range.
    fn cast_into(self) -> Dst;
}

// For constructing MapEntry values from f64 pairs (e.g. at the Python boundary),
// use `num_traits::FromPrimitive::from_f64()` which returns `Option<Self>`.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a numeric value to f64 for error reporting (not on the hot path).
/// Returns `f64::NAN` if the conversion is not possible.
#[inline]
fn to_f64_lossy<T: ToPrimitive>(val: T) -> f64 {
    val.to_f64().unwrap_or(f64::NAN)
}

// ---------------------------------------------------------------------------
// Scalar map lookup
// ---------------------------------------------------------------------------

/// Apply scalar_map lookup for float sources. Returns Some(tgt) on match.
#[inline]
fn apply_scalar_map_float<Src: CastFloat, Dst: CastNum>(
    val: Src,
    map_entries: &[MapEntry<Src, Dst>],
) -> Option<Dst> {
    for entry in map_entries {
        if entry.src.is_nan() {
            if val.is_nan() {
                return Some(entry.tgt);
            }
        } else if val == entry.src {
            return Some(entry.tgt);
        }
    }
    None
}

/// Apply scalar_map lookup for integer sources. Returns Some(tgt) on match.
/// Integers can never be NaN, so only exact equality is checked.
#[inline]
fn apply_scalar_map_int<Src: CastInt, Dst: CastNum>(
    val: Src,
    map_entries: &[MapEntry<Src, Dst>],
) -> Option<Dst> {
    for entry in map_entries {
        if val == entry.src {
            return Some(entry.tgt);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Conversion configs (per-path)
// ---------------------------------------------------------------------------

/// Configuration for float→int conversion.
pub struct FloatToIntConfig<Src, Dst> {
    pub map_entries: Vec<MapEntry<Src, Dst>>,
    pub rounding: RoundingMode,
    pub out_of_range: Option<OutOfRangeMode>,
}

/// Configuration for int→int conversion.
pub struct IntToIntConfig<Src, Dst> {
    pub map_entries: Vec<MapEntry<Src, Dst>>,
    pub out_of_range: Option<OutOfRangeMode>,
}

/// Configuration for float→float conversion.
pub struct FloatToFloatConfig<Src, Dst> {
    pub map_entries: Vec<MapEntry<Src, Dst>>,
    pub rounding: RoundingMode,
    pub out_of_range: Option<OutOfRangeMode>,
}

/// Configuration for int→float conversion.
pub struct IntToFloatConfig<Src, Dst> {
    pub map_entries: Vec<MapEntry<Src, Dst>>,
    pub rounding: RoundingMode,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if a float→float cast overflowed (finite source → infinite result)
/// and handle according to `out_of_range`.
///
/// Per the spec: with `clamp`, "values outside the finite range of the data type
/// MUST be mapped to ±Infinity" — which is what already happened (the `as` cast
/// produces ±Inf on overflow), so clamp is a no-op for float targets.
/// `wrap` is not permitted for float targets.
/// With no mode set, overflow is an error.
#[inline]
fn check_float_overflow<Src, Dst>(
    val: Src,
    result: Dst,
    out_of_range: Option<OutOfRangeMode>,
) -> Result<Dst, CastError>
where
    Src: CastFloat + CastInto<Dst>,
    Dst: CastFloat,
{
    // Overflow: finite source became infinite result
    if val.is_finite() && result.is_infinite() {
        match out_of_range {
            Some(OutOfRangeMode::Clamp) => {
                // Per spec: map to ±Infinity — which is already what we have
                Ok(result)
            }
            _ => {
                // Wrap is not permitted for float targets; None means error
                let lo = Src::dst_min();
                let hi = Src::dst_max();
                Err(CastError::OutOfRange {
                    value: to_f64_lossy(val),
                    lo: to_f64_lossy(lo),
                    hi: to_f64_lossy(hi),
                })
            }
        }
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Layer 1: Four per-element conversion functions
// ---------------------------------------------------------------------------

/// Convert a single float element to an integer.
///
/// Pipeline: scalar_map → NaN check → round → range check/clamp/wrap → cast.
#[inline]
pub fn convert_float_to_int<Src, Dst>(
    val: Src,
    config: &FloatToIntConfig<Src, Dst>,
) -> Result<Dst, CastError>
where
    Src: CastFloat + CastInto<Dst>,
    Dst: CastInt,
{
    // Step 1: scalar_map lookup
    if !config.map_entries.is_empty() {
        if let Some(tgt) = apply_scalar_map_float(val, &config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: reject NaN (NaN comparisons are all false, would slip through range check)
    if val.is_nan() {
        return Err(CastError::NanOrInf {
            value: to_f64_lossy(val),
        });
    }

    // Step 3: round
    let val = val.round_with_mode(config.rounding);

    // Step 4: range check + out-of-range handling
    let lo = Src::dst_min();
    let hi = Src::dst_max();
    match config.out_of_range {
        Some(OutOfRangeMode::Clamp) => {
            let clamped = if val < lo {
                lo
            } else if val > hi {
                hi
            } else {
                val
            };
            Ok(clamped.cast_into())
        }
        Some(OutOfRangeMode::Wrap) => {
            // Inf can't be wrapped (rem_euclid(Inf) is NaN)
            if val.is_infinite() {
                return Err(CastError::NanOrInf {
                    value: to_f64_lossy(val),
                });
            }
            if val < lo || val > hi {
                let range = hi - lo + Src::one();
                let wrapped = (val - lo).rem_euclid(range) + lo;
                Ok(wrapped.cast_into())
            } else {
                Ok(val.cast_into())
            }
        }
        None => {
            if val < lo || val > hi {
                Err(CastError::OutOfRange {
                    value: to_f64_lossy(val),
                    lo: to_f64_lossy(lo),
                    hi: to_f64_lossy(hi),
                })
            } else {
                Ok(val.cast_into())
            }
        }
    }
}

/// Convert a single integer element to another integer type.
///
/// Pipeline: scalar_map → range check/clamp/wrap → cast.
#[inline]
pub fn convert_int_to_int<Src, Dst>(
    val: Src,
    config: &IntToIntConfig<Src, Dst>,
) -> Result<Dst, CastError>
where
    Src: CastInt + CastInto<Dst>,
    Dst: CastInt,
{
    // Step 1: scalar_map lookup
    if !config.map_entries.is_empty() {
        if let Some(tgt) = apply_scalar_map_int(val, &config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: range check + out-of-range handling
    let lo = Src::dst_min();
    let hi = Src::dst_max();
    match config.out_of_range {
        Some(OutOfRangeMode::Clamp) => {
            let clamped = if val < lo {
                lo
            } else if val > hi {
                hi
            } else {
                val
            };
            Ok(clamped.cast_into())
        }
        Some(OutOfRangeMode::Wrap) => {
            // Int→int wrap: Rust's `as` cast truncates to the target width,
            // which is exactly modular arithmetic.
            Ok(val.cast_into())
        }
        None => {
            if val < lo || val > hi {
                Err(CastError::OutOfRange {
                    value: to_f64_lossy(val),
                    lo: to_f64_lossy(lo),
                    hi: to_f64_lossy(hi),
                })
            } else {
                Ok(val.cast_into())
            }
        }
    }
}

/// Convert a single float element to another float type.
///
/// Pipeline: scalar_map → NaN/Inf propagation → cast → rounding adjustment → range check.
///
/// Per the zarr cast_value spec:
/// - NaN values MUST be propagated (if both types support IEEE 754).
/// - Signed zero MUST be preserved.
/// - Rounding applies when the target type has insufficient precision.
/// - Out-of-range handling applies when the value exceeds the target's finite range.
///   Only `clamp` is permitted (not `wrap`).
#[inline]
pub fn convert_float_to_float<Src, Dst>(
    val: Src,
    config: &FloatToFloatConfig<Src, Dst>,
) -> Result<Dst, CastError>
where
    Src: CastFloat + CastInto<Dst>,
    Dst: CastFloat,
{
    // Step 1: scalar_map lookup
    if !config.map_entries.is_empty() {
        if let Some(tgt) = apply_scalar_map_float(val, &config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: NaN and Inf propagate naturally through the cast (IEEE 754).

    // Step 3: cast (Rust `as` uses nearest-even rounding by default)
    let result = val.cast_into();

    // Step 4: if the cast was not exact and a non-nearest-even rounding mode is
    // requested, adjust the result. We detect inexactness by comparing the
    // result's value (promoted back to f64) against the original.
    if !val.is_nan() && config.rounding != RoundingMode::NearestEven {
        let val_f64 = to_f64_lossy(val);
        let result_f64 = to_f64_lossy(result);
        if val_f64 != result_f64 {
            let adjusted = match config.rounding {
                RoundingMode::NearestEven => result, // unreachable, but keep exhaustive
                RoundingMode::TowardsZero => {
                    // Need the value closer to zero
                    if result_f64.abs() > val_f64.abs() {
                        if val_f64 >= 0.0 {
                            result.next_down()
                        } else {
                            result.next_up()
                        }
                    } else {
                        result
                    }
                }
                RoundingMode::TowardsPositive => {
                    // Need the value >= original
                    if result_f64 < val_f64 {
                        result.next_up()
                    } else {
                        result
                    }
                }
                RoundingMode::TowardsNegative => {
                    // Need the value <= original
                    if result_f64 > val_f64 {
                        result.next_down()
                    } else {
                        result
                    }
                }
                RoundingMode::NearestAway => {
                    // Nearest, ties away from zero. The `as` cast already gave
                    // us nearest-even. We only need to adjust on ties where
                    // nearest-even went towards zero but nearest-away should
                    // go away from zero. Check if the midpoint between result
                    // and the adjacent value equals the original.
                    let candidate = if val_f64 > result_f64 {
                        result.next_up()
                    } else {
                        result.next_down()
                    };
                    let mid = (to_f64_lossy(result) + to_f64_lossy(candidate)) / 2.0;
                    if mid == val_f64 {
                        // It's a tie — pick the one farther from zero
                        if to_f64_lossy(candidate.abs()) > to_f64_lossy(result.abs()) {
                            candidate
                        } else {
                            result
                        }
                    } else {
                        result // not a tie, nearest-even already correct
                    }
                }
            };
            // Step 5: out-of-range check (the adjusted value might be Inf)
            return check_float_overflow(val, adjusted, config.out_of_range);
        }
    }

    // Step 5: out-of-range check
    check_float_overflow(val, result, config.out_of_range)
}

/// Convert a single integer element to a float type.
///
/// Pipeline: scalar_map → cast → rounding adjustment.
///
/// Per the zarr cast_value spec, rounding applies when "casting from integer
/// types to floating-point types with insufficient mantissa bits (e.g. int64
/// to float32)". No out-of-range handling is needed because all supported
/// integer ranges fit within the finite range of f32/f64.
#[inline]
pub fn convert_int_to_float<Src, Dst>(
    val: Src,
    config: &IntToFloatConfig<Src, Dst>,
) -> Result<Dst, CastError>
where
    Src: CastInt + CastInto<Dst>,
    Dst: CastFloat,
{
    // Step 1: scalar_map lookup
    if !config.map_entries.is_empty() {
        if let Some(tgt) = apply_scalar_map_int(val, &config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: cast (Rust `as` uses nearest-even)
    let result = val.cast_into();

    // Step 3: if the cast was not exact and a non-nearest-even rounding mode
    // is requested, adjust the result.
    if config.rounding != RoundingMode::NearestEven {
        let val_f64 = to_f64_lossy(val);
        let result_f64 = to_f64_lossy(result);
        if val_f64 != result_f64 {
            return Ok(match config.rounding {
                RoundingMode::NearestEven => result,
                RoundingMode::TowardsZero => {
                    if result_f64.abs() > val_f64.abs() {
                        if val_f64 >= 0.0 {
                            result.next_down()
                        } else {
                            result.next_up()
                        }
                    } else {
                        result
                    }
                }
                RoundingMode::TowardsPositive => {
                    if result_f64 < val_f64 {
                        result.next_up()
                    } else {
                        result
                    }
                }
                RoundingMode::TowardsNegative => {
                    if result_f64 > val_f64 {
                        result.next_down()
                    } else {
                        result
                    }
                }
                RoundingMode::NearestAway => {
                    let candidate = if val_f64 > result_f64 {
                        result.next_up()
                    } else {
                        result.next_down()
                    };
                    let mid = (to_f64_lossy(result) + to_f64_lossy(candidate)) / 2.0;
                    if mid == val_f64 {
                        if to_f64_lossy(candidate.abs()) > to_f64_lossy(result.abs()) {
                            candidate
                        } else {
                            result
                        }
                    } else {
                        result
                    }
                }
            });
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Layer 2: Four slice conversion functions
// ---------------------------------------------------------------------------

/// Convert a slice of float values to integer values. Returns early on first error.
///
/// When the configuration allows it (empty scalar_map, clamp mode,
/// rounding mode other than NearestAway), uses SIMD-accelerated kernels
/// for supported type pairs (f64→u8, f64→i32, f32→u8).
pub fn convert_slice_float_to_int<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &FloatToIntConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastFloat + CastInto<Dst> + 'static,
    Dst: CastInt + 'static,
{
    // SIMD fast path: empty scalar_map + clamp + supported rounding mode.
    // Uses TypeId to dispatch to concrete SIMD kernels at runtime.
    // The `'static` bound is satisfied by all primitive numeric types.
    if config.map_entries.is_empty()
        && config.out_of_range == Some(OutOfRangeMode::Clamp)
        && config.rounding != RoundingMode::NearestAway
    {
        use std::any::TypeId;

        // f64 → u8
        if TypeId::of::<Src>() == TypeId::of::<f64>() && TypeId::of::<Dst>() == TypeId::of::<u8>() {
            // SAFETY: We just verified Src == f64 and Dst == u8 via TypeId.
            let src_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len()) };
            let dst_u8: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len()) };
            if simd::try_f64_to_u8_clamp(src_f64, dst_u8, config.rounding)? {
                return Ok(());
            }
        }

        // f64 → i32
        if TypeId::of::<Src>() == TypeId::of::<f64>() && TypeId::of::<Dst>() == TypeId::of::<i32>()
        {
            let src_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len()) };
            let dst_i32: &mut [i32] =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i32, dst.len()) };
            if simd::try_f64_to_i32_clamp(src_f64, dst_i32, config.rounding)? {
                return Ok(());
            }
        }

        // f32 → u8
        if TypeId::of::<Src>() == TypeId::of::<f32>() && TypeId::of::<Dst>() == TypeId::of::<u8>() {
            let src_f32: &[f32] =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len()) };
            let dst_u8: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len()) };
            if simd::try_f32_to_u8_clamp(src_f32, dst_u8, config.rounding)? {
                return Ok(());
            }
        }
    }

    // SIMD fast path: empty scalar_map + no out_of_range (error mode) + supported rounding.
    // Range-checking variant: errors if any value is out of range.
    if config.map_entries.is_empty()
        && config.out_of_range.is_none()
        && config.rounding != RoundingMode::NearestAway
    {
        use std::any::TypeId;

        // f64 → i32
        if TypeId::of::<Src>() == TypeId::of::<f64>()
            && TypeId::of::<Dst>() == TypeId::of::<i32>()
        {
            let src_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len()) };
            let dst_i32: &mut [i32] =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i32, dst.len()) };
            if simd::try_f64_to_i32_check(src_f64, dst_i32, config.rounding)? {
                return Ok(());
            }
        }
    }

    // Scalar fallback
    for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
        *out_slot = convert_float_to_int(*in_val, config)?;
    }
    Ok(())
}

/// Convert a slice of integer values to integer values. Returns early on first error.
pub fn convert_slice_int_to_int<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &IntToIntConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastInt + CastInto<Dst>,
    Dst: CastInt,
{
    for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
        *out_slot = convert_int_to_int(*in_val, config)?;
    }
    Ok(())
}

/// Convert a slice of float values to float values. Returns early on first error.
///
/// When the configuration allows it (empty scalar_map, nearest-even rounding),
/// uses SIMD-accelerated kernels for supported type pairs (f64->f32).
pub fn convert_slice_float_to_float<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &FloatToFloatConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastFloat + CastInto<Dst> + 'static,
    Dst: CastFloat + 'static,
{
    // SIMD fast path: empty scalar_map + nearest-even rounding + f64→f32.
    if config.map_entries.is_empty() && config.rounding == RoundingMode::NearestEven {
        use std::any::TypeId;

        if TypeId::of::<Src>() == TypeId::of::<f64>()
            && TypeId::of::<Dst>() == TypeId::of::<f32>()
        {
            // SAFETY: We just verified Src == f64 and Dst == f32 via TypeId.
            let src_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len()) };
            let dst_f32: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len()) };
            let error_on_overflow = config.out_of_range != Some(OutOfRangeMode::Clamp);
            if simd::try_f64_to_f32_nearest(src_f64, dst_f32, error_on_overflow)? {
                return Ok(());
            }
        }
    }

    // Scalar fallback
    for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
        *out_slot = convert_float_to_float(*in_val, config)?;
    }
    Ok(())
}

/// Convert a slice of integer values to float values. Returns early on first error.
pub fn convert_slice_int_to_float<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &IntToFloatConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastInt + CastInto<Dst>,
    Dst: CastFloat,
{
    for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
        *out_slot = convert_int_to_float(*in_val, config)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Trait implementations for primitive types
// ---------------------------------------------------------------------------

// ---- CastNum impls ----

macro_rules! impl_cast_num {
    ($($ty:ty),*) => {
        $( impl CastNum for $ty {} )*
    };
}

impl_cast_num!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

// ---- CastInt impls ----

macro_rules! impl_cast_int {
    ($($ty:ty),*) => {
        $( impl CastInt for $ty {} )*
    };
}

impl_cast_int!(i8, i16, i32, i64, u8, u16, u32, u64);

// ---- CastFloat impls ----

macro_rules! impl_cast_float {
    ($($ty:ty),*) => {
        $(
            impl CastFloat for $ty {
                #[inline]
                fn round_with_mode(self, mode: RoundingMode) -> Self {
                    match mode {
                        RoundingMode::NearestEven => <$ty>::round_ties_even(self),
                        RoundingMode::TowardsZero => <$ty>::trunc(self),
                        RoundingMode::TowardsPositive => <$ty>::ceil(self),
                        RoundingMode::TowardsNegative => <$ty>::floor(self),
                        RoundingMode::NearestAway => {
                            // Round half away from zero:
                            // copysign(floor(|x| + 0.5), x)
                            <$ty>::copysign((<$ty>::abs(self) + 0.5).floor(), self)
                        }
                    }
                }
                #[inline]
                fn rem_euclid(self, rhs: Self) -> Self { <$ty>::rem_euclid(self, rhs) }
                #[inline]
                fn next_up(self) -> Self { <$ty>::next_up(self) }
                #[inline]
                fn next_down(self) -> Self { <$ty>::next_down(self) }
            }
        )*
    };
}

impl_cast_float!(f32, f64);

// ---- CastInto impls ----
// We need N×N implementations. Use a macro to generate them.

macro_rules! impl_cast_into {
    ($src:ty => $dst:ty, min: $min:expr, max: $max:expr) => {
        impl CastInto<$dst> for $src {
            #[inline]
            fn dst_min() -> Self {
                $min
            }
            #[inline]
            fn dst_max() -> Self {
                $max
            }
            #[inline]
            fn cast_into(self) -> $dst {
                self as $dst
            }
        }
    };
}

// Helper: for float→int, the min/max are the integer bounds as floats.
// For int→int, the min/max are the integer bounds clamped to the source range.
// For any→float, range checking is skipped (target is float, no range check),
// so min/max values don't matter, but we still need the impl.

// -- int→int --
macro_rules! impl_int_to_int {
    ($src:ty => $dst:ty) => {
        impl_cast_into!(
            $src => $dst,
            min: {
                let dst_min = <$dst>::MIN as i128;
                let src_min = <$src>::MIN as i128;
                (if dst_min < src_min { src_min } else { dst_min }) as $src
            },
            max: {
                let dst_max = <$dst>::MAX as i128;
                let src_max = <$src>::MAX as i128;
                (if dst_max > src_max { src_max } else { dst_max }) as $src
            }
        );
    };
}

// -- float→int --
macro_rules! impl_float_to_int {
    ($src:ty => $dst:ty) => {
        impl_cast_into!(
            $src => $dst,
            min: <$dst>::MIN as $src,
            max: <$dst>::MAX as $src
        );
    };
}

// -- int→float (no range issue — float range always covers integer range) --
macro_rules! impl_int_to_float {
    ($src:ty => $dst:ty) => {
        impl_cast_into!(
            $src => $dst,
            min: 0 as $src,  // unused — no range check for int→float
            max: 0 as $src
        );
    };
}

// -- float→float (needs real range bounds for out_of_range support) --
macro_rules! impl_float_to_float {
    ($src:ty => $dst:ty) => {
        impl_cast_into!(
            $src => $dst,
            min: <$dst>::MIN as $src,
            max: <$dst>::MAX as $src
        );
    };
}

// Generate all combinations for the 10 types: i8,i16,i32,i64,u8,u16,u32,u64,f32,f64

// int sources → int targets
macro_rules! impl_all_int_to_int {
    ($src:ty => $($dst:ty),*) => {
        $( impl_int_to_int!($src => $dst); )*
    };
}

impl_all_int_to_int!(i8 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(i16 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(i32 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(i64 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(u8 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(u16 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(u32 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(u64 => i8, i16, i32, i64, u8, u16, u32, u64);

// float sources → int targets
macro_rules! impl_all_float_to_int {
    ($src:ty => $($dst:ty),*) => {
        $( impl_float_to_int!($src => $dst); )*
    };
}

impl_all_float_to_int!(f32 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_float_to_int!(f64 => i8, i16, i32, i64, u8, u16, u32, u64);

// int sources → float targets
macro_rules! impl_all_int_to_float {
    ($src:ty => $($dst:ty),*) => {
        $( impl_int_to_float!($src => $dst); )*
    };
}

impl_all_int_to_float!(i8 => f32, f64);
impl_all_int_to_float!(i16 => f32, f64);
impl_all_int_to_float!(i32 => f32, f64);
impl_all_int_to_float!(i64 => f32, f64);
impl_all_int_to_float!(u8 => f32, f64);
impl_all_int_to_float!(u16 => f32, f64);
impl_all_int_to_float!(u32 => f32, f64);
impl_all_int_to_float!(u64 => f32, f64);

// float sources → float targets
macro_rules! impl_all_float_to_float {
    ($src:ty => $($dst:ty),*) => {
        $( impl_float_to_float!($src => $dst); )*
    };
}

impl_all_float_to_float!(f32 => f32, f64);
impl_all_float_to_float!(f64 => f32, f64);

// ---------------------------------------------------------------------------
// float16 support (behind the "float16" feature flag)
// ---------------------------------------------------------------------------

#[cfg(feature = "float16")]
mod float16_impls {
    use super::*;
    use half::f16;

    // ---- CastNum for f16 ----
    impl CastNum for f16 {}

    // ---- CastFloat for f16 ----
    // f16 lacks native arithmetic, so all operations are performed by
    // promoting to f32, computing there, and converting back. This is
    // lossless for inputs that are representable as f16.
    impl CastFloat for f16 {
        #[inline]
        fn round_with_mode(self, mode: RoundingMode) -> Self {
            f16::from_f32(self.to_f32().round_with_mode(mode))
        }
        #[inline]
        fn rem_euclid(self, rhs: Self) -> Self {
            f16::from_f32(self.to_f32().rem_euclid(rhs.to_f32()))
        }
        #[inline]
        fn next_up(self) -> Self {
            // IEEE 754 nextUp implemented directly on f16 bits.
            // Avoids round-tripping through f32 which could skip f16 values.
            let bits = self.to_bits();
            if self.is_nan() {
                self
            } else if bits == 0x8000 {
                // -0.0 → smallest positive subnormal
                f16::from_bits(0x0001)
            } else if (bits & 0x8000) == 0 {
                // Positive: increment bit pattern
                f16::from_bits(bits + 1)
            } else {
                // Negative: decrement bit pattern (moves towards zero)
                f16::from_bits(bits - 1)
            }
        }
        #[inline]
        fn next_down(self) -> Self {
            // nextDown(x) = -nextUp(-x)
            -(-self).next_up()
        }
    }

    // ---- CastInto impls for f16 ----
    // Cannot use the `as` cast macro since Rust has no `as f16` or `as T`
    // from f16. All conversions go through f32.

    // -- f16 → int targets --
    macro_rules! impl_f16_to_int {
        ($($dst:ty),*) => {
            $(
                impl CastInto<$dst> for f16 {
                    #[inline]
                    fn dst_min() -> Self { f16::from_f32(<$dst>::MIN as f32) }
                    #[inline]
                    fn dst_max() -> Self { f16::from_f32(<$dst>::MAX as f32) }
                    #[inline]
                    fn cast_into(self) -> $dst { self.to_f32() as $dst }
                }
            )*
        };
    }
    impl_f16_to_int!(i8, i16, i32, i64, u8, u16, u32, u64);

    // -- int → f16 targets --
    macro_rules! impl_int_to_f16 {
        ($($src:ty),*) => {
            $(
                impl CastInto<f16> for $src {
                    #[inline]
                    fn dst_min() -> Self { 0 as $src } // unused — no range check
                    #[inline]
                    fn dst_max() -> Self { 0 as $src }
                    #[inline]
                    fn cast_into(self) -> f16 { f16::from_f32(self as f32) }
                }
            )*
        };
    }
    impl_int_to_f16!(i8, i16, i32, i64, u8, u16, u32, u64);

    // -- f16 → float targets --
    impl CastInto<f32> for f16 {
        #[inline]
        fn dst_min() -> Self {
            f16::from_f32(f32::MIN)
        }
        #[inline]
        fn dst_max() -> Self {
            f16::from_f32(f32::MAX)
        }
        #[inline]
        fn cast_into(self) -> f32 {
            self.to_f32()
        }
    }

    impl CastInto<f64> for f16 {
        #[inline]
        fn dst_min() -> Self {
            f16::from_f32(f64::MIN as f32)
        }
        #[inline]
        fn dst_max() -> Self {
            f16::from_f32(f64::MAX as f32)
        }
        #[inline]
        fn cast_into(self) -> f64 {
            self.to_f32() as f64
        }
    }

    // -- float → f16 targets --
    impl CastInto<f16> for f32 {
        #[inline]
        fn dst_min() -> Self {
            f16::MIN.to_f32()
        }
        #[inline]
        fn dst_max() -> Self {
            f16::MAX.to_f32()
        }
        #[inline]
        fn cast_into(self) -> f16 {
            f16::from_f32(self)
        }
    }

    impl CastInto<f16> for f64 {
        #[inline]
        fn dst_min() -> Self {
            f16::MIN.to_f32() as f64
        }
        #[inline]
        fn dst_max() -> Self {
            f16::MAX.to_f32() as f64
        }
        #[inline]
        fn cast_into(self) -> f16 {
            f16::from_f32(self as f32)
        }
    }

    // -- f16 → f16 (identity) --
    impl CastInto<f16> for f16 {
        #[inline]
        fn dst_min() -> Self {
            f16::MIN
        }
        #[inline]
        fn dst_max() -> Self {
            f16::MAX
        }
        #[inline]
        fn cast_into(self) -> f16 {
            self
        }
    }
}

// Re-export f16 when the feature is enabled so consumers can use it
// through this crate without depending on `half` directly.
#[cfg(feature = "float16")]
pub use half::f16;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build FloatToIntConfig
    fn f2i_cfg<Src: CastFloat, Dst: CastInt>(
        map: Vec<MapEntry<Src, Dst>>,
        rounding: RoundingMode,
        oor: Option<OutOfRangeMode>,
    ) -> FloatToIntConfig<Src, Dst> {
        FloatToIntConfig {
            map_entries: map,
            rounding,
            out_of_range: oor,
        }
    }

    // Helper to build IntToIntConfig
    fn i2i_cfg<Src: CastInt, Dst: CastInt>(
        map: Vec<MapEntry<Src, Dst>>,
        oor: Option<OutOfRangeMode>,
    ) -> IntToIntConfig<Src, Dst> {
        IntToIntConfig {
            map_entries: map,
            out_of_range: oor,
        }
    }

    #[test]
    fn test_float64_to_uint8_basic() {
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        assert_eq!(convert_float_to_int(42.0_f64, &c).unwrap(), 42_u8);
    }

    #[test]
    fn test_float64_to_uint8_rounding() {
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        // 2.5 rounds to 2 (banker's rounding)
        assert_eq!(convert_float_to_int(2.5_f64, &c).unwrap(), 2_u8);
        // 3.5 rounds to 4
        assert_eq!(convert_float_to_int(3.5_f64, &c).unwrap(), 4_u8);
    }

    #[test]
    fn test_float64_to_uint8_clamp() {
        let c = f2i_cfg::<f64, u8>(
            vec![],
            RoundingMode::NearestEven,
            Some(OutOfRangeMode::Clamp),
        );
        assert_eq!(convert_float_to_int(300.0_f64, &c).unwrap(), 255_u8);
        assert_eq!(convert_float_to_int(-10.0_f64, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_float64_to_int8_wrap() {
        let c = f2i_cfg::<f64, i8>(
            vec![],
            RoundingMode::NearestEven,
            Some(OutOfRangeMode::Wrap),
        );
        // 200 wraps: (200 - (-128)) % 256 + (-128) = 328 % 256 + (-128) = 72 + (-128) = -56
        assert_eq!(convert_float_to_int(200.0_f64, &c).unwrap(), -56_i8);
    }

    #[test]
    fn test_int32_to_int8_wrap() {
        let c = i2i_cfg::<i32, i8>(vec![], Some(OutOfRangeMode::Wrap));
        // 200_i32 as i8 = -56 (bit truncation = modular wrap)
        assert_eq!(convert_int_to_int(200_i32, &c).unwrap(), -56_i8);
        assert_eq!(convert_int_to_int(-200_i32, &c).unwrap(), 56_i8);
    }

    #[test]
    fn test_int32_to_uint8_wrap() {
        let c = i2i_cfg::<i32, u8>(vec![], Some(OutOfRangeMode::Wrap));
        assert_eq!(convert_int_to_int(300_i32, &c).unwrap(), 44_u8);
        assert_eq!(convert_int_to_int(-1_i32, &c).unwrap(), 255_u8);
    }

    #[test]
    fn test_out_of_range_error() {
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        assert!(convert_float_to_int(300.0_f64, &c).is_err());
    }

    #[test]
    fn test_nan_to_int_error() {
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        assert!(convert_float_to_int(f64::NAN, &c).is_err());
    }

    #[test]
    fn test_inf_to_int_error() {
        // Without out_of_range, Inf is caught by the range check (OutOfRange error)
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        assert!(convert_float_to_int(f64::INFINITY, &c).is_err());
        assert!(convert_float_to_int(f64::NEG_INFINITY, &c).is_err());
    }

    #[test]
    fn test_inf_to_int_clamp() {
        // With clamp, +Inf clamps to max, -Inf clamps to min
        let c = f2i_cfg::<f64, u8>(
            vec![],
            RoundingMode::NearestEven,
            Some(OutOfRangeMode::Clamp),
        );
        assert_eq!(convert_float_to_int(f64::INFINITY, &c).unwrap(), 255_u8);
        assert_eq!(convert_float_to_int(f64::NEG_INFINITY, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_inf_to_int_wrap_errors() {
        // With wrap, Inf is rejected (rem_euclid(Inf) is NaN)
        let c = f2i_cfg::<f64, i8>(
            vec![],
            RoundingMode::NearestEven,
            Some(OutOfRangeMode::Wrap),
        );
        assert!(convert_float_to_int(f64::INFINITY, &c).is_err());
        assert!(convert_float_to_int(f64::NEG_INFINITY, &c).is_err());
    }

    #[test]
    fn test_scalar_map_nan() {
        let entries = vec![MapEntry {
            src: f64::NAN,
            tgt: 0_u8,
        }];
        let c = f2i_cfg(entries, RoundingMode::NearestEven, None);
        assert_eq!(convert_float_to_int(f64::NAN, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_scalar_map_nan_payloads() {
        // Any NaN should match a NaN map entry, regardless of payload bits.
        let entries = vec![MapEntry {
            src: f64::NAN,
            tgt: 42_u8,
        }];
        let c = f2i_cfg(entries, RoundingMode::NearestEven, None);

        // Standard NaN
        assert_eq!(convert_float_to_int(f64::NAN, &c).unwrap(), 42_u8);

        // NaN with custom payload (different bit pattern, still NaN)
        let nan_payload = f64::from_bits(0x7FF8_0000_0000_0001);
        assert!(nan_payload.is_nan());
        assert_eq!(convert_float_to_int(nan_payload, &c).unwrap(), 42_u8);

        // Negative NaN (sign bit set)
        let neg_nan = f64::from_bits(0xFFF8_0000_0000_0000);
        assert!(neg_nan.is_nan());
        assert_eq!(convert_float_to_int(neg_nan, &c).unwrap(), 42_u8);

        // Signaling NaN (quiet bit clear, payload nonzero)
        let snan = f64::from_bits(0x7FF0_0000_0000_0001);
        assert!(snan.is_nan());
        assert_eq!(convert_float_to_int(snan, &c).unwrap(), 42_u8);
    }

    #[test]
    fn test_scalar_map_exact() {
        let entries = vec![MapEntry {
            src: 42.0_f64,
            tgt: 99_u8,
        }];
        let c = f2i_cfg(entries, RoundingMode::NearestEven, None);
        assert_eq!(convert_float_to_int(42.0_f64, &c).unwrap(), 99_u8);
        // Non-matching value goes through normal path
        assert_eq!(convert_float_to_int(10.0_f64, &c).unwrap(), 10_u8);
    }

    #[test]
    fn test_int32_to_uint8_range_check() {
        let c = i2i_cfg::<i32, u8>(vec![], None);
        assert_eq!(convert_int_to_int(100_i32, &c).unwrap(), 100_u8);
        assert!(convert_int_to_int(300_i32, &c).is_err());
        assert!(convert_int_to_int(-1_i32, &c).is_err());
    }

    #[test]
    fn test_int32_to_float64() {
        let c = IntToFloatConfig::<i32, f64> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
        };
        assert_eq!(convert_int_to_float(42_i32, &c).unwrap(), 42.0_f64);
    }

    #[test]
    fn test_convert_slice_basic() {
        let src = [1.0_f64, 2.0, 3.0, 4.0];
        let mut dst = [0_u8; 4];
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        convert_slice_float_to_int(&src, &mut dst, &c).unwrap();
        assert_eq!(dst, [1, 2, 3, 4]);
    }

    #[test]
    fn test_convert_slice_early_termination() {
        let src = [1.0_f64, 2.0, 300.0, 4.0];
        let mut dst = [0_u8; 4];
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        let result = convert_slice_float_to_int(&src, &mut dst, &c);
        assert!(result.is_err());
        // First two should have been written
        assert_eq!(dst[0], 1);
        assert_eq!(dst[1], 2);
    }

    #[test]
    fn test_all_rounding_modes() {
        // towards-zero
        let c = f2i_cfg::<f64, i8>(vec![], RoundingMode::TowardsZero, None);
        assert_eq!(convert_float_to_int(2.7_f64, &c).unwrap(), 2_i8);
        assert_eq!(convert_float_to_int(-2.7_f64, &c).unwrap(), -2_i8);

        // towards-positive
        let c = f2i_cfg::<f64, i8>(vec![], RoundingMode::TowardsPositive, None);
        assert_eq!(convert_float_to_int(2.1_f64, &c).unwrap(), 3_i8);
        assert_eq!(convert_float_to_int(-2.7_f64, &c).unwrap(), -2_i8);

        // towards-negative
        let c = f2i_cfg::<f64, i8>(vec![], RoundingMode::TowardsNegative, None);
        assert_eq!(convert_float_to_int(2.7_f64, &c).unwrap(), 2_i8);
        assert_eq!(convert_float_to_int(-2.1_f64, &c).unwrap(), -3_i8);

        // nearest-away
        let c = f2i_cfg::<f64, i8>(vec![], RoundingMode::NearestAway, None);
        assert_eq!(convert_float_to_int(2.5_f64, &c).unwrap(), 3_i8);
        assert_eq!(convert_float_to_int(-2.5_f64, &c).unwrap(), -3_i8);
    }

    #[test]
    fn test_int64_to_int32_clamp() {
        let c = i2i_cfg::<i64, i32>(vec![], Some(OutOfRangeMode::Clamp));
        assert_eq!(convert_int_to_int(i64::MAX, &c).unwrap(), i32::MAX);
        assert_eq!(convert_int_to_int(i64::MIN, &c).unwrap(), i32::MIN);
    }

    #[test]
    fn test_float32_to_float64() {
        let c = FloatToFloatConfig::<f32, f64> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        assert_eq!(
            convert_float_to_float(1.25_f32, &c).unwrap(),
            1.25_f32 as f64
        );
    }

    // -- float→float rounding tests --

    #[test]
    fn test_float64_to_float32_nearest_even() {
        // A value that is exactly between two f32 representable values
        // should round to even (the default `as` behavior).
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        // 1.0 is exactly representable — no rounding
        assert_eq!(convert_float_to_float(1.0_f64, &c).unwrap(), 1.0_f32);
    }

    #[test]
    fn test_float64_to_float32_towards_zero() {
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::TowardsZero,
            out_of_range: None,
        };
        // Pick a value not exactly representable in f32.
        // f32 has ~7 decimal digits of precision.
        let val: f64 = 1.0000001_f64; // slightly above 1.0
        let result = convert_float_to_float(val, &c).unwrap();
        // Towards zero: result should be <= val (since val > 0)
        assert!(
            to_f64_lossy(result) <= val,
            "towards-zero: {result} > {val}"
        );
        assert!(to_f64_lossy(result) > 0.0);

        // Negative: towards zero means closer to 0 (larger, i.e. less negative)
        let val_neg: f64 = -1.0000001_f64;
        let result_neg = convert_float_to_float(val_neg, &c).unwrap();
        assert!(
            to_f64_lossy(result_neg) >= val_neg,
            "towards-zero negative: {result_neg} < {val_neg}"
        );
    }

    #[test]
    fn test_float64_to_float32_towards_positive() {
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::TowardsPositive,
            out_of_range: None,
        };
        let val: f64 = 1.0000001_f64;
        let result = convert_float_to_float(val, &c).unwrap();
        // Towards positive: result should be >= val
        assert!(
            to_f64_lossy(result) >= val,
            "towards-positive: {result} < {val}"
        );
    }

    #[test]
    fn test_float64_to_float32_towards_negative() {
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::TowardsNegative,
            out_of_range: None,
        };
        let val: f64 = 1.0000001_f64;
        let result = convert_float_to_float(val, &c).unwrap();
        // Towards negative: result should be <= val
        assert!(
            to_f64_lossy(result) <= val,
            "towards-negative: {result} > {val}"
        );
    }

    // -- float→float out-of-range tests --

    #[test]
    fn test_float64_to_float32_overflow_error() {
        // A value exceeding f32::MAX should error when no out_of_range is set
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        let val = f64::from(f32::MAX) * 2.0;
        assert!(convert_float_to_float(val, &c).is_err());
    }

    #[test]
    fn test_float64_to_float32_overflow_clamp() {
        // With clamp, overflow maps to ±Infinity
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        let val = f64::from(f32::MAX) * 2.0;
        let result = convert_float_to_float(val, &c).unwrap();
        assert!(result.is_infinite());
        assert!(result > 0.0_f32);

        let val_neg = -f64::from(f32::MAX) * 2.0;
        let result_neg = convert_float_to_float(val_neg, &c).unwrap();
        assert!(result_neg.is_infinite());
        assert!(result_neg < 0.0_f32);
    }

    #[test]
    fn test_float64_to_float32_inf_propagates() {
        // ±Inf from source should propagate to target (not trigger out-of-range error)
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        assert!(convert_float_to_float(f64::INFINITY, &c)
            .unwrap()
            .is_infinite());
        assert!(convert_float_to_float(f64::NEG_INFINITY, &c)
            .unwrap()
            .is_infinite());
    }

    #[test]
    fn test_float64_to_float32_nan_propagates() {
        // NaN should propagate without error
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        assert!(convert_float_to_float(f64::NAN, &c).unwrap().is_nan());
    }

    #[test]
    fn test_float64_to_float32_signed_zero() {
        // Signed zero must be preserved
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        let result = convert_float_to_float(-0.0_f64, &c).unwrap();
        assert!(result == 0.0_f32);
        assert!(result.is_sign_negative());
    }

    // -- int→float rounding tests --

    #[test]
    fn test_int64_to_float32_rounding() {
        // i64 values near 2^24 can't all be exactly represented in f32
        // (f32 has 24 bits of mantissa). Pick a value that requires rounding.
        let val: i64 = (1_i64 << 24) + 1; // 16777217 — not exactly representable in f32

        // nearest-even: default behavior
        let c_ne = IntToFloatConfig::<i64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
        };
        let result_ne = convert_int_to_float(val, &c_ne).unwrap();
        assert_eq!(result_ne, val as f32); // same as Rust `as`

        // towards-positive: result should be >= val
        let c_pos = IntToFloatConfig::<i64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::TowardsPositive,
        };
        let result_pos = convert_int_to_float(val, &c_pos).unwrap();
        assert!(
            to_f64_lossy(result_pos) >= val as f64,
            "towards-positive: {result_pos} < {val}"
        );

        // towards-zero: result should be <= val (since val > 0)
        let c_tz = IntToFloatConfig::<i64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::TowardsZero,
        };
        let result_tz = convert_int_to_float(val, &c_tz).unwrap();
        assert!(
            to_f64_lossy(result_tz) <= val as f64,
            "towards-zero: {result_tz} > {val}"
        );
    }

    // ---- float16 tests ----

    #[cfg(feature = "float16")]
    mod float16_tests {
        use super::*;
        use half::f16;

        #[test]
        fn test_f16_to_u8_basic() {
            let c = f2i_cfg::<f16, u8>(vec![], RoundingMode::NearestEven, None);
            assert_eq!(
                convert_float_to_int(f16::from_f32(42.0), &c).unwrap(),
                42_u8
            );
        }

        #[test]
        fn test_f16_to_u8_clamp() {
            let c = f2i_cfg::<f16, u8>(
                vec![],
                RoundingMode::NearestEven,
                Some(OutOfRangeMode::Clamp),
            );
            assert_eq!(
                convert_float_to_int(f16::from_f32(300.0), &c).unwrap(),
                255_u8
            );
            assert_eq!(
                convert_float_to_int(f16::from_f32(-10.0), &c).unwrap(),
                0_u8
            );
        }

        #[test]
        fn test_f16_nan_error() {
            let c = f2i_cfg::<f16, u8>(vec![], RoundingMode::NearestEven, None);
            assert!(convert_float_to_int(f16::NAN, &c).is_err());
        }

        #[test]
        fn test_f16_scalar_map_nan() {
            let c = f2i_cfg::<f16, u8>(
                vec![MapEntry {
                    src: f16::NAN,
                    tgt: 0_u8,
                }],
                RoundingMode::NearestEven,
                None,
            );
            assert_eq!(convert_float_to_int(f16::NAN, &c).unwrap(), 0_u8);
        }

        #[test]
        fn test_f32_to_f16_basic() {
            let c = FloatToFloatConfig {
                map_entries: vec![],
                rounding: RoundingMode::NearestEven,
                out_of_range: None,
            };
            let result: f16 = convert_float_to_float(1.5_f32, &c).unwrap();
            assert_eq!(result, f16::from_f32(1.5));
        }

        #[test]
        fn test_f64_to_f16_overflow_clamp() {
            let c = FloatToFloatConfig {
                map_entries: vec![],
                rounding: RoundingMode::NearestEven,
                out_of_range: Some(OutOfRangeMode::Clamp),
            };
            let result: f16 = convert_float_to_float(1.0e10_f64, &c).unwrap();
            assert!(result.is_infinite());
        }

        #[test]
        fn test_f16_to_f32_lossless() {
            let c = FloatToFloatConfig {
                map_entries: vec![],
                rounding: RoundingMode::NearestEven,
                out_of_range: None,
            };
            let val = f16::from_f32(1.5);
            let result: f32 = convert_float_to_float(val, &c).unwrap();
            assert_eq!(result, 1.5_f32);
        }

        #[test]
        fn test_f16_nan_propagates() {
            let c = FloatToFloatConfig {
                map_entries: vec![],
                rounding: RoundingMode::NearestEven,
                out_of_range: None,
            };
            let result: f32 = convert_float_to_float(f16::NAN, &c).unwrap();
            assert!(result.is_nan());
        }

        #[test]
        fn test_i32_to_f16() {
            let c = IntToFloatConfig {
                map_entries: vec![],
                rounding: RoundingMode::NearestEven,
            };
            let result: f16 = convert_int_to_float(42_i32, &c).unwrap();
            assert_eq!(result, f16::from_f32(42.0));
        }

        #[test]
        fn test_f16_to_i32() {
            let c = f2i_cfg::<f16, i32>(vec![], RoundingMode::NearestEven, None);
            assert_eq!(
                convert_float_to_int(f16::from_f32(-7.0), &c).unwrap(),
                -7_i32
            );
        }

        #[test]
        fn test_f16_slice_conversion() {
            let src = [f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
            let mut dst = [0_u8; 3];
            let c = f2i_cfg::<f16, u8>(vec![], RoundingMode::NearestEven, None);
            convert_slice_float_to_int(&src, &mut dst, &c).unwrap();
            assert_eq!(dst, [1, 2, 3]);
        }

        #[test]
        fn test_f16_next_up_next_down() {
            let one = f16::from_f32(1.0);
            let up = one.next_up();
            let down = one.next_down();
            assert!(up > one);
            assert!(down < one);
            assert_eq!(up.next_down(), one);
            assert_eq!(down.next_up(), one);
        }
    }
}
