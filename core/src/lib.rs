//! zarr-cast-value-core: Pure Rust implementation of the cast_value codec's
//! per-element conversion logic.
//!
//! This crate is independent of Python/PyO3 and can be used by any Rust consumer
//! (e.g. zarrs, zarr-python bindings).

use std::ops::{Add, Sub};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during element conversion.
#[derive(Debug, Clone)]
pub enum CastError {
    /// A NaN or Infinity value was encountered when casting to an integer type.
    NanOrInf { value: f64 },
    /// A value is out of range for the target type and no out_of_range mode was set.
    OutOfRange { value: f64, lo: f64, hi: f64 },
}

impl std::fmt::Display for CastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CastError::NanOrInf { value } => {
                write!(f, "Cannot cast {value} to integer type without scalar_map")
            }
            CastError::OutOfRange { value, lo, hi } => {
                write!(
                    f,
                    "Value {value} out of range [{lo}, {hi}]. \
                     Set out_of_range='clamp' or out_of_range='wrap' to handle this."
                )
            }
        }
    }
}

impl std::error::Error for CastError {}

// ---------------------------------------------------------------------------
// Rounding modes
// ---------------------------------------------------------------------------

/// How to round floating-point values when casting to integer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub trait CastNum: Copy + PartialEq + PartialOrd + std::fmt::Debug {
    /// Convert to f64 for error reporting only (not in the conversion hot path).
    fn to_f64_lossy(self) -> f64;
}

/// An integer numeric type (i8..i64, u8..u64).
/// Marker trait — integers need no special operations beyond `CastNum`.
pub trait CastInt: CastNum {}

/// A floating-point numeric type (f32, f64).
/// Carries all float-specific operations: NaN/Inf checks, rounding, and
/// arithmetic needed for float→int wrap mode.
pub trait CastFloat: CastNum + Add<Output = Self> + Sub<Output = Self> {
    /// Returns true if the value is NaN.
    fn is_nan(self) -> bool;
    /// Returns true if the value is infinite.
    fn is_infinite(self) -> bool;
    /// Returns true if the value is finite (not NaN or infinite).
    fn is_finite(self) -> bool;
    /// Round to integer according to the given rounding mode.
    fn round(self, mode: RoundingMode) -> Self;
    /// Euclidean remainder (used in float→int wrap mode).
    fn rem_euclid(self, rhs: Self) -> Self;
    /// The value 1 in this type.
    fn one() -> Self;
    /// The smallest representable value greater than self (IEEE 754 nextUp).
    fn next_up(self) -> Self;
    /// The largest representable value less than self (IEEE 754 nextDown).
    fn next_down(self) -> Self;
    /// The absolute value.
    fn abs(self) -> Self;
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

/// Helper trait for constructing a value from f64 (needed for constructing
/// MapEntry values from f64 pairs at the Python boundary).
pub trait FromF64 {
    fn from_f64(val: f64) -> Self;
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
pub struct FloatToIntConfig<'a, Src, Dst> {
    pub map_entries: &'a [MapEntry<Src, Dst>],
    pub rounding: RoundingMode,
    pub out_of_range: Option<OutOfRangeMode>,
}

/// Configuration for int→int conversion.
pub struct IntToIntConfig<'a, Src, Dst> {
    pub map_entries: &'a [MapEntry<Src, Dst>],
    pub out_of_range: Option<OutOfRangeMode>,
}

/// Configuration for float→float conversion.
pub struct FloatToFloatConfig<'a, Src, Dst> {
    pub map_entries: &'a [MapEntry<Src, Dst>],
    pub rounding: RoundingMode,
    pub out_of_range: Option<OutOfRangeMode>,
}

/// Configuration for int→float conversion.
pub struct IntToFloatConfig<'a, Src, Dst> {
    pub map_entries: &'a [MapEntry<Src, Dst>],
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
                    value: val.to_f64_lossy(),
                    lo: lo.to_f64_lossy(),
                    hi: hi.to_f64_lossy(),
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
        if let Some(tgt) = apply_scalar_map_float(val, config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: reject NaN (NaN comparisons are all false, would slip through range check)
    if val.is_nan() {
        return Err(CastError::NanOrInf {
            value: val.to_f64_lossy(),
        });
    }

    // Step 3: round
    let val = val.round(config.rounding);

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
                    value: val.to_f64_lossy(),
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
                    value: val.to_f64_lossy(),
                    lo: lo.to_f64_lossy(),
                    hi: hi.to_f64_lossy(),
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
        if let Some(tgt) = apply_scalar_map_int(val, config.map_entries) {
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
                    value: val.to_f64_lossy(),
                    lo: lo.to_f64_lossy(),
                    hi: hi.to_f64_lossy(),
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
        if let Some(tgt) = apply_scalar_map_float(val, config.map_entries) {
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
        let val_f64 = val.to_f64_lossy();
        let result_f64 = result.to_f64_lossy();
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
                    let mid = (result.to_f64_lossy() + candidate.to_f64_lossy()) / 2.0;
                    if mid == val_f64 {
                        // It's a tie — pick the one farther from zero
                        if candidate.abs().to_f64_lossy() > result.abs().to_f64_lossy() {
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
        if let Some(tgt) = apply_scalar_map_int(val, config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: cast (Rust `as` uses nearest-even)
    let result = val.cast_into();

    // Step 3: if the cast was not exact and a non-nearest-even rounding mode
    // is requested, adjust the result.
    if config.rounding != RoundingMode::NearestEven {
        let val_f64 = val.to_f64_lossy();
        let result_f64 = result.to_f64_lossy();
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
                    let mid = (result.to_f64_lossy() + candidate.to_f64_lossy()) / 2.0;
                    if mid == val_f64 {
                        if candidate.abs().to_f64_lossy() > result.abs().to_f64_lossy() {
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
pub fn convert_slice_float_to_int<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &FloatToIntConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastFloat + CastInto<Dst>,
    Dst: CastInt,
{
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
pub fn convert_slice_float_to_float<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &FloatToFloatConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastFloat + CastInto<Dst>,
    Dst: CastFloat,
{
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
        $(
            impl CastNum for $ty {
                #[inline]
                fn to_f64_lossy(self) -> f64 { self as f64 }
            }
        )*
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
                fn is_nan(self) -> bool { <$ty>::is_nan(self) }
                #[inline]
                fn is_infinite(self) -> bool { <$ty>::is_infinite(self) }
                #[inline]
                fn is_finite(self) -> bool { <$ty>::is_finite(self) }
                #[inline]
                fn round(self, mode: RoundingMode) -> Self {
                    match mode {
                        RoundingMode::NearestEven => <$ty>::round_ties_even(self),
                        RoundingMode::TowardsZero => <$ty>::trunc(self),
                        RoundingMode::TowardsPositive => <$ty>::ceil(self),
                        RoundingMode::TowardsNegative => <$ty>::floor(self),
                        RoundingMode::NearestAway => {
                            <$ty>::copysign((<$ty>::abs(self) + 0.5).floor(), self)
                        }
                    }
                }
                #[inline]
                fn rem_euclid(self, rhs: Self) -> Self { <$ty>::rem_euclid(self, rhs) }
                #[inline]
                fn one() -> Self { 1.0 }
                #[inline]
                fn next_up(self) -> Self { <$ty>::next_up(self) }
                #[inline]
                fn next_down(self) -> Self { <$ty>::next_down(self) }
                #[inline]
                fn abs(self) -> Self { <$ty>::abs(self) }
            }
        )*
    };
}

impl_cast_float!(f32, f64);

// ---- FromF64 impls ----

macro_rules! impl_from_f64 {
    ($($ty:ty),*) => {
        $(
            impl FromF64 for $ty {
                #[inline]
                fn from_f64(val: f64) -> Self { val as $ty }
            }
        )*
    };
}

impl_from_f64!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build FloatToIntConfig
    fn f2i_cfg<Src: CastFloat, Dst: CastInt>(
        map: &[MapEntry<Src, Dst>],
        rounding: RoundingMode,
        oor: Option<OutOfRangeMode>,
    ) -> FloatToIntConfig<'_, Src, Dst> {
        FloatToIntConfig {
            map_entries: map,
            rounding,
            out_of_range: oor,
        }
    }

    // Helper to build IntToIntConfig
    fn i2i_cfg<Src: CastInt, Dst: CastInt>(
        map: &[MapEntry<Src, Dst>],
        oor: Option<OutOfRangeMode>,
    ) -> IntToIntConfig<'_, Src, Dst> {
        IntToIntConfig {
            map_entries: map,
            out_of_range: oor,
        }
    }

    #[test]
    fn test_float64_to_uint8_basic() {
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        assert_eq!(convert_float_to_int(42.0_f64, &c).unwrap(), 42_u8);
    }

    #[test]
    fn test_float64_to_uint8_rounding() {
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        // 2.5 rounds to 2 (banker's rounding)
        assert_eq!(convert_float_to_int(2.5_f64, &c).unwrap(), 2_u8);
        // 3.5 rounds to 4
        assert_eq!(convert_float_to_int(3.5_f64, &c).unwrap(), 4_u8);
    }

    #[test]
    fn test_float64_to_uint8_clamp() {
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Clamp));
        assert_eq!(convert_float_to_int(300.0_f64, &c).unwrap(), 255_u8);
        assert_eq!(convert_float_to_int(-10.0_f64, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_float64_to_int8_wrap() {
        let c = f2i_cfg::<f64, i8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Wrap));
        // 200 wraps: (200 - (-128)) % 256 + (-128) = 328 % 256 + (-128) = 72 + (-128) = -56
        assert_eq!(convert_float_to_int(200.0_f64, &c).unwrap(), -56_i8);
    }

    #[test]
    fn test_int32_to_int8_wrap() {
        let c = i2i_cfg::<i32, i8>(&[], Some(OutOfRangeMode::Wrap));
        // 200_i32 as i8 = -56 (bit truncation = modular wrap)
        assert_eq!(convert_int_to_int(200_i32, &c).unwrap(), -56_i8);
        assert_eq!(convert_int_to_int(-200_i32, &c).unwrap(), 56_i8);
    }

    #[test]
    fn test_int32_to_uint8_wrap() {
        let c = i2i_cfg::<i32, u8>(&[], Some(OutOfRangeMode::Wrap));
        assert_eq!(convert_int_to_int(300_i32, &c).unwrap(), 44_u8);
        assert_eq!(convert_int_to_int(-1_i32, &c).unwrap(), 255_u8);
    }

    #[test]
    fn test_out_of_range_error() {
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        assert!(convert_float_to_int(300.0_f64, &c).is_err());
    }

    #[test]
    fn test_nan_to_int_error() {
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        assert!(convert_float_to_int(f64::NAN, &c).is_err());
    }

    #[test]
    fn test_inf_to_int_error() {
        // Without out_of_range, Inf is caught by the range check (OutOfRange error)
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        assert!(convert_float_to_int(f64::INFINITY, &c).is_err());
        assert!(convert_float_to_int(f64::NEG_INFINITY, &c).is_err());
    }

    #[test]
    fn test_inf_to_int_clamp() {
        // With clamp, +Inf clamps to max, -Inf clamps to min
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Clamp));
        assert_eq!(convert_float_to_int(f64::INFINITY, &c).unwrap(), 255_u8);
        assert_eq!(convert_float_to_int(f64::NEG_INFINITY, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_inf_to_int_wrap_errors() {
        // With wrap, Inf is rejected (rem_euclid(Inf) is NaN)
        let c = f2i_cfg::<f64, i8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Wrap));
        assert!(convert_float_to_int(f64::INFINITY, &c).is_err());
        assert!(convert_float_to_int(f64::NEG_INFINITY, &c).is_err());
    }

    #[test]
    fn test_scalar_map_nan() {
        let entries = vec![MapEntry {
            src: f64::NAN,
            tgt: 0_u8,
        }];
        let c = f2i_cfg(&entries, RoundingMode::NearestEven, None);
        assert_eq!(convert_float_to_int(f64::NAN, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_scalar_map_nan_payloads() {
        // Any NaN should match a NaN map entry, regardless of payload bits.
        let entries = vec![MapEntry {
            src: f64::NAN,
            tgt: 42_u8,
        }];
        let c = f2i_cfg(&entries, RoundingMode::NearestEven, None);

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
        let c = f2i_cfg(&entries, RoundingMode::NearestEven, None);
        assert_eq!(convert_float_to_int(42.0_f64, &c).unwrap(), 99_u8);
        // Non-matching value goes through normal path
        assert_eq!(convert_float_to_int(10.0_f64, &c).unwrap(), 10_u8);
    }

    #[test]
    fn test_int32_to_uint8_range_check() {
        let c = i2i_cfg::<i32, u8>(&[], None);
        assert_eq!(convert_int_to_int(100_i32, &c).unwrap(), 100_u8);
        assert!(convert_int_to_int(300_i32, &c).is_err());
        assert!(convert_int_to_int(-1_i32, &c).is_err());
    }

    #[test]
    fn test_int32_to_float64() {
        let c = IntToFloatConfig::<i32, f64> {
            map_entries: &[],
            rounding: RoundingMode::NearestEven,
        };
        assert_eq!(convert_int_to_float(42_i32, &c).unwrap(), 42.0_f64);
    }

    #[test]
    fn test_convert_slice_basic() {
        let src = [1.0_f64, 2.0, 3.0, 4.0];
        let mut dst = [0_u8; 4];
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        convert_slice_float_to_int(&src, &mut dst, &c).unwrap();
        assert_eq!(dst, [1, 2, 3, 4]);
    }

    #[test]
    fn test_convert_slice_early_termination() {
        let src = [1.0_f64, 2.0, 300.0, 4.0];
        let mut dst = [0_u8; 4];
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        let result = convert_slice_float_to_int(&src, &mut dst, &c);
        assert!(result.is_err());
        // First two should have been written
        assert_eq!(dst[0], 1);
        assert_eq!(dst[1], 2);
    }

    #[test]
    fn test_all_rounding_modes() {
        // towards-zero
        let c = f2i_cfg::<f64, i8>(&[], RoundingMode::TowardsZero, None);
        assert_eq!(convert_float_to_int(2.7_f64, &c).unwrap(), 2_i8);
        assert_eq!(convert_float_to_int(-2.7_f64, &c).unwrap(), -2_i8);

        // towards-positive
        let c = f2i_cfg::<f64, i8>(&[], RoundingMode::TowardsPositive, None);
        assert_eq!(convert_float_to_int(2.1_f64, &c).unwrap(), 3_i8);
        assert_eq!(convert_float_to_int(-2.7_f64, &c).unwrap(), -2_i8);

        // towards-negative
        let c = f2i_cfg::<f64, i8>(&[], RoundingMode::TowardsNegative, None);
        assert_eq!(convert_float_to_int(2.7_f64, &c).unwrap(), 2_i8);
        assert_eq!(convert_float_to_int(-2.1_f64, &c).unwrap(), -3_i8);

        // nearest-away
        let c = f2i_cfg::<f64, i8>(&[], RoundingMode::NearestAway, None);
        assert_eq!(convert_float_to_int(2.5_f64, &c).unwrap(), 3_i8);
        assert_eq!(convert_float_to_int(-2.5_f64, &c).unwrap(), -3_i8);
    }

    #[test]
    fn test_int64_to_int32_clamp() {
        let c = i2i_cfg::<i64, i32>(&[], Some(OutOfRangeMode::Clamp));
        assert_eq!(convert_int_to_int(i64::MAX, &c).unwrap(), i32::MAX);
        assert_eq!(convert_int_to_int(i64::MIN, &c).unwrap(), i32::MIN);
    }

    #[test]
    fn test_float32_to_float64() {
        let c = FloatToFloatConfig::<f32, f64> {
            map_entries: &[],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        assert_eq!(
            convert_float_to_float(3.14_f32, &c).unwrap(),
            3.14_f32 as f64
        );
    }

    // -- float→float rounding tests --

    #[test]
    fn test_float64_to_float32_nearest_even() {
        // A value that is exactly between two f32 representable values
        // should round to even (the default `as` behavior).
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: &[],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        // 1.0 is exactly representable — no rounding
        assert_eq!(convert_float_to_float(1.0_f64, &c).unwrap(), 1.0_f32);
    }

    #[test]
    fn test_float64_to_float32_towards_zero() {
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: &[],
            rounding: RoundingMode::TowardsZero,
            out_of_range: None,
        };
        // Pick a value not exactly representable in f32.
        // f32 has ~7 decimal digits of precision.
        let val: f64 = 1.0000001_f64; // slightly above 1.0
        let result = convert_float_to_float(val, &c).unwrap();
        // Towards zero: result should be <= val (since val > 0)
        assert!(
            result.to_f64_lossy() <= val,
            "towards-zero: {result} > {val}"
        );
        assert!(result.to_f64_lossy() > 0.0);

        // Negative: towards zero means closer to 0 (larger, i.e. less negative)
        let val_neg: f64 = -1.0000001_f64;
        let result_neg = convert_float_to_float(val_neg, &c).unwrap();
        assert!(
            result_neg.to_f64_lossy() >= val_neg,
            "towards-zero negative: {result_neg} < {val_neg}"
        );
    }

    #[test]
    fn test_float64_to_float32_towards_positive() {
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: &[],
            rounding: RoundingMode::TowardsPositive,
            out_of_range: None,
        };
        let val: f64 = 1.0000001_f64;
        let result = convert_float_to_float(val, &c).unwrap();
        // Towards positive: result should be >= val
        assert!(
            result.to_f64_lossy() >= val,
            "towards-positive: {result} < {val}"
        );
    }

    #[test]
    fn test_float64_to_float32_towards_negative() {
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: &[],
            rounding: RoundingMode::TowardsNegative,
            out_of_range: None,
        };
        let val: f64 = 1.0000001_f64;
        let result = convert_float_to_float(val, &c).unwrap();
        // Towards negative: result should be <= val
        assert!(
            result.to_f64_lossy() <= val,
            "towards-negative: {result} > {val}"
        );
    }

    // -- float→float out-of-range tests --

    #[test]
    fn test_float64_to_float32_overflow_error() {
        // A value exceeding f32::MAX should error when no out_of_range is set
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: &[],
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
            map_entries: &[],
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
            map_entries: &[],
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
            map_entries: &[],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        assert!(convert_float_to_float(f64::NAN, &c).unwrap().is_nan());
    }

    #[test]
    fn test_float64_to_float32_signed_zero() {
        // Signed zero must be preserved
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: &[],
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
            map_entries: &[],
            rounding: RoundingMode::NearestEven,
        };
        let result_ne = convert_int_to_float(val, &c_ne).unwrap();
        assert_eq!(result_ne, val as f32); // same as Rust `as`

        // towards-positive: result should be >= val
        let c_pos = IntToFloatConfig::<i64, f32> {
            map_entries: &[],
            rounding: RoundingMode::TowardsPositive,
        };
        let result_pos = convert_int_to_float(val, &c_pos).unwrap();
        assert!(
            result_pos.to_f64_lossy() >= val as f64,
            "towards-positive: {result_pos} < {val}"
        );

        // towards-zero: result should be <= val (since val > 0)
        let c_tz = IntToFloatConfig::<i64, f32> {
            map_entries: &[],
            rounding: RoundingMode::TowardsZero,
        };
        let result_tz = convert_int_to_float(val, &c_tz).unwrap();
        assert!(
            result_tz.to_f64_lossy() <= val as f64,
            "towards-zero: {result_tz} > {val}"
        );
    }
}
