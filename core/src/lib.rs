//! zarr-cast-value-core: Pure Rust implementation of the cast_value codec's
//! per-element conversion logic.
//!
//! This crate is independent of Python/PyO3 and can be used by any Rust consumer
//! (e.g. zarrs, zarr-python bindings).

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
                write!(
                    f,
                    "Cannot cast {value} to integer type without scalar_map"
                )
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

impl RoundingMode {
    pub fn from_str(s: &str) -> Result<Self, String> {
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

impl OutOfRangeMode {
    pub fn from_str(s: &str) -> Result<Self, String> {
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
// Conversion configuration
// ---------------------------------------------------------------------------

/// All parameters needed for a convert_slice call, bundled together.
pub struct ConvertConfig<'a, Src, Dst> {
    pub map_entries: &'a [MapEntry<Src, Dst>],
    pub rounding: RoundingMode,
    pub out_of_range: Option<OutOfRangeMode>,
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// A numeric type that can participate in cast_value conversions.
pub trait CastType: Copy + PartialEq + PartialOrd + std::fmt::Debug {
    /// Whether this is a floating-point type (has NaN/Inf).
    const IS_FLOAT: bool;
    /// Whether this is an integer type (needs range checking as target).
    const IS_INTEGER: bool;

    /// Returns true if the value is NaN.
    fn is_nan(self) -> bool;
    /// Returns true if the value is infinite.
    fn is_infinite(self) -> bool;
    /// Convert to f64 for error reporting only (not in the conversion hot path).
    fn to_f64_lossy(self) -> f64;

    /// Euclidean remainder. Only meaningful for float types (used in float→int
    /// wrap mode). Integer types may panic — the call is guarded by IS_FLOAT.
    fn rem_euclid(self, _rhs: Self) -> Self {
        unreachable!("rem_euclid called on integer type")
    }
    /// Subtraction in native type. Only meaningful for float types.
    fn sub(self, _rhs: Self) -> Self {
        unreachable!("sub called on integer type")
    }
    /// Addition in native type. Only meaningful for float types.
    fn add(self, _rhs: Self) -> Self {
        unreachable!("add called on integer type")
    }
    /// The value 1 in this type. Only meaningful for float types.
    fn one() -> Self {
        unreachable!("one called on integer type")
    }
}

/// Rounding support. Float types implement actual rounding;
/// integer types implement this as identity (no-op).
pub trait CastRound: CastType {
    fn round(self, mode: RoundingMode) -> Self;
}

/// Conversion from Src to Dst. Implemented per (Src, Dst) pair.
pub trait CastInto<Dst: CastType>: CastType {
    /// The minimum Dst value, represented in Src's type.
    fn dst_min() -> Self;
    /// The maximum Dst value, represented in Src's type.
    fn dst_max() -> Self;
    /// Convert self to Dst. Caller must ensure value is in range.
    fn cast_into(self) -> Dst;
}

// ---------------------------------------------------------------------------
// convert_element: Layer 1
// ---------------------------------------------------------------------------

/// Apply scalar_map lookup. Returns Some(tgt) on match, None otherwise.
#[inline]
fn apply_scalar_map<Src: CastType, Dst: CastType>(
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

/// Convert a single element from Src to Dst, applying the full pipeline:
/// scalar_map → NaN/Inf check → round → range check → cast.
#[inline]
pub fn convert_element<Src, Dst>(
    val: Src,
    config: &ConvertConfig<Src, Dst>,
) -> Result<Dst, CastError>
where
    Src: CastType + CastRound + CastInto<Dst>,
    Dst: CastType,
{
    // Step 1: scalar_map lookup
    if !config.map_entries.is_empty() {
        if let Some(tgt) = apply_scalar_map(val, config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: reject NaN for float→int
    // NaN must be caught here because NaN comparisons are all false,
    // so NaN would slip through the range check in step 4.
    // Infinity is handled by the range check — it naturally compares
    // as > hi or < lo, so clamp/error work correctly. Wrap mode
    // rejects infinity separately since rem_euclid(Inf) is NaN.
    if Src::IS_FLOAT && Dst::IS_INTEGER && val.is_nan() {
        return Err(CastError::NanOrInf {
            value: val.to_f64_lossy(),
        });
    }

    // Step 3: round (float→int only)
    let val = if Src::IS_FLOAT && Dst::IS_INTEGER {
        val.round(config.rounding)
    } else {
        val
    };

    // Step 4: range check (integer targets only)
    if Dst::IS_INTEGER {
        let lo = Src::dst_min();
        let hi = Src::dst_max();
        match config.out_of_range {
            Some(OutOfRangeMode::Clamp) => {
                // Clamp then cast
                let clamped = if val < lo {
                    lo
                } else if val > hi {
                    hi
                } else {
                    val
                };
                return Ok(clamped.cast_into());
            }
            Some(OutOfRangeMode::Wrap) => {
                if Src::IS_FLOAT {
                    // Float→int wrap: Inf can't be wrapped.
                    if val.is_infinite() {
                        return Err(CastError::NanOrInf {
                            value: val.to_f64_lossy(),
                        });
                    }
                    if val < lo || val > hi {
                        // The value is an integer-valued float (already rounded).
                        // Wrap using native float arithmetic — no f64 promotion.
                        let range = hi.sub(lo).add(Src::one());
                        let wrapped = val.sub(lo).rem_euclid(range).add(lo);
                        return Ok(wrapped.cast_into());
                    }
                } else if val < lo || val > hi {
                    // Int→int wrap: Rust's `as` cast truncates to the target
                    // width, which is exactly modular arithmetic. No f64 needed.
                    return Ok(val.cast_into());
                }
            }
            None => {
                if val < lo || val > hi {
                    return Err(CastError::OutOfRange {
                        value: val.to_f64_lossy(),
                        lo: lo.to_f64_lossy(),
                        hi: hi.to_f64_lossy(),
                    });
                }
            }
        }
    }

    // Step 5: cast
    Ok(val.cast_into())
}

// ---------------------------------------------------------------------------
// convert_slice: Layer 2
// ---------------------------------------------------------------------------

/// Convert a single element without scalar_map lookup.
/// This is the hot path when no map entries are present.
#[inline]
fn convert_element_no_map<Src, Dst>(
    val: Src,
    config: &ConvertConfig<Src, Dst>,
) -> Result<Dst, CastError>
where
    Src: CastType + CastRound + CastInto<Dst>,
    Dst: CastType,
{
    // Skip step 1 entirely — go straight to NaN check
    if Src::IS_FLOAT && Dst::IS_INTEGER && val.is_nan() {
        return Err(CastError::NanOrInf {
            value: val.to_f64_lossy(),
        });
    }
    let val = if Src::IS_FLOAT && Dst::IS_INTEGER {
        val.round(config.rounding)
    } else {
        val
    };
    if Dst::IS_INTEGER {
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
                return Ok(clamped.cast_into());
            }
            Some(OutOfRangeMode::Wrap) => {
                if Src::IS_FLOAT {
                    if val.is_infinite() {
                        return Err(CastError::NanOrInf {
                            value: val.to_f64_lossy(),
                        });
                    }
                    if val < lo || val > hi {
                        let range = hi.sub(lo).add(Src::one());
                        let wrapped = val.sub(lo).rem_euclid(range).add(lo);
                        return Ok(wrapped.cast_into());
                    }
                } else if val < lo || val > hi {
                    return Ok(val.cast_into());
                }
            }
            None => {
                if val < lo || val > hi {
                    return Err(CastError::OutOfRange {
                        value: val.to_f64_lossy(),
                        lo: lo.to_f64_lossy(),
                        hi: hi.to_f64_lossy(),
                    });
                }
            }
        }
    }
    Ok(val.cast_into())
}

/// Convert a slice of Src values to Dst values. Returns early on first error.
///
/// When no scalar_map entries are present, uses a tighter inner loop that
/// skips the map lookup entirely, allowing better optimization.
pub fn convert_slice<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &ConvertConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastType + CastRound + CastInto<Dst>,
    Dst: CastType,
{
    if config.map_entries.is_empty() {
        // Hot path: no scalar_map — skip map lookup per element
        for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
            *out_slot = convert_element_no_map(*in_val, config)?;
        }
    } else {
        // Map path: includes scalar_map lookup per element
        for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
            *out_slot = convert_element(*in_val, config)?;
        }
    }
    Ok(())
}

/// Helper trait for constructing a value from f64 (needed for float→int wrap
/// mode and for constructing MapEntry values from f64 pairs at the Python boundary).
pub trait FromF64 {
    fn from_f64(val: f64) -> Self;
}

// ---------------------------------------------------------------------------
// Trait implementations for primitive types
// ---------------------------------------------------------------------------

// ---- CastType impls ----

macro_rules! impl_cast_type_int {
    ($($ty:ty),*) => {
        $(
            impl CastType for $ty {
                const IS_FLOAT: bool = false;
                const IS_INTEGER: bool = true;
                #[inline] fn is_nan(self) -> bool { false }
                #[inline] fn is_infinite(self) -> bool { false }
                #[inline] fn to_f64_lossy(self) -> f64 { self as f64 }
            }
        )*
    };
}

impl_cast_type_int!(i8, i16, i32, i64, u8, u16, u32, u64);

macro_rules! impl_cast_type_float {
    ($($ty:ty),*) => {
        $(
            impl CastType for $ty {
                const IS_FLOAT: bool = true;
                const IS_INTEGER: bool = false;
                #[inline] fn is_nan(self) -> bool { <$ty>::is_nan(self) }
                #[inline] fn is_infinite(self) -> bool { <$ty>::is_infinite(self) }
                #[inline] fn to_f64_lossy(self) -> f64 { self as f64 }
                #[inline] fn rem_euclid(self, rhs: Self) -> Self { <$ty>::rem_euclid(self, rhs) }
                #[inline] fn sub(self, rhs: Self) -> Self { self - rhs }
                #[inline] fn add(self, rhs: Self) -> Self { self + rhs }
                #[inline] fn one() -> Self { 1.0 }
            }
        )*
    };
}

impl_cast_type_float!(f32, f64);

// ---- CastRound impls ----

macro_rules! impl_cast_round_int {
    ($($ty:ty),*) => {
        $(
            impl CastRound for $ty {
                #[inline]
                fn round(self, _mode: RoundingMode) -> Self { self }
            }
        )*
    };
}

impl_cast_round_int!(i8, i16, i32, i64, u8, u16, u32, u64);

macro_rules! impl_cast_round_float {
    ($($ty:ty),*) => {
        $(
            impl CastRound for $ty {
                #[inline]
                fn round(self, mode: RoundingMode) -> Self {
                    match mode {
                        RoundingMode::NearestEven => {
                            // round_ties_even is available on f32/f64 since Rust 1.77
                            let v = self as f64;
                            v.round_ties_even() as $ty
                        }
                        RoundingMode::TowardsZero => {
                            let v = self as f64;
                            v.trunc() as $ty
                        }
                        RoundingMode::TowardsPositive => {
                            let v = self as f64;
                            v.ceil() as $ty
                        }
                        RoundingMode::TowardsNegative => {
                            let v = self as f64;
                            v.floor() as $ty
                        }
                        RoundingMode::NearestAway => {
                            let v = self as f64;
                            (v.signum() * (v.abs() + 0.5).floor()) as $ty
                        }
                    }
                }
            }
        )*
    };
}

impl_cast_round_float!(f32, f64);

// ---- PartialOrd for range checking ----
// All primitives already implement PartialOrd, so the < > comparisons in
// convert_element work naturally.

// ---- FromF64 impls (for wrap mode) ----

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
// For any→float, range checking is skipped (IS_INTEGER is false for float Dst),
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

// -- any→float (range check not applied, but impl needed) --
macro_rules! impl_to_float {
    ($src:ty => $dst:ty) => {
        impl_cast_into!(
            $src => $dst,
            min: 0 as $src,  // unused, IS_INTEGER is false for float Dst
            max: 0 as $src
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

// all sources → float targets
macro_rules! impl_all_to_float {
    ($src:ty => $($dst:ty),*) => {
        $( impl_to_float!($src => $dst); )*
    };
}

impl_all_to_float!(i8 => f32, f64);
impl_all_to_float!(i16 => f32, f64);
impl_all_to_float!(i32 => f32, f64);
impl_all_to_float!(i64 => f32, f64);
impl_all_to_float!(u8 => f32, f64);
impl_all_to_float!(u16 => f32, f64);
impl_all_to_float!(u32 => f32, f64);
impl_all_to_float!(u64 => f32, f64);
impl_all_to_float!(f32 => f32, f64);
impl_all_to_float!(f64 => f32, f64);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg<Src: CastType, Dst: CastType>(
        map: &[MapEntry<Src, Dst>],
        rounding: RoundingMode,
        oor: Option<OutOfRangeMode>,
    ) -> ConvertConfig<'_, Src, Dst> {
        ConvertConfig {
            map_entries: map,
            rounding,
            out_of_range: oor,
        }
    }

    #[test]
    fn test_float64_to_uint8_basic() {
        let c = cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        assert_eq!(convert_element(42.0_f64, &c).unwrap(), 42_u8);
    }

    #[test]
    fn test_float64_to_uint8_rounding() {
        let c = cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        // 2.5 rounds to 2 (banker's rounding)
        assert_eq!(convert_element(2.5_f64, &c).unwrap(), 2_u8);
        // 3.5 rounds to 4
        assert_eq!(convert_element(3.5_f64, &c).unwrap(), 4_u8);
    }

    #[test]
    fn test_float64_to_uint8_clamp() {
        let c = cfg::<f64, u8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Clamp));
        assert_eq!(convert_element(300.0_f64, &c).unwrap(), 255_u8);
        assert_eq!(convert_element(-10.0_f64, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_float64_to_int8_wrap() {
        let c = cfg::<f64, i8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Wrap));
        // 200 wraps: (200 - (-128)) % 256 + (-128) = 328 % 256 + (-128) = 72 + (-128) = -56
        assert_eq!(convert_element(200.0_f64, &c).unwrap(), -56_i8);
    }

    #[test]
    fn test_int32_to_int8_wrap() {
        let c = cfg::<i32, i8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Wrap));
        // 200_i32 as i8 = -56 (bit truncation = modular wrap)
        assert_eq!(convert_element(200_i32, &c).unwrap(), -56_i8);
        assert_eq!(convert_element(-200_i32, &c).unwrap(), 56_i8);
    }

    #[test]
    fn test_int32_to_uint8_wrap() {
        let c = cfg::<i32, u8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Wrap));
        assert_eq!(convert_element(300_i32, &c).unwrap(), 44_u8);
        assert_eq!(convert_element(-1_i32, &c).unwrap(), 255_u8);
    }

    #[test]
    fn test_out_of_range_error() {
        let c = cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        assert!(convert_element(300.0_f64, &c).is_err());
    }

    #[test]
    fn test_nan_to_int_error() {
        let c = cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        assert!(convert_element(f64::NAN, &c).is_err());
    }

    #[test]
    fn test_inf_to_int_error() {
        // Without out_of_range, Inf is caught by the range check (OutOfRange error)
        let c = cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        assert!(convert_element(f64::INFINITY, &c).is_err());
        assert!(convert_element(f64::NEG_INFINITY, &c).is_err());
    }

    #[test]
    fn test_inf_to_int_clamp() {
        // With clamp, +Inf clamps to max, -Inf clamps to min
        let c = cfg::<f64, u8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Clamp));
        assert_eq!(convert_element(f64::INFINITY, &c).unwrap(), 255_u8);
        assert_eq!(convert_element(f64::NEG_INFINITY, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_inf_to_int_wrap_errors() {
        // With wrap, Inf is rejected (rem_euclid(Inf) is NaN)
        let c = cfg::<f64, i8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Wrap));
        assert!(convert_element(f64::INFINITY, &c).is_err());
        assert!(convert_element(f64::NEG_INFINITY, &c).is_err());
    }

    #[test]
    fn test_scalar_map_nan() {
        let entries = vec![MapEntry {
            src: f64::NAN,
            tgt: 0_u8,
        }];
        let c = cfg(&entries, RoundingMode::NearestEven, None);
        assert_eq!(convert_element(f64::NAN, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_scalar_map_exact() {
        let entries = vec![MapEntry {
            src: 42.0_f64,
            tgt: 99_u8,
        }];
        let c = cfg(&entries, RoundingMode::NearestEven, None);
        assert_eq!(convert_element(42.0_f64, &c).unwrap(), 99_u8);
        // Non-matching value goes through normal path
        assert_eq!(convert_element(10.0_f64, &c).unwrap(), 10_u8);
    }

    #[test]
    fn test_int32_to_uint8_range_check() {
        let c = cfg::<i32, u8>(&[], RoundingMode::NearestEven, None);
        assert_eq!(convert_element(100_i32, &c).unwrap(), 100_u8);
        assert!(convert_element(300_i32, &c).is_err());
        assert!(convert_element(-1_i32, &c).is_err());
    }

    #[test]
    fn test_int32_to_float64() {
        let c = cfg::<i32, f64>(&[], RoundingMode::NearestEven, None);
        assert_eq!(convert_element(42_i32, &c).unwrap(), 42.0_f64);
    }

    #[test]
    fn test_convert_slice_basic() {
        let src = [1.0_f64, 2.0, 3.0, 4.0];
        let mut dst = [0_u8; 4];
        let c = cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        convert_slice(&src, &mut dst, &c).unwrap();
        assert_eq!(dst, [1, 2, 3, 4]);
    }

    #[test]
    fn test_convert_slice_early_termination() {
        let src = [1.0_f64, 2.0, 300.0, 4.0];
        let mut dst = [0_u8; 4];
        let c = cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        let result = convert_slice(&src, &mut dst, &c);
        assert!(result.is_err());
        // First two should have been written
        assert_eq!(dst[0], 1);
        assert_eq!(dst[1], 2);
    }

    #[test]
    fn test_all_rounding_modes() {
        // towards-zero
        let c = cfg::<f64, i8>(&[], RoundingMode::TowardsZero, None);
        assert_eq!(convert_element(2.7_f64, &c).unwrap(), 2_i8);
        assert_eq!(convert_element(-2.7_f64, &c).unwrap(), -2_i8);

        // towards-positive
        let c = cfg::<f64, i8>(&[], RoundingMode::TowardsPositive, None);
        assert_eq!(convert_element(2.1_f64, &c).unwrap(), 3_i8);
        assert_eq!(convert_element(-2.7_f64, &c).unwrap(), -2_i8);

        // towards-negative
        let c = cfg::<f64, i8>(&[], RoundingMode::TowardsNegative, None);
        assert_eq!(convert_element(2.7_f64, &c).unwrap(), 2_i8);
        assert_eq!(convert_element(-2.1_f64, &c).unwrap(), -3_i8);

        // nearest-away
        let c = cfg::<f64, i8>(&[], RoundingMode::NearestAway, None);
        assert_eq!(convert_element(2.5_f64, &c).unwrap(), 3_i8);
        assert_eq!(convert_element(-2.5_f64, &c).unwrap(), -3_i8);
    }

    #[test]
    fn test_int64_to_int32_clamp() {
        let c = cfg::<i64, i32>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Clamp));
        assert_eq!(
            convert_element(i64::MAX, &c).unwrap(),
            i32::MAX
        );
        assert_eq!(
            convert_element(i64::MIN, &c).unwrap(),
            i32::MIN
        );
    }

    #[test]
    fn test_float32_to_float64() {
        let c = cfg::<f32, f64>(&[], RoundingMode::NearestEven, None);
        assert_eq!(convert_element(3.14_f32, &c).unwrap(), 3.14_f32 as f64);
    }
}
