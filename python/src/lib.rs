//! cast-value-rs: PyO3 bindings for the cast_value codec.
//!
//! Exposes `cast_array` and `cast_array_into` to Python, dispatching on
//! numpy dtype pairs to monomorphized conversion calls from the core crate.

use half::f16;
use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyMapping;
use zarr_cast_value::{
    CastError, CastFloat, CastInt, CastInto, FloatToFloatConfig, FloatToIntConfig,
    IntToFloatConfig, IntToIntConfig, MapEntry, OutOfRangeMode, RoundingMode,
};

// ---------------------------------------------------------------------------
// Extraction trait for scalar map values
// ---------------------------------------------------------------------------

/// Trait for extracting a typed value from a Python object.
/// This exists because `half::f16` does not implement `pyo3::FromPyObject`.
/// We implement it for each concrete numeric type used in dispatch.
trait ExtractFromPy: Sized {
    fn extract_from_py(ob: &Bound<'_, PyAny>) -> PyResult<Self>;
}

macro_rules! impl_extract_via_pyo3 {
    ($($ty:ty),*) => {
        $(
            impl ExtractFromPy for $ty {
                #[inline]
                fn extract_from_py(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
                    ob.extract()
                }
            }
        )*
    };
}

impl_extract_via_pyo3!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

/// f16: extract as f32, then convert. Python has no native float16.
impl ExtractFromPy for f16 {
    #[inline]
    fn extract_from_py(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let val: f32 = ob.extract()?;
        Ok(f16::from_f32(val))
    }
}

// ---------------------------------------------------------------------------
// "Parse, don't validate" wrapper types for Python arguments
// ---------------------------------------------------------------------------

/// Wrapper around `RoundingMode` that implements `FromPyObject` for automatic
/// parsing from Python string arguments.
struct PyRoundingMode(RoundingMode);

impl<'py> FromPyObject<'py> for PyRoundingMode {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let s: &str = ob.extract()?;
        s.parse::<RoundingMode>()
            .map(PyRoundingMode)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

/// Wrapper around `OutOfRangeMode` that implements `FromPyObject` for
/// automatic parsing from Python string arguments.
struct PyOutOfRangeMode(OutOfRangeMode);

impl<'py> FromPyObject<'py> for PyOutOfRangeMode {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let s: &str = ob.extract()?;
        s.parse::<OutOfRangeMode>()
            .map(PyOutOfRangeMode)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cast_error_to_pyerr(e: CastError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Parse scalar_map_entries from Python list[list[float]] into typed MapEntry
// ---------------------------------------------------------------------------

/// Parse scalar_map_entries from Python into typed `MapEntry` values.
///
/// Values are extracted directly as the concrete `Src`/`Dst` types via
/// pyo3's `FromPyObject`, avoiding any intermediate f64 conversion that
/// could lose precision for large integers.
///
/// Accepts:
/// - `dict` — e.g. `{float('nan'): 0, float('inf'): 255}`
/// - Any iterable of 2-element sequences — e.g. `[(nan, 0)]`,
///   `[[nan, 0]]`, tuples, or a generator.
fn parse_map_entries<'py, Src, Dst>(
    entries: Option<&Bound<'py, PyAny>>,
    src_dtype_name: &str,
    dst_dtype_name: &str,
) -> PyResult<Vec<MapEntry<Src, Dst>>>
where
    Src: zarr_cast_value::CastNum + ExtractFromPy,
    Dst: zarr_cast_value::CastNum + ExtractFromPy,
{
    let Some(obj) = entries else {
        return Ok(Vec::new());
    };

    // Helper closures that wrap extraction errors with scalar_map context.
    let extract_src = |ob: &Bound<'py, PyAny>| -> PyResult<Src> {
        Src::extract_from_py(ob).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "scalar_map source value {}: expected a value \
                 convertible to source dtype {src_dtype_name}",
                ob.repr().unwrap_or_else(|_| ob.str().unwrap()),
            ))
        })
    };
    let extract_tgt = |ob: &Bound<'py, PyAny>| -> PyResult<Dst> {
        Dst::extract_from_py(ob).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "scalar_map target value {}: expected a value \
                 convertible to target dtype {dst_dtype_name}",
                ob.repr().unwrap_or_else(|_| ob.str().unwrap()),
            ))
        })
    };

    // Mappings yield keys only when iterated, so extract key-value pairs via .items().
    if let Ok(dict) = obj.downcast::<PyMapping>() {
        // PyMapping.len() returns Result even though the protocol guarantees __len__.
        let mapping_len = dict.len().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Failed to get length of scalar_map dict.")
        })?;
        let mut result = Vec::with_capacity(mapping_len);
        for kv in dict.items()?.iter() {
            let (key, val) = kv.extract()?;
            let src = extract_src(&key)?;
            let tgt = extract_tgt(&val)?;
            result.push(MapEntry { src, tgt });
        }
        return Ok(result);
    }

    // For anything else, iterate and extract each item as a 2-element
    // sequence. We index with get_item rather than extracting as a tuple,
    // so that lists, tuples, and other sequences all work.
    let iter = obj.try_iter().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "scalar_map_entries must be a dict or an iterable of (source, target) pairs",
        )
    })?;
    let mut result = Vec::new();
    for item in iter {
        let item = item?;
        let len: usize = item.len().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "Each scalar_map entry must be a (source, target) pair",
            )
        })?;
        if len != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Each scalar_map entry must be a (source, target) pair",
            ));
        }
        let src = extract_src(&item.get_item(0)?)?;
        let tgt = extract_tgt(&item.get_item(1)?)?;
        result.push(MapEntry { src, tgt });
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Dtype name mapping: numpy dtype name → Zarr V3 dtype name
// ---------------------------------------------------------------------------

/// Map a numpy dtype name to the corresponding Zarr V3 data type name.
///
/// For core spec types the names are identical, but this mapping exists so
/// that future extension types (e.g. ml_dtypes' `bfloat16`, `float8_e4m3fn`)
/// can be mapped to their Zarr V3 registered names.
fn numpy_name_to_zarr_dtype(name: &str) -> PyResult<&'static str> {
    match name {
        "int8" => Ok("int8"),
        "int16" => Ok("int16"),
        "int32" => Ok("int32"),
        "int64" => Ok("int64"),
        "uint8" => Ok("uint8"),
        "uint16" => Ok("uint16"),
        "uint32" => Ok("uint32"),
        "uint64" => Ok("uint64"),
        "float16" => Ok("float16"),
        "float32" => Ok("float32"),
        "float64" => Ok("float64"),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Unsupported numpy dtype: {name}",
        ))),
    }
}

/// Extract the Zarr V3 dtype name from a numpy array's dtype.
fn array_dtype_key(arr: &Bound<'_, PyUntypedArray>) -> PyResult<&'static str> {
    let dtype = arr.getattr("dtype")?;
    let name: String = dtype.getattr("name")?.extract()?;
    numpy_name_to_zarr_dtype(&name)
}

// ---------------------------------------------------------------------------
// Per-path conversion helpers (avoid duplicating the numpy I/O boilerplate)
// ---------------------------------------------------------------------------

/// Perform a float→int conversion on numpy arrays.
fn do_float_to_int_alloc<'py, Src, Dst>(
    py: Python<'py>,
    arr: &Bound<'py, PyUntypedArray>,
    rounding: RoundingMode,
    oor: Option<OutOfRangeMode>,
    map_entries_py: Option<&Bound<'py, PyAny>>,
    src_dtype: &str,
    tgt_dtype: &str,
) -> PyResult<PyObject>
where
    Src: CastFloat + CastInto<Dst> + ExtractFromPy + numpy::Element + 'static,
    Dst: CastInt + ExtractFromPy + numpy::Element + 'static,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.downcast::<PyArrayDyn<Src>>()?.readonly();
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = FloatToIntConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py, src_dtype, tgt_dtype)?,
        rounding,
        out_of_range: oor,
    };
    let shape: Vec<usize> = arr.shape().to_vec();
    let output = PyArrayDyn::<Dst>::zeros(py, &shape[..], false);
    {
        // SAFETY: We just created `output` and hold the GIL, so no other
        // code can alias this array. The mutable reference is valid for
        // the duration of this block.
        let mut output_rw = unsafe { output.as_array_mut() };
        let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Failed to get mutable slice from output array")
        })?;
        zarr_cast_value::convert_slice_float_to_int(src_slice, dst_slice, &config)
            .map_err(cast_error_to_pyerr)?;
    }
    Ok(output.into_pyobject(py)?.into_any().unbind())
}

/// Perform an int→int conversion on numpy arrays.
fn do_int_to_int_alloc<'py, Src, Dst>(
    py: Python<'py>,
    arr: &Bound<'py, PyUntypedArray>,
    oor: Option<OutOfRangeMode>,
    map_entries_py: Option<&Bound<'py, PyAny>>,
    src_dtype: &str,
    tgt_dtype: &str,
) -> PyResult<PyObject>
where
    Src: CastInt + CastInto<Dst> + ExtractFromPy + numpy::Element,
    Dst: CastInt + ExtractFromPy + numpy::Element,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.downcast::<PyArrayDyn<Src>>()?.readonly();
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = IntToIntConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py, src_dtype, tgt_dtype)?,
        out_of_range: oor,
    };
    let shape: Vec<usize> = arr.shape().to_vec();
    let output = PyArrayDyn::<Dst>::zeros(py, &shape[..], false);
    {
        // SAFETY: We just created `output` and hold the GIL, so no other
        // code can alias this array.
        let mut output_rw = unsafe { output.as_array_mut() };
        let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Failed to get mutable slice from output array")
        })?;
        zarr_cast_value::convert_slice_int_to_int(src_slice, dst_slice, &config)
            .map_err(cast_error_to_pyerr)?;
    }
    Ok(output.into_pyobject(py)?.into_any().unbind())
}

/// Perform a float→float conversion on numpy arrays.
fn do_float_to_float_alloc<'py, Src, Dst>(
    py: Python<'py>,
    arr: &Bound<'py, PyUntypedArray>,
    rounding: RoundingMode,
    oor: Option<OutOfRangeMode>,
    map_entries_py: Option<&Bound<'py, PyAny>>,
    src_dtype: &str,
    tgt_dtype: &str,
) -> PyResult<PyObject>
where
    Src: CastFloat + CastInto<Dst> + ExtractFromPy + numpy::Element + 'static,
    Dst: CastFloat + ExtractFromPy + numpy::Element + 'static,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.downcast::<PyArrayDyn<Src>>()?.readonly();
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = FloatToFloatConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py, src_dtype, tgt_dtype)?,
        rounding,
        out_of_range: oor,
    };
    let shape: Vec<usize> = arr.shape().to_vec();
    let output = PyArrayDyn::<Dst>::zeros(py, &shape[..], false);
    {
        // SAFETY: We just created `output` and hold the GIL, so no other
        // code can alias this array.
        let mut output_rw = unsafe { output.as_array_mut() };
        let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Failed to get mutable slice from output array")
        })?;
        zarr_cast_value::convert_slice_float_to_float(src_slice, dst_slice, &config)
            .map_err(cast_error_to_pyerr)?;
    }
    Ok(output.into_pyobject(py)?.into_any().unbind())
}

/// Perform an int→float conversion on numpy arrays.
fn do_int_to_float_alloc<'py, Src, Dst>(
    py: Python<'py>,
    arr: &Bound<'py, PyUntypedArray>,
    rounding: RoundingMode,
    map_entries_py: Option<&Bound<'py, PyAny>>,
    src_dtype: &str,
    tgt_dtype: &str,
) -> PyResult<PyObject>
where
    Src: CastInt + CastInto<Dst> + ExtractFromPy + numpy::Element,
    Dst: CastFloat + ExtractFromPy + numpy::Element,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.downcast::<PyArrayDyn<Src>>()?.readonly();
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = IntToFloatConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py, src_dtype, tgt_dtype)?,
        rounding,
    };
    let shape: Vec<usize> = arr.shape().to_vec();
    let output = PyArrayDyn::<Dst>::zeros(py, &shape[..], false);
    {
        // SAFETY: We just created `output` and hold the GIL, so no other
        // code can alias this array.
        let mut output_rw = unsafe { output.as_array_mut() };
        let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Failed to get mutable slice from output array")
        })?;
        zarr_cast_value::convert_slice_int_to_float(src_slice, dst_slice, &config)
            .map_err(cast_error_to_pyerr)?;
    }
    Ok(output.into_pyobject(py)?.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Per-path conversion helpers: into variant (pre-allocated output)
// ---------------------------------------------------------------------------

fn do_float_to_int_into<'py, Src, Dst>(
    py: Python<'py>,
    arr: &Bound<'py, PyUntypedArray>,
    out: &Bound<'py, PyUntypedArray>,
    rounding: RoundingMode,
    oor: Option<OutOfRangeMode>,
    map_entries_py: Option<&Bound<'py, PyAny>>,
    src_dtype: &str,
    tgt_dtype: &str,
) -> PyResult<PyObject>
where
    Src: CastFloat + CastInto<Dst> + ExtractFromPy + numpy::Element + 'static,
    Dst: CastInt + ExtractFromPy + numpy::Element + 'static,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.downcast::<PyArrayDyn<Src>>()?.readonly();
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = FloatToIntConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py, src_dtype, tgt_dtype)?,
        rounding,
        out_of_range: oor,
    };
    let out_arr: &Bound<'_, PyArrayDyn<Dst>> = out.downcast()?;
    {
        // SAFETY: The GIL is held and `out_arr` is a distinct array from
        // `input_arr` (different dtypes). No aliasing occurs.
        let mut output_rw = unsafe { out_arr.as_array_mut() };
        let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Output array must be contiguous and writeable")
        })?;
        zarr_cast_value::convert_slice_float_to_int(src_slice, dst_slice, &config)
            .map_err(cast_error_to_pyerr)?;
    }
    Ok(py.None())
}

fn do_int_to_int_into<'py, Src, Dst>(
    py: Python<'py>,
    arr: &Bound<'py, PyUntypedArray>,
    out: &Bound<'py, PyUntypedArray>,
    oor: Option<OutOfRangeMode>,
    map_entries_py: Option<&Bound<'py, PyAny>>,
    src_dtype: &str,
    tgt_dtype: &str,
) -> PyResult<PyObject>
where
    Src: CastInt + CastInto<Dst> + ExtractFromPy + numpy::Element,
    Dst: CastInt + ExtractFromPy + numpy::Element,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.downcast::<PyArrayDyn<Src>>()?.readonly();
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = IntToIntConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py, src_dtype, tgt_dtype)?,
        out_of_range: oor,
    };
    let out_arr: &Bound<'_, PyArrayDyn<Dst>> = out.downcast()?;
    {
        // SAFETY: The GIL is held and `out_arr` is a distinct array from
        // `input_arr`. No aliasing occurs.
        let mut output_rw = unsafe { out_arr.as_array_mut() };
        let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Output array must be contiguous and writeable")
        })?;
        zarr_cast_value::convert_slice_int_to_int(src_slice, dst_slice, &config)
            .map_err(cast_error_to_pyerr)?;
    }
    Ok(py.None())
}

fn do_float_to_float_into<'py, Src, Dst>(
    py: Python<'py>,
    arr: &Bound<'py, PyUntypedArray>,
    out: &Bound<'py, PyUntypedArray>,
    rounding: RoundingMode,
    oor: Option<OutOfRangeMode>,
    map_entries_py: Option<&Bound<'py, PyAny>>,
    src_dtype: &str,
    tgt_dtype: &str,
) -> PyResult<PyObject>
where
    Src: CastFloat + CastInto<Dst> + ExtractFromPy + numpy::Element + 'static,
    Dst: CastFloat + ExtractFromPy + numpy::Element + 'static,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.downcast::<PyArrayDyn<Src>>()?.readonly();
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = FloatToFloatConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py, src_dtype, tgt_dtype)?,
        rounding,
        out_of_range: oor,
    };
    let out_arr: &Bound<'_, PyArrayDyn<Dst>> = out.downcast()?;
    {
        // SAFETY: The GIL is held and `out_arr` is a distinct array from
        // `input_arr`. No aliasing occurs.
        let mut output_rw = unsafe { out_arr.as_array_mut() };
        let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Output array must be contiguous and writeable")
        })?;
        zarr_cast_value::convert_slice_float_to_float(src_slice, dst_slice, &config)
            .map_err(cast_error_to_pyerr)?;
    }
    Ok(py.None())
}

fn do_int_to_float_into<'py, Src, Dst>(
    py: Python<'py>,
    arr: &Bound<'py, PyUntypedArray>,
    out: &Bound<'py, PyUntypedArray>,
    rounding: RoundingMode,
    map_entries_py: Option<&Bound<'py, PyAny>>,
    src_dtype: &str,
    tgt_dtype: &str,
) -> PyResult<PyObject>
where
    Src: CastInt + CastInto<Dst> + ExtractFromPy + numpy::Element,
    Dst: CastFloat + ExtractFromPy + numpy::Element,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.downcast::<PyArrayDyn<Src>>()?.readonly();
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = IntToFloatConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py, src_dtype, tgt_dtype)?,
        rounding,
    };
    let out_arr: &Bound<'_, PyArrayDyn<Dst>> = out.downcast()?;
    {
        // SAFETY: The GIL is held and `out_arr` is a distinct array from
        // `input_arr`. No aliasing occurs.
        let mut output_rw = unsafe { out_arr.as_array_mut() };
        let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Output array must be contiguous and writeable")
        })?;
        zarr_cast_value::convert_slice_int_to_float(src_slice, dst_slice, &config)
            .map_err(cast_error_to_pyerr)?;
    }
    Ok(py.None())
}

// ---------------------------------------------------------------------------
// N x N dispatch: allocating variant
// ---------------------------------------------------------------------------

/// Dispatch on (src_dtype, tgt_dtype) to call the appropriate conversion
/// function with concrete types. Allocates a new output numpy array.
fn dispatch_alloc<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyUntypedArray>,
    src_dtype: &str,
    tgt_dtype: &str,
    rounding: RoundingMode,
    oor: Option<OutOfRangeMode>,
    map_entries_py: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    // Dispatch uses four path-specific macros to call the right conversion
    // function.
    macro_rules! float_to_int {
        ($src_ty:ty, $dst_ty:ty) => {
            do_float_to_int_alloc::<$src_ty, $dst_ty>(
                py,
                arr,
                rounding,
                oor,
                map_entries_py,
                src_dtype,
                tgt_dtype,
            )
        };
    }
    macro_rules! int_to_int {
        ($src_ty:ty, $dst_ty:ty) => {
            do_int_to_int_alloc::<$src_ty, $dst_ty>(
                py,
                arr,
                oor,
                map_entries_py,
                src_dtype,
                tgt_dtype,
            )
        };
    }
    macro_rules! float_to_float {
        ($src_ty:ty, $dst_ty:ty) => {
            do_float_to_float_alloc::<$src_ty, $dst_ty>(
                py,
                arr,
                rounding,
                oor,
                map_entries_py,
                src_dtype,
                tgt_dtype,
            )
        };
    }
    macro_rules! int_to_float {
        ($src_ty:ty, $dst_ty:ty) => {
            do_int_to_float_alloc::<$src_ty, $dst_ty>(
                py,
                arr,
                rounding,
                map_entries_py,
                src_dtype,
                tgt_dtype,
            )
        };
    }

    // Dispatch on (src_dtype, tgt_dtype) calling the appropriate path.
    macro_rules! dispatch_int_src {
        ($src_ty:ty) => {
            match tgt_dtype {
                "int8" => int_to_int!($src_ty, i8),
                "int16" => int_to_int!($src_ty, i16),
                "int32" => int_to_int!($src_ty, i32),
                "int64" => int_to_int!($src_ty, i64),
                "uint8" => int_to_int!($src_ty, u8),
                "uint16" => int_to_int!($src_ty, u16),
                "uint32" => int_to_int!($src_ty, u32),
                "uint64" => int_to_int!($src_ty, u64),
                "float16" => int_to_float!($src_ty, f16),
                "float32" => int_to_float!($src_ty, f32),
                "float64" => int_to_float!($src_ty, f64),
                _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Unsupported target dtype: {tgt_dtype}"
                ))),
            }
        };
    }

    macro_rules! dispatch_float_src {
        ($src_ty:ty) => {
            match tgt_dtype {
                "int8" => float_to_int!($src_ty, i8),
                "int16" => float_to_int!($src_ty, i16),
                "int32" => float_to_int!($src_ty, i32),
                "int64" => float_to_int!($src_ty, i64),
                "uint8" => float_to_int!($src_ty, u8),
                "uint16" => float_to_int!($src_ty, u16),
                "uint32" => float_to_int!($src_ty, u32),
                "uint64" => float_to_int!($src_ty, u64),
                "float16" => float_to_float!($src_ty, f16),
                "float32" => float_to_float!($src_ty, f32),
                "float64" => float_to_float!($src_ty, f64),
                _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Unsupported target dtype: {tgt_dtype}"
                ))),
            }
        };
    }

    match src_dtype {
        "int8" => dispatch_int_src!(i8),
        "int16" => dispatch_int_src!(i16),
        "int32" => dispatch_int_src!(i32),
        "int64" => dispatch_int_src!(i64),
        "uint8" => dispatch_int_src!(u8),
        "uint16" => dispatch_int_src!(u16),
        "uint32" => dispatch_int_src!(u32),
        "uint64" => dispatch_int_src!(u64),
        "float16" => dispatch_float_src!(f16),
        "float32" => dispatch_float_src!(f32),
        "float64" => dispatch_float_src!(f64),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Unsupported source dtype: {src_dtype}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// N x N dispatch: into variant
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn dispatch_into<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyUntypedArray>,
    out: &Bound<'py, PyUntypedArray>,
    src_dtype: &str,
    tgt_dtype: &str,
    rounding: RoundingMode,
    oor: Option<OutOfRangeMode>,
    map_entries_py: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    macro_rules! float_to_int {
        ($src_ty:ty, $dst_ty:ty) => {
            do_float_to_int_into::<$src_ty, $dst_ty>(
                py,
                arr,
                out,
                rounding,
                oor,
                map_entries_py,
                src_dtype,
                tgt_dtype,
            )
        };
    }
    macro_rules! int_to_int {
        ($src_ty:ty, $dst_ty:ty) => {
            do_int_to_int_into::<$src_ty, $dst_ty>(
                py,
                arr,
                out,
                oor,
                map_entries_py,
                src_dtype,
                tgt_dtype,
            )
        };
    }
    macro_rules! float_to_float {
        ($src_ty:ty, $dst_ty:ty) => {
            do_float_to_float_into::<$src_ty, $dst_ty>(
                py,
                arr,
                out,
                rounding,
                oor,
                map_entries_py,
                src_dtype,
                tgt_dtype,
            )
        };
    }
    macro_rules! int_to_float {
        ($src_ty:ty, $dst_ty:ty) => {
            do_int_to_float_into::<$src_ty, $dst_ty>(
                py,
                arr,
                out,
                rounding,
                map_entries_py,
                src_dtype,
                tgt_dtype,
            )
        };
    }

    macro_rules! dispatch_int_src {
        ($src_ty:ty) => {
            match tgt_dtype {
                "int8" => int_to_int!($src_ty, i8),
                "int16" => int_to_int!($src_ty, i16),
                "int32" => int_to_int!($src_ty, i32),
                "int64" => int_to_int!($src_ty, i64),
                "uint8" => int_to_int!($src_ty, u8),
                "uint16" => int_to_int!($src_ty, u16),
                "uint32" => int_to_int!($src_ty, u32),
                "uint64" => int_to_int!($src_ty, u64),
                "float16" => int_to_float!($src_ty, f16),
                "float32" => int_to_float!($src_ty, f32),
                "float64" => int_to_float!($src_ty, f64),
                _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Unsupported target dtype: {tgt_dtype}"
                ))),
            }
        };
    }

    macro_rules! dispatch_float_src {
        ($src_ty:ty) => {
            match tgt_dtype {
                "int8" => float_to_int!($src_ty, i8),
                "int16" => float_to_int!($src_ty, i16),
                "int32" => float_to_int!($src_ty, i32),
                "int64" => float_to_int!($src_ty, i64),
                "uint8" => float_to_int!($src_ty, u8),
                "uint16" => float_to_int!($src_ty, u16),
                "uint32" => float_to_int!($src_ty, u32),
                "uint64" => float_to_int!($src_ty, u64),
                "float16" => float_to_float!($src_ty, f16),
                "float32" => float_to_float!($src_ty, f32),
                "float64" => float_to_float!($src_ty, f64),
                _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Unsupported target dtype: {tgt_dtype}"
                ))),
            }
        };
    }

    match src_dtype {
        "int8" => dispatch_int_src!(i8),
        "int16" => dispatch_int_src!(i16),
        "int32" => dispatch_int_src!(i32),
        "int64" => dispatch_int_src!(i64),
        "uint8" => dispatch_int_src!(u8),
        "uint16" => dispatch_int_src!(u16),
        "uint32" => dispatch_int_src!(u32),
        "uint64" => dispatch_int_src!(u64),
        "float16" => dispatch_float_src!(f16),
        "float32" => dispatch_float_src!(f32),
        "float64" => dispatch_float_src!(f64),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Unsupported source dtype: {src_dtype}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// cast_array: allocating variant
// ---------------------------------------------------------------------------

/// Cast a numpy array to a new dtype, allocating a new output array.
///
/// # Arguments
///
/// * `arr` - Input numpy array
/// * `target_dtype` - Target dtype string (e.g. "uint8", "float32")
/// * `rounding_mode` - Rounding mode string (e.g. "nearest-even")
/// * `out_of_range_mode` - Optional out-of-range handling ("clamp" or "wrap")
/// * `scalar_map_entries` - Optional list of [source, target] pairs
///
/// # Returns
///
/// A new numpy array with the target dtype.
///
/// # Errors
///
/// Returns `PyValueError` for invalid rounding/out-of-range modes or
/// conversion errors. Returns `PyTypeError` for unsupported dtypes.
#[pyfunction]
#[pyo3(signature = (arr, *, target_dtype, rounding_mode, out_of_range_mode=None, scalar_map_entries=None))]
fn cast_array<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyUntypedArray>,
    target_dtype: &str,
    rounding_mode: PyRoundingMode,
    out_of_range_mode: Option<PyOutOfRangeMode>,
    scalar_map_entries: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let src_dtype = array_dtype_key(arr)?;

    dispatch_alloc(
        py,
        arr,
        src_dtype,
        target_dtype,
        rounding_mode.0,
        out_of_range_mode.map(|m| m.0),
        scalar_map_entries,
    )
}

// ---------------------------------------------------------------------------
// cast_array_into: pre-allocated variant
// ---------------------------------------------------------------------------

/// Cast a numpy array into a pre-allocated output array.
///
/// # Arguments
///
/// * `arr` - Input numpy array
/// * `out` - Pre-allocated output numpy array (must match input shape)
/// * `rounding_mode` - Rounding mode string (e.g. "nearest-even")
/// * `out_of_range_mode` - Optional out-of-range handling ("clamp" or "wrap")
/// * `scalar_map_entries` - Optional list of [source, target] pairs
///
/// # Returns
///
/// None on success.
///
/// # Errors
///
/// Returns `PyValueError` for shape mismatch, invalid modes, or conversion
/// errors. Returns `PyTypeError` for unsupported dtypes.
#[pyfunction]
#[pyo3(signature = (arr, out, *, rounding_mode, out_of_range_mode=None, scalar_map_entries=None))]
fn cast_array_into<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyUntypedArray>,
    out: &Bound<'py, PyUntypedArray>,
    rounding_mode: PyRoundingMode,
    out_of_range_mode: Option<PyOutOfRangeMode>,
    scalar_map_entries: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let src_dtype = array_dtype_key(arr)?;
    let tgt_dtype = array_dtype_key(out)?;

    // Validate shapes match
    let src_shape: Vec<usize> = arr.shape().to_vec();
    let dst_shape: Vec<usize> = out.shape().to_vec();
    if src_shape != dst_shape {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Shape mismatch: input shape {:?} != output shape {:?}",
            src_shape, dst_shape
        )));
    }

    dispatch_into(
        py,
        arr,
        out,
        src_dtype,
        tgt_dtype,
        rounding_mode.0,
        out_of_range_mode.map(|m| m.0),
        scalar_map_entries,
    )
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn _cast_value_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cast_array, m)?)?;
    m.add_function(wrap_pyfunction!(cast_array_into, m)?)?;
    Ok(())
}
