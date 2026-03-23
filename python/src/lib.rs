//! zarr-cast-value: PyO3 bindings for the cast_value codec.
//!
//! Exposes `cast_array` and `cast_array_into` to Python, dispatching on
//! numpy dtype pairs to monomorphized conversion calls from the core crate.

use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zarr_cast_value_core::{
    CastError, CastFloat, CastInt, CastInto, FloatToFloatConfig, FloatToIntConfig,
    IntToFloatConfig, IntToIntConfig, MapEntry, OutOfRangeMode, RoundingMode,
};

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
) -> PyResult<Vec<MapEntry<Src, Dst>>>
where
    Src: zarr_cast_value_core::CastNum + for<'a> FromPyObject<'a>,
    Dst: zarr_cast_value_core::CastNum + for<'a> FromPyObject<'a>,
{
    let Some(obj) = entries else {
        return Ok(Vec::new());
    };

    // Dicts are special: iterating a dict yields keys only, so we
    // need to iterate key-value pairs explicitly.
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut result = Vec::with_capacity(dict.len());
        for (key, val) in dict.iter() {
            let src: Src = key.extract()?;
            let tgt: Dst = val.extract()?;
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
        let src: Src = item.get_item(0)?.extract()?;
        let tgt: Dst = item.get_item(1)?.extract()?;
        result.push(MapEntry { src, tgt });
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Dtype key extraction
// ---------------------------------------------------------------------------

fn dtype_key(kind: char, itemsize: usize) -> PyResult<&'static str> {
    match (kind, itemsize) {
        ('i', 1) => Ok("int8"),
        ('i', 2) => Ok("int16"),
        ('i', 4) => Ok("int32"),
        ('i', 8) => Ok("int64"),
        ('u', 1) => Ok("uint8"),
        ('u', 2) => Ok("uint16"),
        ('u', 4) => Ok("uint32"),
        ('u', 8) => Ok("uint64"),
        ('f', 4) => Ok("float32"),
        ('f', 8) => Ok("float64"),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Unsupported dtype kind={kind} itemsize={itemsize}",
        ))),
    }
}

fn array_dtype_key(arr: &Bound<'_, PyUntypedArray>) -> PyResult<&'static str> {
    let dtype = arr.getattr("dtype")?;
    let kind: char = dtype.getattr("kind")?.extract()?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    dtype_key(kind, itemsize)
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
) -> PyResult<PyObject>
where
    Src: CastFloat + CastInto<Dst> + for<'a> FromPyObject<'a> + numpy::Element,
    Dst: CastInt + for<'a> FromPyObject<'a> + numpy::Element,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.extract()?;
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = FloatToIntConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py)?,
        rounding,
        out_of_range: oor,
    };
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let output = PyArrayDyn::<Dst>::zeros(py, &shape[..], false);
    {
        // SAFETY: We just created `output` and hold the GIL, so no other
        // code can alias this array. The mutable reference is valid for
        // the duration of this block.
        let mut output_rw = unsafe { output.as_array_mut() };
        let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Failed to get mutable slice from output array")
        })?;
        zarr_cast_value_core::convert_slice_float_to_int(src_slice, dst_slice, &config)
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
) -> PyResult<PyObject>
where
    Src: CastInt + CastInto<Dst> + for<'a> FromPyObject<'a> + numpy::Element,
    Dst: CastInt + for<'a> FromPyObject<'a> + numpy::Element,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.extract()?;
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = IntToIntConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py)?,
        out_of_range: oor,
    };
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let output = PyArrayDyn::<Dst>::zeros(py, &shape[..], false);
    {
        // SAFETY: We just created `output` and hold the GIL, so no other
        // code can alias this array.
        let mut output_rw = unsafe { output.as_array_mut() };
        let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Failed to get mutable slice from output array")
        })?;
        zarr_cast_value_core::convert_slice_int_to_int(src_slice, dst_slice, &config)
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
) -> PyResult<PyObject>
where
    Src: CastFloat + CastInto<Dst> + for<'a> FromPyObject<'a> + numpy::Element,
    Dst: CastFloat + for<'a> FromPyObject<'a> + numpy::Element,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.extract()?;
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = FloatToFloatConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py)?,
        rounding,
        out_of_range: oor,
    };
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let output = PyArrayDyn::<Dst>::zeros(py, &shape[..], false);
    {
        // SAFETY: We just created `output` and hold the GIL, so no other
        // code can alias this array.
        let mut output_rw = unsafe { output.as_array_mut() };
        let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Failed to get mutable slice from output array")
        })?;
        zarr_cast_value_core::convert_slice_float_to_float(src_slice, dst_slice, &config)
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
) -> PyResult<PyObject>
where
    Src: CastInt + CastInto<Dst> + for<'a> FromPyObject<'a> + numpy::Element,
    Dst: CastFloat + for<'a> FromPyObject<'a> + numpy::Element,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.extract()?;
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = IntToFloatConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py)?,
        rounding,
    };
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let output = PyArrayDyn::<Dst>::zeros(py, &shape[..], false);
    {
        // SAFETY: We just created `output` and hold the GIL, so no other
        // code can alias this array.
        let mut output_rw = unsafe { output.as_array_mut() };
        let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Failed to get mutable slice from output array")
        })?;
        zarr_cast_value_core::convert_slice_int_to_float(src_slice, dst_slice, &config)
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
) -> PyResult<PyObject>
where
    Src: CastFloat + CastInto<Dst> + for<'a> FromPyObject<'a> + numpy::Element,
    Dst: CastInt + for<'a> FromPyObject<'a> + numpy::Element,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.extract()?;
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = FloatToIntConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py)?,
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
        zarr_cast_value_core::convert_slice_float_to_int(src_slice, dst_slice, &config)
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
) -> PyResult<PyObject>
where
    Src: CastInt + CastInto<Dst> + for<'a> FromPyObject<'a> + numpy::Element,
    Dst: CastInt + for<'a> FromPyObject<'a> + numpy::Element,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.extract()?;
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = IntToIntConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py)?,
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
        zarr_cast_value_core::convert_slice_int_to_int(src_slice, dst_slice, &config)
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
) -> PyResult<PyObject>
where
    Src: CastFloat + CastInto<Dst> + for<'a> FromPyObject<'a> + numpy::Element,
    Dst: CastFloat + for<'a> FromPyObject<'a> + numpy::Element,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.extract()?;
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = FloatToFloatConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py)?,
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
        zarr_cast_value_core::convert_slice_float_to_float(src_slice, dst_slice, &config)
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
) -> PyResult<PyObject>
where
    Src: CastInt + CastInto<Dst> + for<'a> FromPyObject<'a> + numpy::Element,
    Dst: CastFloat + for<'a> FromPyObject<'a> + numpy::Element,
{
    let input_arr: PyReadonlyArrayDyn<'_, Src> = arr.extract()?;
    let src_slice = input_arr
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input array must be contiguous"))?;
    let config = IntToFloatConfig {
        map_entries: parse_map_entries::<Src, Dst>(map_entries_py)?,
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
        zarr_cast_value_core::convert_slice_int_to_float(src_slice, dst_slice, &config)
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
            do_float_to_int_alloc::<$src_ty, $dst_ty>(py, arr, rounding, oor, map_entries_py)
        };
    }
    macro_rules! int_to_int {
        ($src_ty:ty, $dst_ty:ty) => {
            do_int_to_int_alloc::<$src_ty, $dst_ty>(py, arr, oor, map_entries_py)
        };
    }
    macro_rules! float_to_float {
        ($src_ty:ty, $dst_ty:ty) => {
            do_float_to_float_alloc::<$src_ty, $dst_ty>(py, arr, rounding, oor, map_entries_py)
        };
    }
    macro_rules! int_to_float {
        ($src_ty:ty, $dst_ty:ty) => {
            do_int_to_float_alloc::<$src_ty, $dst_ty>(py, arr, rounding, map_entries_py)
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
            do_float_to_int_into::<$src_ty, $dst_ty>(py, arr, out, rounding, oor, map_entries_py)
        };
    }
    macro_rules! int_to_int {
        ($src_ty:ty, $dst_ty:ty) => {
            do_int_to_int_into::<$src_ty, $dst_ty>(py, arr, out, oor, map_entries_py)
        };
    }
    macro_rules! float_to_float {
        ($src_ty:ty, $dst_ty:ty) => {
            do_float_to_float_into::<$src_ty, $dst_ty>(py, arr, out, rounding, oor, map_entries_py)
        };
    }
    macro_rules! int_to_float {
        ($src_ty:ty, $dst_ty:ty) => {
            do_int_to_float_into::<$src_ty, $dst_ty>(py, arr, out, rounding, map_entries_py)
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
    let src_shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let dst_shape: Vec<usize> = out.getattr("shape")?.extract()?;
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
fn zarr_cast_value(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cast_array, m)?)?;
    m.add_function(wrap_pyfunction!(cast_array_into, m)?)?;
    Ok(())
}
