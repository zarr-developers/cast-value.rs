//! End-to-end demonstration of zarrs-style integration with zarr-cast-value-core.
//!
//! This example simulates the full lifecycle of a `cast_value` codec as it
//! would appear in a zarrs codec pipeline, **without depending on zarrs**:
//!
//! 1. Parse real zarr v3 array metadata JSON (containing a `cast_value` codec)
//! 2. Extract the codec configuration: target dtype, rounding, out-of-range,
//!    and scalar_map entries
//! 3. Parse JSON scalar map entries into typed `MapEntry<Src, Dst>` values
//!    (simulating what zarrs does via `DataTypeTraits::fill_value`)
//! 4. Receive raw `&[u8]` input (simulating `ArrayBytes::Fixed`)
//! 5. Dispatch on (src_dtype, dst_dtype) → call the right `convert_slice_*`
//! 6. Return the output as raw `Vec<u8>`

// The example only covers a subset of dtype pairs; unused variants are expected.
#![allow(dead_code)]

use serde::Deserialize;
use zarr_cast_value_core::{
    FloatToFloatConfig, FloatToIntConfig, FromF64, IntToFloatConfig, IntToIntConfig, MapEntry,
    OutOfRangeMode, RoundingMode,
};

// ============================================================================
// Part 1: Zarr v3 metadata types (what zarrs would parse from zarr.json)
// ============================================================================

/// Top-level zarr v3 array metadata (simplified — only the fields we need).
#[derive(Debug, Deserialize)]
struct ArrayMetadata {
    /// The array's declared data type (e.g. "float64").
    data_type: String,
    /// Codec pipeline. We look for the cast_value entry.
    codecs: Vec<CodecEntry>,
}

/// A single entry in the codec pipeline.
#[derive(Debug, Deserialize)]
struct CodecEntry {
    name: String,
    #[serde(default)]
    configuration: serde_json::Value,
}

/// Parsed configuration for the `cast_value` codec.
///
/// Corresponds to the zarr v3 `cast_value` codec spec:
/// ```json
/// {
///   "name": "cast_value",
///   "configuration": {
///     "data_type": "uint8",
///     "rounding": "nearest-even",
///     "out_of_range": "clamp",
///     "scalar_map": {
///       "encode": [["NaN", 0], ["+Infinity", 255]],
///       "decode": [[0, "NaN"], [255, "+Infinity"]]
///     }
///   }
/// }
/// ```
#[derive(Debug, Deserialize)]
struct CastValueConfig {
    /// The spec field is `data_type`, not `target_dtype`.
    data_type: String,
    #[serde(default = "default_rounding")]
    rounding: String,
    #[serde(default)]
    out_of_range: Option<String>,
    #[serde(default)]
    scalar_map: Option<ScalarMapConfig>,
}

fn default_rounding() -> String {
    "nearest-even".to_string()
}

#[derive(Debug, Deserialize)]
struct ScalarMapConfig {
    /// Entries for the encode direction (array dtype → target dtype).
    #[serde(default)]
    encode: Vec<(serde_json::Value, serde_json::Value)>,
    /// Entries for the decode direction (target dtype → array dtype).
    #[serde(default)]
    decode: Vec<(serde_json::Value, serde_json::Value)>,
}

// ============================================================================
// Part 2: Data type enum + JSON scalar parsing
//         (simulates zarrs' DataType + DataTypeTraits::fill_value)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DataType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

impl DataType {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "int8" => Some(Self::Int8),
            "int16" => Some(Self::Int16),
            "int32" => Some(Self::Int32),
            "int64" => Some(Self::Int64),
            "uint8" => Some(Self::UInt8),
            "uint16" => Some(Self::UInt16),
            "uint32" => Some(Self::UInt32),
            "uint64" => Some(Self::UInt64),
            "float32" => Some(Self::Float32),
            "float64" => Some(Self::Float64),
            _ => None,
        }
    }

    fn is_float(self) -> bool {
        matches!(self, DataType::Float32 | DataType::Float64)
    }

    fn element_size(self) -> usize {
        match self {
            DataType::Int8 | DataType::UInt8 => 1,
            DataType::Int16 | DataType::UInt16 => 2,
            DataType::Int32 | DataType::UInt32 | DataType::Float32 => 4,
            DataType::Int64 | DataType::UInt64 | DataType::Float64 => 8,
        }
    }
}

/// Parse a JSON scalar value into an f64, handling special float strings.
///
/// In zarrs, this is done by `DataTypeTraits::fill_value` which parses JSON
/// scalars according to the target data type. Here we simplify to f64 since
/// all our supported types fit in f64 (with the precision caveat for i64/u64
/// values > 2^53, which CastValue.validate() rejects upstream).
fn json_scalar_to_f64(val: &serde_json::Value) -> f64 {
    match val {
        serde_json::Value::Number(n) => n.as_f64().expect("numeric JSON value"),
        serde_json::Value::String(s) => match s.as_str() {
            "NaN" => f64::NAN,
            "+Infinity" | "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            hex if hex.starts_with("0x") || hex.starts_with("0X") => {
                // Hex-encoded float bit pattern (e.g. "0x7fc00001" for a
                // specific NaN payload). For this example we handle f64.
                let bits = u64::from_str_radix(&hex[2..], 16).expect("valid hex");
                f64::from_bits(bits)
            }
            other => panic!("unknown scalar string: {other}"),
        },
        _ => panic!("scalar must be a number or string, got {val}"),
    }
}

/// Parse scalar_map entries from JSON into typed `MapEntry<Src, Dst>`.
///
/// Each entry is a `[src_json, dst_json]` pair. We parse both sides to f64
/// (handling "NaN", "+Infinity", etc.), then convert to the concrete Src/Dst
/// types via `FromF64`.
fn parse_map_entries_from_json<Src, Dst>(
    entries: &[(serde_json::Value, serde_json::Value)],
) -> Vec<MapEntry<Src, Dst>>
where
    Src: FromF64,
    Dst: FromF64,
{
    entries
        .iter()
        .map(|(src_json, dst_json)| {
            let src_f64 = json_scalar_to_f64(src_json);
            let dst_f64 = json_scalar_to_f64(dst_json);
            MapEntry {
                src: Src::from_f64(src_f64),
                tgt: Dst::from_f64(dst_f64),
            }
        })
        .collect()
}

// ============================================================================
// Part 3: Byte reinterpretation (simulates ArrayBytes::Fixed)
// ============================================================================

fn bytes_as_slice<T: Copy>(bytes: &[u8]) -> &[T] {
    let element_size = std::mem::size_of::<T>();
    assert!(
        bytes.len().is_multiple_of(element_size),
        "byte length {} not a multiple of element size {}",
        bytes.len(),
        element_size
    );
    let (prefix, slice, suffix) = unsafe { bytes.align_to::<T>() };
    assert!(prefix.is_empty() && suffix.is_empty(), "alignment error");
    slice
}

fn bytes_as_slice_mut<T: Copy>(bytes: &mut [u8]) -> &mut [T] {
    let element_size = std::mem::size_of::<T>();
    assert!(
        bytes.len().is_multiple_of(element_size),
        "byte length {} not a multiple of element size {}",
        bytes.len(),
        element_size
    );
    let (prefix, slice, suffix) = unsafe { bytes.align_to_mut::<T>() };
    assert!(prefix.is_empty() && suffix.is_empty(), "alignment error");
    slice
}

/// Typed data → native-endian bytes (simulates writing into ArrayBytes::Fixed).
fn to_bytes<T: Copy>(data: &[T]) -> Vec<u8> {
    let ptr = data.as_ptr() as *const u8;
    // SAFETY: T is a primitive numeric type with no padding.
    unsafe { std::slice::from_raw_parts(ptr, std::mem::size_of_val(data)) }.to_vec()
}

// ============================================================================
// Part 4: The codec — parse config, dispatch, convert
//         (what zarrs would implement as ArrayToArrayCodecTraits)
// ============================================================================

/// A parsed, ready-to-use cast_value codec instance.
struct CastValueCodec {
    src_dtype: DataType,
    dst_dtype: DataType,
    rounding: RoundingMode,
    out_of_range: Option<OutOfRangeMode>,
    /// Raw JSON scalar map entries for the encode direction.
    /// Kept as JSON because they're parsed into typed MapEntry<Src,Dst> at
    /// dispatch time, once we know the concrete Src and Dst types.
    encode_map: Vec<(serde_json::Value, serde_json::Value)>,
}

impl CastValueCodec {
    /// Construct from zarr array metadata JSON.
    ///
    /// In zarrs this would be `Codec::from_metadata()`.
    fn from_array_metadata(json: &str) -> Self {
        let meta: ArrayMetadata = serde_json::from_str(json).expect("valid zarr metadata");
        let src_dtype = DataType::from_str(&meta.data_type).expect("supported source dtype");

        // Find the cast_value codec in the pipeline
        let codec_entry = meta
            .codecs
            .iter()
            .find(|c| c.name == "cast_value")
            .expect("metadata must contain a cast_value codec");

        let config: CastValueConfig =
            serde_json::from_value(codec_entry.configuration.clone()).expect("valid codec config");

        let dst_dtype = DataType::from_str(&config.data_type).expect("supported target dtype");
        let rounding: RoundingMode = config.rounding.parse().expect("valid rounding mode");
        let out_of_range: Option<OutOfRangeMode> = config
            .out_of_range
            .as_deref()
            .map(|s| s.parse().expect("valid out_of_range mode"));

        let encode_map = config.scalar_map.map(|sm| sm.encode).unwrap_or_default();

        Self {
            src_dtype,
            dst_dtype,
            rounding,
            out_of_range,
            encode_map,
        }
    }

    /// Encode: convert from the array's data type to the codec's target dtype.
    ///
    /// In zarrs this would be `ArrayToArrayCodecTraits::encode()`.
    ///
    /// Takes raw input bytes (the array chunk in source dtype) and returns
    /// raw output bytes (the chunk in target dtype).
    fn encode(&self, src_bytes: &[u8]) -> Result<Vec<u8>, String> {
        let n_elements = src_bytes.len() / self.src_dtype.element_size();
        let mut dst_bytes = vec![0u8; n_elements * self.dst_dtype.element_size()];

        self.dispatch(src_bytes, &mut dst_bytes)
            .map_err(|e| e.to_string())?;

        Ok(dst_bytes)
    }

    /// Dispatch on (src_dtype, dst_dtype) to call the right conversion path.
    fn dispatch(
        &self,
        src_bytes: &[u8],
        dst_bytes: &mut [u8],
    ) -> Result<(), zarr_cast_value_core::CastError> {
        // Macros to reduce boilerplate in the N x N match below.
        macro_rules! f2i {
            ($src:ty, $dst:ty) => {{
                let map = parse_map_entries_from_json::<$src, $dst>(&self.encode_map);
                let config = FloatToIntConfig {
                    map_entries: &map,
                    rounding: self.rounding,
                    out_of_range: self.out_of_range,
                };
                zarr_cast_value_core::convert_slice_float_to_int(
                    bytes_as_slice::<$src>(src_bytes),
                    bytes_as_slice_mut::<$dst>(dst_bytes),
                    &config,
                )
            }};
        }
        macro_rules! i2i {
            ($src:ty, $dst:ty) => {{
                let map = parse_map_entries_from_json::<$src, $dst>(&self.encode_map);
                let config = IntToIntConfig {
                    map_entries: &map,
                    out_of_range: self.out_of_range,
                };
                zarr_cast_value_core::convert_slice_int_to_int(
                    bytes_as_slice::<$src>(src_bytes),
                    bytes_as_slice_mut::<$dst>(dst_bytes),
                    &config,
                )
            }};
        }
        macro_rules! f2f {
            ($src:ty, $dst:ty) => {{
                let map = parse_map_entries_from_json::<$src, $dst>(&self.encode_map);
                let config = FloatToFloatConfig {
                    map_entries: &map,
                    rounding: self.rounding,
                    out_of_range: self.out_of_range,
                };
                zarr_cast_value_core::convert_slice_float_to_float(
                    bytes_as_slice::<$src>(src_bytes),
                    bytes_as_slice_mut::<$dst>(dst_bytes),
                    &config,
                )
            }};
        }
        macro_rules! i2f {
            ($src:ty, $dst:ty) => {{
                let map = parse_map_entries_from_json::<$src, $dst>(&self.encode_map);
                let config = IntToFloatConfig {
                    map_entries: &map,
                    rounding: self.rounding,
                };
                zarr_cast_value_core::convert_slice_int_to_float(
                    bytes_as_slice::<$src>(src_bytes),
                    bytes_as_slice_mut::<$dst>(dst_bytes),
                    &config,
                )
            }};
        }

        // Dispatch: int source
        macro_rules! dispatch_int_src {
            ($src:ty) => {
                match self.dst_dtype {
                    DataType::Int8 => i2i!($src, i8),
                    DataType::Int16 => i2i!($src, i16),
                    DataType::Int32 => i2i!($src, i32),
                    DataType::Int64 => i2i!($src, i64),
                    DataType::UInt8 => i2i!($src, u8),
                    DataType::UInt16 => i2i!($src, u16),
                    DataType::UInt32 => i2i!($src, u32),
                    DataType::UInt64 => i2i!($src, u64),
                    DataType::Float32 => i2f!($src, f32),
                    DataType::Float64 => i2f!($src, f64),
                }
            };
        }
        // Dispatch: float source
        macro_rules! dispatch_float_src {
            ($src:ty) => {
                match self.dst_dtype {
                    DataType::Int8 => f2i!($src, i8),
                    DataType::Int16 => f2i!($src, i16),
                    DataType::Int32 => f2i!($src, i32),
                    DataType::Int64 => f2i!($src, i64),
                    DataType::UInt8 => f2i!($src, u8),
                    DataType::UInt16 => f2i!($src, u16),
                    DataType::UInt32 => f2i!($src, u32),
                    DataType::UInt64 => f2i!($src, u64),
                    DataType::Float32 => f2f!($src, f32),
                    DataType::Float64 => f2f!($src, f64),
                }
            };
        }

        match self.src_dtype {
            DataType::Int8 => dispatch_int_src!(i8),
            DataType::Int16 => dispatch_int_src!(i16),
            DataType::Int32 => dispatch_int_src!(i32),
            DataType::Int64 => dispatch_int_src!(i64),
            DataType::UInt8 => dispatch_int_src!(u8),
            DataType::UInt16 => dispatch_int_src!(u16),
            DataType::UInt32 => dispatch_int_src!(u32),
            DataType::UInt64 => dispatch_int_src!(u64),
            DataType::Float32 => dispatch_float_src!(f32),
            DataType::Float64 => dispatch_float_src!(f64),
        }
    }
}

// ============================================================================
// Part 5: Main — run end-to-end scenarios with real zarr metadata
// ============================================================================

fn main() {
    println!("=== zarrs integration: end-to-end with zarr metadata ===\n");

    // -----------------------------------------------------------------------
    // Scenario 1: float64 → uint8 with scalar_map, clamping, rounding
    //
    // A sensor array stored as float64, encoded to uint8 for storage.
    // NaN → 0 (missing data sentinel), +Infinity → 255 (saturated).
    // Other values rounded and clamped to [0, 255].
    // -----------------------------------------------------------------------
    println!("--- Scenario 1: float64 -> uint8 sensor data ---");
    let metadata_json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [6],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [6]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": "NaN",
        "codecs": [
            {
                "name": "cast_value",
                "configuration": {
                    "data_type": "uint8",
                    "rounding": "nearest-even",
                    "out_of_range": "clamp",
                    "scalar_map": {
                        "encode": [["NaN", 0], ["+Infinity", 255]],
                        "decode": [[0, "NaN"], [255, "+Infinity"]]
                    }
                }
            },
            {"name": "bytes", "configuration": {"endian": "little"}}
        ]
    }"#;
    println!("metadata:\n{metadata_json}\n");

    let codec = CastValueCodec::from_array_metadata(metadata_json);
    println!(
        "parsed: {:?} -> {:?}, rounding={:?}, oor={:?}, map_entries={}",
        codec.src_dtype,
        codec.dst_dtype,
        codec.rounding,
        codec.out_of_range,
        codec.encode_map.len()
    );

    // Simulate a chunk of sensor readings
    let chunk_data: Vec<f64> = vec![23.7, f64::NAN, 100.4, 300.0, f64::INFINITY, -5.0];
    let src_bytes = to_bytes(&chunk_data);
    println!("input chunk: {:?}", chunk_data);

    let encoded = codec.encode(&src_bytes).expect("encode succeeds");
    println!("encoded bytes: {:?}", encoded.as_slice());
    // Expected: [24, 0, 100, 255, 255, 0]
    //   23.7  → rounds to 24
    //   NaN   → scalar_map → 0
    //   100.4 → rounds to 100
    //   300.0 → clamped to 255
    //   +Inf  → scalar_map → 255
    //   -5.0  → clamped to 0
    println!();

    // -----------------------------------------------------------------------
    // Scenario 2: int32 → uint8 with wrapping (no scalar_map)
    //
    // A modular counter stored as int32, packed to uint8 via modular
    // arithmetic.
    // -----------------------------------------------------------------------
    println!("--- Scenario 2: int32 -> uint8 with wrapping ---");
    let metadata_json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [5],
        "data_type": "int32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [
            {
                "name": "cast_value",
                "configuration": {
                    "data_type": "uint8",
                    "out_of_range": "wrap"
                }
            },
            {"name": "bytes", "configuration": {"endian": "little"}}
        ]
    }"#;
    println!("metadata:\n{metadata_json}\n");

    let codec = CastValueCodec::from_array_metadata(metadata_json);
    let chunk_data: Vec<i32> = vec![0, 255, 256, 300, -1];
    let src_bytes = to_bytes(&chunk_data);
    println!("input chunk: {:?}", chunk_data);

    let encoded = codec.encode(&src_bytes).expect("encode succeeds");
    println!("encoded bytes: {:?}", encoded.as_slice());
    // Expected: [0, 255, 0, 44, 255]
    //   0   → 0
    //   255 → 255
    //   256 → 256 as u8 = 0 (bit truncation)
    //   300 → 300 as u8 = 44
    //   -1  → -1 as u8 = 255
    println!();

    // -----------------------------------------------------------------------
    // Scenario 3: float64 → float32 with scalar_map for NaN preservation
    //
    // Downcast precision; map NaN to NaN so it survives the cast.
    // -----------------------------------------------------------------------
    println!("--- Scenario 3: float64 -> float32 precision downcast ---");
    let metadata_json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": "NaN",
        "codecs": [
            {
                "name": "cast_value",
                "configuration": {
                    "data_type": "float32",
                    "scalar_map": {
                        "encode": [["NaN", "NaN"]],
                        "decode": [["NaN", "NaN"]]
                    }
                }
            },
            {"name": "bytes", "configuration": {"endian": "little"}}
        ]
    }"#;
    println!("metadata:\n{metadata_json}\n");

    let codec = CastValueCodec::from_array_metadata(metadata_json);
    let chunk_data: Vec<f64> = vec![1.0, f64::NAN, 1e30, -0.0];
    let src_bytes = to_bytes(&chunk_data);
    println!("input chunk: {:?}", chunk_data);

    let encoded = codec.encode(&src_bytes).expect("encode succeeds");
    let output = bytes_as_slice::<f32>(&encoded);
    println!("encoded f32s: {:?}", output);
    // Expected: [1.0, NaN, 1e30 (as f32), -0.0]
    println!();

    // -----------------------------------------------------------------------
    // Scenario 4: int16 → float64 (lossless widening)
    //
    // No scalar_map, no rounding, no range check needed. Just cast.
    // -----------------------------------------------------------------------
    println!("--- Scenario 4: int16 -> float64 lossless widening ---");
    let metadata_json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "int16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [
            {
                "name": "cast_value",
                "configuration": {
                    "data_type": "float64"
                }
            },
            {"name": "bytes", "configuration": {"endian": "little"}}
        ]
    }"#;
    println!("metadata:\n{metadata_json}\n");

    let codec = CastValueCodec::from_array_metadata(metadata_json);
    let chunk_data: Vec<i16> = vec![0, -1, 32767, -32768];
    let src_bytes = to_bytes(&chunk_data);
    println!("input chunk: {:?}", chunk_data);

    let encoded = codec.encode(&src_bytes).expect("encode succeeds");
    let output = bytes_as_slice::<f64>(&encoded);
    println!("encoded f64s: {:?}", output);
    // Expected: [0.0, -1.0, 32767.0, -32768.0]
    println!();

    // -----------------------------------------------------------------------
    // Scenario 5: float64 → uint8 error — NaN without scalar_map
    //
    // Demonstrates the error path: NaN can't be cast to an integer without
    // a scalar_map entry to handle it.
    // -----------------------------------------------------------------------
    println!("--- Scenario 5: error — NaN without scalar_map ---");
    let metadata_json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [3],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [
            {
                "name": "cast_value",
                "configuration": {
                    "data_type": "uint8",
                    "rounding": "towards-zero"
                }
            },
            {"name": "bytes", "configuration": {"endian": "little"}}
        ]
    }"#;

    let codec = CastValueCodec::from_array_metadata(metadata_json);
    let chunk_data: Vec<f64> = vec![1.0, f64::NAN, 3.0];
    let src_bytes = to_bytes(&chunk_data);
    println!("input chunk: {:?}", chunk_data);

    match codec.encode(&src_bytes) {
        Err(e) => println!("encode error (expected): {e}"),
        Ok(_) => println!("ERROR: should have failed!"),
    }
    println!();

    // -----------------------------------------------------------------------
    // Scenario 6: int32 → uint8 error — out of range, no mode set
    // -----------------------------------------------------------------------
    println!("--- Scenario 6: error — out of range without mode ---");
    let metadata_json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [3],
        "data_type": "int32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [
            {
                "name": "cast_value",
                "configuration": {
                    "data_type": "uint8"
                }
            },
            {"name": "bytes", "configuration": {"endian": "little"}}
        ]
    }"#;

    let codec = CastValueCodec::from_array_metadata(metadata_json);
    let chunk_data: Vec<i32> = vec![0, 100, 300];
    let src_bytes = to_bytes(&chunk_data);
    println!("input chunk: {:?}", chunk_data);

    match codec.encode(&src_bytes) {
        Err(e) => println!("encode error (expected): {e}"),
        Ok(_) => println!("ERROR: should have failed!"),
    }

    println!("\n=== done ===");
}
