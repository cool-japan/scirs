//! Conversion between oxih5's `Dtype`-typed model and SciRS2's HDF5 model.
//!
//! # Why this module exists
//!
//! The C-linked `hdf5` crate performed *implicit numeric conversion* inside
//! `Dataset::read_raw::<T>()`: asking for `f64` from an `i16` dataset worked,
//! because libhdf5 ran an `H5Tconvert` on the way out. oxih5 deliberately does
//! not do that — [`oxih5::Dataset::as_f64`] matches **only**
//! `Dtype::Float { size: 8 }` and returns `TypeMismatch` for everything else.
//!
//! A literal port of the old call sites would therefore silently narrow SciRS2
//! from "reads any numeric HDF5 dataset" to "reads exactly f64", and because
//! every pre-existing test fixture is f64 the regression would pass CI and only
//! surface on real-world files. [`dataset_to_f64`] and [`dataset_to_i64`]
//! restore the widening explicitly, over the full 1/2/4/8-byte × signed/unsigned
//! integer matrix plus f16/f32/f64, and every read path routes through them.

use oxih5::{ByteOrder, Dataset, Dtype, OxiH5Error};
use oxih5_core::{Charset, RefType};

use super::types_3::{HDF5DataType, StringEncoding};
use crate::error::{IoError, Result};

/// Wrap an oxih5 decode failure with the SciRS2-side context.
fn decode_err(what: &str, err: OxiH5Error) -> IoError {
    IoError::FormatError(format!("Failed to decode {what} data: {err}"))
}

/// Report a datatype that has no representation in the requested Rust type.
fn unsupported(dtype: &Dtype, target: &str) -> IoError {
    IoError::UnsupportedFormat(format!(
        "HDF5 datatype {dtype} cannot be converted to {target}"
    ))
}

/// Re-label a dataset's bytes with a different `Dtype` so oxih5's decoders apply.
///
/// Used for `Enum` and `Bitfield`, whose payloads are ordinary fixed-width
/// integers but whose `Dtype` discriminant the oxih5 accessors do not match.
/// Copies the payload, so it is only used on the (rare) non-integer-tagged paths.
fn retyped(dataset: &Dataset, dtype: Dtype) -> Dataset {
    Dataset {
        data: dataset.data.clone(),
        shape: dataset.shape.clone(),
        dtype,
        attributes: Vec::new(),
        max_dims: None,
    }
}

/// The integer `Dtype` a `Bitfield` payload is bit-identical to.
fn bitfield_as_int(size: usize, order: ByteOrder) -> Dtype {
    Dtype::Int {
        size,
        signed: false,
        order,
    }
}

/// Translate an oxih5 [`Dtype`] into SciRS2's [`HDF5DataType`].
///
/// Infallible: oxih5 models element types of arrays, enums and compound fields
/// as plain nested `Dtype`s, so no round-trip through a datatype handle is
/// needed (the C backend had to call `Datatype::from_descriptor`, which could
/// fail, three separate times).
///
/// Two variants carry more information than the old backend surfaced:
/// * `Dtype::Array { dims }` keeps the **full** dimension list; the C path
///   flattened every array type to a single-axis `vec![len]`.
/// * `Dtype::Enum { members }` is already `Vec<(String, i64)>`, exactly the
///   shape [`HDF5DataType::Enum`] wants.
///
/// The three variants with no old-API analogue are mapped to their true
/// in-memory footprint rather than to a catch-all string: `Opaque` and
/// `Bitfield` are unsigned integers of the declared width, and a `Reference` is
/// an unsigned 8-byte object address (12 bytes for a region reference). Calling
/// a bitfield a UTF-8 string — which the old catch-all arm did — makes
/// [`super::types_3::Dataset::size_bytes`] and every downstream consumer lie.
pub(super) fn convert_dtype(dtype: &Dtype) -> HDF5DataType {
    match dtype {
        Dtype::Int { size, signed, .. } => HDF5DataType::Integer {
            size: *size,
            signed: *signed,
        },
        Dtype::Float { size, .. } => HDF5DataType::Float { size: *size },
        Dtype::String { charset, .. } => HDF5DataType::String {
            encoding: match charset {
                Charset::Ascii => StringEncoding::ASCII,
                Charset::Utf8 => StringEncoding::UTF8,
            },
        },
        Dtype::Compound { fields } => HDF5DataType::Compound {
            fields: fields
                .iter()
                .map(|field| (field.name.clone(), convert_dtype(&field.dtype)))
                .collect(),
        },
        Dtype::Array { base, dims } => HDF5DataType::Array {
            base_type: Box::new(convert_dtype(base)),
            shape: dims.clone(),
        },
        Dtype::Enum { members, .. } => HDF5DataType::Enum {
            values: members.clone(),
        },
        // A variable-length sequence is modelled as an array whose single axis
        // has length 0. `tests::test_hdf5_varlen_array_marker` pins this sentinel.
        Dtype::VarLen { base } => HDF5DataType::Array {
            base_type: Box::new(convert_dtype(base)),
            shape: vec![0],
        },
        Dtype::Opaque { size, .. } | Dtype::Bitfield { size, .. } => HDF5DataType::Integer {
            size: *size,
            signed: false,
        },
        Dtype::Reference { ref_type } => HDF5DataType::Integer {
            size: match ref_type {
                RefType::Object => 8,
                RefType::Region => 12,
            },
            signed: false,
        },
    }
}

/// Decode a dataset of **any** numeric HDF5 datatype into `Vec<f64>`.
///
/// Accepts f16 / f32 / f64 and every 1/2/4/8-byte signed or unsigned integer,
/// plus `Enum` and `Bitfield` payloads over those integer widths. This is the
/// replacement for the C crate's implicit `read_raw::<f64>()` conversion.
///
/// 64-bit integers are widened with `as f64` and therefore lose precision above
/// 2^53, exactly as libhdf5's own `H5Tconvert` does.
///
/// # Errors
///
/// [`IoError::UnsupportedFormat`] naming the datatype when it is not numeric
/// (strings, compounds, references, opaque blobs, odd-width integers), or
/// [`IoError::FormatError`] when the payload is truncated relative to its shape.
pub(crate) fn dataset_to_f64(dataset: &Dataset) -> Result<Vec<f64>> {
    match &dataset.dtype {
        Dtype::Float { size: 2, .. } => Ok(dataset
            .as_f16()
            .map_err(|e| decode_err("float16", e))?
            .into_iter()
            .map(f64::from)
            .collect()),
        Dtype::Float { size: 4, .. } => Ok(dataset
            .as_f32()
            .map_err(|e| decode_err("float32", e))?
            .into_iter()
            .map(f64::from)
            .collect()),
        Dtype::Float { size: 8, .. } => dataset.as_f64().map_err(|e| decode_err("float64", e)),
        Dtype::Int {
            size: 1,
            signed: true,
            ..
        } => Ok(dataset
            .as_i8()
            .map_err(|e| decode_err("int8", e))?
            .into_iter()
            .map(f64::from)
            .collect()),
        Dtype::Int {
            size: 2,
            signed: true,
            ..
        } => Ok(dataset
            .as_i16()
            .map_err(|e| decode_err("int16", e))?
            .into_iter()
            .map(f64::from)
            .collect()),
        Dtype::Int {
            size: 4,
            signed: true,
            ..
        } => Ok(dataset
            .as_i32()
            .map_err(|e| decode_err("int32", e))?
            .into_iter()
            .map(f64::from)
            .collect()),
        Dtype::Int {
            size: 8,
            signed: true,
            ..
        } => Ok(dataset
            .as_i64()
            .map_err(|e| decode_err("int64", e))?
            .into_iter()
            .map(|v| v as f64)
            .collect()),
        Dtype::Int {
            size: 1,
            signed: false,
            ..
        } => Ok(dataset
            .as_u8()
            .map_err(|e| decode_err("uint8", e))?
            .into_iter()
            .map(f64::from)
            .collect()),
        Dtype::Int {
            size: 2,
            signed: false,
            ..
        } => Ok(dataset
            .as_u16()
            .map_err(|e| decode_err("uint16", e))?
            .into_iter()
            .map(f64::from)
            .collect()),
        Dtype::Int {
            size: 4,
            signed: false,
            ..
        } => Ok(dataset
            .as_u32()
            .map_err(|e| decode_err("uint32", e))?
            .into_iter()
            .map(f64::from)
            .collect()),
        Dtype::Int {
            size: 8,
            signed: false,
            ..
        } => Ok(dataset
            .as_u64()
            .map_err(|e| decode_err("uint64", e))?
            .into_iter()
            .map(|v| v as f64)
            .collect()),
        Dtype::Enum { base, .. } => dataset_to_f64(&retyped(dataset, (**base).clone())),
        Dtype::Bitfield { size, order } => {
            dataset_to_f64(&retyped(dataset, bitfield_as_int(*size, *order)))
        }
        other => Err(unsupported(other, "f64")),
    }
}

/// Decode a dataset of **any** numeric HDF5 datatype into `Vec<i64>`.
///
/// Accepts every 1/2/4/8-byte signed or unsigned integer, `Enum` and `Bitfield`
/// payloads over those widths, and f16 / f32 / f64 (truncated toward zero, as
/// libhdf5's float→int conversion does).
///
/// # Errors
///
/// [`IoError::UnsupportedFormat`] naming the datatype when it is not numeric, or
/// [`IoError::ConversionError`] for a `uint64` value above [`i64::MAX`], which
/// has no faithful `i64` representation — reporting it beats wrapping it into a
/// negative number.
pub(crate) fn dataset_to_i64(dataset: &Dataset) -> Result<Vec<i64>> {
    match &dataset.dtype {
        Dtype::Int {
            size: 1,
            signed: true,
            ..
        } => Ok(dataset
            .as_i8()
            .map_err(|e| decode_err("int8", e))?
            .into_iter()
            .map(i64::from)
            .collect()),
        Dtype::Int {
            size: 2,
            signed: true,
            ..
        } => Ok(dataset
            .as_i16()
            .map_err(|e| decode_err("int16", e))?
            .into_iter()
            .map(i64::from)
            .collect()),
        Dtype::Int {
            size: 4,
            signed: true,
            ..
        } => Ok(dataset
            .as_i32()
            .map_err(|e| decode_err("int32", e))?
            .into_iter()
            .map(i64::from)
            .collect()),
        Dtype::Int {
            size: 8,
            signed: true,
            ..
        } => dataset.as_i64().map_err(|e| decode_err("int64", e)),
        Dtype::Int {
            size: 1,
            signed: false,
            ..
        } => Ok(dataset
            .as_u8()
            .map_err(|e| decode_err("uint8", e))?
            .into_iter()
            .map(i64::from)
            .collect()),
        Dtype::Int {
            size: 2,
            signed: false,
            ..
        } => Ok(dataset
            .as_u16()
            .map_err(|e| decode_err("uint16", e))?
            .into_iter()
            .map(i64::from)
            .collect()),
        Dtype::Int {
            size: 4,
            signed: false,
            ..
        } => Ok(dataset
            .as_u32()
            .map_err(|e| decode_err("uint32", e))?
            .into_iter()
            .map(i64::from)
            .collect()),
        Dtype::Int {
            size: 8,
            signed: false,
            ..
        } => dataset
            .as_u64()
            .map_err(|e| decode_err("uint64", e))?
            .into_iter()
            .map(|v| {
                i64::try_from(v).map_err(|_| {
                    IoError::ConversionError(format!(
                        "uint64 value {v} exceeds i64::MAX and cannot be represented"
                    ))
                })
            })
            .collect(),
        Dtype::Float { .. } => Ok(dataset_to_f64(dataset)?
            .into_iter()
            .map(|v| v as i64)
            .collect()),
        Dtype::Enum { base, .. } => dataset_to_i64(&retyped(dataset, (**base).clone())),
        Dtype::Bitfield { size, order } => {
            dataset_to_i64(&retyped(dataset, bitfield_as_int(*size, *order)))
        }
        other => Err(unsupported(other, "i64")),
    }
}

/// Whether [`dataset_to_i64`] is the right decoder for `dtype`.
///
/// True for integers, enums and bitfields — the datatypes [`convert_dtype`]
/// reports as [`HDF5DataType::Integer`] and that decode losslessly to `i64`.
/// `Opaque` and `Reference` also map to `Integer` but have no meaningful
/// numeric value, so they stay raw bytes.
pub(crate) fn is_integral(dtype: &Dtype) -> bool {
    match dtype {
        Dtype::Int { size, .. } | Dtype::Bitfield { size, .. } => {
            matches!(size, 1 | 2 | 4 | 8)
        }
        Dtype::Enum { base, .. } => is_integral(base),
        _ => false,
    }
}

/// Whether [`dataset_to_f64`] is the right decoder for `dtype`.
pub(crate) fn is_floating(dtype: &Dtype) -> bool {
    matches!(
        dtype,
        Dtype::Float {
            size: 2 | 4 | 8,
            ..
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxih5_core::CompoundField;

    /// Build a synthetic dataset from raw little-endian bytes.
    fn ds(data: Vec<u8>, shape: Vec<usize>, dtype: Dtype) -> Dataset {
        Dataset {
            data,
            shape,
            dtype,
            attributes: Vec::new(),
            max_dims: None,
        }
    }

    fn int(size: usize, signed: bool) -> Dtype {
        Dtype::Int {
            size,
            signed,
            order: ByteOrder::Little,
        }
    }

    /// Every integer width must widen to f64 — this is the regression the
    /// migration exists to prevent, since a literal port only handled f64.
    #[test]
    fn test_widen_all_integer_widths_to_f64() {
        let cases: Vec<(Dtype, Vec<u8>, Vec<f64>)> = vec![
            (int(1, true), vec![0xFF, 0x01], vec![-1.0, 1.0]),
            (int(1, false), vec![0xFF, 0x01], vec![255.0, 1.0]),
            (int(2, true), vec![0xFF, 0xFF, 0x02, 0x00], vec![-1.0, 2.0]),
            (
                int(2, false),
                vec![0xFF, 0xFF, 0x02, 0x00],
                vec![65535.0, 2.0],
            ),
            (int(4, true), vec![0xFF, 0xFF, 0xFF, 0xFF], vec![-1.0]),
            (
                int(4, false),
                vec![0xFF, 0xFF, 0xFF, 0xFF],
                vec![4294967295.0],
            ),
            (int(8, true), vec![0xFF; 8], vec![-1.0]),
            (int(8, false), vec![0x02, 0, 0, 0, 0, 0, 0, 0], vec![2.0]),
        ];
        for (dtype, bytes, expected) in cases {
            let n = expected.len();
            let dataset = ds(bytes, vec![n], dtype.clone());
            let got = dataset_to_f64(&dataset)
                .unwrap_or_else(|e| panic!("{dtype} should widen to f64: {e}"));
            assert_eq!(got, expected, "mismatch for {dtype}");
        }
    }

    /// f32 and f16 datasets must widen too, not just the native f64 case.
    #[test]
    fn test_widen_float_widths_to_f64() {
        let f32_ds = ds(
            1.5f32.to_le_bytes().to_vec(),
            vec![1],
            Dtype::Float {
                size: 4,
                order: ByteOrder::Little,
            },
        );
        assert_eq!(dataset_to_f64(&f32_ds).expect("f32 widens"), vec![1.5]);

        // IEEE-754 half precision for 1.5 is 0x3E00.
        let f16_ds = ds(
            0x3E00u16.to_le_bytes().to_vec(),
            vec![1],
            Dtype::Float {
                size: 2,
                order: ByteOrder::Little,
            },
        );
        assert_eq!(dataset_to_f64(&f16_ds).expect("f16 widens"), vec![1.5]);
    }

    /// Big-endian payloads decode correctly (oxih5 honours `ByteOrder`).
    #[test]
    fn test_big_endian_integer_widens() {
        let dataset = ds(
            vec![0x00, 0x00, 0x01, 0x00],
            vec![1],
            Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Big,
            },
        );
        assert_eq!(dataset_to_f64(&dataset).expect("BE decode"), vec![256.0]);
    }

    /// Enum payloads decode through their integer base type.
    #[test]
    fn test_enum_decodes_via_base_type() {
        let dataset = ds(
            vec![0x07, 0x00, 0x00, 0x00],
            vec![1],
            Dtype::Enum {
                base: Box::new(int(4, true)),
                members: vec![("SEVEN".to_string(), 7)],
            },
        );
        assert_eq!(dataset_to_i64(&dataset).expect("enum decode"), vec![7]);
        assert_eq!(dataset_to_f64(&dataset).expect("enum decode"), vec![7.0]);
    }

    /// Bitfield payloads decode as unsigned integers of the declared width.
    #[test]
    fn test_bitfield_decodes_as_unsigned() {
        let dataset = ds(
            vec![0xF0],
            vec![1],
            Dtype::Bitfield {
                size: 1,
                order: ByteOrder::Little,
            },
        );
        assert_eq!(
            dataset_to_i64(&dataset).expect("bitfield decode"),
            vec![240]
        );
    }

    /// A uint64 above i64::MAX is reported, never wrapped to a negative number.
    #[test]
    fn test_uint64_overflow_is_reported_not_wrapped() {
        let dataset = ds(vec![0xFF; 8], vec![1], int(8, false));
        let err = dataset_to_i64(&dataset).expect_err("u64::MAX must not fit i64");
        assert!(
            matches!(err, IoError::ConversionError(_)),
            "expected ConversionError, got {err:?}"
        );
        // The same value is representable (approximately) as f64.
        assert_eq!(
            dataset_to_f64(&dataset).expect("u64 widens to f64"),
            vec![u64::MAX as f64]
        );
    }

    /// Non-numeric datatypes are refused by name rather than silently coerced.
    #[test]
    fn test_non_numeric_dtype_is_refused() {
        let dataset = ds(
            b"hi".to_vec(),
            vec![1],
            Dtype::String {
                fixed_len: Some(2),
                charset: Charset::Utf8,
            },
        );
        let err = dataset_to_f64(&dataset).expect_err("strings are not numeric");
        assert!(matches!(err, IoError::UnsupportedFormat(_)));
    }

    /// Multi-dimensional array types keep every axis; the C backend flattened
    /// them to a single length.
    #[test]
    fn test_convert_dtype_preserves_array_rank() {
        let dtype = Dtype::Array {
            base: Box::new(Dtype::Float {
                size: 4,
                order: ByteOrder::Little,
            }),
            dims: vec![2, 3, 4],
        };
        match convert_dtype(&dtype) {
            HDF5DataType::Array { base_type, shape } => {
                assert_eq!(shape, vec![2, 3, 4], "all three axes must survive");
                assert!(matches!(*base_type, HDF5DataType::Float { size: 4 }));
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    /// Bitfields, opaque blobs and references report their true width instead
    /// of being mislabelled as UTF-8 strings.
    #[test]
    fn test_convert_dtype_maps_non_string_types_truthfully() {
        assert_eq!(
            convert_dtype(&Dtype::Bitfield {
                size: 2,
                order: ByteOrder::Little
            }),
            HDF5DataType::Integer {
                size: 2,
                signed: false
            }
        );
        assert_eq!(
            convert_dtype(&Dtype::Opaque {
                size: 5,
                tag: "blob".to_string()
            }),
            HDF5DataType::Integer {
                size: 5,
                signed: false
            }
        );
        assert_eq!(
            convert_dtype(&Dtype::Reference {
                ref_type: RefType::Object
            }),
            HDF5DataType::Integer {
                size: 8,
                signed: false
            }
        );
        assert_eq!(
            convert_dtype(&Dtype::Reference {
                ref_type: RefType::Region
            }),
            HDF5DataType::Integer {
                size: 12,
                signed: false
            }
        );
    }

    /// Compound field types recurse without a datatype round-trip.
    #[test]
    fn test_convert_dtype_compound_fields() {
        let dtype = Dtype::Compound {
            fields: vec![
                CompoundField {
                    name: "id".to_string(),
                    offset: 0,
                    dtype: int(4, true),
                },
                CompoundField {
                    name: "label".to_string(),
                    offset: 4,
                    dtype: Dtype::String {
                        fixed_len: Some(8),
                        charset: Charset::Ascii,
                    },
                },
            ],
        };
        match convert_dtype(&dtype) {
            HDF5DataType::Compound { fields } => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].0, "id");
                assert_eq!(
                    fields[1].1,
                    HDF5DataType::String {
                        encoding: StringEncoding::ASCII
                    }
                );
            }
            other => panic!("expected Compound, got {other:?}"),
        }
    }

    /// Enum members map straight across, no re-derivation needed.
    #[test]
    fn test_convert_dtype_enum_members() {
        let dtype = Dtype::Enum {
            base: Box::new(int(2, true)),
            members: vec![("RED".to_string(), 0), ("GREEN".to_string(), 1)],
        };
        match convert_dtype(&dtype) {
            HDF5DataType::Enum { values } => {
                assert_eq!(
                    values,
                    vec![("RED".to_string(), 0), ("GREEN".to_string(), 1)]
                );
            }
            other => panic!("expected Enum, got {other:?}"),
        }
    }

    /// Routing predicates agree with what the decoders actually accept.
    #[test]
    fn test_routing_predicates() {
        assert!(is_integral(&int(4, true)));
        assert!(is_integral(&Dtype::Bitfield {
            size: 8,
            order: ByteOrder::Little
        }));
        assert!(is_integral(&Dtype::Enum {
            base: Box::new(int(1, false)),
            members: vec![],
        }));
        // A 3-byte integer has no Rust counterpart, so it must not be routed
        // to dataset_to_i64 (which would fail).
        assert!(!is_integral(&int(3, true)));
        assert!(is_floating(&Dtype::Float {
            size: 8,
            order: ByteOrder::Little
        }));
        assert!(!is_floating(&int(8, true)));
    }
}
