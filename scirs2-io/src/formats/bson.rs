//! BSON (Binary JSON) encoder and decoder.
//!
//! Implements the [BSON specification](https://bsonspec.org/spec.html) in
//! pure Rust with no external crate dependencies beyond what is already in
//! the workspace.
//!
//! ## Layout
//! A BSON document is a sequence of elements preceded by a 4-byte LE int32
//! total-size (including the size field itself) and terminated by a `\x00`
//! null byte.
//!
//! ## BSON type codes
//! | Code | Type |
//! |------|------|
//! | 0x01 | Double (f64) |
//! | 0x02 | UTF-8 string |
//! | 0x03 | Embedded document |
//! | 0x04 | Array |
//! | 0x08 | Boolean |
//! | 0x0A | Null |
//! | 0x10 | Int32 |
//! | 0x12 | Int64 |
//! | 0x11 | Timestamp (uint64) |
//! | 0x05 | Binary |
//! | 0x07 | ObjectId (12 bytes) |

#![allow(dead_code)]

use indexmap::IndexMap;

use crate::error::IoError;

// ─────────────────────────────── Value types ─────────────────────────────────

/// Ordered map of string → [`BsonValue`] pairs representing one BSON document.
pub type BsonDocument = IndexMap<String, BsonValue>;

/// A single BSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum BsonValue {
    /// 0x01 – IEEE 754 double.
    Double(f64),
    /// 0x02 – UTF-8 string.
    String(String),
    /// 0x03 – embedded document.
    Document(BsonDocument),
    /// 0x04 – array (stored as a document whose keys are "0", "1", …).
    Array(Vec<BsonValue>),
    /// 0x08 – boolean.
    Boolean(bool),
    /// 0x0A – null.
    Null,
    /// 0x10 – 32-bit signed integer.
    Int32(i32),
    /// 0x12 – 64-bit signed integer.
    Int64(i64),
    /// 0x11 – 64-bit unsigned timestamp.
    Timestamp(u64),
    /// 0x05 – binary data (generic sub-type 0x00).
    Binary(Vec<u8>),
    /// 0x07 – ObjectId (12 raw bytes).
    ObjectId([u8; 12]),
}

// ─────────────────────────────── Type-code constants ─────────────────────────

const TYPE_DOUBLE: u8 = 0x01;
const TYPE_STRING: u8 = 0x02;
const TYPE_DOCUMENT: u8 = 0x03;
const TYPE_ARRAY: u8 = 0x04;
const TYPE_BINARY: u8 = 0x05;
const TYPE_OBJECT_ID: u8 = 0x07;
const TYPE_BOOLEAN: u8 = 0x08;
const TYPE_NULL: u8 = 0x0A;
const TYPE_TIMESTAMP: u8 = 0x11;
const TYPE_INT32: u8 = 0x10;
const TYPE_INT64: u8 = 0x12;

// ─────────────────────────────── Encoder ─────────────────────────────────────

/// Encode a [`BsonDocument`] into BSON bytes.
pub fn encode_bson(doc: &BsonDocument) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    // Reserve 4 bytes for the document length; we fill them in at the end.
    buf.extend_from_slice(&[0u8; 4]);
    encode_document_body(doc, &mut buf);
    buf.push(0x00); // terminating null byte
    let total = buf.len() as i32;
    buf[0..4].copy_from_slice(&total.to_le_bytes());
    buf
}

fn encode_document_body(doc: &BsonDocument, buf: &mut Vec<u8>) {
    for (key, value) in doc {
        encode_element(key, value, buf);
    }
}

fn encode_element(key: &str, value: &BsonValue, buf: &mut Vec<u8>) {
    match value {
        BsonValue::Double(v) => {
            buf.push(TYPE_DOUBLE);
            encode_cstring(key, buf);
            buf.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        BsonValue::String(s) => {
            buf.push(TYPE_STRING);
            encode_cstring(key, buf);
            encode_bson_string(s, buf);
        }
        BsonValue::Document(inner) => {
            buf.push(TYPE_DOCUMENT);
            encode_cstring(key, buf);
            let sub = encode_bson(inner);
            buf.extend_from_slice(&sub);
        }
        BsonValue::Array(items) => {
            buf.push(TYPE_ARRAY);
            encode_cstring(key, buf);
            // Arrays are encoded as documents with "0", "1", … keys.
            let mut arr_doc: BsonDocument = BsonDocument::new();
            for (i, item) in items.iter().enumerate() {
                arr_doc.insert(i.to_string(), item.clone());
            }
            let sub = encode_bson(&arr_doc);
            buf.extend_from_slice(&sub);
        }
        BsonValue::Boolean(v) => {
            buf.push(TYPE_BOOLEAN);
            encode_cstring(key, buf);
            buf.push(if *v { 0x01 } else { 0x00 });
        }
        BsonValue::Null => {
            buf.push(TYPE_NULL);
            encode_cstring(key, buf);
        }
        BsonValue::Int32(v) => {
            buf.push(TYPE_INT32);
            encode_cstring(key, buf);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        BsonValue::Int64(v) => {
            buf.push(TYPE_INT64);
            encode_cstring(key, buf);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        BsonValue::Timestamp(v) => {
            buf.push(TYPE_TIMESTAMP);
            encode_cstring(key, buf);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        BsonValue::Binary(data) => {
            buf.push(TYPE_BINARY);
            encode_cstring(key, buf);
            let len = data.len() as i32;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.push(0x00); // generic binary sub-type
            buf.extend_from_slice(data);
        }
        BsonValue::ObjectId(oid) => {
            buf.push(TYPE_OBJECT_ID);
            encode_cstring(key, buf);
            buf.extend_from_slice(oid);
        }
    }
}

/// Encode a null-terminated key string.
fn encode_cstring(s: &str, buf: &mut Vec<u8>) {
    buf.extend_from_slice(s.as_bytes());
    buf.push(0x00);
}

/// Encode a BSON string: 4-byte LE length (including null) + bytes + null.
fn encode_bson_string(s: &str, buf: &mut Vec<u8>) {
    let len = (s.len() + 1) as i32; // +1 for the null terminator
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
    buf.push(0x00);
}

// ─────────────────────────────── Decoder ─────────────────────────────────────

/// Decode a BSON document from the front of `data`.
///
/// Returns the decoded document; any trailing bytes beyond the document's
/// declared length are ignored.
pub fn decode_bson(data: &[u8]) -> Result<BsonDocument, IoError> {
    let (doc, _) = decode_document(data, 0)?;
    Ok(doc)
}

/// Decode a document starting at `offset`.  Returns `(doc, next_offset)`.
fn decode_document(data: &[u8], offset: usize) -> Result<(BsonDocument, usize), IoError> {
    if offset + 4 > data.len() {
        return Err(IoError::DeserializationError(
            "BSON: not enough bytes for document length".into(),
        ));
    }
    let len_bytes = [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ];
    let doc_len = i32::from_le_bytes(len_bytes) as usize;
    if doc_len < 5 || offset + doc_len > data.len() {
        return Err(IoError::DeserializationError(format!(
            "BSON: invalid document length {doc_len}"
        )));
    }

    let mut doc = BsonDocument::new();
    let mut pos = offset + 4;
    let end = offset + doc_len - 1; // exclusive of the trailing null byte

    while pos < end {
        let type_byte = data[pos];
        pos += 1;
        if type_byte == 0x00 {
            break; // Terminator encountered before calculated end – accept.
        }

        // Read key (null-terminated).
        let key_start = pos;
        while pos < data.len() && data[pos] != 0x00 {
            pos += 1;
        }
        if pos >= data.len() {
            return Err(IoError::DeserializationError(
                "BSON: unterminated element key".into(),
            ));
        }
        let key = std::str::from_utf8(&data[key_start..pos])
            .map_err(|e| IoError::DeserializationError(format!("BSON: invalid key UTF-8: {e}")))?
            .to_string();
        pos += 1; // skip null terminator

        let (value, next_pos) = decode_value(type_byte, data, pos)?;
        doc.insert(key, value);
        pos = next_pos;
    }

    Ok((doc, offset + doc_len))
}

fn decode_value(type_byte: u8, data: &[u8], offset: usize) -> Result<(BsonValue, usize), IoError> {
    match type_byte {
        TYPE_DOUBLE => {
            if offset + 8 > data.len() {
                return Err(IoError::DeserializationError(
                    "BSON: truncated double".into(),
                ));
            }
            let bits = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            Ok((BsonValue::Double(f64::from_bits(bits)), offset + 8))
        }

        TYPE_STRING => {
            let (s, next) = decode_bson_string(data, offset)?;
            Ok((BsonValue::String(s), next))
        }

        TYPE_DOCUMENT => {
            let (inner, next) = decode_document(data, offset)?;
            Ok((BsonValue::Document(inner), next))
        }

        TYPE_ARRAY => {
            let (inner, next) = decode_document(data, offset)?;
            // Re-assemble the array in index order.
            let mut items: Vec<(usize, BsonValue)> = inner
                .into_iter()
                .filter_map(|(k, v)| k.parse::<usize>().ok().map(|i| (i, v)))
                .collect();
            items.sort_by_key(|(i, _)| *i);
            Ok((
                BsonValue::Array(items.into_iter().map(|(_, v)| v).collect()),
                next,
            ))
        }

        TYPE_BINARY => {
            if offset + 5 > data.len() {
                return Err(IoError::DeserializationError(
                    "BSON: truncated binary header".into(),
                ));
            }
            let len = i32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            let _subtype = data[offset + 4];
            let start = offset + 5;
            if start + len > data.len() {
                return Err(IoError::DeserializationError(
                    "BSON: truncated binary data".into(),
                ));
            }
            Ok((
                BsonValue::Binary(data[start..start + len].to_vec()),
                start + len,
            ))
        }

        TYPE_OBJECT_ID => {
            if offset + 12 > data.len() {
                return Err(IoError::DeserializationError(
                    "BSON: truncated ObjectId".into(),
                ));
            }
            let mut oid = [0u8; 12];
            oid.copy_from_slice(&data[offset..offset + 12]);
            Ok((BsonValue::ObjectId(oid), offset + 12))
        }

        TYPE_BOOLEAN => {
            if offset >= data.len() {
                return Err(IoError::DeserializationError(
                    "BSON: truncated boolean".into(),
                ));
            }
            Ok((BsonValue::Boolean(data[offset] != 0x00), offset + 1))
        }

        TYPE_NULL => Ok((BsonValue::Null, offset)),

        TYPE_TIMESTAMP => {
            if offset + 8 > data.len() {
                return Err(IoError::DeserializationError(
                    "BSON: truncated timestamp".into(),
                ));
            }
            let v = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            Ok((BsonValue::Timestamp(v), offset + 8))
        }

        TYPE_INT32 => {
            if offset + 4 > data.len() {
                return Err(IoError::DeserializationError(
                    "BSON: truncated int32".into(),
                ));
            }
            let v = i32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            Ok((BsonValue::Int32(v), offset + 4))
        }

        TYPE_INT64 => {
            if offset + 8 > data.len() {
                return Err(IoError::DeserializationError(
                    "BSON: truncated int64".into(),
                ));
            }
            let v = i64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            Ok((BsonValue::Int64(v), offset + 8))
        }

        other => Err(IoError::DeserializationError(format!(
            "BSON: unsupported type code 0x{other:02X}"
        ))),
    }
}

/// Decode a BSON string (4-byte LE length + bytes + null) at `offset`.
fn decode_bson_string(data: &[u8], offset: usize) -> Result<(String, usize), IoError> {
    if offset + 4 > data.len() {
        return Err(IoError::DeserializationError(
            "BSON: truncated string length".into(),
        ));
    }
    let len = i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    let start = offset + 4;
    // len includes the null terminator
    if len == 0 || start + len > data.len() {
        return Err(IoError::DeserializationError(
            "BSON: invalid string length".into(),
        ));
    }
    let s = std::str::from_utf8(&data[start..start + len - 1])
        .map_err(|e| IoError::DeserializationError(format!("BSON: invalid string UTF-8: {e}")))?;
    Ok((s.to_string(), start + len))
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(doc: BsonDocument) -> BsonDocument {
        let bytes = encode_bson(&doc);
        decode_bson(&bytes).expect("decode failed")
    }

    #[test]
    fn test_simple_fields() {
        let mut doc = BsonDocument::new();
        doc.insert("name".into(), BsonValue::String("Alice".into()));
        doc.insert("age".into(), BsonValue::Int32(30));
        doc.insert("score".into(), BsonValue::Double(9.8));
        doc.insert("active".into(), BsonValue::Boolean(true));
        doc.insert("extra".into(), BsonValue::Null);

        let out = round_trip(doc.clone());
        assert_eq!(out["name"], BsonValue::String("Alice".into()));
        assert_eq!(out["age"], BsonValue::Int32(30));
        assert_eq!(out["active"], BsonValue::Boolean(true));
        assert_eq!(out["extra"], BsonValue::Null);
        if let BsonValue::Double(v) = out["score"] {
            assert!((v - 9.8).abs() < 1e-9);
        }
    }

    #[test]
    fn test_nested_document() {
        let mut inner = BsonDocument::new();
        inner.insert("x".into(), BsonValue::Int64(42));

        let mut doc = BsonDocument::new();
        doc.insert("inner".into(), BsonValue::Document(inner));

        let out = round_trip(doc);
        if let BsonValue::Document(ref d) = out["inner"] {
            assert_eq!(d["x"], BsonValue::Int64(42));
        } else {
            panic!("expected embedded document");
        }
    }

    #[test]
    fn test_array() {
        let mut doc = BsonDocument::new();
        doc.insert(
            "items".into(),
            BsonValue::Array(vec![
                BsonValue::Int32(1),
                BsonValue::Int32(2),
                BsonValue::Int32(3),
            ]),
        );
        let out = round_trip(doc);
        if let BsonValue::Array(ref arr) = out["items"] {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], BsonValue::Int32(1));
            assert_eq!(arr[2], BsonValue::Int32(3));
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn test_binary() {
        let mut doc = BsonDocument::new();
        doc.insert("data".into(), BsonValue::Binary(vec![0x01, 0x02, 0x03]));
        let out = round_trip(doc);
        assert_eq!(out["data"], BsonValue::Binary(vec![0x01, 0x02, 0x03]));
    }

    #[test]
    fn test_timestamp_and_int64() {
        let mut doc = BsonDocument::new();
        doc.insert("ts".into(), BsonValue::Timestamp(1_700_000_000));
        doc.insert("big".into(), BsonValue::Int64(i64::MAX));
        let out = round_trip(doc);
        assert_eq!(out["ts"], BsonValue::Timestamp(1_700_000_000));
        assert_eq!(out["big"], BsonValue::Int64(i64::MAX));
    }

    #[test]
    fn test_object_id() {
        let oid: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut doc = BsonDocument::new();
        doc.insert("_id".into(), BsonValue::ObjectId(oid));
        let out = round_trip(doc);
        assert_eq!(out["_id"], BsonValue::ObjectId(oid));
    }
}

// ─────────────────────────────── File I/O helpers ────────────────────────────

use std::fs;
use std::path::Path;

/// Read a BSON file from `path` and decode it as a [`BsonDocument`].
pub fn read_bson_file(path: impl AsRef<Path>) -> Result<BsonDocument, IoError> {
    let bytes = fs::read(path.as_ref())
        .map_err(|e| IoError::SerializationError(format!("BSON: cannot read file: {e}")))?;
    decode_bson(&bytes)
}

/// Encode `doc` as BSON bytes and write them to `path`.
pub fn write_bson_file(path: impl AsRef<Path>, doc: &BsonDocument) -> Result<(), IoError> {
    let bytes = encode_bson(doc);
    fs::write(path.as_ref(), &bytes)
        .map_err(|e| IoError::SerializationError(format!("BSON: cannot write file: {e}")))
}

#[cfg(test)]
mod file_tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_roundtrip_file() {
        let mut doc = BsonDocument::new();
        doc.insert("title".into(), BsonValue::String("SciRS2".into()));
        doc.insert("count".into(), BsonValue::Int32(42));
        doc.insert("score".into(), BsonValue::Double(9.95));

        let path = temp_dir().join("bson_roundtrip_test.bson");
        write_bson_file(&path, &doc).expect("write failed");
        let out = read_bson_file(&path).expect("read failed");
        assert_eq!(out["title"], BsonValue::String("SciRS2".into()));
        assert_eq!(out["count"], BsonValue::Int32(42));
        let _ = std::fs::remove_file(&path);
    }
}
