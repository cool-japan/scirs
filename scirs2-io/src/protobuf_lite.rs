//! Lightweight pure-Rust Protocol Buffers encoding and decoding.
//!
//! This module implements the core wire-format operations described in the
//! [Protocol Buffers encoding specification](https://protobuf.dev/programming-guides/encoding/):
//!
//! - Varint encoding / decoding (LEB-128 variant)
//! - Wire-type tagging and field-number encoding
//! - 32-bit and 64-bit fixed-width fields
//! - Length-delimited (bytes / strings / embedded messages) fields
//!
//! It intentionally avoids code generation.  You can use it to serialise ad-hoc
//! binary messages, parse legacy protobufs without a `.proto` compiler, or embed
//! lightweight framing in custom protocols.
//!
//! # Wire types
//!
//! | Wire type | Meaning                    |
//! |-----------|----------------------------|
//! | 0         | Varint                     |
//! | 1         | 64-bit fixed width         |
//! | 2         | Length-delimited (bytes)   |
//! | 5         | 32-bit fixed width         |
//!
//! # Example
//!
//! ```rust
//! use scirs2_io::protobuf_lite::{
//!     encode_varint, decode_varint,
//!     encode_field, decode_fields,
//!     ProtobufField,
//! };
//!
//! // Varint round-trip
//! let encoded = encode_varint(300);
//! let (value, rest) = decode_varint(&encoded).unwrap();
//! assert_eq!(value, 300);
//! assert!(rest.is_empty());
//!
//! // Field encoding
//! let bytes = encode_field(1, ProtobufField::Varint(42));
//! let fields = decode_fields(&bytes).unwrap();
//! assert_eq!(fields[0], (1, ProtobufField::Varint(42)));
//! ```

use crate::error::IoError;

/// Result alias used throughout this module.
pub type ProtoResult<T> = Result<T, IoError>;

// ─────────────────────────── Wire type constants ──────────────────────────────

const WIRE_VARINT: u8 = 0;
const WIRE_FIXED64: u8 = 1;
const WIRE_LEN_DELIM: u8 = 2;
const WIRE_FIXED32: u8 = 5;

// ─────────────────────────── ProtobufField ───────────────────────────────────

/// A single field value in a Protocol Buffer message.
///
/// Each variant corresponds to one of the four wire types used in this
/// implementation.
#[derive(Debug, Clone, PartialEq)]
pub enum ProtobufField {
    /// Wire type 0 — a variable-length integer (up to 64 bits).
    Varint(u64),
    /// Wire type 2 — an arbitrary byte string (also used for embedded messages
    /// and repeated packed fields).
    LengthDelimited(Vec<u8>),
    /// Wire type 1 — a 64-bit value in little-endian byte order.
    Fixed64([u8; 8]),
    /// Wire type 5 — a 32-bit value in little-endian byte order.
    Fixed32([u8; 4]),
}

impl ProtobufField {
    fn wire_type(&self) -> u8 {
        match self {
            ProtobufField::Varint(_) => WIRE_VARINT,
            ProtobufField::LengthDelimited(_) => WIRE_LEN_DELIM,
            ProtobufField::Fixed64(_) => WIRE_FIXED64,
            ProtobufField::Fixed32(_) => WIRE_FIXED32,
        }
    }

    /// Convenience: construct a `LengthDelimited` field from a UTF-8 string.
    pub fn from_str(s: &str) -> Self {
        ProtobufField::LengthDelimited(s.as_bytes().to_vec())
    }

    /// Convenience: construct a `LengthDelimited` field from an embedded message
    /// (previously encoded with `encode_fields`).
    pub fn from_message(encoded: Vec<u8>) -> Self {
        ProtobufField::LengthDelimited(encoded)
    }

    /// Construct a `Fixed64` from a `u64` value (little-endian).
    pub fn from_u64(v: u64) -> Self {
        ProtobufField::Fixed64(v.to_le_bytes())
    }

    /// Construct a `Fixed64` from an `f64` value (little-endian IEEE 754).
    pub fn from_f64(v: f64) -> Self {
        ProtobufField::Fixed64(v.to_le_bytes())
    }

    /// Construct a `Fixed32` from a `u32` value (little-endian).
    pub fn from_u32(v: u32) -> Self {
        ProtobufField::Fixed32(v.to_le_bytes())
    }

    /// Construct a `Fixed32` from an `f32` value (little-endian IEEE 754).
    pub fn from_f32(v: f32) -> Self {
        ProtobufField::Fixed32(v.to_le_bytes())
    }

    /// If this is a `Fixed64`, interpret it as a `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        if let ProtobufField::Fixed64(b) = self {
            Some(f64::from_le_bytes(*b))
        } else {
            None
        }
    }

    /// If this is a `Fixed32`, interpret it as a `f32`.
    pub fn as_f32(&self) -> Option<f32> {
        if let ProtobufField::Fixed32(b) = self {
            Some(f32::from_le_bytes(*b))
        } else {
            None
        }
    }

    /// If this is a `LengthDelimited`, try to interpret the bytes as UTF-8.
    pub fn as_str(&self) -> Option<&str> {
        if let ProtobufField::LengthDelimited(b) = self {
            std::str::from_utf8(b).ok()
        } else {
            None
        }
    }
}

// ─────────────────────────── Varint encoding ─────────────────────────────────

/// Encode a `u64` as a Protocol Buffers varint (LEB-128 variant).
///
/// The result is 1–10 bytes long.
///
/// # Example
///
/// ```rust
/// use scirs2_io::protobuf_lite::encode_varint;
///
/// assert_eq!(encode_varint(0), vec![0x00]);
/// assert_eq!(encode_varint(127), vec![0x7f]);
/// assert_eq!(encode_varint(128), vec![0x80, 0x01]);
/// assert_eq!(encode_varint(300), vec![0xac, 0x02]);
/// ```
pub fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(10);
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
    buf
}

/// Decode a Protocol Buffers varint from the beginning of `data`.
///
/// Returns `(decoded_value, remaining_bytes)`.  Returns an error if `data` is
/// empty or the varint is longer than 10 bytes (i.e. would overflow a `u64`).
///
/// # Example
///
/// ```rust
/// use scirs2_io::protobuf_lite::decode_varint;
///
/// let (v, rest) = decode_varint(&[0xac, 0x02, 0xff]).unwrap();
/// assert_eq!(v, 300);
/// assert_eq!(rest, &[0xff]);
/// ```
pub fn decode_varint(data: &[u8]) -> ProtoResult<(u64, &[u8])> {
    if data.is_empty() {
        return Err(IoError::ParseError("varint: empty input".to_string()));
    }

    let mut result: u64 = 0;
    let mut shift = 0u32;

    for (i, &byte) in data.iter().enumerate() {
        if shift >= 64 {
            return Err(IoError::ParseError(
                "varint: overflow (more than 10 bytes)".to_string(),
            ));
        }
        result |= ((byte & 0x7f) as u64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Ok((result, &data[i + 1..]));
        }
    }

    Err(IoError::ParseError(
        "varint: unterminated (missing stop byte)".to_string(),
    ))
}

// ─────────────────────────── ZigZag helpers ──────────────────────────────────

/// Encode a signed `i64` with ZigZag encoding before storing as a varint.
///
/// This is used for signed integers in Protocol Buffers `sint64` fields.
///
/// # Example
///
/// ```rust
/// use scirs2_io::protobuf_lite::encode_zigzag_i64;
///
/// assert_eq!(encode_zigzag_i64(0), vec![0x00]);
/// assert_eq!(encode_zigzag_i64(-1), vec![0x01]);
/// assert_eq!(encode_zigzag_i64(1), vec![0x02]);
/// ```
pub fn encode_zigzag_i64(value: i64) -> Vec<u8> {
    let zz = ((value << 1) ^ (value >> 63)) as u64;
    encode_varint(zz)
}

/// Decode a ZigZag-encoded varint back to a signed `i64`.
pub fn decode_zigzag_i64(data: &[u8]) -> ProtoResult<(i64, &[u8])> {
    let (zz, rest) = decode_varint(data)?;
    let v = ((zz >> 1) as i64) ^ -((zz & 1) as i64);
    Ok((v, rest))
}

// ─────────────────────────── Field encoding ──────────────────────────────────

/// Encode a single field with its tag number as Protocol Buffers wire bytes.
///
/// The tag is a field number (1–536_870_911) combined with the wire type in the
/// first varint.
///
/// # Example
///
/// ```rust
/// use scirs2_io::protobuf_lite::{encode_field, ProtobufField};
///
/// let bytes = encode_field(1, ProtobufField::Varint(150));
/// // tag=1 wire_type=0 → 0x08, then varint 150 → 0x96 0x01
/// assert_eq!(bytes, vec![0x08, 0x96, 0x01]);
/// ```
pub fn encode_field(tag: u32, field: ProtobufField) -> Vec<u8> {
    let wire_type = field.wire_type() as u32;
    let key = (tag << 3) | wire_type;
    let mut buf = encode_varint(key as u64);

    match field {
        ProtobufField::Varint(v) => buf.extend(encode_varint(v)),
        ProtobufField::LengthDelimited(bytes) => {
            buf.extend(encode_varint(bytes.len() as u64));
            buf.extend_from_slice(&bytes);
        }
        ProtobufField::Fixed64(b) => buf.extend_from_slice(&b),
        ProtobufField::Fixed32(b) => buf.extend_from_slice(&b),
    }

    buf
}

/// Encode multiple fields in order, concatenating them into one byte vector.
pub fn encode_fields(fields: &[(u32, ProtobufField)]) -> Vec<u8> {
    let mut buf = Vec::new();
    for (tag, field) in fields {
        buf.extend(encode_field(*tag, field.clone()));
    }
    buf
}

// ─────────────────────────── Field decoding ──────────────────────────────────

/// Decode all Protocol Buffers fields from a byte slice.
///
/// Returns a `Vec<(tag_number, ProtobufField)>`.  Fields are returned in the
/// order they appear in `data`.  Unknown wire types are returned as an error.
///
/// # Example
///
/// ```rust
/// use scirs2_io::protobuf_lite::{encode_field, decode_fields, ProtobufField};
///
/// let enc = encode_field(2, ProtobufField::from_str("hello"));
/// let fields = decode_fields(&enc).unwrap();
/// assert_eq!(fields[0].0, 2);
/// assert_eq!(fields[0].1.as_str(), Some("hello"));
/// ```
pub fn decode_fields(data: &[u8]) -> ProtoResult<Vec<(u32, ProtobufField)>> {
    let mut out = Vec::new();
    let mut pos = data;

    while !pos.is_empty() {
        // Key varint: (field_number << 3) | wire_type
        let (key, rest) = decode_varint(pos)?;
        let wire_type = (key & 0x07) as u8;
        let field_number = (key >> 3) as u32;

        let (field, remaining) = decode_one_field(wire_type, rest)?;
        out.push((field_number, field));
        pos = remaining;
    }

    Ok(out)
}

fn decode_one_field(wire_type: u8, data: &[u8]) -> ProtoResult<(ProtobufField, &[u8])> {
    match wire_type {
        WIRE_VARINT => {
            let (v, rest) = decode_varint(data)?;
            Ok((ProtobufField::Varint(v), rest))
        }
        WIRE_FIXED64 => {
            if data.len() < 8 {
                return Err(IoError::ParseError(
                    "fixed64: insufficient bytes".to_string(),
                ));
            }
            let mut b = [0u8; 8];
            b.copy_from_slice(&data[..8]);
            Ok((ProtobufField::Fixed64(b), &data[8..]))
        }
        WIRE_LEN_DELIM => {
            let (len, rest) = decode_varint(data)?;
            let len = len as usize;
            if rest.len() < len {
                return Err(IoError::ParseError(format!(
                    "length-delimited: need {len} bytes but only {} available",
                    rest.len()
                )));
            }
            let payload = rest[..len].to_vec();
            Ok((ProtobufField::LengthDelimited(payload), &rest[len..]))
        }
        WIRE_FIXED32 => {
            if data.len() < 4 {
                return Err(IoError::ParseError(
                    "fixed32: insufficient bytes".to_string(),
                ));
            }
            let mut b = [0u8; 4];
            b.copy_from_slice(&data[..4]);
            Ok((ProtobufField::Fixed32(b), &data[4..]))
        }
        _ => Err(IoError::ParseError(format!(
            "unknown wire type: {wire_type}"
        ))),
    }
}

// ─────────────────────────── Message builder ─────────────────────────────────

/// Convenience builder for constructing Protocol Buffer messages without
/// managing tag numbers manually.
///
/// # Example
///
/// ```rust
/// use scirs2_io::protobuf_lite::MessageBuilder;
///
/// let bytes = MessageBuilder::new()
///     .varint(1, 42)
///     .string(2, "hello")
///     .f64(3, 3.14)
///     .build();
///
/// // Decode and verify
/// use scirs2_io::protobuf_lite::decode_fields;
/// let fields = decode_fields(&bytes).unwrap();
/// assert_eq!(fields.len(), 3);
/// ```
#[derive(Debug, Default)]
pub struct MessageBuilder {
    buf: Vec<u8>,
}

impl MessageBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Append a varint field.
    pub fn varint(mut self, tag: u32, value: u64) -> Self {
        self.buf
            .extend(encode_field(tag, ProtobufField::Varint(value)));
        self
    }

    /// Append a signed integer field using ZigZag encoding.
    pub fn sint64(mut self, tag: u32, value: i64) -> Self {
        let zz = ((value << 1) ^ (value >> 63)) as u64;
        self.buf
            .extend(encode_field(tag, ProtobufField::Varint(zz)));
        self
    }

    /// Append a raw bytes / length-delimited field.
    pub fn bytes(mut self, tag: u32, value: Vec<u8>) -> Self {
        self.buf
            .extend(encode_field(tag, ProtobufField::LengthDelimited(value)));
        self
    }

    /// Append a UTF-8 string field (length-delimited).
    pub fn string(mut self, tag: u32, value: &str) -> Self {
        self.buf
            .extend(encode_field(tag, ProtobufField::from_str(value)));
        self
    }

    /// Append a 64-bit float field (fixed64).
    pub fn f64(mut self, tag: u32, value: f64) -> Self {
        self.buf
            .extend(encode_field(tag, ProtobufField::from_f64(value)));
        self
    }

    /// Append a 32-bit float field (fixed32).
    pub fn f32(mut self, tag: u32, value: f32) -> Self {
        self.buf
            .extend(encode_field(tag, ProtobufField::from_f32(value)));
        self
    }

    /// Append an embedded sub-message field (length-delimited).
    pub fn message(mut self, tag: u32, encoded: Vec<u8>) -> Self {
        self.buf
            .extend(encode_field(tag, ProtobufField::from_message(encoded)));
        self
    }

    /// Consume the builder and return the encoded bytes.
    pub fn build(self) -> Vec<u8> {
        self.buf
    }
}

// ─────────────────────────── Tests ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Varint ──────────────────────────────────────────────────────────────

    #[test]
    fn test_varint_single_byte_values() {
        for v in 0u64..128 {
            let enc = encode_varint(v);
            assert_eq!(enc.len(), 1);
            let (dec, rest) = decode_varint(&enc).expect("decode failed");
            assert_eq!(dec, v);
            assert!(rest.is_empty());
        }
    }

    #[test]
    fn test_varint_multi_byte() {
        let cases: &[(u64, &[u8])] = &[
            (128, &[0x80, 0x01]),
            (300, &[0xac, 0x02]),
            (16_383, &[0xff, 0x7f]),
            (16_384, &[0x80, 0x80, 0x01]),
        ];
        for (v, expected) in cases {
            let enc = encode_varint(*v);
            assert_eq!(&enc, expected, "encode mismatch for {v}");
            let (dec, rest) = decode_varint(&enc).expect("decode failed");
            assert_eq!(dec, *v);
            assert!(rest.is_empty());
        }
    }

    #[test]
    fn test_varint_max_u64() {
        let enc = encode_varint(u64::MAX);
        assert_eq!(enc.len(), 10);
        let (dec, rest) = decode_varint(&enc).expect("decode max");
        assert_eq!(dec, u64::MAX);
        assert!(rest.is_empty());
    }

    #[test]
    fn test_decode_varint_with_trailing_bytes() {
        let data = [0xac, 0x02, 0xde, 0xad];
        let (v, rest) = decode_varint(&data).expect("decode");
        assert_eq!(v, 300);
        assert_eq!(rest, &[0xde, 0xad]);
    }

    #[test]
    fn test_decode_varint_empty_is_error() {
        assert!(decode_varint(&[]).is_err());
    }

    #[test]
    fn test_decode_varint_unterminated_is_error() {
        // All bytes have the continuation bit set → no terminator
        let bad = [0x80u8; 11];
        assert!(decode_varint(&bad).is_err());
    }

    // ── ZigZag ──────────────────────────────────────────────────────────────

    #[test]
    fn test_zigzag_roundtrip() {
        let values: &[i64] = &[0, -1, 1, -2147483648, 2147483647, i64::MIN, i64::MAX];
        for &v in values {
            let enc = encode_zigzag_i64(v);
            let (dec, rest) = decode_zigzag_i64(&enc).expect("zigzag decode");
            assert_eq!(dec, v, "zigzag roundtrip for {v}");
            assert!(rest.is_empty());
        }
    }

    // ── Field encoding ───────────────────────────────────────────────────────

    #[test]
    fn test_encode_field_varint_example_from_spec() {
        // Official protobuf example: field 1, varint 150 → 08 96 01
        let enc = encode_field(1, ProtobufField::Varint(150));
        assert_eq!(enc, vec![0x08, 0x96, 0x01]);
    }

    #[test]
    fn test_encode_decode_string_field() {
        let enc = encode_field(2, ProtobufField::from_str("testing"));
        let fields = decode_fields(&enc).expect("decode");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, 2);
        assert_eq!(fields[0].1.as_str(), Some("testing"));
    }

    #[test]
    fn test_encode_decode_f64_field() {
        let value = std::f64::consts::PI;
        let enc = encode_field(3, ProtobufField::from_f64(value));
        let fields = decode_fields(&enc).expect("decode");
        let decoded = fields[0].1.as_f64().expect("as_f64");
        assert!((decoded - value).abs() < 1e-15);
    }

    #[test]
    fn test_encode_decode_f32_field() {
        let value = std::f32::consts::E;
        let enc = encode_field(4, ProtobufField::from_f32(value));
        let fields = decode_fields(&enc).expect("decode");
        let decoded = fields[0].1.as_f32().expect("as_f32");
        assert!((decoded - value).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_fields_roundtrip() {
        let msg = MessageBuilder::new()
            .varint(1, 42)
            .string(2, "hello world")
            .f64(3, std::f64::consts::E)
            .bytes(4, vec![0xde, 0xad, 0xbe, 0xef])
            .build();

        let fields = decode_fields(&msg).expect("decode message");
        assert_eq!(fields.len(), 4);

        assert_eq!(fields[0], (1, ProtobufField::Varint(42)));
        assert_eq!(fields[1].0, 2);
        assert_eq!(fields[1].1.as_str(), Some("hello world"));
        assert_eq!(fields[2].0, 3);
        assert!((fields[2].1.as_f64().unwrap() - std::f64::consts::E).abs() < 1e-9);
        assert_eq!(
            fields[3].1,
            ProtobufField::LengthDelimited(vec![0xde, 0xad, 0xbe, 0xef])
        );
    }

    #[test]
    fn test_embedded_message() {
        let inner = MessageBuilder::new()
            .varint(1, 100)
            .string(2, "inner")
            .build();

        let outer = MessageBuilder::new()
            .varint(1, 999)
            .message(2, inner.clone())
            .build();

        let fields = decode_fields(&outer).expect("decode outer");
        assert_eq!(fields.len(), 2);
        let inner_bytes = if let ProtobufField::LengthDelimited(b) = &fields[1].1 {
            b.clone()
        } else {
            panic!("expected LengthDelimited for embedded message");
        };
        assert_eq!(inner_bytes, inner);

        // Decode the inner message too
        let inner_fields = decode_fields(&inner_bytes).expect("decode inner");
        assert_eq!(inner_fields.len(), 2);
        assert_eq!(inner_fields[0], (1, ProtobufField::Varint(100)));
    }

    #[test]
    fn test_unknown_wire_type_returns_error() {
        // Craft a key with wire type 3 (not defined in our subset)
        let bad_key = encode_varint((1u64 << 3) | 3); // field 1, wire_type 3
        let result = decode_fields(&bad_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_sint64_roundtrip() {
        let values: &[i64] = &[-1000, -1, 0, 1, 1000, i64::MIN / 2, i64::MAX / 2];
        for &v in values {
            let msg = MessageBuilder::new().sint64(1, v).build();
            let fields = decode_fields(&msg).expect("decode sint64");
            // Decode ZigZag from the varint
            if let ProtobufField::Varint(zz) = fields[0].1 {
                let decoded = ((zz >> 1) as i64) ^ -((zz & 1) as i64);
                assert_eq!(decoded, v, "sint64 roundtrip for {v}");
            } else {
                panic!("expected Varint");
            }
        }
    }

    #[test]
    fn test_empty_string_field() {
        let enc = encode_field(5, ProtobufField::from_str(""));
        let fields = decode_fields(&enc).expect("decode empty str");
        assert_eq!(fields[0].1.as_str(), Some(""));
    }

    #[test]
    fn test_decode_fields_empty_input() {
        let fields = decode_fields(&[]).expect("decode empty");
        assert!(fields.is_empty());
    }
}
