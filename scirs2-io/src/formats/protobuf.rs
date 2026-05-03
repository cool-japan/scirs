//! Protocol Buffers wire format — higher-level encoding in the `formats` module.
//!
//! This module builds on top of the workspace's varint primitives
//! ([`crate::protobuf_lite`]) and adds:
//!
//! - [`ProtoValue`]: a recursive value enum covering all four wire types plus
//!   embedded-message and repeated-field convenience variants.
//! - [`ProtoField`]: a numbered field (field_number + [`ProtoValue`]).
//! - [`ProtoMessage`]: a trait that types can implement to gain
//!   `encode` / `decode` support without a protoc code-generator.
//! - [`ProtoMessageBuilder`]: a builder that constructs a `Vec<ProtoField>`
//!   step-by-step and serialises the result to raw bytes.
//! - [`ProtoDescriptor`]: a lightweight runtime schema descriptor (field name +
//!   number + type tag) analogous to a `.proto` file's `message` block.
//! - [`VarintEncoder`] / [`VarintDecoder`]: standalone LEB-128 helpers that
//!   wrap the `protobuf_lite` primitives for use in other encoding layers.
//!
//! # Wire types
//!
//! | Wire type | Enum arm                     |
//! |-----------|------------------------------|
//! | 0         | `ProtoValue::Varint`         |
//! | 1         | `ProtoValue::Fixed64`        |
//! | 2         | `ProtoValue::LengthDelimited`|
//! | 5         | `ProtoValue::Fixed32`        |
//!
//! # Example
//!
//! ```rust
//! use scirs2_io::formats::protobuf::{
//!     ProtoMessageBuilder, ProtoValue, decode_proto_fields,
//! };
//!
//! // Build a simple message: field 1 = varint 42, field 2 = "hello"
//! let bytes = ProtoMessageBuilder::new()
//!     .add_varint(1, 42)
//!     .add_string(2, "hello")
//!     .build();
//!
//! let fields = decode_proto_fields(&bytes).expect("decode");
//! assert_eq!(fields[0].field_number, 1);
//! assert_eq!(fields[1].field_number, 2);
//! if let ProtoValue::LengthDelimited(ref b) = fields[1].value {
//!     assert_eq!(std::str::from_utf8(b).unwrap(), "hello");
//! }
//! ```

#![allow(dead_code)]

use crate::error::IoError;

/// Result type used throughout this module.
pub type ProtoResult<T> = Result<T, IoError>;

// ─────────────────────────────────────── Wire-type constants ─────────────────

const WIRE_VARINT: u8 = 0;
const WIRE_FIXED64: u8 = 1;
const WIRE_LEN_DELIM: u8 = 2;
const WIRE_FIXED32: u8 = 5;

// ─────────────────────────────────────── VarintEncoder ───────────────────────

/// Standalone LEB-128 variable-length integer encoder.
///
/// Wraps the primitive from [`crate::protobuf_lite`] with an object-oriented
/// API for streaming encoding contexts where you want to push individual values
/// into a buffer rather than allocate a fresh `Vec` per call.
#[derive(Debug, Default)]
pub struct VarintEncoder {
    buf: Vec<u8>,
}

impl VarintEncoder {
    /// Create a new, empty encoder.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Encode `value` and append it to the internal buffer.
    pub fn push_u64(&mut self, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                self.buf.push(byte);
                break;
            }
            self.buf.push(byte | 0x80);
        }
    }

    /// Encode a signed `i64` using ZigZag encoding then push.
    pub fn push_i64(&mut self, value: i64) {
        let zz = zigzag_encode(value);
        self.push_u64(zz);
    }

    /// Consume the encoder and return the encoded bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Borrow the internal buffer.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Clear all accumulated bytes without reallocating.
    pub fn clear(&mut self) {
        self.buf.clear();
    }
}

// ─────────────────────────────────────── VarintDecoder ───────────────────────

/// Standalone LEB-128 variable-length integer decoder.
///
/// Provides a cursor-based API: call [`VarintDecoder::read_u64`] or
/// [`VarintDecoder::read_i64`] (ZigZag) to consume one varint at a time.
pub struct VarintDecoder<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> VarintDecoder<'a> {
    /// Create a new decoder over `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current byte offset into the input.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// `true` if all bytes have been consumed.
    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Read one LEB-128 `u64` from the current position.
    ///
    /// Returns `Err` if the input is empty, truncated, or overflows `u64`.
    pub fn read_u64(&mut self) -> ProtoResult<u64> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            if self.pos >= self.data.len() {
                return Err(IoError::FormatError(
                    "varint: unexpected end of input".to_string(),
                ));
            }
            let byte = self.data[self.pos];
            self.pos += 1;
            if shift >= 64 {
                return Err(IoError::FormatError(
                    "varint: value overflows u64 (> 10 bytes)".to_string(),
                ));
            }
            result |= ((byte & 0x7f) as u64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
        }
    }

    /// Read one ZigZag-encoded `i64` varint.
    pub fn read_i64(&mut self) -> ProtoResult<i64> {
        let zz = self.read_u64()?;
        Ok(zigzag_decode(zz))
    }
}

// ─────────────────────────────────────── ZigZag helpers ──────────────────────

/// ZigZag-encode a signed integer to an unsigned integer.
///
/// Maps: 0 → 0, -1 → 1, 1 → 2, -2 → 3, …
#[inline]
pub fn zigzag_encode(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// ZigZag-decode an unsigned integer to a signed integer.
#[inline]
pub fn zigzag_decode(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

// ─────────────────────────────────────── ProtoValue ──────────────────────────

/// A Protocol Buffers value carrying both the data and its wire-type information.
#[derive(Debug, Clone, PartialEq)]
pub enum ProtoValue {
    /// Wire type 0 — variable-length unsigned integer (up to 64 bits).
    Varint(u64),

    /// Wire type 0 — ZigZag-encoded signed integer stored as a varint.
    ///
    /// Use this variant when your `.proto` schema uses `sint32` or `sint64`.
    SignedVarint(i64),

    /// Wire type 1 — 64-bit fixed-width field (little-endian).
    Fixed64([u8; 8]),

    /// Wire type 2 — length-delimited byte string.
    ///
    /// Used for `bytes`, `string`, embedded messages, and packed repeated fields.
    LengthDelimited(Vec<u8>),

    /// Wire type 5 — 32-bit fixed-width field (little-endian).
    Fixed32([u8; 4]),
}

impl ProtoValue {
    /// Return the wire-type integer for this value.
    pub fn wire_type(&self) -> u8 {
        match self {
            ProtoValue::Varint(_) | ProtoValue::SignedVarint(_) => WIRE_VARINT,
            ProtoValue::Fixed64(_) => WIRE_FIXED64,
            ProtoValue::LengthDelimited(_) => WIRE_LEN_DELIM,
            ProtoValue::Fixed32(_) => WIRE_FIXED32,
        }
    }

    /// Interpret the value as a UTF-8 string (only valid for `LengthDelimited`).
    ///
    /// Returns `None` for other variants or if the bytes are not valid UTF-8.
    pub fn as_str(&self) -> Option<&str> {
        if let ProtoValue::LengthDelimited(b) = self {
            std::str::from_utf8(b).ok()
        } else {
            None
        }
    }

    /// Convenience constructor: UTF-8 string → `LengthDelimited`.
    pub fn from_string(s: &str) -> Self {
        ProtoValue::LengthDelimited(s.as_bytes().to_vec())
    }

    /// Convenience constructor: raw bytes → `LengthDelimited`.
    pub fn from_bytes(b: Vec<u8>) -> Self {
        ProtoValue::LengthDelimited(b)
    }

    /// Convenience constructor: embedded message (already encoded) → `LengthDelimited`.
    pub fn from_embedded_message(encoded: Vec<u8>) -> Self {
        ProtoValue::LengthDelimited(encoded)
    }

    /// Convenience constructor: `f64` → `Fixed64` (IEEE 754 LE).
    pub fn from_f64(v: f64) -> Self {
        ProtoValue::Fixed64(v.to_le_bytes())
    }

    /// Convenience constructor: `f32` → `Fixed32` (IEEE 754 LE).
    pub fn from_f32(v: f32) -> Self {
        ProtoValue::Fixed32(v.to_le_bytes())
    }

    /// Interpret as `f64` if this is a `Fixed64` field.
    pub fn as_f64(&self) -> Option<f64> {
        if let ProtoValue::Fixed64(b) = self {
            Some(f64::from_le_bytes(*b))
        } else {
            None
        }
    }

    /// Interpret as `f32` if this is a `Fixed32` field.
    pub fn as_f32(&self) -> Option<f32> {
        if let ProtoValue::Fixed32(b) = self {
            Some(f32::from_le_bytes(*b))
        } else {
            None
        }
    }

    /// Encode this value (not including the tag) into `buf`.
    fn encode_into(&self, buf: &mut Vec<u8>) {
        match self {
            ProtoValue::Varint(v) => {
                let mut enc = VarintEncoder::new();
                enc.push_u64(*v);
                buf.extend_from_slice(enc.as_bytes());
            }
            ProtoValue::SignedVarint(v) => {
                let mut enc = VarintEncoder::new();
                enc.push_i64(*v);
                buf.extend_from_slice(enc.as_bytes());
            }
            ProtoValue::Fixed64(b) => buf.extend_from_slice(b),
            ProtoValue::LengthDelimited(b) => {
                let mut enc = VarintEncoder::new();
                enc.push_u64(b.len() as u64);
                buf.extend_from_slice(enc.as_bytes());
                buf.extend_from_slice(b);
            }
            ProtoValue::Fixed32(b) => buf.extend_from_slice(b),
        }
    }
}

// ─────────────────────────────────────── ProtoField ──────────────────────────

/// A single field in a Protocol Buffers message: a field number plus a value.
#[derive(Debug, Clone, PartialEq)]
pub struct ProtoField {
    /// The field number from the `.proto` schema (1-based, must be ≥ 1).
    pub field_number: u32,
    /// The encoded value of the field.
    pub value: ProtoValue,
}

impl ProtoField {
    /// Construct a new field.
    pub fn new(field_number: u32, value: ProtoValue) -> Self {
        Self {
            field_number,
            value,
        }
    }

    /// Encode the tag (field number + wire type) as a varint.
    fn encode_tag(&self) -> u64 {
        ((self.field_number as u64) << 3) | (self.value.wire_type() as u64)
    }

    /// Encode this field to bytes (tag varint + value bytes) and append to `buf`.
    pub fn encode_into(&self, buf: &mut Vec<u8>) {
        let tag = self.encode_tag();
        let mut enc = VarintEncoder::new();
        enc.push_u64(tag);
        buf.extend_from_slice(enc.as_bytes());
        self.value.encode_into(buf);
    }
}

// ─────────────────────────────────────── ProtoMessageBuilder ─────────────────

/// Builder for constructing Protocol Buffers messages field-by-field.
///
/// # Example
/// ```rust
/// use scirs2_io::formats::protobuf::ProtoMessageBuilder;
///
/// let bytes = ProtoMessageBuilder::new()
///     .add_varint(1, 99)
///     .add_string(2, "hello")
///     .add_f64(3, 3.14159)
///     .build();
/// assert!(!bytes.is_empty());
/// ```
#[derive(Debug, Default)]
pub struct ProtoMessageBuilder {
    fields: Vec<ProtoField>,
}

impl ProtoMessageBuilder {
    /// Create a new, empty builder.
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// Add a varint (wire type 0) field.
    pub fn add_varint(mut self, field_number: u32, value: u64) -> Self {
        self.fields
            .push(ProtoField::new(field_number, ProtoValue::Varint(value)));
        self
    }

    /// Add a ZigZag-encoded signed integer (wire type 0, `sint64` semantics).
    pub fn add_sint64(mut self, field_number: u32, value: i64) -> Self {
        self.fields.push(ProtoField::new(
            field_number,
            ProtoValue::SignedVarint(value),
        ));
        self
    }

    /// Add a bool as a varint (0 = false, 1 = true).
    pub fn add_bool(mut self, field_number: u32, value: bool) -> Self {
        self.add_varint(field_number, if value { 1 } else { 0 })
    }

    /// Add a length-delimited UTF-8 string field.
    pub fn add_string(mut self, field_number: u32, value: &str) -> Self {
        self.fields.push(ProtoField::new(
            field_number,
            ProtoValue::from_string(value),
        ));
        self
    }

    /// Add a length-delimited raw bytes field.
    pub fn add_bytes(mut self, field_number: u32, value: Vec<u8>) -> Self {
        self.fields
            .push(ProtoField::new(field_number, ProtoValue::from_bytes(value)));
        self
    }

    /// Add an embedded message (already encoded to bytes) as a length-delimited field.
    pub fn add_message(mut self, field_number: u32, encoded: Vec<u8>) -> Self {
        self.fields.push(ProtoField::new(
            field_number,
            ProtoValue::from_embedded_message(encoded),
        ));
        self
    }

    /// Add a `f64` as a fixed-64 field (wire type 1, little-endian IEEE 754).
    pub fn add_f64(mut self, field_number: u32, value: f64) -> Self {
        self.fields
            .push(ProtoField::new(field_number, ProtoValue::from_f64(value)));
        self
    }

    /// Add a `f32` as a fixed-32 field (wire type 5, little-endian IEEE 754).
    pub fn add_f32(mut self, field_number: u32, value: f32) -> Self {
        self.fields
            .push(ProtoField::new(field_number, ProtoValue::from_f32(value)));
        self
    }

    /// Add a `u32` fixed 32-bit field (wire type 5).
    pub fn add_fixed32(mut self, field_number: u32, value: u32) -> Self {
        self.fields.push(ProtoField::new(
            field_number,
            ProtoValue::Fixed32(value.to_le_bytes()),
        ));
        self
    }

    /// Add a `u64` fixed 64-bit field (wire type 1).
    pub fn add_fixed64(mut self, field_number: u32, value: u64) -> Self {
        self.fields.push(ProtoField::new(
            field_number,
            ProtoValue::Fixed64(value.to_le_bytes()),
        ));
        self
    }

    /// Add a packed repeated varint field (length-delimited, wire type 2).
    ///
    /// All `values` are encoded contiguously as varints inside one
    /// length-delimited byte string, following the proto3 packed encoding spec.
    pub fn add_packed_varints(mut self, field_number: u32, values: &[u64]) -> Self {
        let mut enc = VarintEncoder::new();
        for &v in values {
            enc.push_u64(v);
        }
        let packed = enc.into_bytes();
        self.fields.push(ProtoField::new(
            field_number,
            ProtoValue::LengthDelimited(packed),
        ));
        self
    }

    /// Serialise all accumulated fields to a `Vec<u8>`.
    pub fn build(self) -> Vec<u8> {
        let mut buf = Vec::new();
        for field in &self.fields {
            field.encode_into(&mut buf);
        }
        buf
    }

    /// Expose the accumulated fields without consuming.
    pub fn fields(&self) -> &[ProtoField] {
        &self.fields
    }
}

// ─────────────────────────────────────── Decode ──────────────────────────────

/// Decode a raw Protocol Buffers byte slice into a list of [`ProtoField`]s.
///
/// Unknown or reserved wire types (3, 4, 6, 7) cause an error.  The decoder
/// correctly handles varints that span multiple bytes, including 10-byte
/// `u64::MAX` varints.
///
/// This is a convenience wrapper around [`ProtoDecoder::decode_all`].
///
/// # Errors
/// Returns [`IoError::FormatError`] on malformed input.
pub fn decode_proto_fields(data: &[u8]) -> ProtoResult<Vec<ProtoField>> {
    ProtoDecoder::new(data).decode_all()
}

// ─────────────────────────────────────── ProtoDecoder (cursor-based) ─────────

/// A cursor-based Protocol Buffers decoder.
///
/// Unlike the free-function [`decode_proto_fields`], this struct maintains an
/// explicit absolute position and allows incremental (field-by-field) decoding
/// from a larger byte stream.
pub struct ProtoDecoder<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ProtoDecoder<'a> {
    /// Create a new decoder over `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current byte offset.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// `true` if all bytes have been consumed.
    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Read one LEB-128 varint from the current position.
    fn read_varint(&mut self) -> ProtoResult<u64> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            if self.pos >= self.data.len() {
                return Err(IoError::FormatError(
                    "protobuf: unexpected end of data reading varint".to_string(),
                ));
            }
            let byte = self.data[self.pos];
            self.pos += 1;
            if shift >= 64 {
                return Err(IoError::FormatError(
                    "protobuf: varint overflows u64".to_string(),
                ));
            }
            result |= ((byte & 0x7f) as u64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
        }
    }

    /// Read exactly `n` bytes and return a slice reference.
    fn read_bytes(&mut self, n: usize) -> ProtoResult<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(IoError::FormatError(format!(
                "protobuf: need {n} bytes at offset {} but only {} remain",
                self.pos,
                self.data.len() - self.pos
            )));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Decode the next field from the stream.
    ///
    /// Returns `None` when exhausted.
    pub fn next_field(&mut self) -> ProtoResult<Option<ProtoField>> {
        if self.is_exhausted() {
            return Ok(None);
        }

        let tag = self.read_varint()?;
        let wire_type = (tag & 0x07) as u8;
        let field_number = (tag >> 3) as u32;

        if field_number == 0 {
            return Err(IoError::FormatError(
                "protobuf: field number 0 is reserved".to_string(),
            ));
        }

        let value = match wire_type {
            WIRE_VARINT => ProtoValue::Varint(self.read_varint()?),
            WIRE_FIXED64 => {
                let bytes = self.read_bytes(8)?;
                let arr: [u8; 8] = bytes.try_into().map_err(|_| {
                    IoError::FormatError("protobuf: fixed64 conversion failed".to_string())
                })?;
                ProtoValue::Fixed64(arr)
            }
            WIRE_LEN_DELIM => {
                let len = self.read_varint()? as usize;
                let bytes = self.read_bytes(len)?;
                ProtoValue::LengthDelimited(bytes.to_vec())
            }
            WIRE_FIXED32 => {
                let bytes = self.read_bytes(4)?;
                let arr: [u8; 4] = bytes.try_into().map_err(|_| {
                    IoError::FormatError("protobuf: fixed32 conversion failed".to_string())
                })?;
                ProtoValue::Fixed32(arr)
            }
            wt => {
                return Err(IoError::FormatError(format!(
                    "protobuf: unsupported wire type {wt} for field {field_number}"
                )));
            }
        };

        Ok(Some(ProtoField::new(field_number, value)))
    }

    /// Decode all remaining fields from the stream.
    pub fn decode_all(&mut self) -> ProtoResult<Vec<ProtoField>> {
        let mut fields = Vec::new();
        while let Some(field) = self.next_field()? {
            fields.push(field);
        }
        Ok(fields)
    }
}

// ─────────────────────────────────────── ProtoMessage trait ──────────────────

/// A trait for types that can be serialised to and deserialised from the
/// Protocol Buffers wire format without a code generator.
///
/// Implementors encode their fields via a [`ProtoMessageBuilder`] and decode
/// from a list of [`ProtoField`]s.
///
/// # Example
///
/// ```rust
/// use scirs2_io::formats::protobuf::{
///     ProtoMessage, ProtoMessageBuilder, ProtoField, ProtoValue, ProtoResult,
/// };
///
/// #[derive(Debug, PartialEq)]
/// struct Point {
///     x: f64,
///     y: f64,
///     label: String,
/// }
///
/// impl ProtoMessage for Point {
///     fn encode(&self) -> Vec<u8> {
///         ProtoMessageBuilder::new()
///             .add_f64(1, self.x)
///             .add_f64(2, self.y)
///             .add_string(3, &self.label)
///             .build()
///     }
///
///     fn decode(fields: &[ProtoField]) -> ProtoResult<Self> {
///         use scirs2_io::error::IoError;
///         let mut x = None;
///         let mut y = None;
///         let mut label = String::new();
///         for field in fields {
///             match field.field_number {
///                 1 => x = field.value.as_f64(),
///                 2 => y = field.value.as_f64(),
///                 3 => label = field.value.as_str().unwrap_or("").to_string(),
///                 _ => {}
///             }
///         }
///         Ok(Point {
///             x: x.ok_or_else(|| IoError::FormatError("missing field x".into()))?,
///             y: y.ok_or_else(|| IoError::FormatError("missing field y".into()))?,
///             label,
///         })
///     }
/// }
///
/// let p = Point { x: 1.0, y: 2.5, label: "origin".into() };
/// let bytes = p.encode();
/// let mut dec = scirs2_io::formats::protobuf::ProtoDecoder::new(&bytes);
/// let fields = dec.decode_all().unwrap();
/// let p2 = Point::decode(&fields).unwrap();
/// assert_eq!(p, p2);
/// ```
pub trait ProtoMessage: Sized {
    /// Serialise `self` to protobuf wire-format bytes.
    fn encode(&self) -> Vec<u8>;

    /// Deserialise from a list of decoded fields.
    fn decode(fields: &[ProtoField]) -> ProtoResult<Self>;

    /// Convenience: decode from raw bytes (parses fields then calls `decode`).
    fn from_bytes(bytes: &[u8]) -> ProtoResult<Self> {
        let mut dec = ProtoDecoder::new(bytes);
        let fields = dec.decode_all()?;
        Self::decode(&fields)
    }

    /// Convenience: encode to raw bytes (same as `encode()` but emphasises the
    /// byte-level output).
    fn to_bytes(&self) -> Vec<u8> {
        self.encode()
    }
}

// ─────────────────────────────────────── ProtoDescriptor ─────────────────────

/// The data-type of a field in a [`ProtoDescriptor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtoType {
    /// Unsigned integer (varint on wire).
    Uint64,
    /// Signed integer, ZigZag-encoded (varint on wire).
    Sint64,
    /// Boolean flag (varint 0/1 on wire).
    Bool,
    /// IEEE-754 double (fixed-64 on wire).
    Double,
    /// IEEE-754 single (fixed-32 on wire).
    Float,
    /// Raw byte string (length-delimited on wire).
    Bytes,
    /// UTF-8 string (length-delimited on wire).
    String,
    /// Embedded message (length-delimited on wire).
    Message,
}

/// A descriptor for a single field inside a message schema.
#[derive(Debug, Clone)]
pub struct ProtoFieldDescriptor {
    /// Human-readable name (not encoded on the wire).
    pub name: String,
    /// Wire field number.
    pub field_number: u32,
    /// Expected data type.
    pub field_type: ProtoType,
    /// If `true`, the field may be absent and will be skipped if so.
    pub optional: bool,
}

impl ProtoFieldDescriptor {
    /// Build a new descriptor.
    pub fn new(
        name: impl Into<String>,
        field_number: u32,
        field_type: ProtoType,
        optional: bool,
    ) -> Self {
        Self {
            name: name.into(),
            field_number,
            field_type,
            optional,
        }
    }
}

/// A lightweight runtime schema descriptor for a protobuf message.
///
/// Acts similarly to a `.proto` file's `message` block: it enumerates the
/// expected fields and their types, enabling generic validation without
/// code generation.
///
/// # Example
/// ```rust
/// use scirs2_io::formats::protobuf::{
///     ProtoDescriptor, ProtoFieldDescriptor, ProtoType, ProtoMessageBuilder,
/// };
///
/// let desc = ProtoDescriptor::new("Person")
///     .field("id",   1, ProtoType::Uint64, false)
///     .field("name", 2, ProtoType::String, false)
///     .field("age",  3, ProtoType::Uint64, true);
///
/// let bytes = ProtoMessageBuilder::new()
///     .add_varint(1, 42)
///     .add_string(2, "Alice")
///     .build();
///
/// let result = desc.validate_bytes(&bytes);
/// assert!(result.is_ok(), "validation failed: {:?}", result);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ProtoDescriptor {
    /// Name of this message type (for diagnostic messages only).
    pub message_name: String,
    /// Ordered list of field descriptors.
    pub fields: Vec<ProtoFieldDescriptor>,
}

impl ProtoDescriptor {
    /// Create a new, empty descriptor for `message_name`.
    pub fn new(message_name: impl Into<String>) -> Self {
        Self {
            message_name: message_name.into(),
            fields: Vec::new(),
        }
    }

    /// Add a field descriptor (builder-style).
    pub fn field(
        mut self,
        name: impl Into<String>,
        field_number: u32,
        field_type: ProtoType,
        optional: bool,
    ) -> Self {
        self.fields.push(ProtoFieldDescriptor::new(
            name,
            field_number,
            field_type,
            optional,
        ));
        self
    }

    /// Validate a decoded list of [`ProtoField`]s against this descriptor.
    ///
    /// Returns `Ok(())` if all required fields are present and no wire-type
    /// mismatches are detected; otherwise `Err` with a diagnostic message.
    pub fn validate(&self, fields: &[ProtoField]) -> ProtoResult<()> {
        // Check that all required fields are present
        for desc in &self.fields {
            if desc.optional {
                continue;
            }
            let found = fields.iter().any(|f| f.field_number == desc.field_number);
            if !found {
                return Err(IoError::FormatError(format!(
                    "protobuf: required field '{}' (number {}) is missing from '{}'",
                    desc.name, desc.field_number, self.message_name
                )));
            }
        }

        // Check wire-type compatibility for all received fields
        for field in fields {
            if let Some(desc) = self
                .fields
                .iter()
                .find(|d| d.field_number == field.field_number)
            {
                let expected_wt = expected_wire_type(&desc.field_type);
                let actual_wt = field.value.wire_type();
                if expected_wt != actual_wt {
                    return Err(IoError::FormatError(format!(
                        "protobuf: field '{}' (number {}) wire-type mismatch: expected {expected_wt}, got {actual_wt}",
                        desc.name, desc.field_number
                    )));
                }
            }
            // Unknown fields are silently ignored (proto compatibility rule).
        }

        Ok(())
    }

    /// Convenience: decode bytes and validate against this descriptor.
    pub fn validate_bytes(&self, data: &[u8]) -> ProtoResult<Vec<ProtoField>> {
        let mut dec = ProtoDecoder::new(data);
        let fields = dec.decode_all()?;
        self.validate(&fields)?;
        Ok(fields)
    }

    /// Look up a field descriptor by name.
    pub fn field_by_name(&self, name: &str) -> Option<&ProtoFieldDescriptor> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Look up a field descriptor by number.
    pub fn field_by_number(&self, number: u32) -> Option<&ProtoFieldDescriptor> {
        self.fields.iter().find(|f| f.field_number == number)
    }
}

/// Map a [`ProtoType`] to its expected wire-type byte.
fn expected_wire_type(pt: &ProtoType) -> u8 {
    match pt {
        ProtoType::Uint64 | ProtoType::Sint64 | ProtoType::Bool => WIRE_VARINT,
        ProtoType::Double => WIRE_FIXED64,
        ProtoType::Float => WIRE_FIXED32,
        ProtoType::Bytes | ProtoType::String | ProtoType::Message => WIRE_LEN_DELIM,
    }
}

// ─────────────────────────────────────── Tests ───────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── VarintEncoder / VarintDecoder ────────────────────────────────────────

    #[test]
    fn test_varint_single_byte_values() {
        let mut enc = VarintEncoder::new();
        for v in 0u64..128 {
            enc.push_u64(v);
        }
        let bytes = enc.into_bytes();
        // Each value 0..127 encodes to exactly 1 byte
        assert_eq!(bytes.len(), 128);
        for (i, &b) in bytes.iter().enumerate() {
            assert_eq!(b, i as u8);
        }
    }

    #[test]
    fn test_varint_multi_byte() {
        let cases: &[(u64, &[u8])] = &[
            (128, &[0x80, 0x01]),
            (300, &[0xac, 0x02]),
            (16_383, &[0xff, 0x7f]),
            (16_384, &[0x80, 0x80, 0x01]),
            (
                u64::MAX,
                &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01],
            ),
        ];
        for &(v, expected) in cases {
            let mut enc = VarintEncoder::new();
            enc.push_u64(v);
            assert_eq!(enc.as_bytes(), expected, "mismatch for {v}");

            let mut dec = VarintDecoder::new(expected);
            let decoded = dec.read_u64().expect("decode");
            assert_eq!(decoded, v);
            assert!(dec.is_exhausted());
        }
    }

    #[test]
    fn test_zigzag_roundtrip() {
        let values: &[i64] = &[
            0,
            -1,
            1,
            -128,
            127,
            i32::MIN as i64,
            i32::MAX as i64,
            i64::MIN,
            i64::MAX,
        ];
        for &v in values {
            let encoded = zigzag_encode(v);
            let decoded = zigzag_decode(encoded);
            assert_eq!(decoded, v, "zigzag roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_signed_varint_encoder_decoder() {
        let values: &[i64] = &[-1000, -1, 0, 1, 1000, i64::MIN / 2, i64::MAX / 2];
        for &v in values {
            let mut enc = VarintEncoder::new();
            enc.push_i64(v);
            let mut dec = VarintDecoder::new(enc.as_bytes());
            let decoded = dec.read_i64().expect("decode");
            assert_eq!(decoded, v, "signed varint roundtrip for {v}");
        }
    }

    // ── ProtoValue ───────────────────────────────────────────────────────────

    #[test]
    fn test_proto_value_wire_types() {
        assert_eq!(ProtoValue::Varint(0).wire_type(), 0);
        assert_eq!(ProtoValue::SignedVarint(0).wire_type(), 0);
        assert_eq!(ProtoValue::Fixed64([0u8; 8]).wire_type(), 1);
        assert_eq!(ProtoValue::LengthDelimited(vec![]).wire_type(), 2);
        assert_eq!(ProtoValue::Fixed32([0u8; 4]).wire_type(), 5);
    }

    #[test]
    fn test_proto_value_as_str() {
        let v = ProtoValue::from_string("hello");
        assert_eq!(v.as_str(), Some("hello"));
        assert_eq!(ProtoValue::Varint(0).as_str(), None);
    }

    #[test]
    fn test_proto_value_f64_roundtrip() {
        let pi = std::f64::consts::PI;
        let v = ProtoValue::from_f64(pi);
        assert!((v.as_f64().unwrap() - pi).abs() < 1e-15);
    }

    #[test]
    fn test_proto_value_f32_roundtrip() {
        let e = std::f32::consts::E;
        let v = ProtoValue::from_f32(e);
        assert!((v.as_f32().unwrap() - e).abs() < 1e-6);
    }

    // ── ProtoMessageBuilder + ProtoDecoder ───────────────────────────────────

    #[test]
    fn test_builder_varint_roundtrip() {
        let bytes = ProtoMessageBuilder::new()
            .add_varint(1, 42)
            .add_varint(2, u64::MAX)
            .build();

        let mut dec = ProtoDecoder::new(&bytes);
        let fields = dec.decode_all().expect("decode");

        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].field_number, 1);
        assert_eq!(fields[0].value, ProtoValue::Varint(42));
        assert_eq!(fields[1].field_number, 2);
        assert_eq!(fields[1].value, ProtoValue::Varint(u64::MAX));
    }

    #[test]
    fn test_builder_string_roundtrip() {
        let msg = "hello, world!";
        let bytes = ProtoMessageBuilder::new().add_string(1, msg).build();

        let mut dec = ProtoDecoder::new(&bytes);
        let fields = dec.decode_all().expect("decode");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].field_number, 1);
        assert_eq!(fields[0].value.as_str(), Some(msg));
    }

    #[test]
    fn test_builder_f64_roundtrip() {
        let val = std::f64::consts::TAU;
        let bytes = ProtoMessageBuilder::new().add_f64(3, val).build();

        let mut dec = ProtoDecoder::new(&bytes);
        let fields = dec.decode_all().expect("decode");
        assert_eq!(fields.len(), 1);
        let decoded = fields[0].value.as_f64().expect("as_f64");
        assert!((decoded - val).abs() < 1e-15);
    }

    #[test]
    fn test_builder_f32_roundtrip() {
        let val = 1.23456_f32;
        let bytes = ProtoMessageBuilder::new().add_f32(4, val).build();

        let mut dec = ProtoDecoder::new(&bytes);
        let fields = dec.decode_all().expect("decode");
        assert_eq!(fields.len(), 1);
        let decoded = fields[0].value.as_f32().expect("as_f32");
        assert!((decoded - val).abs() < 1e-5);
    }

    #[test]
    fn test_builder_bool_roundtrip() {
        let bytes = ProtoMessageBuilder::new()
            .add_bool(1, true)
            .add_bool(2, false)
            .build();

        let mut dec = ProtoDecoder::new(&bytes);
        let fields = dec.decode_all().expect("decode");
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].value, ProtoValue::Varint(1));
        assert_eq!(fields[1].value, ProtoValue::Varint(0));
    }

    #[test]
    fn test_builder_sint64_roundtrip() {
        let values: &[i64] = &[-42, 0, 42, i64::MIN / 2, i64::MAX / 2];
        for &v in values {
            let bytes = ProtoMessageBuilder::new().add_sint64(1, v).build();
            let mut dec = ProtoDecoder::new(&bytes);
            let fields = dec.decode_all().expect("decode");
            assert_eq!(fields.len(), 1);
            // SignedVarint is stored as a ZigZag varint on wire
            if let ProtoValue::Varint(zz) = fields[0].value {
                let decoded = zigzag_decode(zz);
                assert_eq!(decoded, v, "sint64 roundtrip for {v}");
            } else {
                panic!(
                    "expected Varint for SignedVarint field, got {:?}",
                    fields[0].value
                );
            }
        }
    }

    #[test]
    fn test_builder_bytes_roundtrip() {
        let data = vec![0xde, 0xad, 0xbe, 0xef];
        let bytes = ProtoMessageBuilder::new()
            .add_bytes(5, data.clone())
            .build();

        let mut dec = ProtoDecoder::new(&bytes);
        let fields = dec.decode_all().expect("decode");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].value, ProtoValue::LengthDelimited(data));
    }

    #[test]
    fn test_embedded_message_roundtrip() {
        let inner = ProtoMessageBuilder::new()
            .add_varint(1, 100)
            .add_string(2, "inner value")
            .build();

        let outer = ProtoMessageBuilder::new()
            .add_varint(1, 999)
            .add_message(2, inner.clone())
            .build();

        let mut outer_dec = ProtoDecoder::new(&outer);
        let outer_fields = outer_dec.decode_all().expect("decode outer");
        assert_eq!(outer_fields.len(), 2);
        assert_eq!(outer_fields[0].value, ProtoValue::Varint(999));

        let inner_bytes = if let ProtoValue::LengthDelimited(ref b) = outer_fields[1].value {
            b.clone()
        } else {
            panic!("expected LengthDelimited for embedded message");
        };
        assert_eq!(inner_bytes, inner);

        let mut inner_dec = ProtoDecoder::new(&inner_bytes);
        let inner_fields = inner_dec.decode_all().expect("decode inner");
        assert_eq!(inner_fields.len(), 2);
        assert_eq!(inner_fields[0].value, ProtoValue::Varint(100));
        assert_eq!(inner_fields[1].value.as_str(), Some("inner value"));
    }

    #[test]
    fn test_packed_varints_roundtrip() {
        let values = vec![1u64, 2, 3, 300, u64::MAX / 2];
        let bytes = ProtoMessageBuilder::new()
            .add_packed_varints(1, &values)
            .build();

        let mut dec = ProtoDecoder::new(&bytes);
        let fields = dec.decode_all().expect("decode");
        assert_eq!(fields.len(), 1);

        let packed_bytes = if let ProtoValue::LengthDelimited(ref b) = fields[0].value {
            b.clone()
        } else {
            panic!("expected LengthDelimited");
        };

        // Unpack varints
        let mut vdec = VarintDecoder::new(&packed_bytes);
        let mut decoded = Vec::new();
        while !vdec.is_exhausted() {
            decoded.push(vdec.read_u64().expect("read packed varint"));
        }
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_multiple_fields_mixed() {
        let bytes = ProtoMessageBuilder::new()
            .add_varint(1, 42)
            .add_string(2, "test message")
            .add_f64(3, std::f64::consts::PI)
            .add_bytes(4, vec![0xca, 0xfe])
            .add_bool(5, true)
            .add_fixed32(6, 0xdeadbeefu32)
            .add_fixed64(7, 0xdeadbeefcafebabe_u64)
            .build();

        let mut dec = ProtoDecoder::new(&bytes);
        let fields = dec.decode_all().expect("decode");

        assert_eq!(fields.len(), 7);
        assert_eq!(fields[0].field_number, 1);
        assert_eq!(fields[1].field_number, 2);
        assert_eq!(fields[2].field_number, 3);
        assert_eq!(fields[3].field_number, 4);
        assert_eq!(fields[4].field_number, 5);
        assert_eq!(fields[5].field_number, 6);
        assert_eq!(fields[6].field_number, 7);

        assert_eq!(fields[0].value, ProtoValue::Varint(42));
        assert_eq!(fields[1].value.as_str(), Some("test message"));
        assert!((fields[2].value.as_f64().unwrap() - std::f64::consts::PI).abs() < 1e-15);
        assert_eq!(
            fields[3].value,
            ProtoValue::LengthDelimited(vec![0xca, 0xfe])
        );
        assert_eq!(fields[4].value, ProtoValue::Varint(1));
    }

    // ── decode_proto_fields (free function) ──────────────────────────────────

    #[test]
    fn test_decode_proto_fields_empty_input() {
        let fields = decode_proto_fields(&[]).expect("empty decode");
        assert!(fields.is_empty());
    }

    #[test]
    fn test_decode_unknown_wire_type_returns_error() {
        // Wire type 3 (start group, deprecated) — field 1 | wire 3 = 0x0B
        let bad = [0x0Bu8];
        assert!(decode_proto_fields(&bad).is_err());
    }

    // ── ProtoDescriptor ──────────────────────────────────────────────────────

    #[test]
    fn test_descriptor_validates_required_fields() {
        let desc = ProtoDescriptor::new("Test")
            .field("count", 1, ProtoType::Uint64, false)
            .field("label", 2, ProtoType::String, false);

        // Missing field 2 (label) → should fail
        let bytes = ProtoMessageBuilder::new().add_varint(1, 5).build();

        let result = desc.validate_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_descriptor_validates_ok() {
        let desc = ProtoDescriptor::new("Test")
            .field("count", 1, ProtoType::Uint64, false)
            .field("label", 2, ProtoType::String, false)
            .field("score", 3, ProtoType::Double, true);

        let bytes = ProtoMessageBuilder::new()
            .add_varint(1, 42)
            .add_string(2, "hello")
            .build();

        let fields = desc.validate_bytes(&bytes).expect("should validate");
        assert_eq!(fields.len(), 2);
    }

    #[test]
    fn test_descriptor_wire_type_mismatch() {
        let desc = ProtoDescriptor::new("Test").field("value", 1, ProtoType::Double, false); // expects fixed64

        // Encode as varint instead
        let bytes = ProtoMessageBuilder::new().add_varint(1, 42).build();

        let result = desc.validate_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_descriptor_lookup() {
        let desc = ProtoDescriptor::new("Msg")
            .field("id", 1, ProtoType::Uint64, false)
            .field("name", 2, ProtoType::String, true);

        assert_eq!(desc.field_by_name("id").map(|f| f.field_number), Some(1));
        assert_eq!(
            desc.field_by_number(2).map(|f| f.name.as_str()),
            Some("name")
        );
        assert!(desc.field_by_name("missing").is_none());
    }

    // ── ProtoMessage trait ───────────────────────────────────────────────────

    #[test]
    fn test_proto_message_trait_roundtrip() {
        #[derive(Debug, PartialEq)]
        struct Record {
            id: u64,
            name: String,
            score: f64,
        }

        impl ProtoMessage for Record {
            fn encode(&self) -> Vec<u8> {
                ProtoMessageBuilder::new()
                    .add_varint(1, self.id)
                    .add_string(2, &self.name)
                    .add_f64(3, self.score)
                    .build()
            }

            fn decode(fields: &[ProtoField]) -> ProtoResult<Self> {
                let mut id = None;
                let mut name = String::new();
                let mut score = None;
                for field in fields {
                    match field.field_number {
                        1 => {
                            if let ProtoValue::Varint(v) = field.value {
                                id = Some(v);
                            }
                        }
                        2 => {
                            name = field.value.as_str().unwrap_or("").to_string();
                        }
                        3 => {
                            score = field.value.as_f64();
                        }
                        _ => {}
                    }
                }
                Ok(Record {
                    id: id.ok_or_else(|| IoError::FormatError("missing id".into()))?,
                    name,
                    score: score.ok_or_else(|| IoError::FormatError("missing score".into()))?,
                })
            }
        }

        let r = Record {
            id: 99,
            name: "SciRS2".to_string(),
            score: std::f64::consts::PI,
        };

        let bytes = r.to_bytes();
        let r2 = Record::from_bytes(&bytes).expect("decode");
        assert_eq!(r.id, r2.id);
        assert_eq!(r.name, r2.name);
        assert!((r.score - r2.score).abs() < 1e-12);
    }

    // ── Official protobuf spec examples ─────────────────────────────────────

    #[test]
    fn test_official_spec_example_field1_varint150() {
        // From the official protobuf encoding spec:
        // field 1, varint 150 → 0x08 0x96 0x01
        let bytes = ProtoMessageBuilder::new().add_varint(1, 150).build();
        assert_eq!(bytes, vec![0x08, 0x96, 0x01]);
    }

    #[test]
    fn test_official_spec_example_string_testing() {
        // field 2, string "testing" → 0x12 0x07 b't' b'e' b's' b't' b'i' b'n' b'g'
        let bytes = ProtoMessageBuilder::new().add_string(2, "testing").build();
        assert_eq!(bytes[0], 0x12); // tag: field 2, wire type 2
        assert_eq!(bytes[1], 0x07); // length 7
        assert_eq!(&bytes[2..], b"testing");
    }
}
