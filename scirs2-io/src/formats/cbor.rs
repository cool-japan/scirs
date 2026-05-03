//! CBOR (Concise Binary Object Representation) – RFC 7049 compliant.
//!
//! Pure-Rust implementation with no external crate dependencies.
//!
//! ## Features
//! - Full major-type support (0–7)
//! - Additional-info lengths: direct (≤23), 1/2/4/8-byte (24–27), indefinite (31)
//! - [`CborEncoder`] for incremental writing
//! - [`CborDecoder`] for incremental reading
//! - Convenience [`encode_cbor`] / [`decode_cbor`] functions

#![allow(dead_code)]

use crate::error::IoError;

// ─────────────────────────────── Value type ──────────────────────────────────

/// A CBOR data-model value.
#[derive(Debug, Clone, PartialEq)]
pub enum CborValue {
    /// Major type 0 – unsigned integer.
    Unsigned(u64),
    /// Major type 1 – negative integer (stored as the actual negative i64).
    Negative(i64),
    /// Major type 2 – byte string.
    Bytes(Vec<u8>),
    /// Major type 3 – text string (UTF-8).
    Text(String),
    /// Major type 4 – array.
    Array(Vec<CborValue>),
    /// Major type 5 – map (pairs).
    Map(Vec<(CborValue, CborValue)>),
    /// Major type 7 – IEEE 754 double.
    Float(f64),
    /// Major type 7 – boolean simple value (20/21).
    Bool(bool),
    /// Major type 7 – null (simple value 22).
    Null,
    /// Major type 7 – undefined (simple value 23).
    Undefined,
}

// ─────────────────────────────── Encoder ─────────────────────────────────────

/// Incremental CBOR encoder that accumulates bytes into an internal `Vec<u8>`.
pub struct CborEncoder {
    buf: Vec<u8>,
}

impl CborEncoder {
    /// Create a new empty encoder.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Return the encoded bytes, consuming the encoder.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    /// Borrow the accumulated bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    // ── low-level header helpers ──────────────────────────────────────────────

    fn write_type_arg(&mut self, major: u8, arg: u64) {
        let mt = major << 5;
        if arg <= 23 {
            self.buf.push(mt | arg as u8);
        } else if arg <= u8::MAX as u64 {
            self.buf.push(mt | 24);
            self.buf.push(arg as u8);
        } else if arg <= u16::MAX as u64 {
            self.buf.push(mt | 25);
            self.buf.extend_from_slice(&(arg as u16).to_be_bytes());
        } else if arg <= u32::MAX as u64 {
            self.buf.push(mt | 26);
            self.buf.extend_from_slice(&(arg as u32).to_be_bytes());
        } else {
            self.buf.push(mt | 27);
            self.buf.extend_from_slice(&arg.to_be_bytes());
        }
    }

    // ── public write helpers ──────────────────────────────────────────────────

    /// Encode an unsigned integer (major type 0).
    pub fn write_uint(&mut self, v: u64) {
        self.write_type_arg(0, v);
    }

    /// Encode a negative integer (major type 1).
    /// `v` must be negative; passing a non-negative value produces an invalid item.
    pub fn write_negative(&mut self, v: i64) {
        // RFC 7049: value = -1 - n  →  n = (-1 - v) as u64
        let n = ((-1_i128) - v as i128) as u64;
        self.write_type_arg(1, n);
    }

    /// Encode a byte string (major type 2).
    pub fn write_bytes(&mut self, data: &[u8]) {
        self.write_type_arg(2, data.len() as u64);
        self.buf.extend_from_slice(data);
    }

    /// Encode a text string (major type 3).
    pub fn write_text(&mut self, s: &str) {
        self.write_type_arg(3, s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }

    /// Encode a definite-length array header (major type 4).
    pub fn write_array_header(&mut self, len: usize) {
        self.write_type_arg(4, len as u64);
    }

    /// Encode a definite-length map header (major type 5).
    pub fn write_map_header(&mut self, pairs: usize) {
        self.write_type_arg(5, pairs as u64);
    }

    /// Encode an IEEE 754 double (major type 7, additional info 27).
    pub fn write_float64(&mut self, v: f64) {
        self.buf.push(0xfb); // mt=7, ai=27
        self.buf.extend_from_slice(&v.to_bits().to_be_bytes());
    }

    /// Encode a boolean (major type 7, simple 20/21).
    pub fn write_bool(&mut self, v: bool) {
        self.buf.push(if v { 0xf5 } else { 0xf4 });
    }

    /// Encode null (major type 7, simple 22).
    pub fn write_null(&mut self) {
        self.buf.push(0xf6);
    }

    /// Encode undefined (major type 7, simple 23).
    pub fn write_undefined(&mut self) {
        self.buf.push(0xf7);
    }

    /// Encode a complete [`CborValue`] tree.
    pub fn write_value(&mut self, value: &CborValue) {
        match value {
            CborValue::Unsigned(n) => self.write_uint(*n),
            CborValue::Negative(n) => self.write_negative(*n),
            CborValue::Bytes(b) => self.write_bytes(b),
            CborValue::Text(s) => self.write_text(s),
            CborValue::Array(items) => {
                self.write_array_header(items.len());
                for item in items {
                    self.write_value(item);
                }
            }
            CborValue::Map(pairs) => {
                self.write_map_header(pairs.len());
                for (k, v) in pairs {
                    self.write_value(k);
                    self.write_value(v);
                }
            }
            CborValue::Float(f) => self.write_float64(*f),
            CborValue::Bool(b) => self.write_bool(*b),
            CborValue::Null => self.write_null(),
            CborValue::Undefined => self.write_undefined(),
        }
    }
}

// ─────────────────────────────── Decoder ─────────────────────────────────────

/// Incremental CBOR decoder operating over a borrowed byte slice.
pub struct CborDecoder<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> CborDecoder<'a> {
    /// Create a decoder from a byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current byte position.
    pub fn position(&self) -> usize {
        self.pos
    }

    fn peek(&self) -> Result<u8, IoError> {
        self.data
            .get(self.pos)
            .copied()
            .ok_or_else(|| IoError::DeserializationError("CBOR: unexpected end of input".into()))
    }

    fn take_byte(&mut self) -> Result<u8, IoError> {
        let b = self.peek()?;
        self.pos += 1;
        Ok(b)
    }

    fn take_bytes(&mut self, n: usize) -> Result<&'a [u8], IoError> {
        let end = self.pos + n;
        if end > self.data.len() {
            return Err(IoError::DeserializationError(format!(
                "CBOR: need {n} bytes, only {} remaining",
                self.data.len() - self.pos
            )));
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    // ── argument / additional-info decoder ───────────────────────────────────

    /// Decode the argument value from the additional-info nibble.
    /// Returns `(argument, is_indefinite)`.
    fn decode_argument(&mut self, additional: u8) -> Result<(u64, bool), IoError> {
        match additional {
            0..=23 => Ok((additional as u64, false)),
            24 => {
                let b = self.take_byte()?;
                Ok((b as u64, false))
            }
            25 => {
                let bytes = self.take_bytes(2)?;
                Ok((u16::from_be_bytes([bytes[0], bytes[1]]) as u64, false))
            }
            26 => {
                let bytes = self.take_bytes(4)?;
                Ok((
                    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64,
                    false,
                ))
            }
            27 => {
                let bytes = self.take_bytes(8)?;
                Ok((
                    u64::from_be_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ]),
                    false,
                ))
            }
            31 => Ok((0, true)), // indefinite
            other => Err(IoError::DeserializationError(format!(
                "CBOR: reserved additional info {other}"
            ))),
        }
    }

    // ── break-code detector ───────────────────────────────────────────────────

    fn is_break(&self) -> bool {
        self.data.get(self.pos) == Some(&0xff)
    }

    // ── public read entry point ───────────────────────────────────────────────

    /// Decode the next [`CborValue`] from the stream.
    pub fn read_value(&mut self) -> Result<CborValue, IoError> {
        let initial = self.take_byte()?;
        let major = initial >> 5;
        let additional = initial & 0x1f;

        match major {
            // ── major 0: unsigned integer ─────────────────────────────────────
            0 => {
                let (n, _) = self.decode_argument(additional)?;
                Ok(CborValue::Unsigned(n))
            }

            // ── major 1: negative integer ─────────────────────────────────────
            1 => {
                let (n, _) = self.decode_argument(additional)?;
                // value = -1 - n
                let v = (-1_i128) - n as i128;
                if v < i64::MIN as i128 {
                    return Err(IoError::DeserializationError(
                        "CBOR: negative integer out of i64 range".into(),
                    ));
                }
                Ok(CborValue::Negative(v as i64))
            }

            // ── major 2: byte string ──────────────────────────────────────────
            2 => {
                let (len, indefinite) = self.decode_argument(additional)?;
                if indefinite {
                    let mut out = Vec::new();
                    loop {
                        if self.is_break() {
                            self.pos += 1;
                            break;
                        }
                        let chunk = self.read_value()?;
                        if let CborValue::Bytes(b) = chunk {
                            out.extend_from_slice(&b);
                        } else {
                            return Err(IoError::DeserializationError(
                                "CBOR: non-bytes chunk in indefinite byte string".into(),
                            ));
                        }
                    }
                    Ok(CborValue::Bytes(out))
                } else {
                    let bytes = self.take_bytes(len as usize)?;
                    Ok(CborValue::Bytes(bytes.to_vec()))
                }
            }

            // ── major 3: text string ──────────────────────────────────────────
            3 => {
                let (len, indefinite) = self.decode_argument(additional)?;
                if indefinite {
                    let mut out = String::new();
                    loop {
                        if self.is_break() {
                            self.pos += 1;
                            break;
                        }
                        let chunk = self.read_value()?;
                        if let CborValue::Text(s) = chunk {
                            out.push_str(&s);
                        } else {
                            return Err(IoError::DeserializationError(
                                "CBOR: non-text chunk in indefinite text string".into(),
                            ));
                        }
                    }
                    Ok(CborValue::Text(out))
                } else {
                    let bytes = self.take_bytes(len as usize)?;
                    let s = std::str::from_utf8(bytes).map_err(|e| {
                        IoError::DeserializationError(format!("CBOR: invalid UTF-8: {e}"))
                    })?;
                    Ok(CborValue::Text(s.to_string()))
                }
            }

            // ── major 4: array ────────────────────────────────────────────────
            4 => {
                let (len, indefinite) = self.decode_argument(additional)?;
                if indefinite {
                    let mut items = Vec::new();
                    loop {
                        if self.is_break() {
                            self.pos += 1;
                            break;
                        }
                        items.push(self.read_value()?);
                    }
                    Ok(CborValue::Array(items))
                } else {
                    let mut items = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        items.push(self.read_value()?);
                    }
                    Ok(CborValue::Array(items))
                }
            }

            // ── major 5: map ──────────────────────────────────────────────────
            5 => {
                let (len, indefinite) = self.decode_argument(additional)?;
                if indefinite {
                    let mut pairs = Vec::new();
                    loop {
                        if self.is_break() {
                            self.pos += 1;
                            break;
                        }
                        let k = self.read_value()?;
                        let v = self.read_value()?;
                        pairs.push((k, v));
                    }
                    Ok(CborValue::Map(pairs))
                } else {
                    let mut pairs = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        let k = self.read_value()?;
                        let v = self.read_value()?;
                        pairs.push((k, v));
                    }
                    Ok(CborValue::Map(pairs))
                }
            }

            // ── major 6: tagged item ──────────────────────────────────────────
            6 => {
                // Consume and discard the tag number, return the wrapped value.
                let _ = self.decode_argument(additional)?;
                self.read_value()
            }

            // ── major 7: float / simple ───────────────────────────────────────
            7 => match additional {
                20 => Ok(CborValue::Bool(false)),
                21 => Ok(CborValue::Bool(true)),
                22 => Ok(CborValue::Null),
                23 => Ok(CborValue::Undefined),
                25 => {
                    // IEEE 754 half-precision – decode to f64
                    let bytes = self.take_bytes(2)?;
                    let half = u16::from_be_bytes([bytes[0], bytes[1]]);
                    Ok(CborValue::Float(half_to_f64(half)))
                }
                26 => {
                    // IEEE 754 single
                    let bytes = self.take_bytes(4)?;
                    let bits = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    Ok(CborValue::Float(f32::from_bits(bits) as f64))
                }
                27 => {
                    // IEEE 754 double
                    let bytes = self.take_bytes(8)?;
                    let bits = u64::from_be_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ]);
                    Ok(CborValue::Float(f64::from_bits(bits)))
                }
                _ => Err(IoError::DeserializationError(format!(
                    "CBOR: unsupported simple/float additional info {additional}"
                ))),
            },

            _ => Err(IoError::DeserializationError(format!(
                "CBOR: unknown major type {major}"
            ))),
        }
    }
}

// ─────────────────────────────── Half-float helper ───────────────────────────

fn half_to_f64(half: u16) -> f64 {
    let exp = ((half >> 10) & 0x1f) as i32;
    let mant = (half & 0x3ff) as u32;
    let sign: f64 = if half & 0x8000 != 0 { -1.0 } else { 1.0 };
    if exp == 0 {
        sign * 2.0_f64.powi(-14) * (mant as f64 / 1024.0)
    } else if exp == 31 {
        if mant == 0 {
            sign * f64::INFINITY
        } else {
            f64::NAN
        }
    } else {
        sign * 2.0_f64.powi(exp - 15) * (1.0 + mant as f64 / 1024.0)
    }
}

// ─────────────────────────────── Convenience API ─────────────────────────────

/// Encode a [`CborValue`] into a freshly-allocated `Vec<u8>`.
pub fn encode_cbor(value: &CborValue) -> Vec<u8> {
    let mut enc = CborEncoder::new();
    enc.write_value(value);
    enc.finish()
}

/// Decode the first CBOR item from `data`.
///
/// Returns `(value, bytes_consumed)` on success.
pub fn decode_cbor(data: &[u8]) -> Result<(CborValue, usize), IoError> {
    let mut dec = CborDecoder::new(data);
    let value = dec.read_value()?;
    Ok((value, dec.position()))
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(v: CborValue) -> CborValue {
        let bytes = encode_cbor(&v);
        let (decoded, consumed) = decode_cbor(&bytes).expect("decode failed");
        assert_eq!(consumed, bytes.len());
        decoded
    }

    #[test]
    fn test_unsigned_small() {
        assert_eq!(round_trip(CborValue::Unsigned(0)), CborValue::Unsigned(0));
        assert_eq!(round_trip(CborValue::Unsigned(23)), CborValue::Unsigned(23));
    }

    #[test]
    fn test_unsigned_multibyte() {
        assert_eq!(
            round_trip(CborValue::Unsigned(255)),
            CborValue::Unsigned(255)
        );
        assert_eq!(
            round_trip(CborValue::Unsigned(65536)),
            CborValue::Unsigned(65536)
        );
        assert_eq!(
            round_trip(CborValue::Unsigned(u64::MAX)),
            CborValue::Unsigned(u64::MAX)
        );
    }

    #[test]
    fn test_negative() {
        assert_eq!(round_trip(CborValue::Negative(-1)), CborValue::Negative(-1));
        assert_eq!(
            round_trip(CborValue::Negative(-100)),
            CborValue::Negative(-100)
        );
    }

    #[test]
    fn test_bytes() {
        let v = CborValue::Bytes(vec![0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(round_trip(v.clone()), v);
    }

    #[test]
    fn test_text() {
        let v = CborValue::Text("hello CBOR".into());
        assert_eq!(round_trip(v.clone()), v);
    }

    #[test]
    fn test_array() {
        let v = CborValue::Array(vec![
            CborValue::Unsigned(1),
            CborValue::Text("two".into()),
            CborValue::Bool(true),
        ]);
        assert_eq!(round_trip(v.clone()), v);
    }

    #[test]
    fn test_map() {
        let v = CborValue::Map(vec![
            (CborValue::Text("key".into()), CborValue::Unsigned(42)),
            (CborValue::Text("flag".into()), CborValue::Bool(false)),
        ]);
        assert_eq!(round_trip(v.clone()), v);
    }

    #[test]
    fn test_float() {
        let v = CborValue::Float(std::f64::consts::PI);
        if let CborValue::Float(f) = round_trip(v) {
            assert!((f - std::f64::consts::PI).abs() < 1e-15);
        } else {
            panic!("expected Float");
        }
    }

    #[test]
    fn test_null_undefined_bool() {
        assert_eq!(round_trip(CborValue::Null), CborValue::Null);
        assert_eq!(round_trip(CborValue::Undefined), CborValue::Undefined);
        assert_eq!(round_trip(CborValue::Bool(true)), CborValue::Bool(true));
    }

    #[test]
    fn test_nested() {
        let inner = CborValue::Map(vec![(CborValue::Text("x".into()), CborValue::Unsigned(99))]);
        let outer = CborValue::Array(vec![inner, CborValue::Null]);
        assert_eq!(round_trip(outer.clone()), outer);
    }
}

// ─────────────────────────────── File I/O helpers ────────────────────────────

use std::fs;
use std::path::Path;

/// Read a CBOR file from `path` and decode the first item.
pub fn read_cbor_file(path: impl AsRef<Path>) -> Result<CborValue, IoError> {
    let bytes = fs::read(path.as_ref())
        .map_err(|e| IoError::SerializationError(format!("CBOR: cannot read file: {e}")))?;
    let (value, _) = decode_cbor(&bytes)?;
    Ok(value)
}

/// Encode `value` and write the bytes to `path`.
pub fn write_cbor_file(path: impl AsRef<Path>, value: &CborValue) -> Result<(), IoError> {
    let bytes = encode_cbor(value);
    fs::write(path.as_ref(), &bytes)
        .map_err(|e| IoError::SerializationError(format!("CBOR: cannot write file: {e}")))
}

#[cfg(test)]
mod file_tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_roundtrip_file() {
        let value = CborValue::Map(vec![
            (
                CborValue::Text("project".into()),
                CborValue::Text("scirs2".into()),
            ),
            (CborValue::Text("version".into()), CborValue::Unsigned(3)),
        ]);
        let path = temp_dir().join("cbor_roundtrip_test.cbor");
        write_cbor_file(&path, &value).expect("write failed");
        let decoded = read_cbor_file(&path).expect("read failed");
        assert_eq!(decoded, value);
        let _ = std::fs::remove_file(&path);
    }
}
