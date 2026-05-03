//! MessagePack serialization inside the `formats` module.
//!
//! A thin, self-contained implementation of the MessagePack spec in pure Rust.
//! Unlike the top-level `msgpack` module (which bridges to serde_json), this
//! module exposes a low-level `MsgPackValue` + `MsgPackWriter` / `MsgPackReader`
//! API that is useful when you need fine-grained control over the wire encoding.
//!
//! ## Supported formats
//! - Positive fixint (0..127), negative fixint (-32..-1)
//! - nil, bool
//! - uint8/16/32/64, int8/16/32/64
//! - float32 / float64
//! - fixstr, str8/16/32
//! - bin8/16/32
//! - fixarray, array16/32
//! - fixmap, map16/32
//! - ext types (fixext 1/2/4/8/16, ext8/16/32)

use std::io::{Cursor, Read};

use crate::error::{IoError, Result};

// ─────────────────────────────── byte constants ──────────────────────────────

const FMT_NIL: u8 = 0xc0;
const FMT_FALSE: u8 = 0xc2;
const FMT_TRUE: u8 = 0xc3;
const FMT_BIN8: u8 = 0xc4;
const FMT_BIN16: u8 = 0xc5;
const FMT_BIN32: u8 = 0xc6;
const FMT_EXT8: u8 = 0xc7;
const FMT_EXT16: u8 = 0xc8;
const FMT_EXT32: u8 = 0xc9;
const FMT_FLOAT32: u8 = 0xca;
const FMT_FLOAT64: u8 = 0xcb;
const FMT_UINT8: u8 = 0xcc;
const FMT_UINT16: u8 = 0xcd;
const FMT_UINT32: u8 = 0xce;
const FMT_UINT64: u8 = 0xcf;
const FMT_INT8: u8 = 0xd0;
const FMT_INT16: u8 = 0xd1;
const FMT_INT32: u8 = 0xd2;
const FMT_INT64: u8 = 0xd3;
const FMT_FIXEXT1: u8 = 0xd4;
const FMT_FIXEXT2: u8 = 0xd5;
const FMT_FIXEXT4: u8 = 0xd6;
const FMT_FIXEXT8: u8 = 0xd7;
const FMT_FIXEXT16: u8 = 0xd8;
const FMT_STR8: u8 = 0xd9;
const FMT_STR16: u8 = 0xda;
const FMT_STR32: u8 = 0xdb;
const FMT_ARRAY16: u8 = 0xdc;
const FMT_ARRAY32: u8 = 0xdd;
const FMT_MAP16: u8 = 0xde;
const FMT_MAP32: u8 = 0xdf;

// ─────────────────────────────── Value type ──────────────────────────────────

/// A MessagePack value.
#[derive(Debug, Clone, PartialEq)]
pub enum MsgPackValue {
    /// nil / null
    Nil,
    /// Boolean
    Bool(bool),
    /// Signed integer (covers all int types)
    Int(i64),
    /// Unsigned integer (used when value > i64::MAX)
    UInt(u64),
    /// 32-bit float
    Float32(f32),
    /// 64-bit float
    Float64(f64),
    /// UTF-8 string
    Str(String),
    /// Raw binary data
    Bin(Vec<u8>),
    /// Array of values
    Array(Vec<MsgPackValue>),
    /// Map of key-value pairs
    Map(Vec<(MsgPackValue, MsgPackValue)>),
    /// Extension type: (type_id, bytes)
    Ext(i8, Vec<u8>),
}

// ─────────────────────────────── Encoder ─────────────────────────────────────

/// Low-level MessagePack writer.
///
/// Push individual values into an internal buffer, then call [`MsgPackWriter::into_bytes`]
/// to get the encoded byte slice.
///
/// # Example
/// ```
/// use scirs2_io::formats::msgpack::{MsgPackWriter, MsgPackReader};
///
/// let mut w = MsgPackWriter::new();
/// w.write_map_header(1);
/// w.write_str("answer");
/// w.write_int(42);
/// let bytes = w.into_bytes();
///
/// let mut r = MsgPackReader::new(&bytes);
/// let value = r.read_value().expect("decode failed");
/// ```
#[derive(Debug, Default)]
pub struct MsgPackWriter {
    buf: Vec<u8>,
}

impl MsgPackWriter {
    /// Create a new, empty writer.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Consume the writer and return the encoded bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Borrow the current internal buffer.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Write a nil value.
    pub fn write_nil(&mut self) {
        self.buf.push(FMT_NIL);
    }

    /// Write a boolean value.
    pub fn write_bool(&mut self, v: bool) {
        self.buf.push(if v { FMT_TRUE } else { FMT_FALSE });
    }

    /// Write a signed integer using the smallest possible representation.
    pub fn write_int(&mut self, v: i64) {
        if (0..=127).contains(&v) {
            self.buf.push(v as u8);
        } else if (-32..=-1).contains(&v) {
            self.buf.push((v as i8) as u8);
        } else if v >= 0 {
            let u = v as u64;
            if u <= u8::MAX as u64 {
                self.buf.extend_from_slice(&[FMT_UINT8, u as u8]);
            } else if u <= u16::MAX as u64 {
                self.buf.push(FMT_UINT16);
                self.buf.extend_from_slice(&(u as u16).to_be_bytes());
            } else if u <= u32::MAX as u64 {
                self.buf.push(FMT_UINT32);
                self.buf.extend_from_slice(&(u as u32).to_be_bytes());
            } else {
                self.buf.push(FMT_UINT64);
                self.buf.extend_from_slice(&u.to_be_bytes());
            }
        } else if v >= i8::MIN as i64 {
            self.buf.extend_from_slice(&[FMT_INT8, v as u8]);
        } else if v >= i16::MIN as i64 {
            self.buf.push(FMT_INT16);
            self.buf.extend_from_slice(&(v as i16).to_be_bytes());
        } else if v >= i32::MIN as i64 {
            self.buf.push(FMT_INT32);
            self.buf.extend_from_slice(&(v as i32).to_be_bytes());
        } else {
            self.buf.push(FMT_INT64);
            self.buf.extend_from_slice(&v.to_be_bytes());
        }
    }

    /// Write an unsigned integer using the smallest possible representation.
    pub fn write_uint(&mut self, v: u64) {
        if v <= 127 {
            self.buf.push(v as u8);
        } else if v <= u8::MAX as u64 {
            self.buf.extend_from_slice(&[FMT_UINT8, v as u8]);
        } else if v <= u16::MAX as u64 {
            self.buf.push(FMT_UINT16);
            self.buf.extend_from_slice(&(v as u16).to_be_bytes());
        } else if v <= u32::MAX as u64 {
            self.buf.push(FMT_UINT32);
            self.buf.extend_from_slice(&(v as u32).to_be_bytes());
        } else {
            self.buf.push(FMT_UINT64);
            self.buf.extend_from_slice(&v.to_be_bytes());
        }
    }

    /// Write a 64-bit float (f64).
    pub fn write_float(&mut self, v: f64) {
        self.buf.push(FMT_FLOAT64);
        self.buf.extend_from_slice(&v.to_bits().to_be_bytes());
    }

    /// Write a 32-bit float (f32).
    pub fn write_float32(&mut self, v: f32) {
        self.buf.push(FMT_FLOAT32);
        self.buf.extend_from_slice(&v.to_bits().to_be_bytes());
    }

    /// Write a UTF-8 string.
    pub fn write_str(&mut self, s: &str) {
        let bytes = s.as_bytes();
        let n = bytes.len();
        if n <= 31 {
            self.buf.push(0xa0 | (n as u8));
        } else if n <= 255 {
            self.buf.extend_from_slice(&[FMT_STR8, n as u8]);
        } else if n <= 65535 {
            self.buf.push(FMT_STR16);
            self.buf.extend_from_slice(&(n as u16).to_be_bytes());
        } else {
            self.buf.push(FMT_STR32);
            self.buf.extend_from_slice(&(n as u32).to_be_bytes());
        }
        self.buf.extend_from_slice(bytes);
    }

    /// Write raw binary data.
    pub fn write_bin(&mut self, b: &[u8]) {
        let n = b.len();
        if n <= 255 {
            self.buf.extend_from_slice(&[FMT_BIN8, n as u8]);
        } else if n <= 65535 {
            self.buf.push(FMT_BIN16);
            self.buf.extend_from_slice(&(n as u16).to_be_bytes());
        } else {
            self.buf.push(FMT_BIN32);
            self.buf.extend_from_slice(&(n as u32).to_be_bytes());
        }
        self.buf.extend_from_slice(b);
    }

    /// Write an array header.  You must write exactly `n` elements after this call.
    pub fn write_array_header(&mut self, n: usize) {
        if n <= 15 {
            self.buf.push(0x90 | (n as u8));
        } else if n <= 65535 {
            self.buf.push(FMT_ARRAY16);
            self.buf.extend_from_slice(&(n as u16).to_be_bytes());
        } else {
            self.buf.push(FMT_ARRAY32);
            self.buf.extend_from_slice(&(n as u32).to_be_bytes());
        }
    }

    /// Write a map header.  You must write exactly `n` key-value pairs after this call.
    pub fn write_map_header(&mut self, n: usize) {
        if n <= 15 {
            self.buf.push(0x80 | (n as u8));
        } else if n <= 65535 {
            self.buf.push(FMT_MAP16);
            self.buf.extend_from_slice(&(n as u16).to_be_bytes());
        } else {
            self.buf.push(FMT_MAP32);
            self.buf.extend_from_slice(&(n as u32).to_be_bytes());
        }
    }

    /// Write an extension type.
    pub fn write_ext(&mut self, type_id: i8, data: &[u8]) {
        let n = data.len();
        let tid = type_id as u8;
        match n {
            1 => self.buf.extend_from_slice(&[FMT_FIXEXT1, tid]),
            2 => self.buf.extend_from_slice(&[FMT_FIXEXT2, tid]),
            4 => self.buf.extend_from_slice(&[FMT_FIXEXT4, tid]),
            8 => self.buf.extend_from_slice(&[FMT_FIXEXT8, tid]),
            16 => self.buf.extend_from_slice(&[FMT_FIXEXT16, tid]),
            _ => {
                if n <= 255 {
                    self.buf.extend_from_slice(&[FMT_EXT8, n as u8, tid]);
                } else if n <= 65535 {
                    self.buf.push(FMT_EXT16);
                    self.buf.extend_from_slice(&(n as u16).to_be_bytes());
                    self.buf.push(tid);
                } else {
                    self.buf.push(FMT_EXT32);
                    self.buf.extend_from_slice(&(n as u32).to_be_bytes());
                    self.buf.push(tid);
                }
            }
        }
        self.buf.extend_from_slice(data);
    }

    /// Recursively write a [`MsgPackValue`].
    pub fn write_value(&mut self, value: &MsgPackValue) {
        match value {
            MsgPackValue::Nil => self.write_nil(),
            MsgPackValue::Bool(b) => self.write_bool(*b),
            MsgPackValue::Int(i) => self.write_int(*i),
            MsgPackValue::UInt(u) => self.write_uint(*u),
            MsgPackValue::Float32(f) => self.write_float32(*f),
            MsgPackValue::Float64(f) => self.write_float(*f),
            MsgPackValue::Str(s) => self.write_str(s),
            MsgPackValue::Bin(b) => self.write_bin(b),
            MsgPackValue::Array(arr) => {
                self.write_array_header(arr.len());
                for elem in arr {
                    self.write_value(elem);
                }
            }
            MsgPackValue::Map(map) => {
                self.write_map_header(map.len());
                for (k, v) in map {
                    self.write_value(k);
                    self.write_value(v);
                }
            }
            MsgPackValue::Ext(tid, data) => self.write_ext(*tid, data),
        }
    }
}

// ─────────────────────────────── Decoder ─────────────────────────────────────

/// Low-level MessagePack reader.
///
/// Wraps a byte slice and advances a position cursor as values are read.
pub struct MsgPackReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> MsgPackReader<'a> {
    /// Create a new reader over the given byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current read position.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Return true if all bytes have been consumed.
    pub fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(IoError::FormatError(format!(
                "MsgPack: unexpected end of data (need {n} bytes at offset {})",
                self.pos
            )));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    fn read_u16_be(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn read_u32_be(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64_be(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_be_bytes(b.try_into().map_err(|_| {
            IoError::FormatError("MsgPack: bad u64 bytes".into())
        })?))
    }

    fn read_str_bytes(&mut self, n: usize) -> Result<MsgPackValue> {
        let raw = self.read_bytes(n)?;
        let s = std::str::from_utf8(raw)
            .map_err(|e| IoError::FormatError(format!("MsgPack: invalid UTF-8: {e}")))?
            .to_owned();
        Ok(MsgPackValue::Str(s))
    }

    fn read_array_items(&mut self, n: usize) -> Result<MsgPackValue> {
        let mut arr = Vec::with_capacity(n);
        for _ in 0..n {
            arr.push(self.read_value()?);
        }
        Ok(MsgPackValue::Array(arr))
    }

    fn read_map_items(&mut self, n: usize) -> Result<MsgPackValue> {
        let mut map = Vec::with_capacity(n);
        for _ in 0..n {
            let k = self.read_value()?;
            let v = self.read_value()?;
            map.push((k, v));
        }
        Ok(MsgPackValue::Map(map))
    }

    fn read_ext_data(&mut self, n: usize) -> Result<MsgPackValue> {
        let type_byte = self.read_u8()?;
        let type_id = type_byte as i8;
        let data = self.read_bytes(n)?.to_vec();
        Ok(MsgPackValue::Ext(type_id, data))
    }

    /// Read one MessagePack value from the current position.
    pub fn read_value(&mut self) -> Result<MsgPackValue> {
        let first = self.read_u8()?;

        match first {
            // positive fixint 0xxxxxxx
            b if b & 0x80 == 0 => Ok(MsgPackValue::Int(b as i64)),
            // fixmap 1000xxxx
            b if b & 0xf0 == 0x80 => self.read_map_items((b & 0x0f) as usize),
            // fixarray 1001xxxx
            b if b & 0xf0 == 0x90 => self.read_array_items((b & 0x0f) as usize),
            // fixstr 101xxxxx
            b if b & 0xe0 == 0xa0 => self.read_str_bytes((b & 0x1f) as usize),
            // negative fixint 111xxxxx
            b if b & 0xe0 == 0xe0 => Ok(MsgPackValue::Int((b as i8) as i64)),
            // named formats
            FMT_NIL => Ok(MsgPackValue::Nil),
            FMT_FALSE => Ok(MsgPackValue::Bool(false)),
            FMT_TRUE => Ok(MsgPackValue::Bool(true)),
            FMT_BIN8 => {
                let n = self.read_u8()? as usize;
                Ok(MsgPackValue::Bin(self.read_bytes(n)?.to_vec()))
            }
            FMT_BIN16 => {
                let n = self.read_u16_be()? as usize;
                Ok(MsgPackValue::Bin(self.read_bytes(n)?.to_vec()))
            }
            FMT_BIN32 => {
                let n = self.read_u32_be()? as usize;
                Ok(MsgPackValue::Bin(self.read_bytes(n)?.to_vec()))
            }
            FMT_EXT8 => {
                let n = self.read_u8()? as usize;
                self.read_ext_data(n)
            }
            FMT_EXT16 => {
                let n = self.read_u16_be()? as usize;
                self.read_ext_data(n)
            }
            FMT_EXT32 => {
                let n = self.read_u32_be()? as usize;
                self.read_ext_data(n)
            }
            FMT_FLOAT32 => {
                let b = self.read_bytes(4)?;
                let bits = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
                Ok(MsgPackValue::Float32(f32::from_bits(bits)))
            }
            FMT_FLOAT64 => {
                let bits = self.read_u64_be()?;
                Ok(MsgPackValue::Float64(f64::from_bits(bits)))
            }
            FMT_UINT8 => Ok(MsgPackValue::UInt(self.read_u8()? as u64)),
            FMT_UINT16 => Ok(MsgPackValue::UInt(self.read_u16_be()? as u64)),
            FMT_UINT32 => Ok(MsgPackValue::UInt(self.read_u32_be()? as u64)),
            FMT_UINT64 => Ok(MsgPackValue::UInt(self.read_u64_be()?)),
            FMT_INT8 => Ok(MsgPackValue::Int(self.read_u8()? as i8 as i64)),
            FMT_INT16 => {
                let b = self.read_bytes(2)?;
                Ok(MsgPackValue::Int(i16::from_be_bytes([b[0], b[1]]) as i64))
            }
            FMT_INT32 => {
                let b = self.read_bytes(4)?;
                Ok(MsgPackValue::Int(
                    i32::from_be_bytes([b[0], b[1], b[2], b[3]]) as i64,
                ))
            }
            FMT_INT64 => {
                let b = self.read_bytes(8)?;
                Ok(MsgPackValue::Int(i64::from_be_bytes(
                    b.try_into()
                        .map_err(|_| IoError::FormatError("MsgPack: bad i64 bytes".into()))?,
                )))
            }
            FMT_FIXEXT1 => self.read_ext_data(1),
            FMT_FIXEXT2 => self.read_ext_data(2),
            FMT_FIXEXT4 => self.read_ext_data(4),
            FMT_FIXEXT8 => self.read_ext_data(8),
            FMT_FIXEXT16 => self.read_ext_data(16),
            FMT_STR8 => {
                let n = self.read_u8()? as usize;
                self.read_str_bytes(n)
            }
            FMT_STR16 => {
                let n = self.read_u16_be()? as usize;
                self.read_str_bytes(n)
            }
            FMT_STR32 => {
                let n = self.read_u32_be()? as usize;
                self.read_str_bytes(n)
            }
            FMT_ARRAY16 => {
                let n = self.read_u16_be()? as usize;
                self.read_array_items(n)
            }
            FMT_ARRAY32 => {
                let n = self.read_u32_be()? as usize;
                self.read_array_items(n)
            }
            FMT_MAP16 => {
                let n = self.read_u16_be()? as usize;
                self.read_map_items(n)
            }
            FMT_MAP32 => {
                let n = self.read_u32_be()? as usize;
                self.read_map_items(n)
            }
            other => Err(IoError::FormatError(format!(
                "MsgPack: unknown format byte 0x{other:02x} at offset {}",
                self.pos - 1
            ))),
        }
    }
}

// ─────────────────────────── Convenience encode/decode ───────────────────────

/// Encode a `MsgPackValue` to bytes.
pub fn msgpack_encode(value: &MsgPackValue) -> Vec<u8> {
    let mut w = MsgPackWriter::new();
    w.write_value(value);
    w.into_bytes()
}

/// Decode a `MsgPackValue` from a byte slice.  Returns `(value, bytes_consumed)`.
pub fn msgpack_decode(data: &[u8]) -> Result<(MsgPackValue, usize)> {
    let mut r = MsgPackReader::new(data);
    let value = r.read_value()?;
    Ok((value, r.position()))
}

// ─────────────────────────────── round-trip helper ───────────────────────────

/// Use a [`Cursor`] to hold bytes for cursor-based encode/decode operations.
pub fn roundtrip(value: &MsgPackValue) -> Result<MsgPackValue> {
    let encoded = msgpack_encode(value);
    let (decoded, _) = msgpack_decode(&encoded)?;
    Ok(decoded)
}

// ─────────────────────────────── Cursor-backed reader ────────────────────────

/// A streaming MessagePack reader backed by an [`std::io::Read`] source.
pub struct MsgPackStreamReader<R: Read> {
    inner: Cursor<Vec<u8>>,
    _source: std::marker::PhantomData<R>,
}

impl<R: Read> MsgPackStreamReader<R> {
    /// Create from a `Read` source, eagerly reading all bytes into memory.
    pub fn from_reader(mut source: R) -> Result<Self> {
        let mut buf = Vec::new();
        source.read_to_end(&mut buf).map_err(IoError::Io)?;
        Ok(Self {
            inner: Cursor::new(buf),
            _source: std::marker::PhantomData,
        })
    }

    /// Read the next MessagePack value.
    pub fn read_next(&mut self) -> Result<MsgPackValue> {
        let pos = self.inner.position() as usize;
        let buf = self.inner.get_ref();
        let remaining = &buf[pos..];
        let mut reader = MsgPackReader::new(remaining);
        let value = reader.read_value()?;
        let consumed = reader.position();
        self.inner.set_position((pos + consumed) as u64);
        Ok(value)
    }

    /// True if the stream is exhausted.
    pub fn is_done(&self) -> bool {
        self.inner.position() >= self.inner.get_ref().len() as u64
    }
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(v: MsgPackValue) -> MsgPackValue {
        roundtrip(&v).expect("roundtrip failed")
    }

    #[test]
    fn test_nil_roundtrip() {
        assert_eq!(rt(MsgPackValue::Nil), MsgPackValue::Nil);
    }

    #[test]
    fn test_bool_roundtrip() {
        assert_eq!(rt(MsgPackValue::Bool(true)), MsgPackValue::Bool(true));
        assert_eq!(rt(MsgPackValue::Bool(false)), MsgPackValue::Bool(false));
    }

    #[test]
    fn test_positive_fixint() {
        for i in 0i64..=127 {
            assert_eq!(rt(MsgPackValue::Int(i)), MsgPackValue::Int(i));
        }
    }

    #[test]
    fn test_negative_fixint() {
        for i in -32i64..=-1 {
            assert_eq!(rt(MsgPackValue::Int(i)), MsgPackValue::Int(i));
        }
    }

    #[test]
    fn test_int_ranges() {
        let vals: &[i64] = &[
            i8::MIN as i64,
            i8::MAX as i64,
            i16::MIN as i64,
            i16::MAX as i64,
            i32::MIN as i64,
            i32::MAX as i64,
            i64::MIN,
            i64::MAX,
        ];
        for &v in vals {
            let enc = msgpack_encode(&MsgPackValue::Int(v));
            let (dec, consumed) = msgpack_decode(&enc).expect("decode");
            assert_eq!(consumed, enc.len());
            match dec {
                MsgPackValue::Int(d) => assert_eq!(d, v),
                MsgPackValue::UInt(d) => assert_eq!(d as i64, v),
                other => panic!("unexpected {other:?} for {v}"),
            }
        }
    }

    #[test]
    fn test_uint_roundtrip() {
        let vals: &[u64] = &[
            0,
            127,
            128,
            255,
            256,
            65535,
            65536,
            u32::MAX as u64,
            u64::MAX,
        ];
        for &v in vals {
            let enc = msgpack_encode(&MsgPackValue::UInt(v));
            let (dec, _) = msgpack_decode(&enc).expect("decode");
            match dec {
                MsgPackValue::UInt(d) => assert_eq!(d, v),
                MsgPackValue::Int(d) if d >= 0 => assert_eq!(d as u64, v),
                other => panic!("unexpected {other:?} for {v}"),
            }
        }
    }

    #[test]
    fn test_float32_roundtrip() {
        let v = MsgPackValue::Float32(std::f32::consts::PI);
        assert_eq!(rt(v), MsgPackValue::Float32(std::f32::consts::PI));
    }

    #[test]
    fn test_float64_roundtrip() {
        let v = MsgPackValue::Float64(std::f64::consts::PI);
        assert_eq!(rt(v), MsgPackValue::Float64(std::f64::consts::PI));
    }

    #[test]
    fn test_str_fixstr() {
        let s = "hello".to_string();
        assert_eq!(rt(MsgPackValue::Str(s.clone())), MsgPackValue::Str(s));
    }

    #[test]
    fn test_str_str8() {
        let s = "x".repeat(200);
        assert_eq!(rt(MsgPackValue::Str(s.clone())), MsgPackValue::Str(s));
    }

    #[test]
    fn test_str_str16() {
        let s = "y".repeat(256);
        assert_eq!(rt(MsgPackValue::Str(s.clone())), MsgPackValue::Str(s));
    }

    #[test]
    fn test_bin_roundtrip() {
        let b = vec![0u8, 1, 2, 3, 255];
        assert_eq!(rt(MsgPackValue::Bin(b.clone())), MsgPackValue::Bin(b));
    }

    #[test]
    fn test_array_roundtrip() {
        let arr = MsgPackValue::Array(vec![
            MsgPackValue::Int(1),
            MsgPackValue::Str("hello".into()),
            MsgPackValue::Nil,
        ]);
        assert_eq!(rt(arr.clone()), arr);
    }

    #[test]
    fn test_map_roundtrip() {
        let map = MsgPackValue::Map(vec![(
            MsgPackValue::Str("key".into()),
            MsgPackValue::Int(99),
        )]);
        assert_eq!(rt(map.clone()), map);
    }

    #[test]
    fn test_ext_fixext1() {
        let v = MsgPackValue::Ext(7, vec![0xab]);
        assert_eq!(rt(v.clone()), v);
    }

    #[test]
    fn test_ext_fixext8() {
        let v = MsgPackValue::Ext(-1, vec![0u8; 8]);
        assert_eq!(rt(v.clone()), v);
    }

    #[test]
    fn test_writer_reader_integration() {
        let mut w = MsgPackWriter::new();
        w.write_map_header(2);
        w.write_str("name");
        w.write_str("Alice");
        w.write_str("score");
        w.write_int(100);
        let bytes = w.into_bytes();

        let mut r = MsgPackReader::new(&bytes);
        let v = r.read_value().expect("decode");
        assert!(r.is_empty());

        match v {
            MsgPackValue::Map(pairs) => {
                assert_eq!(pairs.len(), 2);
                assert_eq!(pairs[0].0, MsgPackValue::Str("name".into()));
                assert_eq!(pairs[0].1, MsgPackValue::Str("Alice".into()));
                assert_eq!(pairs[1].0, MsgPackValue::Str("score".into()));
                assert_eq!(pairs[1].1, MsgPackValue::Int(100));
            }
            other => panic!("expected map, got {other:?}"),
        }
    }

    #[test]
    fn test_msgpack_decode_returns_consumed() {
        let mut w = MsgPackWriter::new();
        w.write_int(42);
        w.write_nil();
        let bytes = w.into_bytes();

        // Only consume the first value
        let (val, consumed) = msgpack_decode(&bytes).expect("decode");
        assert_eq!(val, MsgPackValue::Int(42));
        assert_eq!(consumed, 1); // positive fixint is 1 byte
    }
}
