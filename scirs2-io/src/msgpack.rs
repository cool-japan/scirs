//! Pure Rust MessagePack serialization and deserialization
//!
//! Implements the full [MessagePack specification](https://msgpack.org/index.html)
//! in pure Rust without any external msgpack crate.  The central type is
//! [`MsgpackValue`], a recursive enum that can represent any MessagePack value.
//!
//! ## Supported formats
//!
//! - **fixint** (0 .. 127, -32 .. -1)
//! - **uint8 / uint16 / uint32 / uint64**
//! - **int8 / int16 / int32 / int64**
//! - **float32 / float64**
//! - **fixstr / str8 / str16 / str32**
//! - **bin8 / bin16 / bin32**
//! - **fixarray / array16 / array32**
//! - **fixmap / map16 / map32**
//! - **nil / true / false**
//! - **ext8 / ext16 / ext32 / fixext 1/2/4/8/16**
//!
//! ## Serde bridge
//!
//! [`to_msgpack`] and [`from_msgpack`] provide a bridge to serde-compatible
//! types by going through [`serde_json::Value`] as an intermediate
//! representation (JSON ↔ MsgpackValue).
//!
//! ## Examples
//!
//! ```rust
//! use scirs2_io::msgpack::{MsgpackValue, encode, decode};
//!
//! let value = MsgpackValue::Map(vec![
//!     (MsgpackValue::Str("key".into()), MsgpackValue::Int(42)),
//! ]);
//! let bytes = encode(&value);
//! let decoded = decode(&bytes).expect("decode ok");
//! assert_eq!(value, decoded);
//! ```

use std::io::{Cursor, Read, Write};

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use serde::{de::DeserializeOwned, Serialize};

use crate::error::{IoError, Result};

// ─────────────────────────────── Format byte constants ───────────────────────

// nil
const NIL: u8 = 0xc0;

// bool
const FALSE: u8 = 0xc2;
const TRUE: u8 = 0xc3;

// float
const FLOAT32: u8 = 0xca;
const FLOAT64: u8 = 0xcb;

// unsigned int
const UINT8: u8 = 0xcc;
const UINT16: u8 = 0xcd;
const UINT32: u8 = 0xce;
const UINT64: u8 = 0xcf;

// signed int
const INT8: u8 = 0xd0;
const INT16: u8 = 0xd1;
const INT32: u8 = 0xd2;
const INT64: u8 = 0xd3;

// fixext
const FIXEXT1: u8 = 0xd4;
const FIXEXT2: u8 = 0xd5;
const FIXEXT4: u8 = 0xd6;
const FIXEXT8: u8 = 0xd7;
const FIXEXT16: u8 = 0xd8;

// str
const STR8: u8 = 0xd9;
const STR16: u8 = 0xda;
const STR32: u8 = 0xdb;

// array
const ARRAY16: u8 = 0xdc;
const ARRAY32: u8 = 0xdd;

// map
const MAP16: u8 = 0xde;
const MAP32: u8 = 0xdf;

// bin
const BIN8: u8 = 0xc4;
const BIN16: u8 = 0xc5;
const BIN32: u8 = 0xc6;

// ext
const EXT8: u8 = 0xc7;
const EXT16: u8 = 0xc8;
const EXT32: u8 = 0xc9;

// mask / prefix ranges
const FIXINT_POS_MASK: u8 = 0x7f; // 0xxxxxxx  →  0..127
const FIXINT_NEG_MASK: u8 = 0xe0; // 111xxxxx  →  -32..-1
const FIXSTR_MASK: u8 = 0xa0; // 101xxxxx  →  0..31 byte str
const FIXARRAY_MASK: u8 = 0x90; // 1001xxxx  →  0..15 element array
const FIXMAP_MASK: u8 = 0x80; // 1000xxxx  →  0..15 entry map

// ─────────────────────────────── Core value type ─────────────────────────────

/// A MessagePack value.
///
/// Covers every type defined in the MessagePack specification.
#[derive(Debug, Clone, PartialEq)]
pub enum MsgpackValue {
    /// MessagePack nil
    Nil,
    /// Boolean
    Bool(bool),
    /// Signed 64-bit integer (covers all msgpack integer widths)
    Int(i64),
    /// Unsigned 64-bit integer (for values that do not fit in i64)
    UInt(u64),
    /// IEEE 754 double-precision float
    Float(f64),
    /// UTF-8 string
    Str(String),
    /// Raw binary data
    Bin(Vec<u8>),
    /// Heterogeneous array
    Array(Vec<MsgpackValue>),
    /// Map (ordered list of key–value pairs; keys are arbitrary MsgpackValues)
    Map(Vec<(MsgpackValue, MsgpackValue)>),
    /// Extension type: (type_code, data)
    Ext(i8, Vec<u8>),
}

// ─────────────────────────────── Encoder ─────────────────────────────────────

/// Serialize `value` to a MessagePack byte vector.
pub fn encode(value: &MsgpackValue) -> Vec<u8> {
    let mut buf = Vec::new();
    // encode_to always succeeds when writing to a Vec
    let _ = encode_to(&mut buf, value);
    buf
}

/// Serialize `value` to `writer`.
///
/// Returns the number of bytes written.
pub fn write_msgpack(writer: &mut dyn Write, value: &MsgpackValue) -> Result<usize> {
    encode_to(writer, value)
}

fn encode_to(w: &mut dyn Write, value: &MsgpackValue) -> Result<usize> {
    match value {
        MsgpackValue::Nil => {
            w.write_u8(NIL).map_err(IoError::Io)?;
            Ok(1)
        }
        MsgpackValue::Bool(b) => {
            w.write_u8(if *b { TRUE } else { FALSE })
                .map_err(IoError::Io)?;
            Ok(1)
        }
        MsgpackValue::Int(i) => encode_int(w, *i),
        MsgpackValue::UInt(u) => encode_uint(w, *u),
        MsgpackValue::Float(f) => {
            w.write_u8(FLOAT64).map_err(IoError::Io)?;
            w.write_f64::<BigEndian>(*f).map_err(IoError::Io)?;
            Ok(9)
        }
        MsgpackValue::Str(s) => encode_str(w, s),
        MsgpackValue::Bin(b) => encode_bin(w, b),
        MsgpackValue::Array(items) => encode_array(w, items),
        MsgpackValue::Map(entries) => encode_map(w, entries),
        MsgpackValue::Ext(type_code, data) => encode_ext(w, *type_code, data),
    }
}

fn encode_int(w: &mut dyn Write, i: i64) -> Result<usize> {
    // positive fixint
    if (0..=127).contains(&i) {
        w.write_u8(i as u8).map_err(IoError::Io)?;
        return Ok(1);
    }
    // negative fixint
    if (-32..0).contains(&i) {
        w.write_u8(i as i8 as u8).map_err(IoError::Io)?;
        return Ok(1);
    }
    if i >= i8::MIN as i64 && i <= i8::MAX as i64 {
        w.write_u8(INT8).map_err(IoError::Io)?;
        w.write_i8(i as i8).map_err(IoError::Io)?;
        return Ok(2);
    }
    if i >= i16::MIN as i64 && i <= i16::MAX as i64 {
        w.write_u8(INT16).map_err(IoError::Io)?;
        w.write_i16::<BigEndian>(i as i16).map_err(IoError::Io)?;
        return Ok(3);
    }
    if i >= i32::MIN as i64 && i <= i32::MAX as i64 {
        w.write_u8(INT32).map_err(IoError::Io)?;
        w.write_i32::<BigEndian>(i as i32).map_err(IoError::Io)?;
        return Ok(5);
    }
    w.write_u8(INT64).map_err(IoError::Io)?;
    w.write_i64::<BigEndian>(i).map_err(IoError::Io)?;
    Ok(9)
}

fn encode_uint(w: &mut dyn Write, u: u64) -> Result<usize> {
    if u <= 127 {
        w.write_u8(u as u8).map_err(IoError::Io)?;
        return Ok(1);
    }
    if u <= u8::MAX as u64 {
        w.write_u8(UINT8).map_err(IoError::Io)?;
        w.write_u8(u as u8).map_err(IoError::Io)?;
        return Ok(2);
    }
    if u <= u16::MAX as u64 {
        w.write_u8(UINT16).map_err(IoError::Io)?;
        w.write_u16::<BigEndian>(u as u16).map_err(IoError::Io)?;
        return Ok(3);
    }
    if u <= u32::MAX as u64 {
        w.write_u8(UINT32).map_err(IoError::Io)?;
        w.write_u32::<BigEndian>(u as u32).map_err(IoError::Io)?;
        return Ok(5);
    }
    w.write_u8(UINT64).map_err(IoError::Io)?;
    w.write_u64::<BigEndian>(u).map_err(IoError::Io)?;
    Ok(9)
}

fn encode_str(w: &mut dyn Write, s: &str) -> Result<usize> {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let header_size = if len <= 31 {
        w.write_u8(FIXSTR_MASK | len as u8).map_err(IoError::Io)?;
        1
    } else if len <= u8::MAX as usize {
        w.write_u8(STR8).map_err(IoError::Io)?;
        w.write_u8(len as u8).map_err(IoError::Io)?;
        2
    } else if len <= u16::MAX as usize {
        w.write_u8(STR16).map_err(IoError::Io)?;
        w.write_u16::<BigEndian>(len as u16).map_err(IoError::Io)?;
        3
    } else if len <= u32::MAX as usize {
        w.write_u8(STR32).map_err(IoError::Io)?;
        w.write_u32::<BigEndian>(len as u32).map_err(IoError::Io)?;
        5
    } else {
        return Err(IoError::SerializationError(
            "string too large for MessagePack (> 4 GiB)".to_string(),
        ));
    };
    w.write_all(bytes).map_err(IoError::Io)?;
    Ok(header_size + len)
}

fn encode_bin(w: &mut dyn Write, data: &[u8]) -> Result<usize> {
    let len = data.len();
    let header_size = if len <= u8::MAX as usize {
        w.write_u8(BIN8).map_err(IoError::Io)?;
        w.write_u8(len as u8).map_err(IoError::Io)?;
        2
    } else if len <= u16::MAX as usize {
        w.write_u8(BIN16).map_err(IoError::Io)?;
        w.write_u16::<BigEndian>(len as u16).map_err(IoError::Io)?;
        3
    } else if len <= u32::MAX as usize {
        w.write_u8(BIN32).map_err(IoError::Io)?;
        w.write_u32::<BigEndian>(len as u32).map_err(IoError::Io)?;
        5
    } else {
        return Err(IoError::SerializationError(
            "binary data too large for MessagePack (> 4 GiB)".to_string(),
        ));
    };
    w.write_all(data).map_err(IoError::Io)?;
    Ok(header_size + len)
}

fn encode_array(w: &mut dyn Write, items: &[MsgpackValue]) -> Result<usize> {
    let n = items.len();
    let header_size = if n <= 15 {
        w.write_u8(FIXARRAY_MASK | n as u8).map_err(IoError::Io)?;
        1
    } else if n <= u16::MAX as usize {
        w.write_u8(ARRAY16).map_err(IoError::Io)?;
        w.write_u16::<BigEndian>(n as u16).map_err(IoError::Io)?;
        3
    } else if n <= u32::MAX as usize {
        w.write_u8(ARRAY32).map_err(IoError::Io)?;
        w.write_u32::<BigEndian>(n as u32).map_err(IoError::Io)?;
        5
    } else {
        return Err(IoError::SerializationError(
            "array too large for MessagePack (> 4 GiB elements)".to_string(),
        ));
    };
    let mut total = header_size;
    for item in items {
        total += encode_to(w, item)?;
    }
    Ok(total)
}

fn encode_map(w: &mut dyn Write, entries: &[(MsgpackValue, MsgpackValue)]) -> Result<usize> {
    let n = entries.len();
    let header_size = if n <= 15 {
        w.write_u8(FIXMAP_MASK | n as u8).map_err(IoError::Io)?;
        1
    } else if n <= u16::MAX as usize {
        w.write_u8(MAP16).map_err(IoError::Io)?;
        w.write_u16::<BigEndian>(n as u16).map_err(IoError::Io)?;
        3
    } else if n <= u32::MAX as usize {
        w.write_u8(MAP32).map_err(IoError::Io)?;
        w.write_u32::<BigEndian>(n as u32).map_err(IoError::Io)?;
        5
    } else {
        return Err(IoError::SerializationError(
            "map too large for MessagePack (> 4 GiB entries)".to_string(),
        ));
    };
    let mut total = header_size;
    for (k, v) in entries {
        total += encode_to(w, k)?;
        total += encode_to(w, v)?;
    }
    Ok(total)
}

fn encode_ext(w: &mut dyn Write, type_code: i8, data: &[u8]) -> Result<usize> {
    let len = data.len();
    let header_size = match len {
        1 => {
            w.write_u8(FIXEXT1).map_err(IoError::Io)?;
            w.write_i8(type_code).map_err(IoError::Io)?;
            2
        }
        2 => {
            w.write_u8(FIXEXT2).map_err(IoError::Io)?;
            w.write_i8(type_code).map_err(IoError::Io)?;
            2
        }
        4 => {
            w.write_u8(FIXEXT4).map_err(IoError::Io)?;
            w.write_i8(type_code).map_err(IoError::Io)?;
            2
        }
        8 => {
            w.write_u8(FIXEXT8).map_err(IoError::Io)?;
            w.write_i8(type_code).map_err(IoError::Io)?;
            2
        }
        16 => {
            w.write_u8(FIXEXT16).map_err(IoError::Io)?;
            w.write_i8(type_code).map_err(IoError::Io)?;
            2
        }
        l if l <= u8::MAX as usize => {
            w.write_u8(EXT8).map_err(IoError::Io)?;
            w.write_u8(l as u8).map_err(IoError::Io)?;
            w.write_i8(type_code).map_err(IoError::Io)?;
            3
        }
        l if l <= u16::MAX as usize => {
            w.write_u8(EXT16).map_err(IoError::Io)?;
            w.write_u16::<BigEndian>(l as u16).map_err(IoError::Io)?;
            w.write_i8(type_code).map_err(IoError::Io)?;
            4
        }
        l if l <= u32::MAX as usize => {
            w.write_u8(EXT32).map_err(IoError::Io)?;
            w.write_u32::<BigEndian>(l as u32).map_err(IoError::Io)?;
            w.write_i8(type_code).map_err(IoError::Io)?;
            6
        }
        _ => {
            return Err(IoError::SerializationError(
                "ext data too large for MessagePack (> 4 GiB)".to_string(),
            ));
        }
    };
    w.write_all(data).map_err(IoError::Io)?;
    Ok(header_size + len)
}

// ─────────────────────────────── Decoder ─────────────────────────────────────

/// Deserialize a [`MsgpackValue`] from a byte slice.
///
/// Returns an error if the bytes are truncated or contain an unknown format byte.
pub fn decode(bytes: &[u8]) -> Result<MsgpackValue> {
    let mut cursor = Cursor::new(bytes);
    let value = decode_from(&mut cursor)?;
    Ok(value)
}

/// Deserialize a [`MsgpackValue`] from `reader`.
pub fn read_msgpack(reader: &mut dyn Read) -> Result<MsgpackValue> {
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).map_err(IoError::Io)?;
    decode(&buf)
}

fn decode_from(cur: &mut Cursor<&[u8]>) -> Result<MsgpackValue> {
    let byte = read_byte(cur)?;

    match byte {
        // nil
        NIL => Ok(MsgpackValue::Nil),

        // bool
        FALSE => Ok(MsgpackValue::Bool(false)),
        TRUE => Ok(MsgpackValue::Bool(true)),

        // float
        FLOAT32 => {
            let f = cur
                .read_f32::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack float32 read: {e}")))?;
            Ok(MsgpackValue::Float(f as f64))
        }
        FLOAT64 => {
            let f = cur
                .read_f64::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack float64 read: {e}")))?;
            Ok(MsgpackValue::Float(f))
        }

        // unsigned int
        UINT8 => {
            let v = read_byte(cur)? as u64;
            Ok(MsgpackValue::UInt(v))
        }
        UINT16 => {
            let v = cur
                .read_u16::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack uint16 read: {e}")))?
                as u64;
            Ok(MsgpackValue::UInt(v))
        }
        UINT32 => {
            let v = cur
                .read_u32::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack uint32 read: {e}")))?
                as u64;
            Ok(MsgpackValue::UInt(v))
        }
        UINT64 => {
            let v = cur
                .read_u64::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack uint64 read: {e}")))?;
            Ok(MsgpackValue::UInt(v))
        }

        // signed int
        INT8 => {
            let v = cur
                .read_i8()
                .map_err(|e| IoError::FormatError(format!("msgpack int8 read: {e}")))?
                as i64;
            Ok(MsgpackValue::Int(v))
        }
        INT16 => {
            let v = cur
                .read_i16::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack int16 read: {e}")))?
                as i64;
            Ok(MsgpackValue::Int(v))
        }
        INT32 => {
            let v = cur
                .read_i32::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack int32 read: {e}")))?
                as i64;
            Ok(MsgpackValue::Int(v))
        }
        INT64 => {
            let v = cur
                .read_i64::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack int64 read: {e}")))?;
            Ok(MsgpackValue::Int(v))
        }

        // str8 / str16 / str32
        STR8 => {
            let len = read_byte(cur)? as usize;
            Ok(MsgpackValue::Str(read_utf8(cur, len)?))
        }
        STR16 => {
            let len = cur
                .read_u16::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack str16 len: {e}")))?
                as usize;
            Ok(MsgpackValue::Str(read_utf8(cur, len)?))
        }
        STR32 => {
            let len = cur
                .read_u32::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack str32 len: {e}")))?
                as usize;
            Ok(MsgpackValue::Str(read_utf8(cur, len)?))
        }

        // bin8 / bin16 / bin32
        BIN8 => {
            let len = read_byte(cur)? as usize;
            Ok(MsgpackValue::Bin(read_bytes(cur, len)?))
        }
        BIN16 => {
            let len = cur
                .read_u16::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack bin16 len: {e}")))?
                as usize;
            Ok(MsgpackValue::Bin(read_bytes(cur, len)?))
        }
        BIN32 => {
            let len = cur
                .read_u32::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack bin32 len: {e}")))?
                as usize;
            Ok(MsgpackValue::Bin(read_bytes(cur, len)?))
        }

        // array16 / array32
        ARRAY16 => {
            let n = cur
                .read_u16::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack array16 len: {e}")))?
                as usize;
            read_array(cur, n)
        }
        ARRAY32 => {
            let n = cur
                .read_u32::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack array32 len: {e}")))?
                as usize;
            read_array(cur, n)
        }

        // map16 / map32
        MAP16 => {
            let n = cur
                .read_u16::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack map16 len: {e}")))?
                as usize;
            read_map(cur, n)
        }
        MAP32 => {
            let n = cur
                .read_u32::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack map32 len: {e}")))?
                as usize;
            read_map(cur, n)
        }

        // fixext
        FIXEXT1 => {
            let tc = cur
                .read_i8()
                .map_err(|e| IoError::FormatError(format!("msgpack fixext1 type: {e}")))?;
            Ok(MsgpackValue::Ext(tc, read_bytes(cur, 1)?))
        }
        FIXEXT2 => {
            let tc = cur
                .read_i8()
                .map_err(|e| IoError::FormatError(format!("msgpack fixext2 type: {e}")))?;
            Ok(MsgpackValue::Ext(tc, read_bytes(cur, 2)?))
        }
        FIXEXT4 => {
            let tc = cur
                .read_i8()
                .map_err(|e| IoError::FormatError(format!("msgpack fixext4 type: {e}")))?;
            Ok(MsgpackValue::Ext(tc, read_bytes(cur, 4)?))
        }
        FIXEXT8 => {
            let tc = cur
                .read_i8()
                .map_err(|e| IoError::FormatError(format!("msgpack fixext8 type: {e}")))?;
            Ok(MsgpackValue::Ext(tc, read_bytes(cur, 8)?))
        }
        FIXEXT16 => {
            let tc = cur
                .read_i8()
                .map_err(|e| IoError::FormatError(format!("msgpack fixext16 type: {e}")))?;
            Ok(MsgpackValue::Ext(tc, read_bytes(cur, 16)?))
        }

        // ext8 / ext16 / ext32
        EXT8 => {
            let len = read_byte(cur)? as usize;
            let tc = cur
                .read_i8()
                .map_err(|e| IoError::FormatError(format!("msgpack ext8 type: {e}")))?;
            Ok(MsgpackValue::Ext(tc, read_bytes(cur, len)?))
        }
        EXT16 => {
            let len = cur
                .read_u16::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack ext16 len: {e}")))?
                as usize;
            let tc = cur
                .read_i8()
                .map_err(|e| IoError::FormatError(format!("msgpack ext16 type: {e}")))?;
            Ok(MsgpackValue::Ext(tc, read_bytes(cur, len)?))
        }
        EXT32 => {
            let len = cur
                .read_u32::<BigEndian>()
                .map_err(|e| IoError::FormatError(format!("msgpack ext32 len: {e}")))?
                as usize;
            let tc = cur
                .read_i8()
                .map_err(|e| IoError::FormatError(format!("msgpack ext32 type: {e}")))?;
            Ok(MsgpackValue::Ext(tc, read_bytes(cur, len)?))
        }

        // positive fixint (0xxxxxxx)
        b if b & 0x80 == 0 => Ok(MsgpackValue::Int((b & FIXINT_POS_MASK) as i64)),

        // negative fixint (111xxxxx)
        b if b & 0xe0 == FIXINT_NEG_MASK => {
            // sign-extend the 5-bit twos-complement value
            let signed = (b as i8) as i64;
            Ok(MsgpackValue::Int(signed))
        }

        // fixstr (101xxxxx)
        b if b & 0xe0 == FIXSTR_MASK => {
            let len = (b & 0x1f) as usize;
            Ok(MsgpackValue::Str(read_utf8(cur, len)?))
        }

        // fixarray (1001xxxx)
        b if b & 0xf0 == FIXARRAY_MASK => {
            let n = (b & 0x0f) as usize;
            read_array(cur, n)
        }

        // fixmap (1000xxxx)
        b if b & 0xf0 == FIXMAP_MASK => {
            let n = (b & 0x0f) as usize;
            read_map(cur, n)
        }

        // Unused / never-used format bytes
        other => Err(IoError::FormatError(format!(
            "unknown MessagePack format byte: {other:#04x}"
        ))),
    }
}

// ─────────────────────────────── Cursor helpers ──────────────────────────────

fn read_byte(cur: &mut Cursor<&[u8]>) -> Result<u8> {
    cur.read_u8()
        .map_err(|e| IoError::FormatError(format!("unexpected end of msgpack data: {e}")))
}

fn read_bytes(cur: &mut Cursor<&[u8]>, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    cur.read_exact(&mut buf)
        .map_err(|e| IoError::FormatError(format!("truncated msgpack data ({len} bytes): {e}")))?;
    Ok(buf)
}

fn read_utf8(cur: &mut Cursor<&[u8]>, len: usize) -> Result<String> {
    let bytes = read_bytes(cur, len)?;
    String::from_utf8(bytes)
        .map_err(|e| IoError::FormatError(format!("invalid UTF-8 in msgpack str: {e}")))
}

fn read_array(cur: &mut Cursor<&[u8]>, n: usize) -> Result<MsgpackValue> {
    let mut items = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        items.push(decode_from(cur)?);
    }
    Ok(MsgpackValue::Array(items))
}

fn read_map(cur: &mut Cursor<&[u8]>, n: usize) -> Result<MsgpackValue> {
    let mut entries = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        let k = decode_from(cur)?;
        let v = decode_from(cur)?;
        entries.push((k, v));
    }
    Ok(MsgpackValue::Map(entries))
}

// ─────────────────────────────── Serde bridge ────────────────────────────────

/// Serialize any `serde::Serialize` type to MessagePack bytes.
///
/// The value is first converted to a [`serde_json::Value`] then encoded with
/// the pure-Rust MessagePack encoder in this module.
pub fn to_msgpack<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let json_val = serde_json::to_value(value)
        .map_err(|e| IoError::SerializationError(format!("to_msgpack serde_json: {e}")))?;
    let mp_val = json_value_to_msgpack(&json_val);
    Ok(encode(&mp_val))
}

/// Deserialize any `serde::de::DeserializeOwned` type from MessagePack bytes.
///
/// The bytes are decoded to a [`MsgpackValue`], converted to a
/// [`serde_json::Value`], and then deserialized via serde.
pub fn from_msgpack<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let mp_val = decode(bytes)?;
    let json_val = msgpack_to_json_value(&mp_val)?;
    serde_json::from_value(json_val)
        .map_err(|e| IoError::DeserializationError(format!("from_msgpack serde_json: {e}")))
}

// ─────────────────────────────── JSON ↔ MsgpackValue ─────────────────────────

/// Convert a [`serde_json::Value`] to a [`MsgpackValue`].
pub fn json_value_to_msgpack(v: &serde_json::Value) -> MsgpackValue {
    match v {
        serde_json::Value::Null => MsgpackValue::Nil,
        serde_json::Value::Bool(b) => MsgpackValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                MsgpackValue::Int(i)
            } else if let Some(u) = n.as_u64() {
                MsgpackValue::UInt(u)
            } else {
                MsgpackValue::Float(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        serde_json::Value::String(s) => MsgpackValue::Str(s.clone()),
        serde_json::Value::Array(arr) => {
            MsgpackValue::Array(arr.iter().map(json_value_to_msgpack).collect())
        }
        serde_json::Value::Object(obj) => MsgpackValue::Map(
            obj.iter()
                .map(|(k, v)| (MsgpackValue::Str(k.clone()), json_value_to_msgpack(v)))
                .collect(),
        ),
    }
}

/// Convert a [`MsgpackValue`] to a [`serde_json::Value`].
///
/// - Binary data is represented as an array of unsigned integers.
/// - Extension types are represented as `{"__ext_type": <code>, "data": [bytes]}`.
/// - UInt values that exceed `i64::MAX` are represented as `f64` (JSON has no
///   distinct unsigned integer type).
pub fn msgpack_to_json_value(v: &MsgpackValue) -> Result<serde_json::Value> {
    match v {
        MsgpackValue::Nil => Ok(serde_json::Value::Null),
        MsgpackValue::Bool(b) => Ok(serde_json::Value::Bool(*b)),
        MsgpackValue::Int(i) => Ok(serde_json::json!(*i)),
        MsgpackValue::UInt(u) => {
            if *u <= i64::MAX as u64 {
                Ok(serde_json::json!(*u as i64))
            } else {
                Ok(serde_json::json!(*u as f64))
            }
        }
        MsgpackValue::Float(f) => {
            let n = serde_json::Number::from_f64(*f).ok_or_else(|| {
                IoError::ConversionError(format!("non-finite float cannot be JSON: {f}"))
            })?;
            Ok(serde_json::Value::Number(n))
        }
        MsgpackValue::Str(s) => Ok(serde_json::Value::String(s.clone())),
        MsgpackValue::Bin(b) => {
            // Represent binary as array of u8 values
            let arr: Vec<serde_json::Value> =
                b.iter().map(|&byte| serde_json::json!(byte)).collect();
            Ok(serde_json::Value::Array(arr))
        }
        MsgpackValue::Array(items) => {
            let arr: Result<Vec<serde_json::Value>> =
                items.iter().map(msgpack_to_json_value).collect();
            Ok(serde_json::Value::Array(arr?))
        }
        MsgpackValue::Map(entries) => {
            let mut obj = serde_json::Map::new();
            for (k, v) in entries {
                let key = match k {
                    MsgpackValue::Str(s) => s.clone(),
                    other => {
                        // Non-string keys: convert to their string representation
                        format!("{other:?}")
                    }
                };
                obj.insert(key, msgpack_to_json_value(v)?);
            }
            Ok(serde_json::Value::Object(obj))
        }
        MsgpackValue::Ext(type_code, data) => {
            // Encode as a special JSON object
            let bytes: Vec<serde_json::Value> =
                data.iter().map(|&b| serde_json::json!(b)).collect();
            Ok(serde_json::json!({
                "__msgpack_ext_type": *type_code as i64,
                "data": bytes,
            }))
        }
    }
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── roundtrip helpers ────────────────────────────────────────────────────

    fn rt(v: &MsgpackValue) -> MsgpackValue {
        let bytes = encode(v);
        decode(&bytes).expect("decode")
    }

    // ── nil / bool ───────────────────────────────────────────────────────────

    #[test]
    fn test_nil_roundtrip() {
        assert_eq!(rt(&MsgpackValue::Nil), MsgpackValue::Nil);
    }

    #[test]
    fn test_bool_roundtrip() {
        assert_eq!(rt(&MsgpackValue::Bool(true)), MsgpackValue::Bool(true));
        assert_eq!(rt(&MsgpackValue::Bool(false)), MsgpackValue::Bool(false));
    }

    // ── integer: positive fixint ─────────────────────────────────────────────

    #[test]
    fn test_positive_fixint_boundaries() {
        for i in [0i64, 1, 63, 127] {
            let v = MsgpackValue::Int(i);
            assert_eq!(rt(&v), v, "fixint {i}");
            // positive fixint is always 1 byte
            assert_eq!(encode(&v).len(), 1, "fixint len {i}");
        }
    }

    // ── integer: negative fixint ─────────────────────────────────────────────

    #[test]
    fn test_negative_fixint_boundaries() {
        for i in [-1i64, -16, -32] {
            let v = MsgpackValue::Int(i);
            assert_eq!(rt(&v), v, "neg fixint {i}");
            assert_eq!(encode(&v).len(), 1, "neg fixint len {i}");
        }
    }

    // ── integer widths ───────────────────────────────────────────────────────

    #[test]
    fn test_int8_roundtrip() {
        for i in [i8::MIN as i64, i8::MAX as i64, -33i64, 128] {
            let v = MsgpackValue::Int(i);
            assert_eq!(rt(&v), v, "int8 {i}");
        }
    }

    #[test]
    fn test_int16_roundtrip() {
        for i in [i16::MIN as i64, i16::MAX as i64, 256i64, -300] {
            let v = MsgpackValue::Int(i);
            assert_eq!(rt(&v), v, "int16 {i}");
        }
    }

    #[test]
    fn test_int32_roundtrip() {
        for i in [i32::MIN as i64, i32::MAX as i64, 70_000i64, -70_000] {
            let v = MsgpackValue::Int(i);
            assert_eq!(rt(&v), v, "int32 {i}");
        }
    }

    #[test]
    fn test_int64_roundtrip() {
        for i in [i64::MIN, i64::MAX, i64::from(i32::MAX) + 1] {
            let v = MsgpackValue::Int(i);
            assert_eq!(rt(&v), v, "int64 {i}");
        }
    }

    // ── uint widths ──────────────────────────────────────────────────────────

    #[test]
    fn test_uint8_roundtrip() {
        let v = MsgpackValue::UInt(200);
        assert_eq!(rt(&v), v);
        assert_eq!(encode(&v).len(), 2); // UINT8 + value
    }

    #[test]
    fn test_uint16_roundtrip() {
        let v = MsgpackValue::UInt(1000);
        assert_eq!(rt(&v), v);
        assert_eq!(encode(&v).len(), 3);
    }

    #[test]
    fn test_uint32_roundtrip() {
        let v = MsgpackValue::UInt(100_000);
        assert_eq!(rt(&v), v);
        assert_eq!(encode(&v).len(), 5);
    }

    #[test]
    fn test_uint64_roundtrip() {
        let v = MsgpackValue::UInt(u64::MAX);
        assert_eq!(rt(&v), v);
        assert_eq!(encode(&v).len(), 9);
    }

    // ── float ────────────────────────────────────────────────────────────────

    #[test]
    fn test_float64_roundtrip() {
        for f in [
            0.0_f64,
            1.0,
            -1.0,
            f64::MAX,
            f64::MIN_POSITIVE,
            std::f64::consts::PI,
        ] {
            let v = MsgpackValue::Float(f);
            assert_eq!(rt(&v), v, "float64 {f}");
        }
    }

    // ── string ───────────────────────────────────────────────────────────────

    #[test]
    fn test_fixstr_roundtrip() {
        let v = MsgpackValue::Str("hello".into());
        assert_eq!(rt(&v), v);
        // fixstr: 1 header + 5 bytes
        assert_eq!(encode(&v).len(), 6);
    }

    #[test]
    fn test_str8_roundtrip() {
        // 32 bytes → needs str8
        let s: String = "x".repeat(32);
        let v = MsgpackValue::Str(s.clone());
        let encoded = encode(&v);
        assert_eq!(encoded[0], STR8);
        assert_eq!(rt(&v), v);
    }

    #[test]
    fn test_str16_roundtrip() {
        let s: String = "y".repeat(300);
        let v = MsgpackValue::Str(s);
        let encoded = encode(&v);
        assert_eq!(encoded[0], STR16);
        assert_eq!(rt(&v), v);
    }

    #[test]
    fn test_empty_string() {
        let v = MsgpackValue::Str(String::new());
        assert_eq!(rt(&v), v);
    }

    // ── binary ───────────────────────────────────────────────────────────────

    #[test]
    fn test_bin8_roundtrip() {
        let v = MsgpackValue::Bin(vec![0x00, 0xff, 0x42]);
        assert_eq!(rt(&v), v);
        assert_eq!(encode(&v)[0], BIN8);
    }

    #[test]
    fn test_bin16_roundtrip() {
        let v = MsgpackValue::Bin(vec![0u8; 256]);
        let encoded = encode(&v);
        assert_eq!(encoded[0], BIN16);
        assert_eq!(rt(&v), v);
    }

    #[test]
    fn test_bin32_roundtrip() {
        let v = MsgpackValue::Bin(vec![1u8; 65536]);
        let encoded = encode(&v);
        assert_eq!(encoded[0], BIN32);
        assert_eq!(rt(&v), v);
    }

    // ── array ────────────────────────────────────────────────────────────────

    #[test]
    fn test_fixarray_roundtrip() {
        let v = MsgpackValue::Array(vec![
            MsgpackValue::Int(1),
            MsgpackValue::Str("two".into()),
            MsgpackValue::Bool(true),
        ]);
        assert_eq!(rt(&v), v);
    }

    #[test]
    fn test_array16_roundtrip() {
        let items: Vec<MsgpackValue> = (0..16).map(|i| MsgpackValue::Int(i)).collect();
        let v = MsgpackValue::Array(items);
        let encoded = encode(&v);
        assert_eq!(encoded[0], ARRAY16);
        assert_eq!(rt(&v), v);
    }

    #[test]
    fn test_nested_array() {
        let inner = MsgpackValue::Array(vec![MsgpackValue::Int(99)]);
        let outer = MsgpackValue::Array(vec![inner, MsgpackValue::Nil]);
        assert_eq!(rt(&outer), outer);
    }

    // ── map ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_fixmap_roundtrip() {
        let v = MsgpackValue::Map(vec![
            (MsgpackValue::Str("a".into()), MsgpackValue::Int(1)),
            (MsgpackValue::Str("b".into()), MsgpackValue::Bool(false)),
        ]);
        assert_eq!(rt(&v), v);
    }

    #[test]
    fn test_map16_roundtrip() {
        let entries: Vec<(MsgpackValue, MsgpackValue)> = (0..16)
            .map(|i| (MsgpackValue::Int(i), MsgpackValue::Int(i * 2)))
            .collect();
        let v = MsgpackValue::Map(entries);
        let encoded = encode(&v);
        assert_eq!(encoded[0], MAP16);
        assert_eq!(rt(&v), v);
    }

    // ── ext ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_fixext1_roundtrip() {
        let v = MsgpackValue::Ext(42, vec![0xab]);
        assert_eq!(rt(&v), v);
        assert_eq!(encode(&v).len(), 3); // FIXEXT1 + type + 1 data
    }

    #[test]
    fn test_fixext8_roundtrip() {
        let v = MsgpackValue::Ext(-1, vec![0u8; 8]);
        assert_eq!(rt(&v), v);
    }

    #[test]
    fn test_ext8_roundtrip() {
        let v = MsgpackValue::Ext(5, vec![0xffu8; 3]);
        assert_eq!(rt(&v), v);
        assert_eq!(encode(&v)[0], EXT8);
    }

    // ── serde bridge ─────────────────────────────────────────────────────────

    #[test]
    fn test_to_from_msgpack_struct() {
        #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
        struct Point {
            x: f64,
            y: f64,
        }

        let p = Point { x: 1.5, y: -2.5 };
        let bytes = to_msgpack(&p).expect("serialize");
        let p2: Point = from_msgpack(&bytes).expect("deserialize");
        assert!((p.x - p2.x).abs() < 1e-10);
        assert!((p.y - p2.y).abs() < 1e-10);
    }

    #[test]
    fn test_to_from_msgpack_vec() {
        let data = vec![1i64, 2, 3, 4, 5];
        let bytes = to_msgpack(&data).expect("serialize");
        let out: Vec<i64> = from_msgpack(&bytes).expect("deserialize");
        assert_eq!(data, out);
    }

    // ── write_msgpack / read_msgpack ─────────────────────────────────────────

    #[test]
    fn test_write_read_msgpack() {
        let value = MsgpackValue::Map(vec![
            (MsgpackValue::Str("count".into()), MsgpackValue::Int(7)),
            (
                MsgpackValue::Str("label".into()),
                MsgpackValue::Str("test".into()),
            ),
        ]);

        let mut buf = Vec::new();
        write_msgpack(&mut buf, &value).expect("write");
        let decoded = read_msgpack(&mut buf.as_slice()).expect("read");
        assert_eq!(value, decoded);
    }

    // ── error cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_decode_empty_input() {
        let result = decode(&[]);
        assert!(result.is_err(), "empty input should error");
    }

    #[test]
    fn test_decode_truncated_str() {
        // fixstr of length 5, but only 3 bytes of data
        let mut bytes = Vec::new();
        bytes.push(FIXSTR_MASK | 5);
        bytes.extend_from_slice(b"abc"); // missing 2 bytes
        let result = decode(&bytes);
        assert!(result.is_err(), "truncated str should error");
    }

    #[test]
    fn test_decode_unknown_byte() {
        // 0xc1 is a never-used format byte in the spec
        let result = decode(&[0xc1]);
        assert!(result.is_err(), "unknown format byte should error");
    }
}
