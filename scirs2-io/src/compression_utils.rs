//! Extended compression utilities
//!
//! Provides a suite of lossless encoding primitives that complement the full
//! compression pipeline in [`crate::compression`]:
//!
//! * **Gzip** – convenience wrappers around `oxiarc_deflate` DEFLATE (pure Rust)
//! * **LZ4 block** – frame-less block compression via `oxiarc_archive::lz4`
//! * **Run-Length Encoding (RLE)** – classical byte-stream RLE
//! * **Delta encoding** – i64 and f64 variants for monotone / smooth sequences
//! * **Varint (LEB-128)** – variable-length unsigned integer encoding
//!
//! All routines return `Result<_>` and never panic.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use oxiarc_archive::lz4;
use oxiarc_deflate::deflate as deflate_compress_result;
use oxiarc_deflate::inflate as deflate_decompress_result;

use crate::error::{IoError, Result};

// ─────────────────────────────── Gzip ────────────────────────────────────────

/// Default DEFLATE compression level used by the gzip helpers (0–10).
const GZIP_DEFAULT_LEVEL: u8 = 6;

/// Compress `data` with GZIP (DEFLATE + gzip framing) using `oxiarc_deflate`.
pub fn gzip_compress(data: &[u8]) -> Result<Vec<u8>> {
    // oxiarc_deflate compresses the raw deflate stream; we wrap it in a minimal
    // gzip container (RFC 1952) so the output is a valid .gz file / stream.
    let deflated = deflate_compress_result(data, GZIP_DEFAULT_LEVEL)
        .map_err(|e| IoError::CompressionError(format!("DEFLATE compression failed: {e}")))?;
    let mut out = Vec::with_capacity(10 + deflated.len() + 8);

    // Gzip header (minimal, no filename, no comment)
    out.extend_from_slice(&[
        0x1f, 0x8b, // Magic
        0x08, // Compression method: deflate
        0x00, // Flags: none
        0x00, 0x00, 0x00, 0x00, // Modification time
        0x00, // Extra flags
        0xff, // OS: unknown
    ]);
    out.extend_from_slice(&deflated);

    // CRC-32 and ISIZE (both little-endian)
    let crc = crc32fast::hash(data);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());

    Ok(out)
}

/// Decompress a gzip-framed buffer produced by [`gzip_compress`].
pub fn gzip_decompress(data: &[u8]) -> Result<Vec<u8>> {
    // Strip the 10-byte gzip header
    if data.len() < 18 {
        return Err(IoError::DecompressionError(
            "gzip data too short to contain valid header".to_string(),
        ));
    }
    if data[0] != 0x1f || data[1] != 0x8b {
        return Err(IoError::DecompressionError(
            "not a gzip stream (bad magic bytes)".to_string(),
        ));
    }
    // Flags byte: handle FNAME / FEXTRA / FCOMMENT extensions
    let flags = data[3];
    let mut pos: usize = 10;

    // FEXTRA
    if flags & 0x04 != 0 {
        if pos + 2 > data.len() {
            return Err(IoError::DecompressionError("truncated FEXTRA".to_string()));
        }
        let xlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2 + xlen;
    }
    // FNAME (null-terminated)
    if flags & 0x08 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1; // consume the null terminator
    }
    // FCOMMENT (null-terminated)
    if flags & 0x10 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }
    // FHCRC
    if flags & 0x02 != 0 {
        pos += 2;
    }

    // Remaining: deflate stream + 8-byte trailer
    if pos + 8 > data.len() {
        return Err(IoError::DecompressionError(
            "gzip data truncated before trailer".to_string(),
        ));
    }
    let deflate_end = data.len() - 8;
    let compressed_body = &data[pos..deflate_end];

    let decompressed = deflate_decompress_result(compressed_body)
        .map_err(|e| IoError::DecompressionError(format!("DEFLATE decompression failed: {e}")))?;

    // Verify CRC-32
    let stored_crc = u32::from_le_bytes([
        data[deflate_end],
        data[deflate_end + 1],
        data[deflate_end + 2],
        data[deflate_end + 3],
    ]);
    let actual_crc = crc32fast::hash(&decompressed);
    if stored_crc != actual_crc {
        return Err(IoError::DecompressionError(format!(
            "CRC-32 mismatch: stored {stored_crc:#010x}, computed {actual_crc:#010x}"
        )));
    }

    Ok(decompressed)
}

/// Write `data` as a gzip file at `path`.
pub fn write_gzip(path: &Path, data: &[u8]) -> Result<()> {
    let compressed = gzip_compress(data)?;
    let file = File::create(path)
        .map_err(|e| IoError::FileError(format!("cannot create {:?}: {e}", path)))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(&compressed)
        .map_err(|e| IoError::FileError(format!("write failed: {e}")))?;
    writer
        .flush()
        .map_err(|e| IoError::FileError(format!("flush failed: {e}")))
}

/// Read and decompress a gzip file from `path`.
pub fn read_gzip(path: &Path) -> Result<Vec<u8>> {
    let mut file =
        File::open(path).map_err(|e| IoError::FileError(format!("cannot open {:?}: {e}", path)))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|e| IoError::FileError(format!("read failed: {e}")))?;
    gzip_decompress(&buf)
}

// ─────────────────────────────── LZ4 block ───────────────────────────────────

/// Compress `data` with LZ4 frame format via `oxiarc_archive::lz4`.
pub fn lz4_compress(data: &[u8]) -> Result<Vec<u8>> {
    let mut writer = lz4::Lz4Writer::new(Vec::new());
    writer
        .write_compressed(data)
        .map_err(|e| IoError::CompressionError(format!("LZ4 compress failed: {e}")))?;
    Ok(writer.into_inner())
}

/// Decompress LZ4 frame data produced by [`lz4_compress`].
pub fn lz4_decompress(data: &[u8]) -> Result<Vec<u8>> {
    use std::io::Cursor;
    let cursor = Cursor::new(data);
    let mut reader = lz4::Lz4Reader::new(cursor)
        .map_err(|e| IoError::DecompressionError(format!("LZ4 reader init failed: {e}")))?;
    reader
        .decompress()
        .map_err(|e| IoError::DecompressionError(format!("LZ4 decompress failed: {e}")))
}

// ─────────────────────────────── RLE ─────────────────────────────────────────

/// Run-Length Encode a byte slice.
///
/// Format per run: `[count: u8][byte]`.  Runs are capped at 255 bytes.
/// Non-repeated bytes are emitted as single runs of length 1.
pub fn rle_encode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    let mut idx = 0;
    while idx < data.len() {
        let byte = data[idx];
        let mut count = 1usize;
        while idx + count < data.len() && data[idx + count] == byte && count < 255 {
            count += 1;
        }
        out.push(count as u8);
        out.push(byte);
        idx += count;
    }
    out
}

/// Decode a buffer produced by [`rle_encode`].
pub fn rle_decode(encoded: &[u8]) -> Result<Vec<u8>> {
    if !encoded.len().is_multiple_of(2) {
        return Err(IoError::DecompressionError(
            "RLE encoded data has odd byte count – possibly corrupted".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(encoded.len());
    let mut idx = 0;
    while idx + 1 < encoded.len() {
        let count = encoded[idx] as usize;
        let byte = encoded[idx + 1];
        if count == 0 {
            return Err(IoError::DecompressionError(format!(
                "RLE run of length 0 at offset {idx} – corrupted data"
            )));
        }
        for _ in 0..count {
            out.push(byte);
        }
        idx += 2;
    }
    Ok(out)
}

// ─────────────────────────────── Delta encoding ──────────────────────────────

/// Delta-encode a slice of `i64` values.
///
/// The first element is stored as-is; each subsequent element stores the
/// difference from its predecessor.
pub fn delta_encode_i64(data: &[i64]) -> Vec<i64> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    out.push(data[0]);
    for i in 1..data.len() {
        out.push(data[i].wrapping_sub(data[i - 1]));
    }
    out
}

/// Reconstruct the original sequence from a delta-encoded `i64` slice.
pub fn delta_decode_i64(encoded: &[i64]) -> Vec<i64> {
    if encoded.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(encoded.len());
    out.push(encoded[0]);
    for i in 1..encoded.len() {
        out.push(out[i - 1].wrapping_add(encoded[i]));
    }
    out
}

/// Scale factor used when converting `f64` deltas to integer representation.
const F64_DELTA_SCALE: f64 = 1_000_000.0;

/// Delta-encode `f64` data by scaling to integers and computing differences.
///
/// The encoding stores: `[first_raw: f64, delta_1_scaled: f64, ...]` where each
/// delta is `round((x_i - x_{i-1}) * SCALE)`.  This trades some precision for
/// better compression of smooth or monotone sequences.
pub fn delta_encode_f64(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    out.push(data[0]);
    for i in 1..data.len() {
        let delta = (data[i] - data[i - 1]) * F64_DELTA_SCALE;
        out.push(delta.round());
    }
    out
}

/// Reconstruct `f64` data from a [`delta_encode_f64`] output.
pub fn delta_decode_f64(encoded: &[f64]) -> Vec<f64> {
    if encoded.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(encoded.len());
    out.push(encoded[0]);
    for i in 1..encoded.len() {
        let prev = out[i - 1];
        let delta = encoded[i] / F64_DELTA_SCALE;
        out.push(prev + delta);
    }
    out
}

// ─────────────────────────────── Varint (LEB-128) ────────────────────────────

/// Encode an unsigned 64-bit integer as a variable-length little-endian base-128 value.
///
/// Each byte stores 7 payload bits; the MSB is set on all bytes except the last.
pub fn varint_encode(mut value: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
    out
}

/// Decode a LEB-128 varint from the beginning of `bytes`.
///
/// Returns `(value, bytes_consumed)`.  Returns an error if the buffer is
/// exhausted before the varint terminates or if the value overflows `u64`.
pub fn varint_decode(bytes: &[u8]) -> Result<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        if shift >= 64 {
            return Err(IoError::ParseError(
                "varint overflows u64 (more than 10 bytes)".to_string(),
            ));
        }
        let low_bits = (byte & 0x7f) as u64;
        value |= low_bits << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Ok((value, i + 1));
        }
    }
    Err(IoError::ParseError(
        "varint is truncated – no terminating byte found".to_string(),
    ))
}

/// Encode a slice of `u64` values as a concatenated sequence of varints.
pub fn encode_varint_slice(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::new();
    for &v in values {
        out.extend(varint_encode(v));
    }
    out
}

/// Decode a concatenated varint sequence produced by [`encode_varint_slice`].
///
/// The number of values decoded equals the number of varints in `bytes`.
pub fn decode_varint_slice(bytes: &[u8]) -> Result<Vec<u64>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        let (val, consumed) = varint_decode(&bytes[pos..])?;
        out.push(val);
        pos += consumed;
    }
    Ok(out)
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Gzip ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_gzip_compress_decompress_roundtrip() {
        let original = b"Hello, gzip world! ".repeat(100);
        let compressed = gzip_compress(&original).expect("compress");
        assert!(compressed.len() < original.len(), "should compress");
        let restored = gzip_decompress(&compressed).expect("decompress");
        assert_eq!(restored.as_slice(), original.as_slice());
    }

    #[test]
    fn test_gzip_empty_input() {
        let compressed = gzip_compress(b"").expect("compress empty");
        let restored = gzip_decompress(&compressed).expect("decompress empty");
        assert!(restored.is_empty());
    }

    #[test]
    fn test_write_read_gzip_file() {
        let dir = std::env::temp_dir().join("scirs2_io_gzip_test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("data.bin.gz");

        let data = b"Scientific data with lots of repeated values aaaabbbbcccc".to_vec();
        write_gzip(&path, &data).expect("write gzip");
        let loaded = read_gzip(&path).expect("read gzip");
        assert_eq!(loaded, data);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── LZ4 ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_lz4_compress_decompress_roundtrip() {
        let original: Vec<u8> = (0u8..255).cycle().take(1024).collect();
        let compressed = lz4_compress(&original).expect("lz4 compress");
        let restored = lz4_decompress(&compressed).expect("lz4 decompress");
        assert_eq!(restored, original);
    }

    // ── RLE ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_rle_encode_decode_roundtrip() {
        let data = b"AAAABBBCCDEEEEEF".to_vec();
        let encoded = rle_encode(&data);
        let decoded = rle_decode(&encoded).expect("rle decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rle_empty_input() {
        let encoded = rle_encode(b"");
        assert!(encoded.is_empty());
        let decoded = rle_decode(&encoded).expect("decode empty");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_rle_all_different() {
        let data: Vec<u8> = (0u8..10).collect();
        let encoded = rle_encode(&data);
        // Each byte gets a run of 1: 2 bytes per input byte
        assert_eq!(encoded.len(), 20);
        let decoded = rle_decode(&encoded).expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rle_long_run_caps_at_255() {
        let data = vec![0xABu8; 512];
        let encoded = rle_encode(&data);
        // 512 / 255 = 2 full runs + 1 partial = 3 pairs = 6 bytes
        assert_eq!(encoded.len(), 6);
        let decoded = rle_decode(&encoded).expect("decode long run");
        assert_eq!(decoded, data);
    }

    // ── Delta i64 ────────────────────────────────────────────────────────────

    #[test]
    fn test_delta_encode_decode_i64_roundtrip() {
        let data = vec![100i64, 102, 107, 115, 130, 148];
        let encoded = delta_encode_i64(&data);
        let decoded = delta_decode_i64(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_delta_i64_monotone_decreasing() {
        let data = vec![1000i64, 990, 975, 950, 900];
        let encoded = delta_encode_i64(&data);
        // All deltas should be negative
        for &d in &encoded[1..] {
            assert!(d < 0);
        }
        assert_eq!(delta_decode_i64(&encoded), data);
    }

    #[test]
    fn test_delta_i64_empty() {
        assert!(delta_encode_i64(&[]).is_empty());
        assert!(delta_decode_i64(&[]).is_empty());
    }

    // ── Delta f64 ────────────────────────────────────────────────────────────

    #[test]
    fn test_delta_encode_decode_f64_roundtrip() {
        let data = vec![1.0f64, 1.001, 1.003, 1.006, 1.010];
        let encoded = delta_encode_f64(&data);
        let decoded = delta_decode_f64(&encoded);
        for (orig, got) in data.iter().zip(decoded.iter()) {
            assert!((orig - got).abs() < 1e-5, "orig={orig}, got={got}");
        }
    }

    #[test]
    fn test_delta_f64_empty() {
        assert!(delta_encode_f64(&[]).is_empty());
        assert!(delta_decode_f64(&[]).is_empty());
    }

    // ── Varint ───────────────────────────────────────────────────────────────

    #[test]
    fn test_varint_encode_decode_specific_values() {
        for &v in &[0u64, 1, 127, 128, 255, 256, 65535, 65536, u64::MAX] {
            let enc = varint_encode(v);
            let (dec, consumed) = varint_decode(&enc).expect("decode");
            assert_eq!(dec, v, "value={v}");
            assert_eq!(consumed, enc.len(), "consumed bytes for value={v}");
        }
    }

    #[test]
    fn test_varint_single_byte_range() {
        // 0..=127 must encode to exactly 1 byte
        for v in 0u64..=127 {
            assert_eq!(varint_encode(v).len(), 1);
        }
    }

    #[test]
    fn test_varint_slice_roundtrip() {
        let values = vec![0u64, 1, 127, 128, 300, 65536, 1_000_000_000];
        let encoded = encode_varint_slice(&values);
        let decoded = decode_varint_slice(&encoded).expect("decode slice");
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_varint_decode_empty_slice() {
        let decoded = decode_varint_slice(b"").expect("empty");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_varint_decode_truncated_error() {
        // A byte with MSB set but no continuation
        let bad = vec![0x80u8];
        assert!(varint_decode(&bad).is_err());
    }
}
