//! Advanced compression primitives for scientific data
//!
//! Provides high-level compressor structs and encoding utilities:
//! - `ZstdCompressor` – pure-Rust Zstandard-like compression via `oxiarc_archive`
//! - `LZ4Compressor` – pure-Rust LZ4 frame compression via `oxiarc_archive`
//! - `RunLengthEncoding` – RLE for sparse/repetitive byte data
//! - `DeltaEncoding` – delta encoding for sorted numeric streams
//! - `FrameCompressor` – frame-based streaming compression/decompression
//! - `CompressionBenchmark` – measure compression ratio and throughput

#![allow(dead_code)]
#![allow(missing_docs)]

use crate::error::{IoError, Result};
use oxiarc_archive::{lz4, zstd};
use std::io::Write as _;
use std::time::Instant;

// ---------------------------------------------------------------------------
// ZstdCompressor
// ---------------------------------------------------------------------------

/// Pure-Rust Zstandard-like compressor backed by `oxiarc_archive::zstd`.
///
/// Wraps the streaming writer/reader pair to provide a convenient compress/decompress API.
#[derive(Debug, Clone)]
pub struct ZstdCompressor {
    /// Compression level (clamped to a valid range internally)
    pub level: u8,
}

impl ZstdCompressor {
    /// Create a compressor with the default compression level (3).
    pub fn new() -> Self {
        Self { level: 3 }
    }

    /// Create a compressor with a specific level.
    pub fn with_level(level: u8) -> Self {
        Self { level }
    }

    /// Compress `data` in one shot and return the compressed bytes.
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let writer = zstd::ZstdWriter::new();
        writer
            .compress(data)
            .map_err(|e| IoError::CompressionError(format!("zstd compress: {e}")))
    }

    /// Decompress `data` compressed with this (or any compatible) Zstd compressor.
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        use std::io::Cursor;
        let cursor = Cursor::new(data);
        let mut reader = zstd::ZstdReader::new(cursor)
            .map_err(|e| IoError::DecompressionError(format!("zstd reader: {e}")))?;
        reader
            .decompress()
            .map_err(|e| IoError::DecompressionError(format!("zstd decompress: {e}")))
    }

    /// Compression ratio: `uncompressed_size / compressed_size`.
    /// Returns 0.0 if compression fails.
    pub fn ratio(&self, data: &[u8]) -> f64 {
        match self.compress(data) {
            Ok(c) if !c.is_empty() => data.len() as f64 / c.len() as f64,
            _ => 0.0,
        }
    }
}

impl Default for ZstdCompressor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LZ4Compressor
// ---------------------------------------------------------------------------

/// Pure-Rust LZ4 frame compressor backed by `oxiarc_archive::lz4`.
///
/// LZ4 prioritises speed over compression ratio and is ideal for high-throughput
/// real-time compression of scientific streams.
#[derive(Debug, Clone)]
pub struct LZ4Compressor;

impl LZ4Compressor {
    /// Create an LZ4 compressor.
    pub fn new() -> Self {
        Self
    }

    /// Compress `data` in one shot.
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut writer = lz4::Lz4Writer::new(Vec::new());
        writer
            .write_compressed(data)
            .map_err(|e| IoError::CompressionError(format!("lz4 compress: {e}")))?;
        Ok(writer.into_inner())
    }

    /// Decompress `data` previously compressed by this compressor.
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        use std::io::Cursor;
        let cursor = Cursor::new(data);
        let mut reader = lz4::Lz4Reader::new(cursor)
            .map_err(|e| IoError::DecompressionError(format!("lz4 reader: {e}")))?;
        reader
            .decompress()
            .map_err(|e| IoError::DecompressionError(format!("lz4 decompress: {e}")))
    }

    /// Compression ratio: `uncompressed_size / compressed_size`.
    pub fn ratio(&self, data: &[u8]) -> f64 {
        match self.compress(data) {
            Ok(c) if !c.is_empty() => data.len() as f64 / c.len() as f64,
            _ => 0.0,
        }
    }
}

impl Default for LZ4Compressor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RunLengthEncoding
// ---------------------------------------------------------------------------

/// Run-length encoding for byte sequences.
///
/// Encoding format: a sequence of `(count: u8, byte: u8)` pairs.
/// Runs longer than 255 are split into multiple pairs.
///
/// Best suited for sparse or highly repetitive data (e.g. binary masks, indicator arrays).
pub struct RunLengthEncoding;

impl RunLengthEncoding {
    /// Encode `data` as RLE.  Returns the encoded bytes.
    pub fn encode(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut count = 1u8;
        let mut current = data[0];

        for &b in &data[1..] {
            if b == current && count < 255 {
                count += 1;
            } else {
                out.push(count);
                out.push(current);
                current = b;
                count = 1;
            }
        }
        out.push(count);
        out.push(current);
        out
    }

    /// Decode RLE-encoded bytes.  Returns the original byte sequence.
    pub fn decode(encoded: &[u8]) -> Result<Vec<u8>> {
        if !encoded.len().is_multiple_of(2) {
            return Err(IoError::DecompressionError(
                "RLE: encoded length must be even".to_string(),
            ));
        }
        let mut out = Vec::new();
        let mut i = 0;
        while i + 1 < encoded.len() {
            let count = encoded[i] as usize;
            let byte = encoded[i + 1];
            for _ in 0..count {
                out.push(byte);
            }
            i += 2;
        }
        Ok(out)
    }

    /// Compression ratio achieved on `data`.
    pub fn ratio(data: &[u8]) -> f64 {
        let encoded = Self::encode(data);
        if encoded.is_empty() {
            return 1.0;
        }
        data.len() as f64 / encoded.len() as f64
    }
}

// ---------------------------------------------------------------------------
// DeltaEncoding
// ---------------------------------------------------------------------------

/// Delta encoding for sorted or slowly-varying numeric data.
///
/// The first value is stored verbatim; subsequent values store the difference
/// from the previous value.  This dramatically reduces entropy for monotonically
/// sorted or slowly-varying integer or float sequences, making them highly
/// compressible with a secondary compressor.
pub struct DeltaEncoding;

impl DeltaEncoding {
    /// Encode a slice of `i64` values as deltas.
    ///
    /// The encoded format is `[n:u64_le][v0:i64_le][d1:i64_le]...[dn-1:i64_le]`
    /// where `n` is the number of elements.
    pub fn encode_i64(data: &[i64]) -> Vec<u8> {
        let n = data.len();
        let mut out = Vec::with_capacity(8 + n * 8);
        out.extend_from_slice(&(n as u64).to_le_bytes());
        if n == 0 {
            return out;
        }
        out.extend_from_slice(&data[0].to_le_bytes());
        for i in 1..n {
            let delta = data[i].wrapping_sub(data[i - 1]);
            out.extend_from_slice(&delta.to_le_bytes());
        }
        out
    }

    /// Decode delta-encoded `i64` values.
    pub fn decode_i64(encoded: &[u8]) -> Result<Vec<i64>> {
        if encoded.len() < 8 {
            return Err(IoError::DecompressionError("Delta: too short".to_string()));
        }
        let n = u64::from_le_bytes(
            encoded[..8]
                .try_into()
                .map_err(|_| IoError::DecompressionError("Delta: bad length prefix".to_string()))?,
        ) as usize;
        if n == 0 {
            return Ok(Vec::new());
        }
        if encoded.len() < 8 + n * 8 {
            return Err(IoError::DecompressionError(
                "Delta: encoded data too short".to_string(),
            ));
        }
        let mut out = Vec::with_capacity(n);
        let first = i64::from_le_bytes(
            encoded[8..16]
                .try_into()
                .map_err(|_| IoError::DecompressionError("Delta: bad first value".to_string()))?,
        );
        out.push(first);
        let mut prev = first;
        for i in 1..n {
            let offset = 8 + i * 8;
            let delta =
                i64::from_le_bytes(encoded[offset..offset + 8].try_into().map_err(|_| {
                    IoError::DecompressionError("Delta: bad delta value".to_string())
                })?);
            let val = prev.wrapping_add(delta);
            out.push(val);
            prev = val;
        }
        Ok(out)
    }

    /// Encode a slice of `f64` values as integer-quantised deltas.
    ///
    /// Values are multiplied by `scale` and rounded to `i64` before delta encoding.
    /// Decoding divides by `scale` to recover approximate floats.
    pub fn encode_f64(data: &[f64], scale: f64) -> Vec<u8> {
        let ints: Vec<i64> = data.iter().map(|&v| (v * scale).round() as i64).collect();
        Self::encode_i64(&ints)
    }

    /// Decode delta-encoded `f64` values (inverse of `encode_f64`).
    pub fn decode_f64(encoded: &[u8], scale: f64) -> Result<Vec<f64>> {
        let ints = Self::decode_i64(encoded)?;
        Ok(ints.iter().map(|&v| v as f64 / scale).collect())
    }
}

// ---------------------------------------------------------------------------
// FrameCompressor
// ---------------------------------------------------------------------------

/// Frame-based streaming compression.
///
/// Splits input data into fixed-size frames and compresses each frame
/// independently. The frame format is:
/// ```text
/// [magic: 4 bytes "FRCM"][version: u8][codec: u8][num_frames: u32_le]
///   per frame: [compressed_size: u32_le][original_size: u32_le][data: ...]
/// ```
/// This format allows random frame access and partial decompression.
pub struct FrameCompressor {
    /// Frame size in bytes (default: 64 KiB)
    pub frame_size: usize,
    /// Compression codec
    pub codec: FrameCodec,
}

/// Codec used by `FrameCompressor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameCodec {
    /// No compression (passthrough, for benchmarking)
    None,
    /// LZ4 compression per frame
    Lz4,
    /// Zstd compression per frame
    Zstd,
    /// RLE compression per frame
    Rle,
}

const FRAME_MAGIC: &[u8; 4] = b"FRCM";
const FRAME_VERSION: u8 = 1;

impl FrameCompressor {
    /// Create a frame compressor with default settings (64 KiB frames, LZ4).
    pub fn new() -> Self {
        Self {
            frame_size: 64 * 1024,
            codec: FrameCodec::Lz4,
        }
    }

    /// Set frame size.
    pub fn with_frame_size(mut self, size: usize) -> Self {
        self.frame_size = size.max(64); // Minimum 64 bytes
        self
    }

    /// Set codec.
    pub fn with_codec(mut self, codec: FrameCodec) -> Self {
        self.codec = codec;
        self
    }

    /// Compress `data` into a framed stream.
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let frames: Vec<&[u8]> = data.chunks(self.frame_size).collect();
        let num_frames = frames.len() as u32;

        // Compress each frame
        let mut compressed_frames: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
        for frame in &frames {
            let c = self.compress_frame(frame)?;
            compressed_frames.push(c);
        }

        // Serialise
        let total_payload: usize = compressed_frames.iter().map(|f| 8 + f.len()).sum();
        let mut out = Vec::with_capacity(10 + total_payload);
        out.extend_from_slice(FRAME_MAGIC);
        out.push(FRAME_VERSION);
        out.push(self.codec as u8);
        out.extend_from_slice(&num_frames.to_le_bytes());

        for (cf, orig) in compressed_frames.iter().zip(frames.iter()) {
            out.extend_from_slice(&(cf.len() as u32).to_le_bytes());
            out.extend_from_slice(&(orig.len() as u32).to_le_bytes());
            out.extend_from_slice(cf);
        }
        Ok(out)
    }

    /// Decompress a framed stream produced by `compress`.
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 10 {
            return Err(IoError::DecompressionError("Frame: too short".to_string()));
        }
        if &data[..4] != FRAME_MAGIC {
            return Err(IoError::DecompressionError("Frame: bad magic".to_string()));
        }
        if data[4] != FRAME_VERSION {
            return Err(IoError::DecompressionError(format!(
                "Frame: unsupported version {}",
                data[4]
            )));
        }
        let codec = FrameCodec::from_u8(data[5]).ok_or_else(|| {
            IoError::DecompressionError(format!("Frame: unknown codec {}", data[5]))
        })?;
        let num_frames = u32::from_le_bytes([data[6], data[7], data[8], data[9]]) as usize;

        let mut out = Vec::new();
        let mut pos = 10usize;
        for _ in 0..num_frames {
            if pos + 8 > data.len() {
                return Err(IoError::DecompressionError(
                    "Frame: truncated header".to_string(),
                ));
            }
            let comp_size =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            let _orig_size =
                u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                    as usize;
            pos += 8;
            if pos + comp_size > data.len() {
                return Err(IoError::DecompressionError(
                    "Frame: truncated data".to_string(),
                ));
            }
            let frame_data = &data[pos..pos + comp_size];
            pos += comp_size;

            let decompressed = Self::decompress_frame_with_codec(frame_data, codec)?;
            out.extend_from_slice(&decompressed);
        }
        Ok(out)
    }

    fn compress_frame(&self, frame: &[u8]) -> Result<Vec<u8>> {
        match self.codec {
            FrameCodec::None => Ok(frame.to_vec()),
            FrameCodec::Lz4 => {
                let c = LZ4Compressor::new();
                c.compress(frame)
            }
            FrameCodec::Zstd => {
                let c = ZstdCompressor::new();
                c.compress(frame)
            }
            FrameCodec::Rle => Ok(RunLengthEncoding::encode(frame)),
        }
    }

    fn decompress_frame_with_codec(frame: &[u8], codec: FrameCodec) -> Result<Vec<u8>> {
        match codec {
            FrameCodec::None => Ok(frame.to_vec()),
            FrameCodec::Lz4 => LZ4Compressor::new().decompress(frame),
            FrameCodec::Zstd => ZstdCompressor::new().decompress(frame),
            FrameCodec::Rle => RunLengthEncoding::decode(frame),
        }
    }
}

impl Default for FrameCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameCodec {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Lz4),
            2 => Some(Self::Zstd),
            3 => Some(Self::Rle),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// CompressionBenchmark
// ---------------------------------------------------------------------------

/// Benchmark results for a single compression trial.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Codec name
    pub codec: String,
    /// Input (uncompressed) size in bytes
    pub input_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio (input / compressed)
    pub ratio: f64,
    /// Compression throughput in MiB/s
    pub compress_mibps: f64,
    /// Decompression throughput in MiB/s
    pub decompress_mibps: f64,
    /// Compression latency in microseconds
    pub compress_us: u64,
    /// Decompression latency in microseconds
    pub decompress_us: u64,
}

impl BenchmarkResult {
    /// Human-readable one-line summary.
    pub fn summary(&self) -> String {
        format!(
            "{}: ratio={:.2}x  compress={:.1} MiB/s  decompress={:.1} MiB/s  ({} → {} bytes)",
            self.codec,
            self.ratio,
            self.compress_mibps,
            self.decompress_mibps,
            self.input_size,
            self.compressed_size,
        )
    }
}

/// Utility for benchmarking compression algorithms on a data sample.
pub struct CompressionBenchmark {
    /// Number of warm-up runs before timing
    pub warmup_runs: usize,
    /// Number of timed runs for averaging
    pub timed_runs: usize,
}

impl CompressionBenchmark {
    /// Create a benchmark with default settings (1 warm-up, 5 timed runs).
    pub fn new() -> Self {
        Self {
            warmup_runs: 1,
            timed_runs: 5,
        }
    }

    /// Set warm-up run count.
    pub fn with_warmup(mut self, n: usize) -> Self {
        self.warmup_runs = n;
        self
    }

    /// Set timed run count.
    pub fn with_timed_runs(mut self, n: usize) -> Self {
        self.timed_runs = n.max(1);
        self
    }

    /// Benchmark `ZstdCompressor` on `data`.
    pub fn bench_zstd(&self, data: &[u8]) -> Result<BenchmarkResult> {
        let c = ZstdCompressor::new();
        let compressed = c.compress(data)?;
        self.measure(
            "Zstd",
            data,
            &compressed,
            || c.compress(data),
            || c.decompress(&compressed),
        )
    }

    /// Benchmark `LZ4Compressor` on `data`.
    pub fn bench_lz4(&self, data: &[u8]) -> Result<BenchmarkResult> {
        let c = LZ4Compressor::new();
        let compressed = c.compress(data)?;
        self.measure(
            "LZ4",
            data,
            &compressed,
            || c.compress(data),
            || c.decompress(&compressed),
        )
    }

    /// Benchmark `RunLengthEncoding` on `data`.
    pub fn bench_rle(&self, data: &[u8]) -> Result<BenchmarkResult> {
        let encoded = RunLengthEncoding::encode(data);
        self.measure(
            "RLE",
            data,
            &encoded,
            || Ok(RunLengthEncoding::encode(data)),
            || RunLengthEncoding::decode(&encoded),
        )
    }

    /// Benchmark `FrameCompressor` (LZ4) on `data`.
    pub fn bench_frame_lz4(&self, data: &[u8]) -> Result<BenchmarkResult> {
        let fc = FrameCompressor::new();
        let compressed = fc.compress(data)?;
        self.measure(
            "Frame-LZ4",
            data,
            &compressed,
            || fc.compress(data),
            || fc.decompress(&compressed),
        )
    }

    /// Run all built-in benchmarks and return the results.
    pub fn run_all(&self, data: &[u8]) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();
        if let Ok(r) = self.bench_zstd(data) {
            results.push(r);
        }
        if let Ok(r) = self.bench_lz4(data) {
            results.push(r);
        }
        if let Ok(r) = self.bench_rle(data) {
            results.push(r);
        }
        if let Ok(r) = self.bench_frame_lz4(data) {
            results.push(r);
        }
        results
    }

    fn measure<C, D>(
        &self,
        name: &str,
        data: &[u8],
        compressed: &[u8],
        compress_fn: C,
        decompress_fn: D,
    ) -> Result<BenchmarkResult>
    where
        C: Fn() -> Result<Vec<u8>>,
        D: Fn() -> Result<Vec<u8>>,
    {
        // Warm up
        for _ in 0..self.warmup_runs {
            let _ = compress_fn();
            let _ = decompress_fn();
        }

        // Time compression
        let mut total_compress_us = 0u64;
        for _ in 0..self.timed_runs {
            let t = Instant::now();
            compress_fn()?;
            total_compress_us += t.elapsed().as_micros() as u64;
        }
        let avg_compress_us = total_compress_us / self.timed_runs as u64;

        // Time decompression
        let mut total_decompress_us = 0u64;
        for _ in 0..self.timed_runs {
            let t = Instant::now();
            decompress_fn()?;
            total_decompress_us += t.elapsed().as_micros() as u64;
        }
        let avg_decompress_us = total_decompress_us / self.timed_runs as u64;

        let input_size = data.len();
        let compressed_size = compressed.len();
        let ratio = if compressed_size == 0 {
            0.0
        } else {
            input_size as f64 / compressed_size as f64
        };

        let to_mibps = |us: u64| -> f64 {
            if us == 0 {
                return f64::INFINITY;
            }
            (input_size as f64 / (1024.0 * 1024.0)) / (us as f64 / 1_000_000.0)
        };

        Ok(BenchmarkResult {
            codec: name.to_string(),
            input_size,
            compressed_size,
            ratio,
            compress_mibps: to_mibps(avg_compress_us),
            decompress_mibps: to_mibps(avg_decompress_us),
            compress_us: avg_compress_us,
            decompress_us: avg_decompress_us,
        })
    }
}

impl Default for CompressionBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const REPETITIVE: &[u8] = b"aaaaaaaaabbbbbbbbbbccccccccddddddddeeeeeeee";
    const TEXT: &[u8] =
        b"The quick brown fox jumps over the lazy dog. Scientific data compression!";

    #[test]
    fn test_zstd_roundtrip() {
        let c = ZstdCompressor::new();
        let enc = c.compress(REPETITIVE).unwrap();
        let dec = c.decompress(&enc).unwrap();
        assert_eq!(dec, REPETITIVE);
    }

    #[test]
    fn test_lz4_roundtrip() {
        let c = LZ4Compressor::new();
        let enc = c.compress(TEXT).unwrap();
        let dec = c.decompress(&enc).unwrap();
        assert_eq!(dec, TEXT);
    }

    #[test]
    fn test_rle_roundtrip() {
        let dec = RunLengthEncoding::decode(&RunLengthEncoding::encode(REPETITIVE)).unwrap();
        assert_eq!(dec, REPETITIVE);
    }

    #[test]
    fn test_rle_empty() {
        let enc = RunLengthEncoding::encode(&[]);
        assert!(enc.is_empty());
        let dec = RunLengthEncoding::decode(&[]).unwrap();
        assert!(dec.is_empty());
    }

    #[test]
    fn test_rle_long_run() {
        let data = vec![42u8; 512]; // Two runs of 255 + one of 2
        let enc = RunLengthEncoding::encode(&data);
        let dec = RunLengthEncoding::decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn test_delta_i64_roundtrip() {
        let values: Vec<i64> = (0..100).map(|i| i * 7 + 1000).collect();
        let enc = DeltaEncoding::encode_i64(&values);
        let dec = DeltaEncoding::decode_i64(&enc).unwrap();
        assert_eq!(dec, values);
    }

    #[test]
    fn test_delta_f64_roundtrip() {
        let values: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let enc = DeltaEncoding::encode_f64(&values, 1000.0);
        let dec = DeltaEncoding::decode_f64(&enc, 1000.0).unwrap();
        for (a, b) in values.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 0.001, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_delta_empty() {
        let enc = DeltaEncoding::encode_i64(&[]);
        let dec = DeltaEncoding::decode_i64(&enc).unwrap();
        assert!(dec.is_empty());
    }

    #[test]
    fn test_frame_compressor_roundtrip_lz4() {
        let data: Vec<u8> = (0u8..=255).cycle().take(200_000).collect();
        let fc = FrameCompressor::new()
            .with_frame_size(16 * 1024)
            .with_codec(FrameCodec::Lz4);
        let enc = fc.compress(&data).unwrap();
        let dec = fc.decompress(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn test_frame_compressor_roundtrip_rle() {
        let data = vec![0xABu8; 50_000];
        let fc = FrameCompressor::new().with_codec(FrameCodec::Rle);
        let enc = fc.compress(&data).unwrap();
        let dec = fc.decompress(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn test_frame_compressor_bad_magic() {
        let bad = b"XXXX\x01\x01\x00\x00\x00\x00".to_vec();
        let fc = FrameCompressor::new();
        assert!(fc.decompress(&bad).is_err());
    }

    #[test]
    fn test_benchmark_runs() {
        let data: Vec<u8> = (0u8..=255).cycle().take(10_000).collect();
        let bm = CompressionBenchmark::new()
            .with_warmup(0)
            .with_timed_runs(1);
        let results = bm.run_all(&data);
        // At least zstd and lz4 should succeed
        assert!(results.len() >= 2);
        for r in &results {
            assert!(r.ratio > 0.0);
            assert!(!r.summary().is_empty());
        }
    }
}
