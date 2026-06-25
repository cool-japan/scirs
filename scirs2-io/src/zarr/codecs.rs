//! Codec pipeline for Zarr chunk encoding/decoding.
//!
//! Codecs transform raw typed data into byte buffers suitable for storage.
//! A pipeline is an ordered list of codecs applied from first to last on write
//! and from last to first on read.

use crate::error::{IoError, Result};

/// Trait for a single codec in the pipeline.
pub trait Codec: std::fmt::Debug + Send + Sync {
    /// Codec name for metadata.
    fn name(&self) -> &str;

    /// Encode (write path).
    fn encode(&self, data: &[u8]) -> Result<Vec<u8>>;

    /// Decode (read path).
    fn decode(&self, data: &[u8]) -> Result<Vec<u8>>;
}

/// Byte-order codec: converts between native and target endianness.
#[derive(Debug, Clone, Copy)]
pub struct BytesCodec {
    /// Target endianness.
    pub endian: Endian,
    /// Element size in bytes (needed for byte-swapping).
    pub element_size: usize,
}

/// Endianness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    /// Little-endian.
    Little,
    /// Big-endian.
    Big,
}

impl BytesCodec {
    /// Create a new bytes codec.
    pub fn new(endian: Endian, element_size: usize) -> Self {
        Self {
            endian,
            element_size,
        }
    }

    fn needs_swap(&self) -> bool {
        match self.endian {
            Endian::Little => cfg!(target_endian = "big"),
            Endian::Big => cfg!(target_endian = "little"),
        }
    }

    fn swap_bytes(data: &[u8], elem_size: usize) -> Vec<u8> {
        if elem_size <= 1 {
            return data.to_vec();
        }
        let mut out = data.to_vec();
        for chunk in out.chunks_exact_mut(elem_size) {
            chunk.reverse();
        }
        out
    }
}

impl Codec for BytesCodec {
    fn name(&self) -> &str {
        "bytes"
    }

    fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        if self.needs_swap() {
            Ok(Self::swap_bytes(data, self.element_size))
        } else {
            Ok(data.to_vec())
        }
    }

    fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Symmetric: swapping twice is identity
        if self.needs_swap() {
            Ok(Self::swap_bytes(data, self.element_size))
        } else {
            Ok(data.to_vec())
        }
    }
}

/// Transpose codec: reorders array elements between C-order and F-order.
///
/// For simplicity this operates on the raw byte buffer by transposing element
/// indices according to the chunk shape.
#[derive(Debug, Clone)]
pub struct TransposeCodec {
    /// Chunk shape (in elements).
    shape: Vec<usize>,
    /// Element size in bytes.
    element_size: usize,
}

impl TransposeCodec {
    /// Create a transpose codec for the given chunk shape and element size.
    pub fn new(shape: Vec<usize>, element_size: usize) -> Self {
        Self {
            shape,
            element_size,
        }
    }

    fn total_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Convert a multi-dimensional C-order index to F-order linear index.
    fn c_to_f_index(&self, c_linear: usize) -> usize {
        let ndim = self.shape.len();
        if ndim == 0 {
            return 0;
        }
        // Compute multi-dim indices from C-order
        let mut indices = vec![0usize; ndim];
        let mut rem = c_linear;
        for d in (0..ndim).rev() {
            indices[d] = rem % self.shape[d];
            rem /= self.shape[d];
        }
        // Compute F-order linear index
        let mut f_linear = 0usize;
        let mut stride = 1usize;
        for d in 0..ndim {
            f_linear += indices[d] * stride;
            stride *= self.shape[d];
        }
        f_linear
    }

    /// Convert F-order linear index to C-order linear index.
    fn f_to_c_index(&self, f_linear: usize) -> usize {
        let ndim = self.shape.len();
        if ndim == 0 {
            return 0;
        }
        // Compute multi-dim indices from F-order
        let mut indices = vec![0usize; ndim];
        let mut rem = f_linear;
        for d in 0..ndim {
            indices[d] = rem % self.shape[d];
            rem /= self.shape[d];
        }
        // Compute C-order linear index
        let mut c_linear = 0usize;
        let mut stride = 1usize;
        for d in (0..ndim).rev() {
            c_linear += indices[d] * stride;
            stride *= self.shape[d];
        }
        c_linear
    }
}

impl Codec for TransposeCodec {
    fn name(&self) -> &str {
        "transpose"
    }

    fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        let n = self.total_elements();
        let expected = n * self.element_size;
        if data.len() != expected {
            return Err(IoError::FormatError(format!(
                "Transpose encode: expected {} bytes, got {}",
                expected,
                data.len()
            )));
        }
        let mut out = vec![0u8; expected];
        for c_idx in 0..n {
            let f_idx = self.c_to_f_index(c_idx);
            let src = c_idx * self.element_size;
            let dst = f_idx * self.element_size;
            out[dst..dst + self.element_size].copy_from_slice(&data[src..src + self.element_size]);
        }
        Ok(out)
    }

    fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        let n = self.total_elements();
        let expected = n * self.element_size;
        if data.len() != expected {
            return Err(IoError::FormatError(format!(
                "Transpose decode: expected {} bytes, got {}",
                expected,
                data.len()
            )));
        }
        let mut out = vec![0u8; expected];
        for f_idx in 0..n {
            let c_idx = self.f_to_c_index(f_idx);
            let src = f_idx * self.element_size;
            let dst = c_idx * self.element_size;
            out[dst..dst + self.element_size].copy_from_slice(&data[src..src + self.element_size]);
        }
        Ok(out)
    }
}

/// Zstd compression codec using oxiarc-zstd (pure Rust, COOLJAPAN Policy).
///
/// The `level` field is stored for metadata compatibility but oxiarc-zstd
/// uses a fixed compression strategy internally.
#[derive(Debug, Clone, Copy)]
pub struct ZstdCodec {
    /// Compression level (stored for metadata, oxiarc-zstd uses default).
    pub level: i32,
}

impl ZstdCodec {
    /// Create a new Zstd codec with the given nominal compression level.
    pub fn new(level: i32) -> Self {
        Self { level }
    }
}

impl Default for ZstdCodec {
    fn default() -> Self {
        Self { level: 3 }
    }
}

impl Codec for ZstdCodec {
    fn name(&self) -> &str {
        "zstd"
    }

    fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        oxiarc_zstd::compress(data)
            .map_err(|e| IoError::CompressionError(format!("Zstd compression failed: {e}")))
    }

    fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        oxiarc_zstd::decompress(data)
            .map_err(|e| IoError::DecompressionError(format!("Zstd decompression failed: {e}")))
    }
}

/// Byte-shuffle filter: rearranges bytes for better compression.
///
/// Groups byte `k` of each element together, so e.g. for 4-byte floats
/// all MSBs are contiguous, then next bytes, etc.
#[derive(Debug, Clone, Copy)]
pub struct ShuffleCodec {
    /// Element size in bytes.
    pub element_size: usize,
}

impl ShuffleCodec {
    /// Create a shuffle codec for the given element size.
    pub fn new(element_size: usize) -> Self {
        Self { element_size }
    }
}

impl Codec for ShuffleCodec {
    fn name(&self) -> &str {
        "shuffle"
    }

    fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        if self.element_size <= 1 {
            return Ok(data.to_vec());
        }
        let n_elements = data.len() / self.element_size;
        if !data.len().is_multiple_of(self.element_size) {
            return Err(IoError::FormatError(format!(
                "Shuffle encode: data length {} not divisible by element size {}",
                data.len(),
                self.element_size
            )));
        }
        let mut out = vec![0u8; data.len()];
        for elem_idx in 0..n_elements {
            for byte_idx in 0..self.element_size {
                let src = elem_idx * self.element_size + byte_idx;
                let dst = byte_idx * n_elements + elem_idx;
                out[dst] = data[src];
            }
        }
        Ok(out)
    }

    fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        if self.element_size <= 1 {
            return Ok(data.to_vec());
        }
        let n_elements = data.len() / self.element_size;
        if !data.len().is_multiple_of(self.element_size) {
            return Err(IoError::FormatError(format!(
                "Shuffle decode: data length {} not divisible by element size {}",
                data.len(),
                self.element_size
            )));
        }
        let mut out = vec![0u8; data.len()];
        for elem_idx in 0..n_elements {
            for byte_idx in 0..self.element_size {
                let src = byte_idx * n_elements + elem_idx;
                let dst = elem_idx * self.element_size + byte_idx;
                out[dst] = data[src];
            }
        }
        Ok(out)
    }
}

/// An ordered pipeline of codecs applied to chunk data.
#[derive(Debug)]
pub struct CodecPipeline {
    codecs: Vec<Box<dyn Codec>>,
}

impl CodecPipeline {
    /// Create an empty codec pipeline.
    pub fn new() -> Self {
        Self { codecs: Vec::new() }
    }

    /// Append a codec to the pipeline.
    pub fn push<C: Codec + 'static>(&mut self, codec: C) {
        self.codecs.push(Box::new(codec));
    }

    /// Number of codecs in the pipeline.
    pub fn len(&self) -> usize {
        self.codecs.len()
    }

    /// Whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.codecs.is_empty()
    }

    /// Encode: apply codecs in forward order.
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut buf = data.to_vec();
        for codec in &self.codecs {
            buf = codec.encode(&buf)?;
        }
        Ok(buf)
    }

    /// Decode: apply codecs in reverse order.
    pub fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut buf = data.to_vec();
        for codec in self.codecs.iter().rev() {
            buf = codec.decode(&buf)?;
        }
        Ok(buf)
    }
}

impl Default for CodecPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_codec_no_swap() {
        let codec = BytesCodec::new(
            if cfg!(target_endian = "little") {
                Endian::Little
            } else {
                Endian::Big
            },
            4,
        );
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let encoded = codec.encode(&data).expect("encode");
        assert_eq!(encoded, data);
        let decoded = codec.decode(&encoded).expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_bytes_codec_swap() {
        let non_native = if cfg!(target_endian = "little") {
            Endian::Big
        } else {
            Endian::Little
        };
        let codec = BytesCodec::new(non_native, 2);
        let data = vec![0x01, 0x02, 0x03, 0x04];
        let encoded = codec.encode(&data).expect("encode");
        assert_eq!(encoded, vec![0x02, 0x01, 0x04, 0x03]);
        let decoded = codec.decode(&encoded).expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_transpose_codec_roundtrip() {
        // 2x3 array of f32
        let codec = TransposeCodec::new(vec![2, 3], 4);
        // C-order: [[1,2,3],[4,5,6]] as raw bytes
        let mut data = Vec::new();
        for val in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            data.extend_from_slice(&val.to_ne_bytes());
        }
        let encoded = codec.encode(&data).expect("encode");
        // F-order should differ from C-order for 2D
        assert_ne!(encoded, data);
        let decoded = codec.decode(&encoded).expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_zstd_codec_roundtrip() {
        let codec = ZstdCodec::new(3);
        // Use highly compressible data (repeated pattern)
        let data: Vec<u8> = vec![42u8; 4096];
        let compressed = codec.encode(&data).expect("compress");
        // Repeated data should compress well
        assert!(compressed.len() < data.len());
        let decompressed = codec.decode(&compressed).expect("decompress");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_shuffle_codec_roundtrip() {
        let codec = ShuffleCodec::new(4);
        let data: Vec<u8> = (0..32).collect();
        let encoded = codec.encode(&data).expect("encode");
        assert_ne!(encoded, data);
        let decoded = codec.decode(&encoded).expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_shuffle_single_byte_passthrough() {
        let codec = ShuffleCodec::new(1);
        let data = vec![10, 20, 30];
        let encoded = codec.encode(&data).expect("encode");
        assert_eq!(encoded, data);
    }

    #[test]
    fn test_codec_pipeline_chain() {
        let mut pipeline = CodecPipeline::new();
        pipeline.push(ShuffleCodec::new(8));
        pipeline.push(ZstdCodec::new(1));
        assert_eq!(pipeline.len(), 2);

        let data: Vec<u8> = (0..800).map(|i| (i % 256) as u8).collect();
        let encoded = pipeline.encode(&data).expect("pipeline encode");
        let decoded = pipeline.decode(&encoded).expect("pipeline decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_codec_pipeline_empty() {
        let pipeline = CodecPipeline::new();
        assert!(pipeline.is_empty());
        let data = vec![1, 2, 3];
        let encoded = pipeline.encode(&data).expect("encode");
        assert_eq!(encoded, data);
        let decoded = pipeline.decode(&data).expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_transpose_codec_1d() {
        // 1D: transpose should be identity
        let codec = TransposeCodec::new(vec![8], 4);
        let data: Vec<u8> = (0..32).collect();
        let encoded = codec.encode(&data).expect("encode");
        assert_eq!(encoded, data);
    }
}
