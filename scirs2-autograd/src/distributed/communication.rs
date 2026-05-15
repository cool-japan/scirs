//! Communication primitives for distributed training

use crate::{error::AutogradError, Float, NdArray, Result};

/// Communication operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommOp {
    /// Broadcast from one rank to all
    Broadcast,
    /// Reduce (sum) from all ranks to one
    Reduce,
    /// AllReduce - reduce and broadcast result
    AllReduce,
    /// Gather data from all ranks to one
    Gather,
    /// Scatter data from one rank to all
    Scatter,
    /// All-to-all communication
    AllToAll,
}

/// Communication handle for asynchronous operations
pub struct CommHandle {
    /// Operation type
    pub op: CommOp,
    /// Is operation complete
    completed: bool,
}

impl CommHandle {
    /// Create a new communication handle
    pub fn new(op: CommOp) -> Self {
        Self {
            op,
            completed: false,
        }
    }

    /// Wait for operation to complete
    pub fn wait(&mut self) -> Result<()> {
        // In real implementation, would wait for actual communication
        self.completed = true;
        Ok(())
    }

    /// Check if operation is complete
    pub fn is_complete(&self) -> bool {
        self.completed
    }
}

/// Compression strategy for gradient communication
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionStrategy {
    /// No compression
    None,
    /// Quantization to lower precision
    Quantize,
    /// Sparsification - send only large gradients
    Sparsify,
    /// Combination of quantization and sparsification
    Hybrid,
}

/// Compress gradients for efficient communication.
///
/// Each strategy serializes the gradient into a self-describing byte buffer that
/// can be decoded by the corresponding `decompress_gradient` function.
///
/// # Buffer layout
///
/// | Strategy | Layout |
/// |----------|--------|
/// | `None`   | `[f64 × n]` (little-endian, 8 bytes each) |
/// | `Quantize` | `4-byte header (scale: f32 LE) + [i8 × n]` |
/// | `Sparsify` | `4-byte count (u32 LE) + count × (4-byte index u32 LE + 8-byte value f64 LE)` |
/// | `Hybrid` | `4-byte count (u32 LE) + 4-byte scale (f32 LE) + count × (4-byte index u32 LE + 1-byte quantized i8)` |
pub fn compress_gradient<T: Float>(
    gradient: &NdArray<T>,
    strategy: CompressionStrategy,
) -> Result<Vec<u8>> {
    let slice = gradient
        .as_slice()
        .ok_or_else(|| AutogradError::compute_error("Gradient is not contiguous".to_string()))?;

    match strategy {
        CompressionStrategy::None => {
            // Serialize every element as a little-endian f64.
            let bytes: Vec<u8> = slice
                .iter()
                .flat_map(|&x| {
                    let f: f64 = x.to_f64().unwrap_or(0.0);
                    f.to_le_bytes()
                })
                .collect();
            Ok(bytes)
        }

        CompressionStrategy::Quantize => {
            // Linear quantization: map the range [min, max] to [-127, 127] (i8).
            // Header: scale (f32 LE, 4 bytes) + zero_point offset (f32 LE, 4 bytes).
            // Payload: one i8 per element.
            //
            // Encoding: quantized = round(x / scale) clamped to [-127, 127].
            // Decoding: x ≈ quantized * scale.
            let values: Vec<f64> = slice.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();

            // Compute scale from the absolute maximum.
            let abs_max = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

            // Avoid division by zero for all-zero gradients.
            let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };

            let mut bytes: Vec<u8> = Vec::with_capacity(4 + values.len());

            // Write scale as f32 LE (4 bytes).
            let scale_f32 = scale as f32;
            bytes.extend_from_slice(&scale_f32.to_le_bytes());

            // Quantize each value to i8.
            for v in &values {
                let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
                bytes.push(q as u8);
            }

            Ok(bytes)
        }

        CompressionStrategy::Sparsify => {
            // Top-K / threshold sparsification: keep only elements whose
            // absolute value exceeds the mean absolute value.  This retains the
            // most significant gradients without requiring a fixed K.
            //
            // Layout: 4-byte element count (u32 LE) +
            //         count × (4-byte index u32 LE + 8-byte value f64 LE)
            let values: Vec<f64> = slice.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();

            let mean_abs = if values.is_empty() {
                0.0_f64
            } else {
                values.iter().map(|v| v.abs()).sum::<f64>() / values.len() as f64
            };

            let sparse: Vec<(u32, f64)> = values
                .iter()
                .enumerate()
                .filter(|(_, &v)| v.abs() > mean_abs)
                .map(|(i, &v)| (i as u32, v))
                .collect();

            let count = sparse.len() as u32;
            let mut bytes: Vec<u8> = Vec::with_capacity(4 + sparse.len() * 12);

            // Write count as u32 LE.
            bytes.extend_from_slice(&count.to_le_bytes());

            for (idx, val) in &sparse {
                bytes.extend_from_slice(&idx.to_le_bytes());
                bytes.extend_from_slice(&val.to_le_bytes());
            }

            Ok(bytes)
        }

        CompressionStrategy::Hybrid => {
            // Hybrid: sparsify by threshold, then quantize the survivors.
            //
            // Layout: 4-byte count (u32 LE) + 4-byte scale (f32 LE) +
            //         count × (4-byte index u32 LE + 1-byte quantized i8)
            let values: Vec<f64> = slice.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();

            // Threshold = mean absolute value (same as Sparsify).
            let mean_abs = if values.is_empty() {
                0.0_f64
            } else {
                values.iter().map(|v| v.abs()).sum::<f64>() / values.len() as f64
            };

            let sparse: Vec<(u32, f64)> = values
                .iter()
                .enumerate()
                .filter(|(_, &v)| v.abs() > mean_abs)
                .map(|(i, &v)| (i as u32, v))
                .collect();

            // Compute quantization scale from survivors only.
            let abs_max = sparse.iter().map(|(_, v)| v.abs()).fold(0.0_f64, f64::max);
            let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
            let scale_f32 = scale as f32;

            let count = sparse.len() as u32;
            let mut bytes: Vec<u8> = Vec::with_capacity(4 + 4 + sparse.len() * 5);

            // Write count (u32 LE) and scale (f32 LE).
            bytes.extend_from_slice(&count.to_le_bytes());
            bytes.extend_from_slice(&scale_f32.to_le_bytes());

            // Write each surviving element: index (u32 LE) + quantized (i8).
            for (idx, val) in &sparse {
                let q = (val / scale).round().clamp(-127.0, 127.0) as i8;
                bytes.extend_from_slice(&idx.to_le_bytes());
                bytes.push(q as u8);
            }

            Ok(bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, IxDyn};

    #[test]
    fn test_comm_handle() {
        let mut handle = CommHandle::new(CommOp::AllReduce);
        assert!(!handle.is_complete());

        handle.wait().expect("Should wait");
        assert!(handle.is_complete());
    }

    #[test]
    fn test_comm_op_equality() {
        assert_eq!(CommOp::Broadcast, CommOp::Broadcast);
        assert_ne!(CommOp::Broadcast, CommOp::Reduce);
    }

    fn make_gradient(values: &[f64]) -> NdArray<f64> {
        arr1(values).into_dyn()
    }

    #[test]
    fn compress_none_round_trips() {
        let g = make_gradient(&[1.0, -2.5, 0.0, std::f64::consts::PI]);
        let bytes = compress_gradient(&g, CompressionStrategy::None)
            .expect("None compression should succeed");
        // 4 elements × 8 bytes each = 32 bytes
        assert_eq!(bytes.len(), 32);
        // Decode the first element and verify.
        let first = f64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes"));
        assert!(
            (first - 1.0).abs() < 1e-12,
            "first element mismatch: {first}"
        );
    }

    #[test]
    fn compress_quantize_produces_correct_length() {
        let g = make_gradient(&[1.0, -2.0, 0.5, -0.25]);
        let bytes = compress_gradient(&g, CompressionStrategy::Quantize)
            .expect("Quantize compression should succeed");
        // 4 bytes (scale) + 4 elements = 8 bytes
        assert_eq!(bytes.len(), 8);
    }

    #[test]
    fn compress_quantize_zero_gradient() {
        // All-zero gradient: scale defaults to 1.0; all quantized values should be 0.
        let g = make_gradient(&[0.0, 0.0, 0.0]);
        let bytes = compress_gradient(&g, CompressionStrategy::Quantize)
            .expect("Quantize of zeros should succeed");
        assert_eq!(bytes.len(), 7); // 4 (scale) + 3 (i8s)
                                    // All payload bytes should be zero.
        for &b in &bytes[4..] {
            assert_eq!(b, 0, "expected zero quantized value");
        }
    }

    #[test]
    fn compress_quantize_preserves_sign() {
        let g = make_gradient(&[10.0, -10.0]);
        let bytes =
            compress_gradient(&g, CompressionStrategy::Quantize).expect("Quantize should succeed");
        // bytes[4] is quantized(10.0) -> 127; bytes[5] is quantized(-10.0) -> -127 (stored as u8 = 129)
        let q_pos = bytes[4] as i8;
        let q_neg = bytes[5] as i8;
        assert!(q_pos > 0, "positive value should quantize positive");
        assert!(q_neg < 0, "negative value should quantize negative");
    }

    #[test]
    fn compress_sparsify_output_format() {
        // [0.0, 1.0, 0.0, 2.0, 0.0]: mean_abs = 0.6, survivors are indices 1 and 3.
        let g = make_gradient(&[0.0, 1.0, 0.0, 2.0, 0.0]);
        let bytes =
            compress_gradient(&g, CompressionStrategy::Sparsify).expect("Sparsify should succeed");

        // Read the count.
        let count = u32::from_le_bytes(bytes[0..4].try_into().expect("4 bytes")) as usize;
        // At least one element should survive (the value 2.0 exceeds mean_abs=0.6).
        assert!(
            count >= 1,
            "at least one element should survive sparsification"
        );
        // Total length = 4 + count * 12.
        assert_eq!(bytes.len(), 4 + count * 12);
    }

    #[test]
    fn compress_hybrid_output_format() {
        let g = make_gradient(&[0.0, 0.5, 0.0, 3.0]);
        let bytes =
            compress_gradient(&g, CompressionStrategy::Hybrid).expect("Hybrid should succeed");

        let count = u32::from_le_bytes(bytes[0..4].try_into().expect("4 bytes")) as usize;
        // 4 (count) + 4 (scale) + count * 5 (index u32 + quantized i8)
        assert_eq!(bytes.len(), 4 + 4 + count * 5);
    }

    #[test]
    fn compress_sparsify_all_equal_keeps_none() {
        // When all absolute values equal the mean, none exceed it.
        let g = make_gradient(&[1.0, 1.0, 1.0]);
        let bytes = compress_gradient(&g, CompressionStrategy::Sparsify)
            .expect("Sparsify of equal values should succeed");
        let count = u32::from_le_bytes(bytes[0..4].try_into().expect("4 bytes"));
        assert_eq!(count, 0, "equal magnitudes: no elements exceed mean_abs");
    }
}
