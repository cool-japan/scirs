//! Quantization types and enums
//!
//! This module contains the core types used throughout the quantization system,
//! including quantization methods, parameters, and data types.

/// Supported methods of quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMethod {
    /// Uniform quantization maps the input range to uniform discrete levels
    /// with equal spacing between consecutive levels
    Uniform,

    /// Symmetric quantization is centered around zero and has equal positive and
    /// negative ranges, making it suitable for weight matrices
    Symmetric,

    /// Affine quantization uses the formula q = scale * (x - zero_point)
    /// allowing better representation of asymmetric distributions
    Affine,

    /// Power-of-two quantization uses powers of 2 for the scale factor,
    /// enabling efficient implementation with bitshifts
    PowerOfTwo,

    /// Int4 quantization uses 4-bit signed integers, packing two values into each byte
    /// for memory efficiency. This is useful for model compression in ML applications.
    Int4,

    /// UInt4 quantization uses 4-bit unsigned integers, packing two values into each byte.
    /// This provides a positive-only range with maximum memory efficiency.
    UInt4,

    /// Float16 quantization uses IEEE 754 16-bit half-precision floating point format.
    /// It provides a good balance between precision and memory efficiency for ML models.
    Float16,

    /// BFloat16 quantization uses the "brain floating point" 16-bit format,
    /// which has the same exponent size as f32 but fewer mantissa bits.
    /// This is especially well-suited for deep learning applications.
    BFloat16,

    /// Per-channel symmetric quantization applies different symmetric quantization
    /// parameters to each channel (column), improving accuracy for matrices with
    /// varying distributions across channels.
    PerChannelSymmetric,

    /// Per-channel affine quantization applies different affine quantization
    /// parameters to each channel (column), allowing for better representation of
    /// asymmetric distributions that vary by channel.
    PerChannelAffine,
}

/// Parameters for the quantization process
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// The number of bits used for quantization
    pub bits: u8,

    /// The scale factor used to convert between quantized and float values
    /// For per-channel quantization, this is the default scale for debugging
    pub scale: f32,

    /// The zero point used for asymmetric quantization (for affine quantization)
    /// For per-channel quantization, this is the default zero point for debugging
    pub zero_point: i32,

    /// The minimum value of the original data
    /// For per-channel quantization, this is across all channels
    pub min_val: f32,

    /// The maximum value of the original data
    /// For per-channel quantization, this is across all channels
    pub max_val: f32,

    /// The quantization method used
    pub method: QuantizationMethod,

    /// The data type used for storage
    pub data_type: QuantizedDataType,

    /// Per-channel scale factors (only used for per-channel quantization)
    pub channel_scales: Option<Vec<f32>>,

    /// Per-channel zero points (only used for per-channel affine quantization)
    pub channel_zero_points: Option<Vec<i32>>,
}

/// The storage type used for quantized data
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizedDataType {
    /// 8-bit signed integers
    Int8,
    /// 4-bit signed integers (packed into i8 array)
    Int4,
    /// 4-bit unsigned integers (packed into i8 array)
    UInt4,
    /// 16-bit IEEE 754 half-precision floating point (f16)
    Float16,
    /// 16-bit Brain floating point (bf16)
    BFloat16,
}
