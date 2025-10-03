//! Matrix-free operations for quantized tensors
//!
//! This module provides matrix-free operations for quantized tensors, enabling
//! efficient memory usage and computation for large models. It combines the benefits
//! of quantization (reduced memory footprint) with matrix-free operations (no need to
//! materialize large matrices).

use crate::error::{LinalgError, LinalgResult};
use crate::matrixfree::{LinearOperator, MatrixFreeOp};
use crate::quantization::calibration::determine_data_type;
use crate::quantization::{QuantizationMethod, QuantizationParams};
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
use scirs2_core::numeric::{AsPrimitive, Float, FromPrimitive, NumAssign, One, Zero};
use std::fmt::Debug;
use std::iter::Sum;
use std::sync::Arc;

/// Type alias for the matrix-vector product function
pub type MatVecFn<F> = Arc<dyn Fn(&ArrayView1<F>) -> LinalgResult<Array1<F>> + Send + Sync>;

/// A matrix-free operator that represents a quantized matrix
pub struct QuantizedMatrixFreeOp<F>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug + 'static,
{
    /// The shape of the matrix (rows, columns)
    shape: (usize, usize),

    /// The quantization parameters
    params: QuantizationParams,

    /// Matrix application function that doesn't require storing the full matrix
    op_fn: MatVecFn<F>,

    /// Flag indicating whether the operator is symmetric
    symmetric: bool,

    /// Flag indicating whether the operator is positive definite
    positive_definite: bool,
}

impl<F> QuantizedMatrixFreeOp<F>
where
    F: Float
        + NumAssign
        + Zero
        + Sum
        + One
        + ScalarOperand
        + Send
        + Sync
        + Debug
        + FromPrimitive
        + AsPrimitive<f32>
        + 'static,
    f32: AsPrimitive<F>,
{
    /// Create a new quantized matrix-free operator from a function
    ///
    /// This allows direct specification of how the operator acts on vectors
    /// without materializing the quantized matrix.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows in the matrix
    /// * `cols` - Number of columns in the matrix
    /// * `bits` - Bit width for quantization
    /// * `method` - Quantization method
    /// * `op_fn` - Function that implements the matrix-vector product in the quantized domain
    ///
    /// # Returns
    ///
    /// A new `QuantizedMatrixFreeOp` instance
    pub fn new<O>(
        rows: usize,
        cols: usize,
        bits: u8,
        method: QuantizationMethod,
        op_fn: O,
    ) -> LinalgResult<Self>
    where
        O: Fn(&ArrayView1<F>) -> LinalgResult<Array1<F>> + Send + Sync + 'static,
    {
        // Create default quantization parameters - these will be refined when data is observed
        let min_val: f32 = 0.0;
        let max_val: f32 = 1.0;
        let (scale, zero_point) = if method == QuantizationMethod::Symmetric {
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
            let zero_point = (-min_val / scale).round() as i32;
            (scale, zero_point)
        };

        // Create the quantization parameters
        let params = QuantizationParams {
            bits,
            scale,
            zero_point,
            min_val,
            max_val,
            method,
            data_type: determine_data_type(bits),
            channel_scales: None,
            channel_zero_points: None,
        };

        Ok(QuantizedMatrixFreeOp {
            shape: (rows, cols),
            params,
            op_fn: Arc::new(op_fn),
            symmetric: false,
            positive_definite: false,
        })
    }

    /// Create a quantized matrix-free operator from an explicit matrix
    ///
    /// This quantizes the matrix and creates a matrix-free operator that
    /// applies the quantized matrix without materializing it for each operation.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Matrix to quantize
    /// * `bits` - Bit width for quantization
    /// * `method` - Quantization method
    ///
    /// # Returns
    ///
    /// A new `QuantizedMatrixFreeOp` instance
    pub fn frommatrix(
        matrix: &ArrayView2<F>,
        bits: u8,
        method: QuantizationMethod,
    ) -> LinalgResult<Self> {
        // Convert matrix to f32 for quantization
        let matrix_f32: Array1<f32> = matrix.iter().map(|&x| x.as_()).collect();

        // Get min/max values for quantization parameters
        let (min_val, max_val) = if method == QuantizationMethod::Symmetric {
            let max_abs = matrix_f32.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
            (-max_abs, max_abs)
        } else {
            let min_val = matrix_f32.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
            let max_val = matrix_f32
                .iter()
                .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
            (min_val, max_val)
        };

        // Calculate quantization parameters
        let (scale, zero_point) = if method == QuantizationMethod::Symmetric {
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
            let zero_point = (-min_val / scale).round() as i32;
            (scale, zero_point)
        };

        // Create the quantization parameters
        let params = QuantizationParams {
            bits,
            scale,
            zero_point,
            min_val,
            max_val,
            method,
            data_type: determine_data_type(bits),
            channel_scales: None,
            channel_zero_points: None,
        };

        // Copy the dimensions before we move the matrix
        let shape = matrix.dim();

        // Quantize matrix ahead of time
        let quantized_data: Vec<i8> = matrix_f32
            .iter()
            .map(|&val| {
                if method == QuantizationMethod::Symmetric {
                    (val / scale)
                        .round()
                        .clamp(-(1 << (bits - 1)) as f32, ((1 << (bits - 1)) - 1) as f32)
                        as i8
                } else {
                    ((val / scale) + zero_point as f32)
                        .round()
                        .clamp(0.0, ((1 << bits) - 1) as f32) as i8
                }
            })
            .collect();

        // Create the matrix-vector product function
        let op_fn = move |x: &ArrayView1<F>| -> LinalgResult<Array1<F>> {
            if x.len() != shape.1 {
                return Err(LinalgError::ShapeError(format!(
                    "Input vector has wrong length: expected {}, got {}",
                    shape.1,
                    x.len()
                )));
            }

            // Convert input to f32
            let x_f32: Vec<f32> = x.iter().map(|&val| val.as_()).collect();

            // Apply the quantized matrix-vector product
            let mut result = Array1::zeros(shape.0);

            // Implement matrix-vector product manually
            for i in 0..shape.0 {
                let mut sum = 0.0f32;
                for j in 0..shape.1 {
                    let q_val = quantized_data[i * shape.1 + j] as f32;
                    let dequantized = if method == QuantizationMethod::Symmetric {
                        q_val * scale
                    } else {
                        (q_val - zero_point as f32) * scale
                    };
                    sum += dequantized * x_f32[j];
                }
                result[i] = F::from_f32(sum).unwrap_or(F::zero());
            }

            Ok(result)
        };

        // Determine if the operator is symmetric
        let symmetric = method == QuantizationMethod::Symmetric
            && shape.0 == shape.1
            && ismatrix_symmetric(matrix);

        Ok(QuantizedMatrixFreeOp {
            shape,
            params,
            op_fn: Arc::new(op_fn),
            symmetric,
            positive_definite: false, // We can't reliably detect this from the matrix
        })
    }

    /// Mark the operator as symmetric
    ///
    /// # Returns
    ///
    /// Self with the symmetric flag set to true
    pub fn symmetric(mut self) -> Self {
        if self.shape.0 != self.shape.1 {
            panic!("Only square operators can be symmetric");
        }
        self.symmetric = true;
        self
    }

    /// Mark the operator as positive definite
    ///
    /// # Returns
    ///
    /// Self with the positive_definite flag set to true
    pub fn positive_definite(mut self) -> Self {
        if !self.symmetric {
            panic!("Only symmetric operators can be positive definite");
        }
        self.positive_definite = true;
        self
    }

    /// Get the quantization parameters
    pub fn params(&self) -> &QuantizationParams {
        &self.params
    }

    /// Create a memory-efficient operator for block-diagonal matrices
    ///
    /// This is particularly useful for large models with block structure,
    /// as it avoids materializing the full matrix.
    ///
    /// # Arguments
    ///
    /// * `blocks` - A vector of smaller matrices to place on the diagonal
    /// * `bits` - Bit width for quantization
    /// * `method` - Quantization method
    ///
    /// # Returns
    ///
    /// A new `QuantizedMatrixFreeOp` instance
    pub fn block_diagonal(
        blocks: Vec<ArrayView2<F>>,
        bits: u8,
        method: QuantizationMethod,
    ) -> LinalgResult<Self> {
        if blocks.is_empty() {
            return Err(LinalgError::ValueError("Empty blocks vector".to_string()));
        }

        // Calculate total dimensions
        let total_rows = blocks.iter().map(|b| b.dim().0).sum();
        let total_cols = blocks.iter().map(|b| b.dim().1).sum();

        // Quantize each block separately
        let mut block_data = Vec::new();

        for block in &blocks {
            // Convert to f32
            let block_f32: Vec<f32> = block.iter().map(|&x| x.as_()).collect();

            // Get min/max for this block
            let (min_val, max_val) = if method == QuantizationMethod::Symmetric {
                let max_abs = block_f32.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
                (-max_abs, max_abs)
            } else {
                let min_val = block_f32.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
                let max_val = block_f32
                    .iter()
                    .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                (min_val, max_val)
            };

            // Calculate quantization parameters for this block
            let (scale, zero_point) = if method == QuantizationMethod::Symmetric {
                let abs_max = max_val.abs().max(min_val.abs());
                let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
                (scale, 0)
            } else {
                let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
                let zero_point = (-min_val / scale).round() as i32;
                (scale, zero_point)
            };

            // Quantize the block
            let quantized: Vec<i8> = block_f32
                .iter()
                .map(|&val| {
                    if method == QuantizationMethod::Symmetric {
                        (val / scale)
                            .round()
                            .clamp(-(1 << (bits - 1)) as f32, ((1 << (bits - 1)) - 1) as f32)
                            as i8
                    } else {
                        ((val / scale) + zero_point as f32)
                            .round()
                            .clamp(0.0, ((1 << bits) - 1) as f32) as i8
                    }
                })
                .collect();

            // Store the dimensions, quantized data, and quantization parameters
            block_data.push((block.dim(), quantized, scale, zero_point));
        }

        // Create a function that applies the block-diagonal matrix
        let block_data_clone = block_data.clone();
        let blocks_method = method;

        let op_fn = move |x: &ArrayView1<F>| -> LinalgResult<Array1<F>> {
            if x.len() != total_cols {
                return Err(LinalgError::ShapeError(format!(
                    "Input vector has wrong length: expected {}, got {}",
                    total_cols,
                    x.len()
                )));
            }

            // Convert input to f32
            let x_f32: Vec<f32> = x.iter().map(|&val| val.as_()).collect();

            // Prepare result vector
            let mut result = Array1::zeros(total_rows);

            // Process each block
            let mut row_offset = 0;
            let mut col_offset = 0;

            for (shape, quantized, scale, zero_point) in block_data_clone.iter() {
                let block_rows = shape.0;
                let block_cols = shape.1;

                // Apply this block to the corresponding section of the input vector
                for i in 0..block_rows {
                    let mut sum = 0.0f32;
                    for j in 0..block_cols {
                        let x_idx = col_offset + j;
                        if x_idx < x_f32.len() {
                            let q_val = quantized[i * block_cols + j] as f32;
                            let dequantized = if blocks_method == QuantizationMethod::Symmetric {
                                q_val * (*scale)
                            } else {
                                (q_val - (*zero_point) as f32) * (*scale)
                            };
                            sum += dequantized * x_f32[x_idx];
                        }
                    }

                    let result_idx = row_offset + i;
                    if result_idx < result.len() {
                        result[result_idx] = F::from_f32(sum).unwrap_or(F::zero());
                    }
                }

                row_offset += block_rows;
                col_offset += block_cols;
            }

            Ok(result)
        };

        // Calculate global min/max values for our parameters
        let global_min_val = block_data
            .iter()
            .map(|(_, _, scale, zero_point)| {
                if method == QuantizationMethod::Symmetric {
                    -(*scale) * ((1 << (bits - 1)) - 1) as f32
                } else {
                    -(*zero_point) as f32 * (*scale)
                }
            })
            .fold(f32::INFINITY, |a, b| a.min(b));

        let global_max_val = block_data
            .iter()
            .map(|(_, _, scale_, _)| {
                if method == QuantizationMethod::Symmetric {
                    (*scale_) * ((1 << (bits - 1)) - 1) as f32
                } else {
                    (*scale_) * ((1 << bits) - 1) as f32
                }
            })
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));

        // Create the quantization parameters
        let (scale, zero_point) = if method == QuantizationMethod::Symmetric {
            let abs_max = global_max_val.abs().max(global_min_val.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (global_max_val - global_min_val) / ((1 << bits) - 1) as f32;
            let zero_point = (-global_min_val / scale).round() as i32;
            (scale, zero_point)
        };

        let params = QuantizationParams {
            bits,
            scale,
            zero_point,
            min_val: global_min_val,
            max_val: global_max_val,
            method,
            data_type: determine_data_type(bits),
            channel_scales: None,
            channel_zero_points: None,
        };

        // Check if all blocks are square and the operator could be symmetric
        let all_square = blocks.iter().all(|b| b.dim().0 == b.dim().1);
        let symmetric = method == QuantizationMethod::Symmetric && all_square;

        Ok(QuantizedMatrixFreeOp {
            shape: (total_rows, total_cols),
            params,
            op_fn: Arc::new(op_fn),
            symmetric,
            positive_definite: false,
        })
    }

    /// Create a memory-efficient operator for structured sparse matrices
    ///
    /// This is particularly useful for large sparse models, as it stores
    /// only the non-zero elements and their indices.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows in the matrix
    /// * `cols` - Number of columns in the matrix
    /// * `indices` - Pairs of (row, column) indices for non-zero elements
    /// * `values` - Values at the corresponding indices
    /// * `bits` - Bit width for quantization
    /// * `method` - Quantization method
    ///
    /// # Returns
    ///
    /// A new `QuantizedMatrixFreeOp` instance
    pub fn sparse(
        rows: usize,
        cols: usize,
        indices: Vec<(usize, usize)>,
        values: &ArrayView1<F>,
        bits: u8,
        method: QuantizationMethod,
    ) -> LinalgResult<Self> {
        if indices.len() != values.len() {
            return Err(LinalgError::ShapeError(
                "Indices and values must have the same length".to_string(),
            ));
        }

        // Validate indices
        for &(i, j) in &indices {
            if i >= rows || j >= cols {
                return Err(LinalgError::ShapeError(format!(
                    "Index ({i}, {j}) out of bounds for matrix of shape ({rows}, {cols})"
                )));
            }
        }

        // Convert values to f32 for quantization
        let values_f32: Vec<f32> = values.iter().map(|&val| val.as_()).collect();

        // Get min/max values for quantization parameters
        let (min_val, max_val) = if method == QuantizationMethod::Symmetric {
            let max_abs = values_f32.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
            (-max_abs, max_abs)
        } else {
            let min_val = values_f32.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
            let max_val = values_f32
                .iter()
                .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
            (min_val, max_val)
        };

        // Calculate quantization parameters
        let (scale, zero_point) = if method == QuantizationMethod::Symmetric {
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
            let zero_point = (-min_val / scale).round() as i32;
            (scale, zero_point)
        };

        // Quantize the values
        let quantized_data: Vec<i8> = values_f32
            .iter()
            .map(|&val| {
                if method == QuantizationMethod::Symmetric {
                    (val / scale)
                        .round()
                        .clamp(-(1 << (bits - 1)) as f32, ((1 << (bits - 1)) - 1) as f32)
                        as i8
                } else {
                    ((val / scale) + zero_point as f32)
                        .round()
                        .clamp(0.0, ((1 << bits) - 1) as f32) as i8
                }
            })
            .collect();

        // Create a copy of indices for the closure
        let indices_owned = indices.clone();
        let sparse_method = method;

        // Create a function that applies the sparse matrix
        let op_fn = move |x: &ArrayView1<F>| -> LinalgResult<Array1<F>> {
            if x.len() != cols {
                return Err(LinalgError::ShapeError(format!(
                    "Input vector has wrong length: expected {}, got {}",
                    cols,
                    x.len()
                )));
            }

            // Convert input to f32
            let x_f32: Vec<f32> = x.iter().map(|&val| val.as_()).collect();

            // Prepare result vector
            let mut result = Array1::zeros(rows);

            // Apply the sparse matrix
            for (idx, &(i, j)) in indices_owned.iter().enumerate() {
                if idx < quantized_data.len() {
                    let q_val = quantized_data[idx] as f32;
                    let dequantized = if sparse_method == QuantizationMethod::Symmetric {
                        q_val * scale
                    } else {
                        (q_val - zero_point as f32) * scale
                    };

                    result[i] += F::from_f32(dequantized * x_f32[j]).unwrap_or(F::zero());
                }
            }

            Ok(result)
        };

        // Create quantization parameters
        let params = QuantizationParams {
            bits,
            scale,
            zero_point,
            min_val,
            max_val,
            method,
            data_type: determine_data_type(bits),
            channel_scales: None,
            channel_zero_points: None,
        };

        // Check if the matrix could be symmetric
        let symmetric = rows == cols
            && method == QuantizationMethod::Symmetric
            && indices
                .iter()
                .all(|&(i, j)| i == j || indices.contains(&(j, i)));

        Ok(QuantizedMatrixFreeOp {
            shape: (rows, cols),
            params,
            op_fn: Arc::new(op_fn),
            symmetric,
            positive_definite: false,
        })
    }

    /// Create a memory-efficient operator for a banded matrix
    ///
    /// This is particularly useful for banded matrices like tridiagonal
    /// or pentadiagonal matrices, as it stores only the bands.
    ///
    /// # Arguments
    ///
    /// * `n` - Size of the square matrix
    /// * `bands` - Vector of (offset, band_values) pairs, where offset is the diagonal offset
    ///   (0 for main diagonal, 1 for first super-diagonal, -1 for first sub-diagonal)
    /// * `bits` - Bit width for quantization
    /// * `method` - Quantization method
    ///
    /// # Returns
    ///
    /// A new `QuantizedMatrixFreeOp` instance
    pub fn banded(
        n: usize,
        bands: Vec<(isize, ArrayView1<F>)>,
        bits: u8,
        method: QuantizationMethod,
    ) -> LinalgResult<Self> {
        // Validate bands
        for &(offset, ref band) in &bands {
            let expected_len = n - offset.unsigned_abs();
            if band.len() != expected_len {
                return Err(LinalgError::ShapeError(format!(
                    "Band with offset {} should have length {}, got {}",
                    offset,
                    expected_len,
                    band.len()
                )));
            }
        }

        // Quantize each band
        let mut band_data = Vec::new();

        for (offset, band) in &bands {
            // Convert to f32
            let band_f32: Vec<f32> = band.iter().map(|&x| x.as_()).collect();

            // Get min/max for this band
            let (min_val, max_val) = if method == QuantizationMethod::Symmetric {
                let max_abs = band_f32.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
                (-max_abs, max_abs)
            } else {
                let min_val = band_f32.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
                let max_val = band_f32
                    .iter()
                    .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                (min_val, max_val)
            };

            // Calculate quantization parameters for this band
            let (scale, zero_point) = if method == QuantizationMethod::Symmetric {
                let abs_max = max_val.abs().max(min_val.abs());
                let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
                (scale, 0)
            } else {
                let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
                let zero_point = (-min_val / scale).round() as i32;
                (scale, zero_point)
            };

            // Quantize the band
            let quantized: Vec<i8> = band_f32
                .iter()
                .map(|&val| {
                    if method == QuantizationMethod::Symmetric {
                        (val / scale)
                            .round()
                            .clamp(-(1 << (bits - 1)) as f32, ((1 << (bits - 1)) - 1) as f32)
                            as i8
                    } else {
                        ((val / scale) + zero_point as f32)
                            .round()
                            .clamp(0.0, ((1 << bits) - 1) as f32) as i8
                    }
                })
                .collect();

            // Store the offset, quantized data, and quantization parameters
            band_data.push((*offset, quantized, scale, zero_point));
        }

        // Create a function that applies the banded matrix
        let band_data_clone = band_data.clone();
        let banded_method = method;

        let op_fn = move |x: &ArrayView1<F>| -> LinalgResult<Array1<F>> {
            if x.len() != n {
                return Err(LinalgError::ShapeError(format!(
                    "Expected vector of length {}, got {}",
                    n,
                    x.len()
                )));
            }

            // Convert input to f32
            let x_f32: Vec<f32> = x.iter().map(|&val| val.as_()).collect();

            // Prepare result vector
            let mut result = Array1::zeros(n);

            // Apply each band
            for (offset, quantized, scale, zero_point) in &band_data_clone {
                let band_len = quantized.len();

                if *offset >= 0 {
                    // Super-diagonal or main diagonal
                    let offset_usize = *offset as usize;
                    for i in 0..band_len {
                        if i < n && (i + offset_usize) < n {
                            let q_val = quantized[i] as f32;
                            let dequantized = if banded_method == QuantizationMethod::Symmetric {
                                q_val * (*scale)
                            } else {
                                (q_val - (*zero_point) as f32) * (*scale)
                            };

                            result[i] += F::from_f32(dequantized * x_f32[i + offset_usize])
                                .unwrap_or(F::zero());
                        }
                    }
                } else {
                    // Sub-diagonal
                    let offset_usize = (-*offset) as usize;
                    for i in 0..band_len {
                        if (i + offset_usize) < n && i < n {
                            let q_val = quantized[i] as f32;
                            let dequantized = if banded_method == QuantizationMethod::Symmetric {
                                q_val * (*scale)
                            } else {
                                (q_val - (*zero_point) as f32) * (*scale)
                            };

                            result[i + offset_usize] +=
                                F::from_f32(dequantized * x_f32[i]).unwrap_or(F::zero());
                        }
                    }
                }
            }

            Ok(result)
        };

        // Calculate global min/max values for our parameters
        let global_min_val = band_data
            .iter()
            .map(|(_, _, scale, zero_point)| {
                if method == QuantizationMethod::Symmetric {
                    -(*scale) * ((1 << (bits - 1)) - 1) as f32
                } else {
                    -(*zero_point) as f32 * (*scale)
                }
            })
            .fold(f32::INFINITY, |a, b| a.min(b));

        let global_max_val = band_data
            .iter()
            .map(|(_, _, scale_, _)| {
                if method == QuantizationMethod::Symmetric {
                    (*scale_) * ((1 << (bits - 1)) - 1) as f32
                } else {
                    (*scale_) * ((1 << bits) - 1) as f32
                }
            })
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));

        // Create the quantization parameters
        let params = QuantizationParams {
            bits,
            scale: 1.0, // These are placeholder values since we store per-band parameters
            zero_point: 0,
            min_val: global_min_val,
            max_val: global_max_val,
            method,
            data_type: determine_data_type(bits),
            channel_scales: None,
            channel_zero_points: None,
        };

        // Check if the matrix could be symmetric
        let symmetric = method == QuantizationMethod::Symmetric
            && band_data.iter().all(|(offset, _, _, _)| {
                // For a symmetric banded matrix, if there's a band at offset k,
                // there must also be a band at offset -k
                *offset == 0 || band_data.iter().any(|(o, _, _, _)| *o == -*offset)
            });

        Ok(QuantizedMatrixFreeOp {
            shape: (n, n),
            params,
            op_fn: Arc::new(op_fn),
            symmetric,
            positive_definite: false,
        })
    }
}

impl<F> MatrixFreeOp<F> for QuantizedMatrixFreeOp<F>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug + 'static,
{
    fn apply(&self, x: &ArrayView1<F>) -> LinalgResult<Array1<F>> {
        if x.len() != self.shape.1 {
            return Err(LinalgError::ShapeError(format!(
                "Input vector has wrong length: expected {}, got {}",
                self.shape.1,
                x.len()
            )));
        }
        (self.op_fn)(x)
    }

    fn nrows(&self) -> usize {
        self.shape.0
    }

    fn ncols(&self) -> usize {
        self.shape.1
    }

    fn is_symmetric(&self) -> bool {
        self.symmetric
    }

    fn is_positive_definite(&self) -> bool {
        self.positive_definite
    }
}

/// Convert a QuantizedMatrixFreeOp to a generic LinearOperator
///
/// This is useful when you want to use the quantized operator with
/// algorithms that expect a LinearOperator.
///
/// # Arguments
///
/// * `op` - The quantized matrix-free operator
///
/// # Returns
///
/// A LinearOperator that wraps the quantized operator
#[allow(dead_code)]
pub fn quantized_to_linear_operator<F>(op: &QuantizedMatrixFreeOp<F>) -> LinearOperator<F>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug + 'static,
{
    let rows = op.nrows();
    let cols = op.ncols();
    let is_symmetric = op.is_symmetric();
    let is_positive_definite = op.is_positive_definite();

    // We need to clone op.op_fn, but can't directly due to the trait bound
    // So we create a new closure that delegates to the original
    let op_clone = op.clone();

    let linear_op = if rows == cols {
        LinearOperator::new(rows, move |x: &ArrayView1<F>| match op_clone.apply(x) {
            Ok(result) => result,
            Err(_) => Array1::zeros(rows),
        })
    } else {
        LinearOperator::new_rectangular(rows, cols, move |x: &ArrayView1<F>| {
            match op_clone.apply(x) {
                Ok(result) => result,
                Err(_) => Array1::zeros(rows),
            }
        })
    };

    // Add flags if applicable
    if is_symmetric {
        let linear_op = linear_op.symmetric();
        if is_positive_definite {
            linear_op.positive_definite()
        } else {
            linear_op
        }
    } else {
        linear_op
    }
}

// Add clone support to QuantizedMatrixFreeOp
impl<F> Clone for QuantizedMatrixFreeOp<F>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug + 'static,
{
    fn clone(&self) -> Self {
        QuantizedMatrixFreeOp {
            shape: self.shape,
            params: self.params.clone(),
            op_fn: Arc::clone(&self.op_fn),
            symmetric: self.symmetric,
            positive_definite: self.positive_definite,
        }
    }
}

/// Check if a matrix is symmetric
#[allow(dead_code)]
fn ismatrix_symmetric<F>(matrix: &ArrayView2<F>) -> bool
where
    F: Float + PartialEq,
{
    let (rows, cols) = matrix.dim();
    if rows != cols {
        return false;
    }

    for i in 0..rows {
        for j in i + 1..cols {
            if matrix[[i, j]] != matrix[[j, i]] {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_quantizedmatrix_free_op_frommatrix() {
        // Create a test matrix
        let matrix = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        // Create a quantized matrix-free operator
        let op =
            QuantizedMatrixFreeOp::frommatrix(&matrix.view(), 8, QuantizationMethod::Symmetric)
                .unwrap();

        // Apply to a vector
        let x = array![1.0f32, 2.0, 3.0];
        let y = op.apply(&x.view()).unwrap();

        // Compute expected result with regular matrix multiplication
        let expected = matrix.dot(&x);

        // Check that the results are close
        assert_eq!(y.len(), expected.len());
        for i in 0..y.len() {
            assert_relative_eq!(y[i], expected[i], epsilon = 1.0);
        }
    }

    #[test]
    fn test_quantizedmatrix_free_op_block_diagonal() {
        // Create test matrices for the blocks
        let block1 = array![[1.0f32, 2.0], [3.0, 4.0]];

        let block2 = array![[5.0f32]];

        // Create a block-diagonal operator
        let op = QuantizedMatrixFreeOp::block_diagonal(
            vec![block1.view(), block2.view()],
            8,
            QuantizationMethod::Symmetric,
        )
        .unwrap();

        // Apply to a vector
        let x = array![1.0f32, 2.0, 3.0];
        let y = op.apply(&x.view()).unwrap();

        // Expected result would be:
        // [ block1[0,0]*x[0] + block1[0,1]*x[1], block1[1,0]*x[0] + block1[1,1]*x[1], block2[0,0]*x[2] ]
        // = [ 1.0*1.0 + 2.0*2.0, 3.0*1.0 + 4.0*2.0, 5.0*3.0 ]
        // = [ 5.0, 11.0, 15.0 ]
        let expected = array![5.0f32, 11.0, 15.0];

        assert_eq!(y.len(), expected.len());
        for i in 0..y.len() {
            assert_relative_eq!(y[i], expected[i], epsilon = 1.0);
        }
    }

    #[test]
    fn test_quantizedmatrix_free_op_sparse() {
        // Create a sparse matrix:
        // [ 1.0 0.0 2.0 ]
        // [ 0.0 3.0 0.0 ]
        // [ 4.0 0.0 5.0 ]
        let indices = vec![(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)];
        let values = array![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let op = QuantizedMatrixFreeOp::sparse(
            3,
            3,
            indices,
            &values.view(),
            8,
            QuantizationMethod::Symmetric,
        )
        .unwrap();

        // Apply to a vector
        let x = array![1.0f32, 2.0, 3.0];
        let y = op.apply(&x.view()).unwrap();

        // Expected result:
        // [ 1.0*1.0 + 0.0*2.0 + 2.0*3.0, 0.0*1.0 + 3.0*2.0 + 0.0*3.0, 4.0*1.0 + 0.0*2.0 + 5.0*3.0 ]
        // = [ 1.0 + 6.0, 6.0, 4.0 + 15.0 ]
        // = [ 7.0, 6.0, 19.0 ]
        let expected = array![7.0f32, 6.0, 19.0];

        assert_eq!(y.len(), expected.len());
        for i in 0..y.len() {
            assert_relative_eq!(y[i], expected[i], epsilon = 1.0);
        }
    }

    #[test]
    fn test_quantizedmatrix_free_op_banded() {
        // Create a tridiagonal matrix:
        // [ 2.0 1.0 0.0 ]
        // [ 1.0 3.0 1.0 ]
        // [ 0.0 1.0 4.0 ]

        let main_diag = array![2.0f32, 3.0, 4.0];
        let super_diag = array![1.0f32, 1.0];
        let sub_diag = array![1.0f32, 1.0];

        let bands = vec![
            (0, main_diag.view()),
            (1, super_diag.view()),
            (-1, sub_diag.view()),
        ];

        let op = QuantizedMatrixFreeOp::banded(3, bands, 8, QuantizationMethod::Symmetric).unwrap();

        // Apply to a vector
        let x = array![1.0f32, 2.0, 3.0];
        let y = op.apply(&x.view()).unwrap();

        // Expected result:
        // [ 2.0*1.0 + 1.0*2.0, 1.0*1.0 + 3.0*2.0 + 1.0*3.0, 1.0*2.0 + 4.0*3.0 ]
        // = [ 2.0 + 2.0, 1.0 + 6.0 + 3.0, 2.0 + 12.0 ]
        // = [ 4.0, 10.0, 14.0 ]
        let expected = array![4.0f32, 10.0, 14.0];

        assert_eq!(y.len(), expected.len());
        for i in 0..y.len() {
            assert_relative_eq!(y[i], expected[i], epsilon = 1.0);
        }
    }

    #[test]
    fn test_quantized_to_linear_operator() {
        // Create a test matrix
        let matrix = array![[1.0f32, 2.0], [2.0, 3.0]];

        // Create a symmetric quantized matrix-free operator
        let quantized_op =
            QuantizedMatrixFreeOp::frommatrix(&matrix.view(), 8, QuantizationMethod::Symmetric)
                .unwrap()
                .symmetric()
                .positive_definite();

        // Convert to a LinearOperator
        let linear_op = quantized_to_linear_operator(&quantized_op);

        // Check that properties are preserved
        assert_eq!(linear_op.nrows(), quantized_op.nrows());
        assert_eq!(linear_op.ncols(), quantized_op.ncols());
        assert_eq!(linear_op.is_symmetric(), quantized_op.is_symmetric());
        assert_eq!(
            linear_op.is_positive_definite(),
            quantized_op.is_positive_definite()
        );

        // Apply to a vector and check results
        let x = array![1.0f32, 2.0];
        let y_quantized = quantized_op.apply(&x.view()).unwrap();
        let y_linear = linear_op.apply(&x.view()).unwrap();

        assert_eq!(y_quantized.len(), y_linear.len());
        for i in 0..y_quantized.len() {
            assert_relative_eq!(y_quantized[i], y_linear[i], epsilon = 1e-6);
        }
    }
}
