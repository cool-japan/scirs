//! Out-of-core operations for quantized tensors
//!
//! This module provides memory-efficient implementations of matrix operations
//! for quantized tensors that don't require loading the entire matrix into
//! memory at once. It's particularly useful for extremely large matrices
//! that exceed available RAM even when quantized.

use crate::error::{LinalgError, LinalgResult};
use crate::matrixfree::MatrixFreeOp;
use crate::quantization::calibration::determine_data_type;
use crate::quantization::quantized_matrixfree::QuantizedMatrixFreeOp;
use crate::quantization::solvers::quantized_conjugate_gradient;
use crate::quantization::{QuantizationMethod, QuantizationParams};
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::ndarray::{s, Array1, ArrayView1, ArrayView2};
use scirs2_core::numeric::{AsPrimitive, Float, FromPrimitive, NumAssign, One, Zero};
use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::iter::Sum;

/// Chunk size for out-of-core processing (in number of matrix rows)
const CHUNK_SIZE: usize = 1000;

/// A memory-mapped quantized matrix that operates on chunks of data
/// to avoid loading the entire matrix into memory at once
pub struct ChunkedQuantizedMatrix<F> {
    /// The shape of the matrix (rows, columns)
    shape: (usize, usize),

    /// The quantization parameters
    params: QuantizationParams,

    /// Path to the backing file storing quantized matrix data
    file_path: String,

    /// Flag indicating whether the operator is symmetric
    symmetric: bool,

    /// Flag indicating whether the operator is positive definite
    positive_definite: bool,

    /// Phantom marker for the type parameter
    _phantom: std::marker::PhantomData<F>,
}

impl<F> ChunkedQuantizedMatrix<F>
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
    /// Create a new chunked quantized matrix from an explicit matrix
    ///
    /// Quantizes the matrix and stores it to disk in chunks to support
    /// out-of-core processing.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Matrix to quantize (will be processed in chunks)
    /// * `bits` - Bit width for quantization
    /// * `method` - Quantization method
    /// * `file_path` - Path to store the quantized data
    ///
    /// # Returns
    ///
    /// A new `ChunkedQuantizedMatrix` instance
    pub fn new(
        matrix: &ArrayView2<F>,
        bits: u8,
        method: QuantizationMethod,
        file_path: &str,
    ) -> LinalgResult<Self> {
        let shape = matrix.dim();
        let rows = shape.0;
        let cols = shape.1;

        // Analyze the matrix to determine global min/max values
        let mut global_min = f32::INFINITY;
        let mut global_max = f32::NEG_INFINITY;

        for chunk_start in (0..rows).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(rows);
            let chunk = matrix.slice(s![chunk_start..chunk_end, ..]);

            // Convert to f32
            let chunk_f32: Vec<f32> = chunk.iter().map(|&x| x.as_()).collect();

            if method == QuantizationMethod::Symmetric {
                // Find maximum absolute value
                for &val in &chunk_f32 {
                    let abs_val = val.abs();
                    if abs_val > global_max {
                        global_max = abs_val;
                    }
                }
            } else {
                // Find min and max
                for &val in &chunk_f32 {
                    if val < global_min {
                        global_min = val;
                    }
                    if val > global_max {
                        global_max = val;
                    }
                }
            }
        }

        if method == QuantizationMethod::Symmetric {
            global_min = -global_max;
        }

        // Calculate global quantization parameters
        let (scale, zero_point) = if method == QuantizationMethod::Symmetric {
            let scale = global_max / ((1 << (bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (global_max - global_min) / ((1 << bits) - 1) as f32;
            let zero_point = (-global_min / scale).round() as i32;
            (scale, zero_point)
        };

        // Create quantization parameters
        let params = QuantizationParams {
            bits,
            scale,
            zero_point,
            min_val: global_min,
            max_val: global_max,
            method,
            data_type: determine_data_type(bits),
            channel_scales: None,
            channel_zero_points: None,
        };

        // Write the quantized matrix to disk in chunks
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(file_path)
            .map_err(|e| LinalgError::ComputationError(format!("Failed to open file: {e}")))?;

        let mut writer = BufWriter::new(file);

        // First write the matrix shape and quantization parameters
        writer
            .write_all(&(rows as u64).to_le_bytes())
            .map_err(|e| LinalgError::ComputationError(format!("Failed to write rows: {e}")))?;
        writer
            .write_all(&(cols as u64).to_le_bytes())
            .map_err(|e| LinalgError::ComputationError(format!("Failed to write columns: {e}")))?;
        writer
            .write_all(&scale.to_le_bytes())
            .map_err(|e| LinalgError::ComputationError(format!("Failed to write scale: {e}")))?;
        writer.write_all(&zero_point.to_le_bytes()).map_err(|e| {
            LinalgError::ComputationError(format!("Failed to write zeropoint: {e}"))
        })?;

        // Then write the quantized matrix data in chunks
        for chunk_start in (0..rows).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(rows);
            let chunk = matrix.slice(s![chunk_start..chunk_end, ..]);

            // Quantize the chunk
            let quantized_data = quantize_chunk(&chunk, &params)?;

            // Write the quantized chunk - convert i8 to u8 for writing
            // We need to use a bit representation conversion (reinterpret the bytes)
            // rather than a numeric cast to preserve the bit pattern
            let u8_data: Vec<u8> = quantized_data.iter().map(|&x| x.cast_unsigned()).collect();
            writer.write_all(&u8_data).map_err(|e| {
                LinalgError::ComputationError(format!("Failed to write chunk: {e}"))
            })?;
        }

        writer
            .flush()
            .map_err(|e| LinalgError::ComputationError(format!("Failed to flush buffer: {e}")))?;

        // Determine if the matrix is symmetric
        let symmetric =
            method == QuantizationMethod::Symmetric && rows == cols && ismatrix_symmetric(matrix);

        Ok(ChunkedQuantizedMatrix {
            shape: (rows, cols),
            params,
            file_path: file_path.to_string(),
            symmetric,
            positive_definite: false,
            _phantom: std::marker::PhantomData,
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

    /// Apply the matrix to a vector
    ///
    /// This method loads the matrix from disk in chunks and applies each chunk
    /// to the input vector, accumulating the results to avoid loading the entire
    /// matrix into memory at once.
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector
    ///
    /// # Returns
    ///
    /// Result vector
    pub fn apply(&self, x: &ArrayView1<F>) -> LinalgResult<Array1<F>> {
        if x.len() != self.shape.1 {
            return Err(LinalgError::ShapeError(format!(
                "Input vector has wrong length: expected {}, got {}",
                self.shape.1,
                x.len()
            )));
        }

        // Convert input to f32
        let x_f32: Vec<f32> = x.iter().map(|&val| val.as_()).collect();

        // Open the file for reading
        let file = File::open(&self.file_path)
            .map_err(|e| LinalgError::ComputationError(format!("Failed to open file: {e}")))?;

        let mut reader = BufReader::new(file);

        // Skip header information (2 u64 values for dimensions, f32 for scale, i32 for zero_point)
        reader
            .seek(SeekFrom::Start(24))
            .map_err(|e| LinalgError::ComputationError(format!("Failed to seek: {e}")))?;

        // Prepare result vector
        let rows = self.shape.0;
        let cols = self.shape.1;
        let mut result = Array1::zeros(rows);

        // Process the matrix in chunks
        for chunk_start in (0..rows).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(rows);
            let chunk_rows = chunk_end - chunk_start;

            // Read the quantized chunk
            let chunksize = chunk_rows * cols;
            let mut u8_data = vec![0u8; chunksize];

            reader
                .read_exact(&mut u8_data)
                .map_err(|e| LinalgError::ComputationError(format!("Failed to read chunk: {e}")))?;

            // Convert back to i8 for processing using a bit pattern conversion
            // rather than a numeric cast to preserve the sign bit
            let quantized_data: Vec<i8> = u8_data.iter().map(|&x| x.cast_signed()).collect();

            // Apply the chunk to the input vector
            for i in 0..chunk_rows {
                let mut sum = 0.0f32;

                for j in 0..cols {
                    let q_val = quantized_data[i * cols + j] as f32;
                    let dequantized = if self.params.method == QuantizationMethod::Symmetric {
                        q_val * self.params.scale
                    } else {
                        (q_val - self.params.zero_point as f32) * self.params.scale
                    };

                    sum += dequantized * x_f32[j];
                }

                result[chunk_start + i] = F::from_f32(sum).unwrap_or(F::zero());
            }
        }

        Ok(result)
    }

    /// Convert to a regular QuantizedMatrixFreeOp
    ///
    /// This loads the entire matrix into memory, so it should only be used
    /// for matrices that can fit in memory.
    ///
    /// # Returns
    ///
    /// A QuantizedMatrixFreeOp instance
    pub fn tomatrix_free_op(&self) -> LinalgResult<QuantizedMatrixFreeOp<F>> {
        let rows = self.shape.0;
        let cols = self.shape.1;

        // Clone the file path to avoid borrow issues
        let file_path = self.file_path.clone();
        let params = self.params.clone();
        let symmetric = self.symmetric;
        let positive_definite = self.positive_definite;

        // Create a new QuantizedMatrixFreeOp with a closure that reads from the file
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

            // Open the file for reading
            let file = File::open(&file_path)
                .map_err(|e| LinalgError::ComputationError(format!("Failed to open file: {e}")))?;

            let mut reader = BufReader::new(file);

            // Skip header information
            reader
                .seek(SeekFrom::Start(24))
                .map_err(|e| LinalgError::ComputationError(format!("Failed to seek: {e}")))?;

            // Prepare result vector
            let mut result = Array1::zeros(rows);

            // Process the matrix in chunks
            for chunk_start in (0..rows).step_by(CHUNK_SIZE) {
                let chunk_end = (chunk_start + CHUNK_SIZE).min(rows);
                let chunk_rows = chunk_end - chunk_start;

                // Read the quantized chunk
                let chunksize = chunk_rows * cols;
                let mut u8_data = vec![0u8; chunksize];

                reader.read_exact(&mut u8_data).map_err(|e| {
                    LinalgError::ComputationError(format!("Failed to read chunk: {e}"))
                })?;

                // Convert back to i8 for processing using a bit pattern conversion
                // rather than a numeric cast to preserve the sign bit
                let quantized_data: Vec<i8> = u8_data.iter().map(|&x| x.cast_signed()).collect();

                // Apply the chunk to the input vector
                for i in 0..chunk_rows {
                    let mut sum = 0.0f32;

                    for j in 0..cols {
                        let q_val = quantized_data[i * cols + j] as f32;
                        let dequantized = if params.method == QuantizationMethod::Symmetric {
                            q_val * params.scale
                        } else {
                            (q_val - params.zero_point as f32) * params.scale
                        };

                        sum += dequantized * x_f32[j];
                    }

                    result[chunk_start + i] = F::from_f32(sum).unwrap_or(F::zero());
                }
            }

            Ok(result)
        };

        // Create a QuantizedMatrixFreeOp with the closure
        let op =
            QuantizedMatrixFreeOp::new(rows, cols, self.params.bits, self.params.method, op_fn)?;

        // Add symmetric and positive definite flags if applicable
        let op = if symmetric {
            let op = op.symmetric();
            if positive_definite {
                op.positive_definite()
            } else {
                op
            }
        } else {
            op
        };

        Ok(op)
    }

    /// Solve a linear system using out-of-core conjugate gradient
    ///
    /// # Arguments
    ///
    /// * `b` - Right-hand side vector
    /// * `max_iter` - Maximum number of iterations
    /// * `tol` - Convergence tolerance
    /// * `adaptive_precision` - Whether to adaptively adjust precision during iterations
    ///
    /// # Returns
    ///
    /// Solution vector
    pub fn solve_conjugate_gradient(
        &self,
        b: &Array1<F>,
        max_iter: usize,
        tol: F,
        adaptive_precision: bool,
    ) -> LinalgResult<Array1<F>> {
        // For small matrices, we can use the regular solver
        let rows = self.shape.0;
        let cols = self.shape.1;

        if rows * cols <= CHUNK_SIZE * CHUNK_SIZE {
            // Convert to a regular MatrixFreeOp and use the standard solver
            let op = self.tomatrix_free_op()?;
            return quantized_conjugate_gradient(&op, b, max_iter, tol, adaptive_precision);
        }

        // For large matrices, we implement a streaming version of conjugate gradient
        if !self.symmetric {
            return Err(LinalgError::ValueError(
                "Out-of-core conjugate gradient requires a symmetric matrix".to_string(),
            ));
        }

        if rows != cols {
            return Err(LinalgError::ShapeError(format!(
                "Expected square matrix, got shape {rows}x{cols}"
            )));
        }

        if b.len() != rows {
            return Err(LinalgError::ShapeError(format!(
                "Shape mismatch: matrix shape {}x{}, vector shape {}",
                rows,
                cols,
                b.len()
            )));
        }

        // Initialize solution with zeros
        let mut x = Array1::zeros(rows);

        // Initial residual r = b - Ax
        let ax = self.apply(&x.view())?;
        let mut r = b.clone();
        r -= &ax;

        // Initial search direction p = r
        let mut p = r.clone();

        // Initial residual norm squared
        let mut rsold = r.dot(&r);

        // If b is zero or initial guess is very close to solution
        let b_norm = (b.dot(b)).sqrt();
        if b_norm < F::epsilon() || rsold.sqrt() < tol * b_norm {
            return Ok(x);
        }

        // Tracking variables for adaptive _precision
        let mut successive_slow_progress = 0;
        let mut previous_residual = rsold;

        for _iter in 0..max_iter {
            // Compute A*p (this is the out-of-core part)
            let ap = self.apply(&p.view())?;

            // Compute step size alpha
            let pap = p.dot(&ap);

            // Safety check
            if pap.abs() < F::epsilon() {
                break;
            }

            let alpha = rsold / pap;

            // Update solution x = x + alpha*p
            x = &x + &(&p * alpha);

            // Update residual r = r - alpha*A*p
            r = &r - &(&ap * alpha);

            // Compute new residual norm squared
            let mut rsnew = r.dot(&r);

            // Check convergence
            if rsnew.sqrt() < tol * b_norm {
                break;
            }

            // Adaptive _precision strategy
            if adaptive_precision {
                // Check if we're making good progress
                let ratio = rsnew / previous_residual;

                // If progress is slow for multiple iterations, we might need to reset
                if ratio > F::from(0.9).unwrap() {
                    successive_slow_progress += 1;
                } else {
                    successive_slow_progress = 0;
                }

                // If we've had multiple iterations with slow progress, perform a residual refresh
                if successive_slow_progress >= 5 {
                    // Re-compute residual directly from r = b - Ax to avoid accumulated error
                    let ax = self.apply(&x.view())?;
                    r = b.clone();
                    r -= &ax;

                    // Reset progress counter
                    successive_slow_progress = 0;

                    // Recompute rsnew
                    rsnew = r.dot(&r);

                    // Check convergence again after refresh
                    if rsnew.sqrt() < tol * b_norm {
                        break;
                    }
                }

                previous_residual = rsnew;
            }

            // Compute direction update beta
            let beta = rsnew / rsold;

            // Update search direction p = r + beta*p
            p = &r + &(&p * beta);

            // Update old residual norm
            rsold = rsnew;
        }

        Ok(x)
    }

    /// Create a new chunked quantized matrix from an existing file
    ///
    /// This loads the metadata from an existing quantized matrix file
    /// but doesn't load the entire matrix into memory.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the quantized matrix file
    ///
    /// # Returns
    ///
    /// A new `ChunkedQuantizedMatrix` instance
    pub fn from_file(filepath: &str) -> LinalgResult<Self> {
        // Open the file for reading
        let file = File::open(filepath)
            .map_err(|e| LinalgError::ComputationError(format!("Failed to open file: {e}")))?;

        let mut reader = BufReader::new(file);

        // Read header information
        let mut rows_bytes = [0u8; 8];
        let mut cols_bytes = [0u8; 8];
        let mut scale_bytes = [0u8; 4];
        let mut zero_point_bytes = [0u8; 4];

        reader
            .read_exact(&mut rows_bytes)
            .map_err(|e| LinalgError::ComputationError(format!("Failed to read rows: {e}")))?;
        reader
            .read_exact(&mut cols_bytes)
            .map_err(|e| LinalgError::ComputationError(format!("Failed to read columns: {e}")))?;
        reader
            .read_exact(&mut scale_bytes)
            .map_err(|e| LinalgError::ComputationError(format!("Failed to read scale: {e}")))?;
        reader
            .read_exact(&mut zero_point_bytes)
            .map_err(|e| LinalgError::ComputationError(format!("Failed to read zeropoint: {e}")))?;

        let rows = u64::from_le_bytes(rows_bytes) as usize;
        let cols = u64::from_le_bytes(cols_bytes) as usize;
        let scale = f32::from_le_bytes(scale_bytes);
        let zero_point = i32::from_le_bytes(zero_point_bytes);

        // Create QuantizationParams
        // We need to infer min_val and max_val from scale and zero_point
        // We assume symmetric quantization if zero_point is 0
        let (method, min_val, max_val) = if zero_point == 0 {
            // Symmetric quantization
            let max_val = scale * 127.0;
            (QuantizationMethod::Symmetric, -max_val, max_val)
        } else {
            // Affine quantization
            let min_val = -zero_point as f32 * scale;
            let max_val = (255 - zero_point) as f32 * scale;
            (QuantizationMethod::Affine, min_val, max_val)
        };

        let params = QuantizationParams {
            bits: 8, // We assume 8-bit quantization
            scale,
            zero_point,
            min_val,
            max_val,
            method,
            data_type: determine_data_type(8),
            channel_scales: None,
            channel_zero_points: None,
        };

        // We can't determine symmetry or positive definiteness from just the file
        // Without loading the whole matrix, so we'll default to false
        Ok(ChunkedQuantizedMatrix {
            shape: (rows, cols),
            params,
            file_path: filepath.to_string(),
            symmetric: false,
            positive_definite: false,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<F> MatrixFreeOp<F> for ChunkedQuantizedMatrix<F>
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
        + 'static
        + FromPrimitive
        + AsPrimitive<f32>,
    f32: AsPrimitive<F>,
{
    fn apply(&self, x: &ArrayView1<F>) -> LinalgResult<Array1<F>> {
        // Call the struct's apply method rather than recursively calling this method
        ChunkedQuantizedMatrix::apply(self, x)
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

/// Quantize a chunk of a matrix
#[allow(dead_code)]
fn quantize_chunk<F>(chunk: &ArrayView2<F>, params: &QuantizationParams) -> LinalgResult<Vec<i8>>
where
    F: Float + AsPrimitive<f32>,
{
    let rows = chunk.dim().0;
    let cols = chunk.dim().1;
    let mut quantized = vec![0i8; rows * cols];

    for i in 0..rows {
        for j in 0..cols {
            let val = chunk[[i, j]].as_();
            let q_val = if params.method == QuantizationMethod::Symmetric {
                // For symmetric quantization, clamp to [-127, 127] for 8-bit
                // or appropriate range for other bit widths
                let min_val = -(1 << (params.bits - 1)) + 1;
                let max_val = (1 << (params.bits - 1)) - 1;
                ((val / params.scale).round()).clamp(min_val as f32, max_val as f32) as i8
            } else {
                // For affine quantization, clamp to [0, 2^bits - 1]
                ((val / params.scale + params.zero_point as f32).round())
                    .clamp(0.0, ((1 << params.bits) - 1) as f32) as i8
            };

            quantized[i * cols + j] = q_val;
        }
    }

    Ok(quantized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;
    use std::env::temp_dir;
    use std::path::PathBuf;

    // Helper to get a temporary file path
    fn get_temp_file_path(name: &str) -> PathBuf {
        let mut path = temp_dir();
        path.push(format!("quantizedmatrix_{}.bin", name));
        path
    }

    #[test]
    fn test_chunked_quantizedmatrix() {
        // Create a test matrix
        let matrix = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        // Get a temporary file path
        let file_path = get_temp_file_path("test1");

        // Create a chunked quantized matrix
        let chunked = ChunkedQuantizedMatrix::new(
            &matrix.view(),
            8,
            QuantizationMethod::Symmetric,
            file_path.to_str().unwrap(),
        )
        .unwrap();

        // Apply to a vector
        let x = array![1.0f32, 2.0, 3.0];
        let y = chunked.apply(&x.view()).unwrap();

        // Compute expected result with regular matrix multiplication
        let expected = matrix.dot(&x);

        // Check that the results are close
        assert_eq!(y.len(), expected.len());
        for i in 0..y.len() {
            assert_relative_eq!(y[i], expected[i], epsilon = 1.0);
        }

        // Clean up
        std::fs::remove_file(file_path).unwrap_or_default();
    }

    #[test]
    fn test_chunked_quantizedmatrix_from_file() {
        // Create a test matrix
        let matrix = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        // Get a temporary file path
        let file_path = get_temp_file_path("test2");

        // Create and save a chunked quantized matrix
        let _chunked = ChunkedQuantizedMatrix::new(
            &matrix.view(),
            8,
            QuantizationMethod::Symmetric,
            file_path.to_str().unwrap(),
        )
        .unwrap();

        // Load the chunked quantized matrix from file
        let loaded = ChunkedQuantizedMatrix::from_file(file_path.to_str().unwrap()).unwrap();

        // Apply to a vector
        let x = array![1.0f32, 2.0, 3.0];
        let y = loaded.apply(&x.view()).unwrap();

        // Compute expected result with regular matrix multiplication
        let expected = matrix.dot(&x);

        // Check that the results are close
        assert_eq!(y.len(), expected.len());
        for i in 0..y.len() {
            assert_relative_eq!(y[i], expected[i], epsilon = 1.0);
        }

        // Clean up
        std::fs::remove_file(file_path).unwrap_or_default();
    }

    #[test]
    fn test_solve_conjugate_gradient() {
        // Create a symmetric positive definite matrix
        let matrix = array![[4.0f32, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 5.0]];

        // Get a temporary file path
        let file_path = get_temp_file_path("test3");

        // Create a chunked quantized matrix
        let chunked = ChunkedQuantizedMatrix::new(
            &matrix.view(),
            8,
            QuantizationMethod::Symmetric,
            file_path.to_str().unwrap(),
        )
        .unwrap()
        .symmetric()
        .positive_definite();

        // Create a right-hand side vector
        let b = array![1.0f32, 2.0, 3.0];

        // Solve using out-of-core conjugate gradient
        let x = chunked
            .solve_conjugate_gradient(&b, 100, 1e-6, false)
            .unwrap();

        // Compute residual
        let ax = matrix.dot(&x);
        let residual = &ax - &b;
        let residual_norm = (residual.dot(&residual)).sqrt();

        // Check that the residual is small
        assert!(residual_norm < 0.1);

        // Clean up
        std::fs::remove_file(file_path).unwrap_or_default();
    }
}
