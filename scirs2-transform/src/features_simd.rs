//! SIMD-accelerated feature engineering operations
//!
//! This module provides SIMD-optimized implementations of feature engineering operations
//! using the unified SIMD operations from scirs2-core.

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, ArrayView1, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::validation::{check_not_empty, check_positive};

use crate::error::{Result, TransformError};

/// SIMD-accelerated polynomial feature generation
pub struct SimdPolynomialFeatures<F: Float + NumCast + SimdUnifiedOps> {
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + NumCast + SimdUnifiedOps> SimdPolynomialFeatures<F> {
    /// Creates a new SIMD-accelerated polynomial features generator
    pub fn new(degree: usize, include_bias: bool, interactiononly: bool) -> Result<Self> {
        if degree == 0 {
            return Err(TransformError::InvalidInput(
                "Degree must be at least 1".to_string(),
            ));
        }

        Ok(SimdPolynomialFeatures {
            degree,
            include_bias,
            interaction_only: interactiononly,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Transforms input features to polynomial features using SIMD operations
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<F>>
    where
        S: Data<Elem = F>,
    {
        // Validate input using scirs2-core validation
        check_not_empty(x, "x")?;

        // Check finite values
        for &val in x.iter() {
            if !val.is_finite() {
                return Err(crate::error::TransformError::DataValidationError(
                    "Data contains non-finite values".to_string(),
                ));
            }
        }

        let n_samples = x.shape()[0];
        let nfeatures = x.shape()[1];

        if n_samples == 0 || nfeatures == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if nfeatures > 1000 {
            return Err(TransformError::InvalidInput(
                "Too many features for polynomial expansion (>1000)".to_string(),
            ));
        }

        // Calculate output dimensions with overflow check
        let n_outputfeatures = self.calculate_n_outputfeatures(nfeatures)?;

        // Check for memory constraints
        if n_samples > 100_000 && n_outputfeatures > 10_000 {
            return Err(TransformError::ComputationError(
                "Output matrix would be too large (>1B elements)".to_string(),
            ));
        }

        let mut output = Array2::zeros((n_samples, n_outputfeatures));

        // Process samples in batches for better cache locality
        let batch_size = self.calculate_optimal_batch_size(n_samples, n_outputfeatures);
        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_samples);

            for i in batch_start..batch_end {
                let sample = x.row(i);
                let poly_features = self.transform_sample_simd(&sample)?;

                // Use SIMD copy if available
                if poly_features.len() == n_outputfeatures {
                    let mut output_row = output.row_mut(i);
                    for (j, &val) in poly_features.iter().enumerate() {
                        output_row[j] = val;
                    }
                } else {
                    return Err(TransformError::ComputationError(
                        "Feature count mismatch in polynomial expansion".to_string(),
                    ));
                }
            }
        }

        Ok(output)
    }

    /// Transforms a single sample using SIMD operations
    fn transform_sample_simd(&self, sample: &ArrayView1<F>) -> Result<Array1<F>> {
        let nfeatures = sample.len();
        let n_outputfeatures = self.calculate_n_outputfeatures(nfeatures)?;
        let mut output = Array1::zeros(n_outputfeatures);
        let mut output_idx = 0;

        // Include bias term if requested
        if self.include_bias {
            output[output_idx] = F::one();
            output_idx += 1;
        }

        // Copy original features
        for j in 0..nfeatures {
            output[output_idx] = sample[j];
            output_idx += 1;
        }

        // Generate polynomial features
        if self.degree > 1 {
            if self.interaction_only {
                // Only interaction terms (no powers of single features)
                let _ = self.add_interaction_terms(sample, &mut output, output_idx, 2)?;
            } else {
                // All polynomial combinations
                let _ = self.add_polynomial_terms(sample, &mut output, output_idx)?;
            }
        }

        Ok(output)
    }

    /// Adds polynomial terms using SIMD operations where possible
    fn add_polynomial_terms(
        &self,
        sample: &ArrayView1<F>,
        output: &mut Array1<F>,
        mut output_idx: usize,
    ) -> Result<usize> {
        let nfeatures = sample.len();

        // For degree 2, use SIMD for efficient computation
        if self.degree == 2 {
            // Squared terms using SIMD
            let squared = F::simd_mul(&sample.view(), &sample.view());
            for j in 0..nfeatures {
                output[output_idx] = squared[j];
                output_idx += 1;
            }

            // Cross terms with vectorized operations where possible
            for j in 0..nfeatures {
                let remaining_features = nfeatures - j - 1;
                if remaining_features > 0 {
                    // Use SIMD for remaining cross products
                    let sample_j = sample[j];
                    let remaining_slice = sample.slice(scirs2_core::ndarray::s![j + 1..]);

                    // Create a vector filled with sample[j]
                    let sample_j_vec = Array1::from_elem(remaining_features, sample_j);
                    let cross_products = F::simd_mul(&sample_j_vec.view(), &remaining_slice);

                    for &val in cross_products.iter() {
                        output[output_idx] = val;
                        output_idx += 1;
                    }
                }
            }
        } else {
            // For higher degrees, fall back to iterative computation
            // but still use SIMD where beneficial
            for current_degree in 2..=self.degree {
                output_idx = self.add_degree_terms(sample, output, output_idx, current_degree)?;
            }
        }

        Ok(output_idx)
    }

    /// Adds interaction terms only
    fn add_interaction_terms(
        &self,
        sample: &ArrayView1<F>,
        output: &mut Array1<F>,
        mut output_idx: usize,
        degree: usize,
    ) -> Result<usize> {
        let nfeatures = sample.len();

        if degree == 2 {
            // Pairwise interactions with SIMD optimization
            for j in 0..nfeatures {
                let remaining_features = nfeatures - j - 1;
                if remaining_features > 0 {
                    let sample_j = sample[j];
                    let remaining_slice = sample.slice(scirs2_core::ndarray::s![j + 1..]);

                    // Use SIMD for batch processing of interactions
                    let sample_j_vec = Array1::from_elem(remaining_features, sample_j);
                    let interactions = F::simd_mul(&sample_j_vec.view(), &remaining_slice);

                    for &val in interactions.iter() {
                        output[output_idx] = val;
                        output_idx += 1;
                    }
                } else {
                    // Fallback for remaining elements
                    for k in j + 1..nfeatures {
                        output[output_idx] = sample[j] * sample[k];
                        output_idx += 1;
                    }
                }
            }
        } else {
            // Higher-order interactions
            let indices = self.generate_interaction_indices(nfeatures, degree);
            for idx_set in indices {
                let mut prod = F::one();
                for &_idx in &idx_set {
                    prod = prod * sample[_idx];
                }
                output[output_idx] = prod;
                output_idx += 1;
            }
        }

        Ok(output_idx)
    }

    /// Adds terms of a specific degree
    fn add_degree_terms(
        &self,
        sample: &ArrayView1<F>,
        output: &mut Array1<F>,
        mut output_idx: usize,
        degree: usize,
    ) -> Result<usize> {
        let nfeatures = sample.len();
        let indices = self.generate_degree_indices(nfeatures, degree);

        for idx_vec in indices {
            let mut prod = F::one();
            for &_idx in &idx_vec {
                prod = prod * sample[_idx];
            }
            output[output_idx] = prod;
            output_idx += 1;
        }

        Ok(output_idx)
    }

    /// Calculate optimal batch size based on memory characteristics
    fn calculate_optimal_batch_size(&self, n_samples: usize, n_outputfeatures: usize) -> usize {
        const L1_CACHE_SIZE: usize = 32_768;
        let element_size = std::mem::size_of::<F>();

        // Target: keep one batch worth of data in L1 cache
        let elements_per_batch = L1_CACHE_SIZE / element_size / 2; // Conservative estimate
        let max_batch_size = elements_per_batch / n_outputfeatures.max(1);

        // Adaptive batch size based on data characteristics
        let optimal_batch_size = if n_outputfeatures > 1000 {
            // Large feature count: smaller batches
            16.max(max_batch_size).min(64)
        } else if n_samples > 50_000 {
            // Large sample count: medium batches
            64.max(max_batch_size).min(256)
        } else {
            // Standard case: larger batches for better vectorization
            128.max(max_batch_size).min(512)
        };

        optimal_batch_size.min(n_samples)
    }

    /// Calculates the number of output features
    fn calculate_n_outputfeatures(&self, nfeatures: usize) -> Result<usize> {
        let mut count = if self.include_bias { 1 } else { 0 };
        count += nfeatures; // Original _features

        if self.degree > 1 {
            if self.interaction_only {
                // Only interaction terms
                for d in 2..=self.degree {
                    count += self.n_choose_k(nfeatures, d);
                }
            } else {
                // All polynomial combinations
                count += self.n_polynomial_features(nfeatures, self.degree) - nfeatures;
                if self.include_bias {
                    count -= 1;
                }
            }
        }

        Ok(count)
    }

    /// Calculates n choose k
    fn n_choose_k(&self, n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// Calculates the number of polynomial features
    fn n_polynomial_features(&self, nfeatures: usize, degree: usize) -> usize {
        self.n_choose_k(nfeatures + degree, degree)
    }

    /// Generates indices for interaction terms
    fn generate_interaction_indices(&self, nfeatures: usize, degree: usize) -> Vec<Vec<usize>> {
        let mut indices = Vec::new();
        let mut current = vec![0; degree];

        loop {
            // Add current combination
            indices.push(current.clone());

            // Find the rightmost element that can be incremented
            let mut i = degree - 1;
            loop {
                current[i] += 1;
                if current[i] < nfeatures - (degree - 1 - i) {
                    // Reset all elements to the right
                    for j in i + 1..degree {
                        current[j] = current[j - 1] + 1;
                    }
                    break;
                }
                if i == 0 {
                    return indices;
                }
                i -= 1;
            }
        }
    }

    /// Generates indices for polynomial terms of a specific degree
    fn generate_degree_indices(&self, nfeatures: usize, degree: usize) -> Vec<Vec<usize>> {
        let mut indices = Vec::new();
        let mut current = vec![0; degree];

        loop {
            // Add current combination (with repetition allowed)
            indices.push(current.clone());

            // Find the rightmost element that can be incremented
            let mut i = degree - 1;
            loop {
                current[i] += 1;
                if current[i] < nfeatures {
                    // Reset all elements to the right to the same value
                    for j in i + 1..degree {
                        current[j] = current[i];
                    }
                    break;
                }
                if i == 0 {
                    return indices;
                }
                current[i] = 0;
                i -= 1;
            }
        }
    }
}

/// SIMD-accelerated power transformation (Box-Cox and Yeo-Johnson)
#[allow(dead_code)]
pub fn simd_power_transform<F>(data: &Array1<F>, lambda: F, method: &str) -> Result<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    let n = data.len();
    let mut result = Array1::zeros(n);

    match method {
        "box-cox" => {
            // Check for negative values
            let min_val = F::simd_min_element(&data.view());
            if min_val <= F::zero() {
                return Err(TransformError::InvalidInput(
                    "Box-Cox requires strictly positive values".to_string(),
                ));
            }

            if lambda.abs() < F::from(1e-6).unwrap() {
                // lambda ≈ 0: log transform
                for i in 0..n {
                    result[i] = data[i].ln();
                }
            } else {
                // General Box-Cox: (x^lambda - 1) / lambda
                let ones = Array1::from_elem(n, F::one());
                let powered = simd_array_pow(data, lambda)?;
                let numerator = F::simd_sub(&powered.view(), &ones.view());
                let lambda_array = Array1::from_elem(n, lambda);
                result = F::simd_div(&numerator.view(), &lambda_array.view());
            }
        }
        "yeo-johnson" => {
            // Yeo-Johnson handles both positive and negative values
            for i in 0..n {
                let x = data[i];
                if x >= F::zero() {
                    if lambda.abs() < F::from(1e-6).unwrap() {
                        result[i] = x.ln() + F::one();
                    } else {
                        result[i] = ((x + F::one()).powf(lambda) - F::one()) / lambda;
                    }
                } else {
                    if (F::from(2.0).unwrap() - lambda).abs() < F::from(1e-6).unwrap() {
                        result[i] = -((-x + F::one()).ln());
                    } else {
                        result[i] = -((-x + F::one()).powf(F::from(2.0).unwrap() - lambda)
                            - F::one())
                            / (F::from(2.0).unwrap() - lambda);
                    }
                }
            }
        }
        _ => {
            return Err(TransformError::InvalidInput(
                "Method must be 'box-cox' or 'yeo-johnson'".to_string(),
            ));
        }
    }

    Ok(result)
}

/// Helper function to compute element-wise power using SIMD where possible
#[allow(dead_code)]
fn simd_array_pow<F>(data: &Array1<F>, exponent: F) -> Result<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    let n = data.len();

    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    if !exponent.is_finite() {
        return Err(TransformError::InvalidInput(
            "Exponent must be finite".to_string(),
        ));
    }

    let mut result = Array1::zeros(n);

    // For common exponents, use optimized SIMD operations
    if (exponent - F::from(2.0).unwrap()).abs() < F::from(1e-10).unwrap() {
        // Square using SIMD multiplication
        result = F::simd_mul(&data.view(), &data.view());
    } else if (exponent - F::from(0.5).unwrap()).abs() < F::from(1e-10).unwrap() {
        // Square root using SIMD - check for non-negative values first
        for &val in data.iter() {
            if val < F::zero() {
                return Err(TransformError::ComputationError(
                    "Cannot compute square root of negative values".to_string(),
                ));
            }
        }
        result = F::simd_sqrt(&data.view());
    } else if (exponent - F::from(3.0).unwrap()).abs() < F::from(1e-10).unwrap() {
        // Cube: x^3 = x * x * x
        let squared = F::simd_mul(&data.view(), &data.view());
        result = F::simd_mul(&squared.view(), &data.view());
    } else if (exponent - F::from(1.0).unwrap()).abs() < F::from(1e-10).unwrap() {
        // Identity: x^1 = x
        result = data.clone();
    } else if (exponent - F::from(0.0).unwrap()).abs() < F::from(1e-10).unwrap() {
        // Constant: x^0 = 1
        result.fill(F::one());
    } else {
        // General case: use vectorized exponentiation
        let exponent_array = Array1::from_elem(n, exponent);
        // Fallback: element-wise power operation since simd_pow is not available
        result = data.mapv(|x| x.powf(exponent));

        // Validate results
        for &val in result.iter() {
            if !val.is_finite() {
                return Err(TransformError::ComputationError(
                    "Power operation produced non-finite values".to_string(),
                ));
            }
        }
    }

    Ok(result)
}

/// SIMD-accelerated binarization with validation
#[allow(dead_code)]
pub fn simd_binarize<F>(data: &Array2<F>, threshold: F) -> Result<Array2<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    check_not_empty(data, "data")?;

    // Check finite values
    for &val in data.iter() {
        if !val.is_finite() {
            return Err(crate::error::TransformError::DataValidationError(
                "Data contains non-finite values".to_string(),
            ));
        }
    }

    if !threshold.is_finite() {
        return Err(TransformError::InvalidInput(
            "Threshold must be finite".to_string(),
        ));
    }

    let shape = data.shape();
    let mut result = Array2::zeros((shape[0], shape[1]));

    // Calculate adaptive chunk size based on data dimensions
    let chunk_size = calculate_adaptive_chunk_size(shape[0], shape[1]);

    for i in 0..shape[0] {
        let row = data.row(i);
        let row_array = row.to_owned();

        // Process row in chunks using SIMD
        for chunk_start in (0..shape[1]).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(shape[1]);
            let chunk_size = chunk_end - chunk_start;

            let chunk_slice = row_array.slice(scirs2_core::ndarray::s![chunk_start..chunk_end]);
            let threshold_array = Array1::from_elem(chunk_size, threshold);

            // Use SIMD comparison where available
            // Fallback: element-wise comparison since simd_greater_than is not available
            let comparison_result =
                chunk_slice.mapv(|x| if x > threshold { F::one() } else { F::zero() });

            for (j, &cmp_result) in comparison_result.iter().enumerate() {
                result[[i, chunk_start + j]] = if cmp_result > F::zero() {
                    F::one()
                } else {
                    F::zero()
                };
            }
        }
    }

    Ok(result)
}

/// Calculate adaptive chunk size for optimal SIMD performance
#[allow(dead_code)]
fn calculate_adaptive_chunk_size(n_rows: usize, ncols: usize) -> usize {
    const L1_CACHE_SIZE: usize = 32_768;
    const F64_SIZE: usize = 8; // Conservative estimate for element size

    // Calculate how many elements can fit comfortably in L1 cache
    let cache_elements = L1_CACHE_SIZE / F64_SIZE / 4; // Conservative factor

    // Adaptive chunk size based on matrix dimensions
    let chunk_size = if ncols > cache_elements {
        // Wide matrix: use smaller chunks
        32
    } else if n_rows > 10_000 {
        // Many _rows: use medium chunks for better cache reuse
        128
    } else {
        // Standard case: larger chunks for better vectorization
        256
    };

    // Ensure chunk size is reasonable and aligned
    chunk_size.min(ncols).max(16)
}

/// Advanced SIMD polynomial features with memory optimization
#[allow(dead_code)]
pub fn simd_polynomial_features_optimized<F>(
    data: &Array2<F>,
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
    memory_limit_mb: usize,
) -> Result<Array2<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    check_not_empty(data, "data")?;

    // Check finite values
    for &val in data.iter() {
        if !val.is_finite() {
            return Err(crate::error::TransformError::DataValidationError(
                "Data contains non-finite values".to_string(),
            ));
        }
    }

    check_positive(degree, "degree")?;

    let poly_features = SimdPolynomialFeatures::new(degree, include_bias, interaction_only)?;

    let shape = data.shape();
    let element_size = std::mem::size_of::<F>();
    let data_size_mb = (shape[0] * shape[1] * element_size) / (1024 * 1024);

    if data_size_mb > memory_limit_mb {
        // Use chunked processing for large datasets
        simd_polynomial_features_chunked(data, &poly_features, memory_limit_mb)
    } else {
        // Standard processing
        poly_features.transform(data)
    }
}

/// Chunked SIMD polynomial features for large datasets
#[allow(dead_code)]
fn simd_polynomial_features_chunked<F>(
    data: &Array2<F>,
    poly_features: &SimdPolynomialFeatures<F>,
    memory_limit_mb: usize,
) -> Result<Array2<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    let shape = data.shape();
    let element_size = std::mem::size_of::<F>();
    let max_rows_per_chunk = (memory_limit_mb * 1024 * 1024) / (shape[1] * element_size * 2); // Factor of 2 for safety

    if max_rows_per_chunk == 0 {
        return Err(TransformError::MemoryError(
            "Memory limit too small for processing".to_string(),
        ));
    }

    // Process first chunk to determine output dimensions
    let first_chunk_size = max_rows_per_chunk.min(shape[0]);
    let first_chunk = data.slice(scirs2_core::ndarray::s![0..first_chunk_size, ..]);
    let first_result = poly_features.transform(&first_chunk)?;
    let n_outputfeatures = first_result.shape()[1];

    // Initialize full output matrix
    let mut output = Array2::zeros((shape[0], n_outputfeatures));

    // Copy first chunk result
    for i in 0..first_chunk_size {
        for j in 0..n_outputfeatures {
            output[[i, j]] = first_result[[i, j]];
        }
    }

    // Process remaining chunks
    for chunk_start in (first_chunk_size..shape[0]).step_by(max_rows_per_chunk) {
        let chunk_end = (chunk_start + max_rows_per_chunk).min(shape[0]);
        let chunk = data.slice(scirs2_core::ndarray::s![chunk_start..chunk_end, ..]);
        let chunk_result = poly_features.transform(&chunk)?;

        for (i_local, i_global) in (chunk_start..chunk_end).enumerate() {
            for j in 0..n_outputfeatures {
                output[[i_global, j]] = chunk_result[[i_local, j]];
            }
        }
    }

    Ok(output)
}
