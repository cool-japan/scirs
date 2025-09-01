//! Example demonstrating GPU backend support for accelerated processing

use ndarray::Array2;
use scirs2_ndimage::{
    auto_backend, backend::GaussianFilterOp, gaussian_filter, Backend, BackendBuilder,
    BackendConfig, BackendExecutor, BackendOp, NdimageResult,
};

#[allow(dead_code)]
fn main() {
    // Example 1: Automatic backend selection
    automatic_backend_selection();

    // Example 2: Force GPU backend
    force_gpu_backend();

    // Example 3: Custom backend configuration
    custom_backend_config();

    // Example 4: Fallback handling
    gpu_with_fallback();
}

/// Automatic backend selection based on array size
#[allow(dead_code)]
fn automatic_backend_selection() {
    println!("Example 1: Automatic backend selection");

    // Create executor with automatic backend selection
    let executor = auto_backend().expect("Failed to create backend executor");

    // Small array - will use CPU
    let small_array = Array2::from_elem((100, 100), 1.0f32);
    let op = GaussianFilterOp::new(vec![2.0, 2.0], Some(4.0));

    println!("Processing small array (100x100) - should use CPU");
    let op2 = GaussianFilterOp::new(vec![2.0, 2.0], Some(4.0));
    let _result = executor.execute(&small_array.view(), op).unwrap();

    // Large array - will use GPU if available
    let large_array = Array2::from_elem((5000, 5000), 1.0f32);
    println!("Processing large array (5000x5000) - should use GPU if available");
    let _result = executor.execute(&large_array.view(), op2).unwrap();
}

/// Force GPU backend for all operations
#[allow(dead_code)]
fn force_gpu_backend() {
    println!("\nExample 2: Force GPU backend");

    #[cfg(feature = "cuda")]
    {
        let executor = BackendBuilder::new()
            .backend(Backend::Cuda)
            .gpu_threshold(0) // Use GPU even for small arrays
            .allow_fallback(false) // Don't fall back to CPU
            .device_id(0) // Use first GPU
            .build()
            .expect("Failed to create CUDA backend");

        let array = Array2::from_elem((1000, 1000), 1.0f32);
        let op = GaussianFilterOp::new(vec![5.0, 5.0], None);

        println!("Forcing CUDA backend for 1000x1000 array");
        match executor.execute(&array.view(), op) {
            Ok(_) => println!("Successfully processed on GPU"),
            Err(e) => println!("GPU processing failed: {}", e),
        }
    }

    #[cfg(not(feature = "cuda"))]
    println!("CUDA support not compiled in");
}

/// Custom backend configuration
#[allow(dead_code)]
fn custom_backend_config() {
    println!("\nExample 3: Custom backend configuration");

    let config = BackendConfig {
        backend: Backend::Auto,
        gpu_threshold: 500_000, // Use GPU for arrays > 500k elements
        gpu_memory_limit: Some(2 * 1024 * 1024 * 1024), // Limit to 2GB
        allow_fallback: true,
        device_id: None, // Auto-select device
    };

    let executor = BackendExecutor::new(config).expect("Failed to create backend");

    // Process multiple arrays with different sizes
    let sizes = vec![(100, 100), (1000, 1000), (2000, 2000)];

    for (h, w) in sizes {
        let array = Array2::from_elem((h, w), 1.0f64);
        let op = GaussianFilterOp::new(vec![3.0, 3.0], None);

        println!("Processing {}x{} array ({} elements)", h, w, h * w);

        let _result = executor.execute(&array.view(), op).unwrap();
    }
}

/// GPU with CPU fallback
#[allow(dead_code)]
fn gpu_with_fallback() {
    println!("\nExample 4: GPU with CPU fallback");

    let executor = BackendBuilder::new()
        .backend(Backend::Auto)
        .allow_fallback(true) // Enable fallback
        .gpu_memory_limit(100) // Very low limit to trigger fallback
        .build()
        .expect("Failed to create backend");

    // Large array that might exceed GPU memory limit
    let large_array = Array2::from_elem((10000, 10000), 1.0f32);
    let op = GaussianFilterOp::new(vec![10.0, 10.0], None);

    println!("Processing very large array with fallback enabled");
    match executor.execute(&large_array.view(), op) {
        Ok(_) => println!("Successfully processed (possibly with CPU fallback)"),
        Err(e) => println!("Processing failed: {}", e),
    }
}

/// Custom operation with backend support
#[derive(Clone)]
#[allow(dead_code)]
struct CustomBlurOp {
    iterations: usize,
    sigma: f32,
}

impl BackendOp<f32, ndarray::Ix2> for CustomBlurOp {
    fn execute_cpu(&self, input: &ndarray::ArrayView2<f32>) -> NdimageResult<Array2<f32>> {
        let mut result = input.to_owned();

        // Apply Gaussian blur multiple times
        for _ in 0..self.iterations {
            result = gaussian_filter(&result.mapv(|x| x as f64), self.sigma as f64, None, None)?
                .mapv(|x| x as f32);
        }

        Ok(result)
    }

    #[cfg(feature = "gpu")]
    fn execute_gpu(
        &self,
        input: &ndarray::ArrayView2<f32>,
        _backend: Backend,
    ) -> NdimageResult<Array2<f32>> {
        // In a real implementation, this would use GPU kernels
        println!(
            "GPU implementation would apply {} iterations",
            self.iterations
        );
        self.execute_cpu(input) // Fallback to CPU for now
    }

    fn memory_requirement(&self, inputshape: &[usize]) -> usize {
        let elements: usize = inputshape.iter().product();
        // Need memory for input + output + temp buffer for each iteration
        elements * std::mem::size_of::<f32>() * (2 + self.iterations)
    }

    fn benefits_from_gpu(&self, arraysize: usize) -> bool {
        // Multiple iterations benefit more from GPU
        arraysize > 50_000 || (arraysize > 10_000 && self.iterations > 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_builder() {
        let executor = BackendBuilder::new()
            .backend(Backend::Cpu)
            .gpu_threshold(100_000)
            .build()
            .unwrap();

        let array = Array2::zeros((10, 10));
        let op = GaussianFilterOp::new(vec![1.0, 1.0], None);

        let result = executor.execute(&array.view(), op).unwrap();
        assert_eq!(result.shape(), array.shape());
    }

    #[test]
    fn test_custom_op() {
        let executor = auto_backend().unwrap();
        let array = Array2::ones((50, 50));
        let op = CustomBlurOp {
            iterations: 2,
            sigma: 1.5,
        };

        let result = executor.execute(&array.view(), op).unwrap();
        assert_eq!(result.shape(), array.shape());
        // Values should be blurred (not exactly 1.0 anymore)
        assert!((result[[25, 25]] - 1.0).abs() < 0.1);
    }
}
