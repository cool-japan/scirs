//! Example demonstrating streaming operations for large images

use scirs2_core::ndarray::{Array2, ArrayView2, ArrayViewMut2};
use scirs2_ndimage::{
    stream_process_file, streaming::OverlapInfo, StreamConfig, StreamableOp,
    StreamingGaussianFilter,
};
use std::path::Path;

/// Example: Processing very large images that don't fit in memory
#[allow(dead_code)]
fn main() {
    // Example 1: Process a large image file with streaming Gaussian filter
    process_largeimage_gaussian();

    // Example 2: Process with Fourier domain filters
    process_largeimage_fourier();

    // Example 3: Custom streaming operation
    process_with_custom_op();
}

/// Process a large image with streaming Gaussian filter
#[allow(dead_code)]
fn process_largeimage_gaussian() {
    println!("Processing large image with streaming Gaussian filter...");

    // Configure streaming with 256MB chunks
    let config = StreamConfig {
        chunk_size: 256 * 1024 * 1024,
        overlap: vec![10, 10], // 10 pixel overlap for smooth boundaries
        use_mmap: true,
        cache_chunks: 4,
        temp_dir: Some("/tmp".to_string()),
    };

    // Create streaming Gaussian filter
    let op = StreamingGaussianFilter::new(vec![5.0, 5.0], Some(4.0));

    // Process a hypothetical large image file
    // In a real scenario, these would be actual file paths
    let input_path = Path::new("largeimage_10gb.raw");
    let output_path = Path::new("largeimage_filtered.raw");
    let shape = &[50000, 50000]; // 50k x 50k image

    // This would process the image in chunks without loading it all into memory
    println!("Would process image of size {:?} in chunks...", shape);
    // stream_process_file::<f64, scirs2_core::ndarray::Ix2_>(
    //     input_path,
    //     output_path,
    //     shape,
    //     op,
    //     Some(config),
    // ).unwrap();
}

/// Process with Fourier domain filters
#[allow(dead_code)]
fn process_largeimage_fourier() {
    println!("\nProcessing with Fourier domain filters...");

    let config = StreamConfig {
        chunk_size: 512 * 1024 * 1024, // Larger chunks for FFT efficiency
        overlap: vec![8, 8],           // Minimal overlap for Fourier filters
        ..Default::default()
    };

    let input_path = Path::new("largeimage.raw");
    let output_path = Path::new("largeimage_fourier_gaussian.raw");
    let shape = &[30000, 30000];
    let sigma = &[10.0, 10.0];

    // This would apply Fourier Gaussian filter to large images
    println!("Would apply Fourier Gaussian with sigma {:?}...", sigma);
    // fourier_gaussian_file::<f64>(
    //     input_path,
    //     output_path,
    //     shape,
    //     sigma,
    //     Some(config),
    // ).unwrap();
}

/// Custom streaming operation
#[allow(dead_code)]
fn process_with_custom_op() {
    println!("\nProcessing with custom streaming operation...");

    // Custom edge enhancement operation
    struct EdgeEnhancementOp {
        strength: f64,
    }

    impl StreamableOp<f64, scirs2_core::ndarray::Ix2> for EdgeEnhancementOp {
        fn apply_chunk(
            &self,
            chunk: &ArrayView2<f64>,
        ) -> scirs2_ndimage::NdimageResult<Array2<f64>> {
            // Apply Laplacian for edge detection
            let edges = scirs2_ndimage::laplace(&chunk.to_owned(), None, None)?;

            // Enhance edges
            let enhanced = chunk.to_owned() - edges * self.strength;
            Ok(enhanced)
        }

        fn required_overlap(&self) -> Vec<usize> {
            vec![3, 3] // Need 3 pixel overlap for Laplacian
        }

        fn merge_overlap(
            &self,
            self_output: &mut ArrayViewMut2<f64>,
            _new_chunk: &ArrayView2<f64>,
            _overlap_info: &OverlapInfo,
        ) -> scirs2_ndimage::NdimageResult<()> {
            // Simple blending in overlap regions
            Ok(())
        }
    }

    let op = EdgeEnhancementOp { strength: 0.5 };
    let config = StreamConfig::default();

    println!("Custom edge enhancement operation configured with strength 0.5");
    // Would process with custom operation...
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;
    use scirs2_ndimage::StreamProcessor;

    #[test]
    fn test_streaming_small_array() {
        // Test with a small array to verify functionality
        let input = arr2(&[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);

        let config = StreamConfig {
            chunk_size: 64, // Very small chunks for testing
            overlap: vec![1, 1],
            ..Default::default()
        };

        let processor = StreamProcessor::<f64>::new(config);
        let op = StreamingGaussianFilter::new(vec![1.0, 1.0], None);

        let result = processor.process_in_memory(&input.view(), op).unwrap();
        assert_eq!(result.shape(), input.shape());

        // Check that values are smoothed
        assert!(result[[1, 1]] != input[[1, 1]]);
    }
}
