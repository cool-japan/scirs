//! VoiRS Compatibility Test Suite
//!
//! This test suite validates that all VoiRS migration patterns work correctly
//! with SciRS2 APIs. It covers:
//!
//! 1. Real FFT with trait objects
//! 2. Random number generation (seeding, shuffling)
//! 3. Parallel operations
//! 4. Complex number handling
//!
//! These tests match the actual usage patterns from the VoiRS codebase.

use scirs2_core::numeric::Complex64;
use scirs2_core::parallel_ops::*;
use scirs2_core::random::{Random, SliceRandom};
use scirs2_core::Complex64 as ScirsComplex64;
use scirs2_fft::real_planner::{ComplexToReal, RealFftPlanner, RealToComplex};
use scirs2_fft::{irfft, rfft};
use std::f64::consts::PI;
use std::sync::Arc;

// ============================================
// 1. REAL FFT TESTS (VoiRS Spatial Audio Patterns)
// ============================================

/// Test basic real FFT functional API (workaround for Beta 4)
#[test]
fn test_real_fft_functional_api() {
    // VoiRS pattern: Simple FFT processing
    let signal: Vec<f32> = (0..1024)
        .map(|i| (2.0 * std::f32::consts::PI * i as f32 / 1024.0).sin())
        .collect();

    // Forward FFT
    let spectrum = rfft(&signal, None).unwrap();
    assert_eq!(spectrum.len(), 513); // 1024/2 + 1

    // Inverse FFT
    let recovered = irfft(&spectrum, Some(1024)).unwrap();
    assert_eq!(recovered.len(), 1024);

    // Verify round-trip accuracy
    for (i, &orig) in signal.iter().enumerate() {
        let diff = (orig as f64 - recovered[i]).abs();
        assert!(diff < 1e-6, "Round-trip error at {}: {}", i, diff);
    }
}

/// Test RealFftPlanner with trait objects (Beta 4 API)
#[test]
fn test_real_fft_planner_trait_objects() {
    // VoiRS pattern: Cached FFT plans with trait objects
    struct AudioProcessor {
        forward: Arc<dyn RealToComplex<f64>>,
        backward: Arc<dyn ComplexToReal<f64>>,
    }

    impl AudioProcessor {
        fn new(size: usize) -> Self {
            let mut planner = RealFftPlanner::<f64>::new();
            Self {
                forward: planner.plan_fft_forward(size),
                backward: planner.plan_fft_inverse(size),
            }
        }

        fn process(&self, input: &[f64]) -> Vec<f64> {
            let mut spectrum = vec![Complex64::new(0.0, 0.0); self.forward.output_len()];
            self.forward.process(input, &mut spectrum);

            let mut output = vec![0.0; self.backward.len()];
            self.backward.process(&spectrum, &mut output);

            output
        }
    }

    // Test the processor
    let processor = AudioProcessor::new(1024);
    let input: Vec<f64> = (0..1024).map(|i| i as f64).collect();
    let output = processor.process(&input);

    // Verify round-trip
    for (i, (&orig, &recov)) in input.iter().zip(output.iter()).enumerate() {
        let diff = (orig - recov).abs();
        assert!(diff < 1e-10, "Mismatch at {}: {} vs {}", i, orig, recov);
    }
}

/// Test VoiRS WFS (Wavefield Synthesis) pattern
#[test]
fn test_voirs_wfs_pattern() {
    // Simulates voirs-spatial/src/wfs.rs usage
    struct WavefieldSynthesizer {
        fft_size: usize,
        forward: Arc<dyn RealToComplex<f64>>,
        backward: Arc<dyn ComplexToReal<f64>>,
    }

    impl WavefieldSynthesizer {
        fn new(size: usize) -> Self {
            let mut planner = RealFftPlanner::<f64>::new();
            Self {
                fft_size: size,
                forward: planner.plan_fft_forward(size),
                backward: planner.plan_fft_inverse(size),
            }
        }

        fn synthesize(&self, input: &[f64]) -> Vec<f64> {
            assert_eq!(input.len(), self.fft_size);

            let mut spectrum = vec![Complex64::new(0.0, 0.0); self.forward.output_len()];
            self.forward.process(input, &mut spectrum);

            // Apply transfer function (example: low-pass filter)
            let cutoff = spectrum.len() / 4;
            for (i, val) in spectrum.iter_mut().enumerate() {
                if i > cutoff {
                    *val = Complex64::new(0.0, 0.0);
                }
            }

            let mut output = vec![0.0; self.fft_size];
            self.backward.process(&spectrum, &mut output);

            output
        }
    }

    let synthesizer = WavefieldSynthesizer::new(512);
    let input: Vec<f64> = (0..512)
        .map(|i| (2.0 * PI * 5.0 * i as f64 / 512.0).sin())
        .collect();

    let output = synthesizer.synthesize(&input);
    assert_eq!(output.len(), 512);

    // After low-pass filtering, output should be smoother
    let input_variance = calculate_variance(&input);
    let output_variance = calculate_variance(&output);
    assert!(output_variance < input_variance * 1.5);
}

/// Test f32 precision (VoiRS often uses f32 for memory efficiency)
#[test]
fn test_real_fft_f32_precision() {
    let mut planner = RealFftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(256);
    let inverse = planner.plan_fft_inverse(256);

    let input: Vec<f32> = (0..256).map(|i| (i as f32 / 256.0).sin()).collect();

    let mut spectrum = vec![scirs2_core::numeric::Complex::new(0.0f32, 0.0); forward.output_len()];
    forward.process(&input, &mut spectrum);

    let mut output = vec![0.0f32; 256];
    inverse.process(&spectrum, &mut output);

    for (i, (&orig, &recov)) in input.iter().zip(output.iter()).enumerate() {
        let diff = (orig - recov).abs();
        assert!(diff < 1e-5, "f32 precision error at {}: {}", i, diff);
    }
}

// ============================================
// 2. RANDOM NUMBER GENERATION TESTS
// ============================================

/// Test seeding reproducibility (VoiRS dataset shuffling)
#[test]
fn test_random_seeding_reproducibility() {
    // VoiRS pattern: voirs-dataset/src/datasets/jvs.rs
    let mut rng1 = Random::seed(42);
    let mut rng2 = Random::seed(42);

    let sample1: f64 = rng1.random_f64();
    let sample2: f64 = rng2.random_f64();

    assert_eq!(
        sample1, sample2,
        "Seeded RNGs should produce identical values"
    );

    // Generate sequences
    let seq1: Vec<f64> = (0..100).map(|_| rng1.random_f64()).collect();
    let mut rng3 = Random::seed(42);
    rng3.random_f64(); // Skip first value
    let seq2: Vec<f64> = (0..100).map(|_| rng3.random_f64()).collect();

    assert_eq!(seq1, seq2, "Seeded sequences should be identical");
}

/// Test shuffle reproducibility (VoiRS dataset augmentation)
#[test]
fn test_shuffle_reproducibility() {
    // VoiRS pattern: voirs-dataset/src/datasets/ljspeech.rs
    let mut data1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut data2 = data1.clone();

    let mut rng1 = Random::seed(123);
    let mut rng2 = Random::seed(123);

    rng1.shuffle(&mut data1);
    rng2.shuffle(&mut data2);

    assert_eq!(data1, data2, "Seeded shuffles should be identical");
    assert_ne!(
        data1,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Data should be shuffled"
    );
}

/// Test SliceRandom trait direct usage
#[test]
fn test_slice_random_trait() {
    let mut data = vec![1, 2, 3, 4, 5];
    let original = data.clone();

    let mut rng = Random::default();

    // Test shuffle method from SliceRandom trait (via Random wrapper)
    rng.shuffle(&mut data);

    // Data should be shuffled (very unlikely to be in original order)
    let is_shuffled = data != original;
    assert!(
        is_shuffled || data.len() < 3,
        "Data should likely be shuffled"
    );

    // All original elements should still be present
    let mut sorted_data = data.clone();
    sorted_data.sort();
    let mut sorted_original = original;
    sorted_original.sort();
    assert_eq!(sorted_data, sorted_original);
}

/// Test VoiRS domain adapter pattern
#[test]
fn test_voirs_domain_adapter_pattern() {
    // Simulates voirs-dataset/src/ml/domain/adapter.rs
    struct DomainAdapter {
        rng: Random<scirs2_core::random::rngs::StdRng>,
    }

    impl DomainAdapter {
        fn new(seed: u64) -> Self {
            Self {
                rng: Random::seed(seed),
            }
        }

        fn sample_indices(&mut self, total: usize, count: usize) -> Vec<usize> {
            let mut indices: Vec<usize> = (0..total).collect();
            self.rng.shuffle(&mut indices);
            indices.truncate(count);
            indices
        }
    }

    let mut adapter = DomainAdapter::new(999);
    let indices = adapter.sample_indices(100, 10);

    assert_eq!(indices.len(), 10);
    assert!(indices.iter().all(|&i| i < 100));

    // Test reproducibility
    let mut adapter1 = DomainAdapter::new(111);
    let mut adapter2 = DomainAdapter::new(111);
    let indices1 = adapter1.sample_indices(50, 5);
    let indices2 = adapter2.sample_indices(50, 5);
    assert_eq!(indices1, indices2);
}

// ============================================
// 3. PARALLEL OPERATIONS TESTS
// ============================================

/// Test parallel iterator basic functionality
#[test]
fn test_parallel_iterator() {
    // VoiRS pattern: voirs-dataset/src/processing/pipeline.rs
    let data: Vec<i32> = (0..10000).collect();

    let result: Vec<i32> = data.par_iter().map(|&x| x * x).collect();

    assert_eq!(result.len(), 10000);
    assert_eq!(result[0], 0);
    assert_eq!(result[99], 99 * 99);
    assert_eq!(result[9999], 9999 * 9999);
}

/// Test parallel chunks processing (VoiRS audio chunking)
#[test]
fn test_parallel_chunks() {
    // VoiRS pattern: Processing audio frames in parallel
    let audio_data: Vec<f32> = (0..8192).map(|i| (i as f32).sin()).collect();

    let chunk_size = 512;
    let results: Vec<f32> = audio_data
        .par_chunks(chunk_size)
        .flat_map(|chunk| {
            // Process each chunk (e.g., apply window)
            chunk.iter().map(|&x| x * 0.5).collect::<Vec<_>>()
        })
        .collect();

    assert_eq!(results.len(), audio_data.len());
}

/// Test current_num_threads (rayon compatibility)
#[test]
fn test_current_num_threads() {
    use scirs2_core::parallel_ops::current_num_threads;

    let threads = current_num_threads();
    assert!(threads > 0);
    assert!(threads <= 256); // Reasonable upper bound

    // Also test the num_threads alias
    let threads2 = num_threads();
    assert_eq!(threads, threads2);
}

/// Test par_iter on slices (VoiRS spectrogram processing)
#[test]
fn test_par_iter_slices() {
    let samples: Vec<f64> = (0..1000).map(|i| i as f64).collect();

    let sum: f64 = samples.par_iter().map(|&x| x * x).sum();

    let expected: f64 = samples.iter().map(|&x| x * x).sum();
    assert_eq!(sum, expected);
}

/// Test VoiRS parallel pipeline pattern
#[test]
fn test_voirs_parallel_pipeline() {
    // Simulates voirs-vocoder/src/parallel/mod.rs usage
    struct ParallelProcessor {
        num_workers: usize,
    }

    impl ParallelProcessor {
        fn new() -> Self {
            Self {
                num_workers: current_num_threads(),
            }
        }

        fn process_batch(&self, data: &[f64]) -> Vec<f64> {
            data.par_iter().map(|&x| x.powi(2) + x.sqrt()).collect()
        }
    }

    let processor = ParallelProcessor::new();
    assert!(processor.num_workers > 0);

    let data: Vec<f64> = (1..1001).map(|i| i as f64).collect();
    let result = processor.process_batch(&data);
    assert_eq!(result.len(), 1000);
}

// ============================================
// 4. COMPLEX NUMBER TESTS
// ============================================

/// Test Complex64 re-export
#[test]
fn test_complex_reexport() {
    // VoiRS should use scirs2_core::Complex64, not num_complex directly
    let z1 = ScirsComplex64::new(1.0, 2.0);
    let z2 = Complex64::new(3.0, 4.0);

    let sum = z1 + z2;
    assert_eq!(sum.re, 4.0);
    assert_eq!(sum.im, 6.0);
}

/// Test complex array operations (VoiRS HRTF processing)
#[test]
fn test_complex_array_operations() {
    // VoiRS pattern: voirs-spatial/src/hrtf.rs
    let size = 128;
    let mut spectrum = vec![Complex64::new(0.0, 0.0); size];

    // Initialize with test data
    for (i, val) in spectrum.iter_mut().enumerate() {
        *val = Complex64::new(i as f64, (size - i) as f64);
    }

    // Apply complex operations
    let processed: Vec<Complex64> = spectrum
        .iter()
        .map(|&z| z * Complex64::new(0.5, 0.0) + Complex64::new(1.0, 1.0))
        .collect();

    assert_eq!(processed.len(), size);
    // For i=0: (0+128i)*0.5 + (1+i) = (0+64i) + (1+i) = (1+65i)
    assert_eq!(processed[0], Complex64::new(1.0, 65.0));
    // For i=10: (10+118i)*0.5 + (1+i) = (5+59i) + (1+i) = (6+60i)
    assert_eq!(processed[10], Complex64::new(6.0, 60.0));
}

// ============================================
// 5. INTEGRATION TESTS (Full VoiRS Patterns)
// ============================================

/// Test complete VoiRS audio processing pipeline
#[test]
fn test_complete_audio_pipeline() {
    // Simulates full VoiRS processing chain
    struct AudioPipeline {
        fft_size: usize,
        forward: Arc<dyn RealToComplex<f64>>,
        backward: Arc<dyn ComplexToReal<f64>>,
        rng: Random<scirs2_core::random::rngs::StdRng>,
    }

    impl AudioPipeline {
        fn new(fft_size: usize, seed: u64) -> Self {
            let mut planner = RealFftPlanner::<f64>::new();
            Self {
                fft_size,
                forward: planner.plan_fft_forward(fft_size),
                backward: planner.plan_fft_inverse(fft_size),
                rng: Random::seed(seed),
            }
        }

        fn process_with_noise(&mut self, input: &[f64]) -> Vec<f64> {
            assert_eq!(input.len(), self.fft_size);

            // FFT
            let mut spectrum = vec![Complex64::new(0.0, 0.0); self.forward.output_len()];
            self.forward.process(input, &mut spectrum);

            // Add noise to spectrum
            for val in spectrum.iter_mut() {
                let noise_re = self.rng.random_f64() * 0.01 - 0.005;
                let noise_im = self.rng.random_f64() * 0.01 - 0.005;
                *val += Complex64::new(noise_re, noise_im);
            }

            // IFFT
            let mut output = vec![0.0; self.fft_size];
            self.backward.process(&spectrum, &mut output);

            output
        }
    }

    let mut pipeline = AudioPipeline::new(1024, 42);
    let input: Vec<f64> = (0..1024)
        .map(|i| (2.0 * PI * 10.0 * i as f64 / 1024.0).sin())
        .collect();

    let output = pipeline.process_with_noise(&input);
    assert_eq!(output.len(), 1024);

    // Output should be similar to input (small noise added)
    let mse: f64 = input
        .iter()
        .zip(output.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        / input.len() as f64;

    assert!(mse < 0.01, "MSE too high: {}", mse);
}

/// Test parallel + FFT combination (VoiRS batch processing)
#[test]
fn test_parallel_fft_processing() {
    // Process multiple signals in parallel
    let signals: Vec<Vec<f64>> = (0..8)
        .map(|j| {
            (0..512)
                .map(|i| (2.0 * PI * (j + 1) as f64 * i as f64 / 512.0).sin())
                .collect()
        })
        .collect();

    let results: Vec<Vec<Complex64>> = signals
        .par_iter()
        .map(|signal| rfft(signal, None).unwrap())
        .collect();

    assert_eq!(results.len(), 8);
    for result in results {
        assert_eq!(result.len(), 257); // 512/2 + 1
    }
}

// ============================================
// HELPER FUNCTIONS
// ============================================

fn calculate_variance(data: &[f64]) -> f64 {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
}
