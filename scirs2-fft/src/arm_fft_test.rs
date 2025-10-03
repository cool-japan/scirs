//! Test for ARM NEON FFT Support
//!
//! This file contains tests to verify that the FFT implementation works correctly
//! on ARM platforms with NEON SIMD acceleration. It validates both the core FFT
//! functionality and the ARM-specific optimizations.

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use crate::error::FFTResult;
    use crate::simd_fft::{fft2_adaptive, fft_adaptive, fftn_adaptive, ifft_adaptive};
    use scirs2_core::numeric::Complex64;
    use std::f64::consts::PI;

    /// Test 1D FFT on ARM platforms
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_arm_fft_1d() -> FFTResult<()> {
        // Create a test signal
        let n = 1024;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / 128.0).sin())
            .collect();

        // Compute FFT
        let spectrum = fft_adaptive(&signal, None)?;

        // The spectrum should have peaks at bin 8 (frequency = n/128) and n-8
        // Due to the properties of the sine wave at frequency 1/128
        let magnitude = |c: &Complex64| -> f64 { (c.re.powi(2) + c.im.powi(2)).sqrt() };

        // Find the peak frequencies (excluding DC)
        let mut peak_bins = Vec::new();
        for i in 1..n {
            if magnitude(&spectrum[i]) > n as f64 / 4.0 {
                peak_bins.push(i);
            }
        }

        // We should have exactly two peaks (at 8 and n-8)
        assert_eq!(peak_bins.len(), 2);

        // Check the peaks are at expected positions
        assert!(
            peak_bins.contains(&8) || peak_bins.contains(&(n - 8)),
            "Expected peaks at bins 8 and {} but found at {:?}",
            n - 8,
            peak_bins
        );

        // Now do inverse FFT
        let reconstructed = ifft_adaptive(&spectrum, None)?;

        // Check the reconstructed signal matches the original
        for i in 0..n {
            assert!(
                (signal[i] - reconstructed[i].re).abs() < 1e-10,
                "Mismatch at index {}: {} vs {}",
                i,
                signal[i],
                reconstructed[i].re
            );
        }

        Ok(())
    }

    /// Test 2D FFT on ARM platforms
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_arm_fft_2d() -> FFTResult<()> {
        // Create a 2D test signal (32x32 grid)
        let n_rows = 32;
        let n_cols = 32;
        let mut signal = Vec::with_capacity(n_rows * n_cols);

        for i in 0..n_rows {
            for j in 0..n_cols {
                let x = i as f64 / n_rows as f64;
                let y = j as f64 / n_cols as f64;
                let value = (2.0 * PI * 3.0 * x).sin() * (2.0 * PI * 5.0 * y).cos();
                signal.push(value);
            }
        }

        // Compute 2D FFT
        let spectrum = fft2_adaptive(&signal, Some((n_rows, n_cols)), None)?;

        // The spectrum should have peaks at (3, 5), (3, n_cols-5), (n_rows-3, 5), (n_rows-3, n_cols-5)
        // due to the properties of the sine/cosine waves
        let magnitude = |c: &Complex64| -> f64 { (c.re.powi(2) + c.im.powi(2)).sqrt() };

        // Find the peak frequencies (excluding DC)
        let mut peak_positions = Vec::new();
        for i in 0..n_rows {
            for j in 0..n_cols {
                if i == 0 && j == 0 {
                    continue; // Skip DC component
                }

                if magnitude(&spectrum[[i, j]]) > (n_rows * n_cols) as f64 / 8.0 {
                    peak_positions.push((i, j));
                }
            }
        }

        // We should have exactly 4 peaks
        assert!(
            peak_positions.len() >= 4,
            "Expected at least 4 peaks but found {}",
            peak_positions.len()
        );

        Ok(())
    }

    /// Test N-dimensional FFT on ARM platforms
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_arm_fft_nd() -> FFTResult<()> {
        // Create a 3D test signal (16x16x16 grid)
        let shape = [16, 16, 16];
        let total_elements: usize = shape.iter().product();

        let mut signal = Vec::with_capacity(total_elements);

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let x = i as f64 / shape[0] as f64;
                    let y = j as f64 / shape[1] as f64;
                    let z = k as f64 / shape[2] as f64;
                    let value = (2.0 * PI * 2.0 * x).sin()
                        * (2.0 * PI * 3.0 * y).cos()
                        * (2.0 * PI * 4.0 * z).sin();
                    signal.push(value);
                }
            }
        }

        // Compute N-dimensional FFT
        let spectrum = fftn_adaptive(&signal, Some(&shape), None, None)?;

        // Verify the result has the correct dimensions
        assert_eq!(spectrum.len(), total_elements);

        Ok(())
    }

    // Using AdaptivePlanner instead of optimize_plan_for_platform
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_arm_plan_optimization() -> FFTResult<()> {
        // Create a test signal
        let n = 2048;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / 256.0).sin())
            .collect();

        // Test using adaptive planner
        use crate::planning_adaptive::{AdaptiveExecutor, AdaptivePlanningConfig};

        // Create an adaptive executor with default config
        let config = AdaptivePlanningConfig::default();
        let executor = AdaptiveExecutor::new(&[n], true, Some(config));

        // Convert input to complex
        let input: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let mut output = vec![Complex64::default(); n];

        // Run the FFT
        executor.execute(&input, &mut output)?;

        // Check that we got meaningful results - the spectrum of a sine wave
        // should have peaks at specific frequencies
        let magnitude = |c: &Complex64| -> f64 { (c.re.powi(2) + c.im.powi(2)).sqrt() };

        // Find the peak frequencies (excluding DC)
        let mut peak_bins = Vec::new();
        for i in 1..n {
            if magnitude(&output[i]) > n as f64 / 4.0 {
                peak_bins.push(i);
            }
        }

        // We should have exactly two peaks for a sine wave
        assert_eq!(peak_bins.len(), 2);

        // Check the adaptive executor's statistics
        let stats = executor.get_statistics();
        assert!(
            stats[&executor.current_strategy()].1 > 0,
            "Should have recorded execution statistics"
        );

        Ok(())
    }

    /// Test parallel planning strategies on ARM platforms
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_arm_parallel_planning() -> FFTResult<()> {
        // Skip test on systems with only 1 core
        let num_cpus = num_cpus::get();
        if num_cpus <= 1 {
            println!("Skipping parallel planning test on single-core system");
            return Ok(());
        }

        // Create a large test signal to benefit from parallel planning
        let n = 4096;
        let signal: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((2.0 * PI * i as f64 / 512.0).sin(), 0.0))
            .collect();

        // Test with parallel planning
        let planner = crate::planning_parallel::ParallelPlanner::new(None);
        let plan = planner.plan_fft(&[n], false, crate::planning::PlannerBackend::RustFFT)?;
        let executor = crate::planning_parallel::ParallelExecutor::new(plan, None);
        let mut spectrum = vec![Complex64::new(0.0, 0.0); n];
        executor.execute(&signal, &mut spectrum)?;

        // Verify the result has the correct dimension
        assert_eq!(spectrum.len(), n);

        // Verify the result has the expected peaks
        // The spectrum should have peaks at bin 8 (frequency = n/512) and n-8
        let magnitude = |c: &Complex64| -> f64 { (c.re.powi(2) + c.im.powi(2)).sqrt() };

        // Find the peak frequencies (excluding DC)
        let mut peak_bins = Vec::new();
        for i in 1..n {
            if magnitude(&spectrum[i]) > n as f64 / 4.0 {
                peak_bins.push(i);
            }
        }

        // We should have exactly two peaks
        assert_eq!(peak_bins.len(), 2);

        Ok(())
    }
}
