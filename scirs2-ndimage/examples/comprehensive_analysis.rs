//! Comprehensive image analysis example demonstrating advanced features
//!
//! This example showcases the enhanced capabilities of scirs2-ndimage including:
//! - Advanced image quality assessment
//! - Texture analysis
//! - Multi-scale analysis
//! - Visualization and reporting
//! - Advanced filtering with wavelets and Gabor filters
//! - GPU backend device detection

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_ndimage::{
    // Analysis functions
    analysis::{
        image_entropy, image_quality_assessment, image_sharpness, multi_scale_analysis,
        peak_signal_to_noise_ratio, structural_similarity_index, texture_analysis,
        MultiScaleConfig,
    },
    // Backend detection
    backend::{auto_backend, DeviceManager},
    // Error handling
    error::NdimageResult,
    // Advanced filters
    filters::{
        advanced::{gabor_filter, log_gabor_filter, steerable_filter, GaborParams},
        wavelets::{wavelet_denoise, WaveletFamily, WaveletFilter},
        BorderMode,
    },
    // Standard filters
    filters::{gaussian_filter, sobel},
    // Visualization functions
    visualization::{
        create_colormap, generate_report, plot_histogram, ColorMap, PlotConfig, ReportConfig,
        ReportFormat,
    },
};

#[allow(dead_code)]
fn main() -> NdimageResult<()> {
    println!("=== SciRS2 N-Dimensional Image Processing - Comprehensive Analysis Demo ===\n");

    // 1. Device Detection and Backend Capabilities
    demonstrate_device_detection()?;

    // 2. Create a synthetic test image
    let testimage = create_testimage();
    println!(
        "Created synthetic test image ({}×{})",
        testimage.nrows(),
        testimage.ncols()
    );

    // 3. Advanced Image Quality Assessment
    demonstrate_quality_assessment(&testimage)?;

    // 4. Texture Analysis
    demonstratetexture_analysis(&testimage)?;

    // 5. Advanced Filtering Techniques
    demonstrate_advanced_filtering(&testimage)?;

    // 6. Wavelet Analysis and Denoising
    demonstrate_wavelet_analysis(&testimage)?;

    // 7. Multi-scale Analysis
    demonstrate_multiscale_analysis(&testimage)?;

    // 8. Visualization and Reporting
    demonstrate_visualization(&testimage)?;

    // 9. Performance Comparison
    demonstrate_performance_comparison(&testimage)?;

    println!("\n=== Analysis Complete ===");
    println!("This example demonstrates the comprehensive image processing");
    println!("capabilities of SciRS2 including advanced analysis, visualization,");
    println!("and GPU backend support for scientific image processing workflows.");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_device_detection() -> NdimageResult<()> {
    println!("1. GPU Backend Device Detection");
    println!("-------------------------------");

    // Try to create device manager
    match scirs2_ndimage::backend::device_detection::get_device_manager() {
        Ok(manager_arc) => {
            let manager = manager_arc.lock().unwrap();

            // Check different backends
            let backends = [
                #[cfg(feature = "cuda")]
                scirs2_ndimage::backend::Backend::Cuda,
                #[cfg(feature = "opencl")]
                scirs2_ndimage::backend::Backend::OpenCL,
                #[cfg(all(target_os = "macos", feature = "metal"))]
                scirs2_ndimage::backend::Backend::Metal,
                scirs2_ndimage::backend::Backend::Cpu,
            ];

            for backend in &backends {
                let available = manager.is_backend_available(*backend);
                let device_count = manager.device_count(*backend);
                println!(
                    "  {:?}: {} (devices: {})",
                    backend,
                    if available {
                        "Available"
                    } else {
                        "Not Available"
                    },
                    device_count
                );

                if available && device_count > 0 {
                    if let Some(device_info) = manager.get_device_info(*backend, 0) {
                        println!("    Primary device: {}", device_info.name);
                        println!(
                            "    Total memory: {:.1} GB",
                            device_info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
                        );
                        if let Some(compute_cap) = device_info.compute_capability {
                            println!(
                                "    Compute capability: {}.{}",
                                compute_cap.0, compute_cap.1
                            );
                        }
                    }
                }
            }

            // Try auto backend selection
            match auto_backend() {
                Ok(_executor) => println!("  Auto backend executor: Successfully created"),
                Err(e) => println!("  Auto backend executor: Failed ({})", e),
            }
        }
        Err(e) => {
            println!("  Device manager: Failed to initialize ({})", e);
            println!("  Note: This is expected in environments without GPU libraries");
        }
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn create_testimage() -> Array2<f64> {
    let (height, width) = (128, 128);
    let mut image = Array2::zeros((height, width));

    // Create a complex test pattern with multiple features
    for i in 0..height {
        for j in 0..width {
            let x = j as f64 / width as f64;
            let y = i as f64 / height as f64;

            // Combine multiple patterns
            let mut value = 0.0;

            // Radial gradient
            let center_x = 0.5;
            let center_y = 0.5;
            let dist = ((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt();
            value += 0.3 * (1.0 - dist);

            // Sine wave pattern
            value += 0.2 * (8.0 * std::f64::consts::PI * x).sin();

            // Checkerboard pattern
            if ((i / 16) + (j / 16)) % 2 == 0 {
                value += 0.3;
            }

            // Gaussian blob
            let blob_dist = ((x - 0.7).powi(2) + (y - 0.3).powi(2)).sqrt();
            value += 0.4 * (-blob_dist * blob_dist * 50.0).exp();

            // Add some noise
            value += 0.05 * scirs2_core::random::random::<f64>() - 0.025;

            image[[i, j]] = value.clamp(0.0, 1.0);
        }
    }

    image
}

#[allow(dead_code)]
fn demonstrate_quality_assessment(image: &Array2<f64>) -> NdimageResult<()> {
    println!("2. Advanced Image Quality Assessment");
    println!("------------------------------------");

    // Create a slightly modified version for comparison
    let noisyimage =
        image.mapv(|x| (x + 0.02 * scirs2_core::random::random::<f64>()).clamp(0.0, 1.0));

    // Compute comprehensive quality metrics
    let metrics = image_quality_assessment(&image.view(), &noisyimage.view())?;

    println!("  Quality Metrics (comparing original vs. noisy):");
    println!("    PSNR:           {:.2} dB", metrics.psnr);
    println!("    SSIM:           {:.4}", metrics.ssim);
    println!("    MSE:            {:.6}", metrics.mse);
    println!("    RMSE:           {:.6}", metrics.rmse);
    println!("    MAE:            {:.6}", metrics.mae);
    println!("    SNR:            {:.2} dB", metrics.snr);
    println!("    CNR:            {:.2}", metrics.cnr);
    println!("    Entropy:        {:.2} bits", metrics.entropy);
    println!("    Sharpness:      {:.6}", metrics.sharpness);
    println!("    Local Variance: {:.6}", metrics.local_variance);

    // Individual metric demonstrations
    let entropy = image_entropy(&image.view())?;
    let sharpness = image_sharpness(&image.view())?;
    let psnr = peak_signal_to_noise_ratio(&image.view(), &noisyimage.view())?;
    let ssim = structural_similarity_index(&image.view(), &noisyimage.view())?;

    println!("\n  Individual Metric Validation:");
    println!("    Image entropy:  {:.2} bits", entropy);
    println!("    Image sharpness:{:.6}", sharpness);
    println!("    PSNR validation:{:.2} dB", psnr);
    println!("    SSIM validation:{:.4}", ssim);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstratetexture_analysis(image: &Array2<f64>) -> NdimageResult<()> {
    println!("3. Comprehensive Texture Analysis");
    println!("----------------------------------");

    let texturemetrics = texture_analysis(&image.view(), Some(5))?;

    println!("  Texture Metrics:");
    println!("    GLCM Contrast:     {:.6}", texturemetrics.glcm_contrast);
    println!(
        "    GLCM Dissimilarity:{:.6}",
        texturemetrics.glcm_dissimilarity
    );
    println!(
        "    GLCM Homogeneity:  {:.6}",
        texturemetrics.glcm_homogeneity
    );
    println!("    GLCM Energy:       {:.6}", texturemetrics.glcm_energy);
    println!(
        "    GLCM Correlation:  {:.6}",
        texturemetrics.glcm_correlation
    );
    println!(
        "    LBP Uniformity:    {:.6}",
        texturemetrics.lbp_uniformity
    );
    println!("    Gabor Mean:        {:.6}", texturemetrics.gabor_mean);
    println!("    Gabor Std:         {:.6}", texturemetrics.gabor_std);
    println!(
        "    Fractal Dimension: {:.3}",
        texturemetrics.fractal_dimension
    );

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_advanced_filtering(image: &Array2<f64>) -> NdimageResult<()> {
    println!("4. Advanced Filtering Techniques");
    println!("--------------------------------");

    // Gabor filtering for texture analysis
    let gabor_params = GaborParams {
        wavelength: 8.0,
        orientation: std::f64::consts::PI / 4.0, // 45 degrees
        sigma_x: 3.0,
        sigma_y: 3.0,
        phase: 0.0,
        aspect_ratio: None,
    };

    let gabor_result = gabor_filter(
        &image.view(),
        &gabor_params,
        Some(15),
        Some(BorderMode::Reflect),
    )?;
    println!("  Gabor filter: Applied (kernel size: 15×15, orientation: 45°)");
    println!(
        "    Output range: [{:.3}, {:.3}]",
        gabor_result.iter().cloned().fold(f64::INFINITY, f64::min),
        gabor_result
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    // Log-Gabor filtering (frequency domain)
    match log_gabor_filter(&image.view(), 0.1, 1.0, 0.0, std::f64::consts::PI / 6.0) {
        Ok(log_gabor_result) => {
            println!("  Log-Gabor filter: Applied (center freq: 0.1, bandwidth: 1.0 octave)");
            println!(
                "    Output range: [{:.3}, {:.3}]",
                log_gabor_result
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min),
                log_gabor_result
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max)
            );
        }
        Err(e) => println!("  Log-Gabor filter: Skipped ({})", e),
    }

    // Steerable filtering
    let steerable_result = steerable_filter(
        &image.view(),
        2,
        std::f64::consts::PI / 6.0,
        2.0,
        Some(BorderMode::Reflect),
    )?;
    println!("  Steerable filter: Applied (order: 2, orientation: 30°, sigma: 2.0)");
    println!(
        "    Output range: [{:.3}, {:.3}]",
        steerable_result
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min),
        steerable_result
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    // Standard filters for comparison
    let gaussian_result = gaussian_filter(image, 2.0, None, None)?;
    println!("  Gaussian filter: Applied (sigma: 2.0)");
    println!(
        "    Output range: [{:.3}, {:.3}]",
        gaussian_result
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min),
        gaussian_result
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    let sobel_result = sobel(image, 0, None)?;
    println!("  Sobel edge detection: Applied");
    println!(
        "    Output range: [{:.3}, {:.3}]",
        sobel_result.iter().cloned().fold(f64::INFINITY, f64::min),
        sobel_result
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_wavelet_analysis(image: &Array2<f64>) -> NdimageResult<()> {
    println!("5. Wavelet Analysis and Denoising");
    println!("----------------------------------");

    // Add noise for denoising demonstration
    let noisyimage =
        image.mapv(|x| (x + 0.05 * scirs2_core::random::random::<f64>()).clamp(0.0, 1.0));

    // Test different wavelet families
    let wavelets = [
        ("Haar", WaveletFamily::Haar),
        ("Daubechies-2", WaveletFamily::Daubechies(2)),
        ("Daubechies-4", WaveletFamily::Daubechies(4)),
    ];

    for (name, family) in &wavelets {
        match WaveletFilter::new(*family).and_then(|filter| {
            wavelet_denoise(&noisyimage.view(), &filter, 0.05, 3, BorderMode::Reflect)
        }) {
            Ok(denoised) => {
                let original_mse = noisyimage
                    .iter()
                    .zip(image.iter())
                    .map(|(&noisy, &orig)| (noisy - orig).powi(2))
                    .sum::<f64>()
                    / (image.len() as f64);

                let denoised_mse = denoised
                    .iter()
                    .zip(image.iter())
                    .map(|(&denoised_val, &orig)| (denoised_val - orig).powi(2))
                    .sum::<f64>()
                    / (image.len() as f64);

                println!("  {} wavelet denoising:", name);
                println!("    Noisy MSE:    {:.6}", original_mse);
                println!("    Denoised MSE: {:.6}", denoised_mse);
                println!(
                    "    Improvement:  {:.1}%",
                    (1.0 - denoised_mse / original_mse) * 100.0
                );
            }
            Err(e) => println!("  {} wavelet: Failed ({})", name, e),
        }
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_multiscale_analysis(image: &Array2<f64>) -> NdimageResult<()> {
    println!("6. Multi-scale Analysis");
    println!("-----------------------");

    let config = MultiScaleConfig {
        num_scales: 4,
        scale_factor: 2.0,
        min_size: 16,
    };

    let multiscale_results = multi_scale_analysis(&image.view(), &config)?;

    println!(
        "  Multi-scale quality analysis ({} scales):",
        multiscale_results.len()
    );
    for (i, metrics) in multiscale_results.iter().enumerate() {
        let scale_size = image.nrows() / (2_usize.pow(i as u32));
        println!(
            "    Scale {}: {}×{} - Entropy: {:.2}, Sharpness: {:.4}",
            i + 1,
            scale_size,
            scale_size,
            metrics.entropy,
            metrics.sharpness
        );
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_visualization(image: &Array2<f64>) -> NdimageResult<()> {
    println!("7. Visualization and Reporting");
    println!("------------------------------");

    // Create histogram data
    let flat_data: Array1<f64> = image.iter().cloned().collect();

    // Generate histogram plot
    let plot_config = PlotConfig {
        title: "Image Intensity Histogram".to_string(),
        xlabel: "Intensity".to_string(),
        ylabel: "Frequency".to_string(),
        num_bins: 50,
        format: ReportFormat::Text,
        ..PlotConfig::default()
    };

    match plot_histogram(&flat_data.view(), &plot_config) {
        Ok(histogram_plot) => {
            println!("  Histogram plot generated:");
            // Show just the first few lines
            for line in histogram_plot.lines().take(5) {
                println!("    {}", line);
            }
            println!("    ... (truncated)");
        }
        Err(e) => println!("  Histogram generation failed: {}", e),
    }

    // Generate comprehensive report
    let report_config = ReportConfig {
        title: "SciRS2 Image Analysis Report".to_string(),
        author: "Comprehensive Analysis Example".to_string(),
        format: ReportFormat::Markdown,
        ..ReportConfig::default()
    };

    match generate_report(&image.view(), None, None, &report_config) {
        Ok(report) => {
            println!("\n  Analysis report generated:");
            // Show first few lines of the report
            for line in report.lines().take(10) {
                println!("    {}", line);
            }
            println!("    ... (report continues)");
        }
        Err(e) => println!("  Report generation failed: {}", e),
    }

    // Demonstrate colormap creation
    let viridis_colors = create_colormap(ColorMap::Viridis, 5);
    println!("\n  Viridis colormap (5 colors):");
    for (i, color) in viridis_colors.iter().enumerate() {
        println!(
            "    Color {}: {} (R:{}, G:{}, B:{})",
            i,
            color.to_hex(),
            color.r,
            color.g,
            color.b
        );
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_performance_comparison(image: &Array2<f64>) -> NdimageResult<()> {
    println!("8. Performance Characteristics");
    println!("------------------------------");

    use std::time::Instant;

    // Gaussian filter timing
    let start = Instant::now();
    let _gaussian = gaussian_filter(image, 2.0, None, None)?;
    let gaussian_time = start.elapsed();

    // Sobel filter timing
    let start = Instant::now();
    let _sobel = sobel(image, 0, None)?;
    let sobel_time = start.elapsed();

    // Gabor filter timing
    let gabor_params = GaborParams::default();
    let start = Instant::now();
    let _gabor = gabor_filter(&image.view(), &gabor_params, Some(15), None)?;
    let gabor_time = start.elapsed();

    println!(
        "  Processing times for {}×{} image:",
        image.nrows(),
        image.ncols()
    );
    println!("    Gaussian filter: {:?}", gaussian_time);
    println!("    Sobel filter:    {:?}", sobel_time);
    println!("    Gabor filter:    {:?}", gabor_time);

    // Memory usage estimation
    let image_size_mb = (image.len() * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
    println!("\n  Memory characteristics:");
    println!("    Input image:     {:.2} MB", image_size_mb);
    println!(
        "    Typical overhead:{:.2} MB (for intermediate arrays)",
        image_size_mb * 2.0
    );

    println!();
    Ok(())
}

// Include rand for the random number generation in the example
