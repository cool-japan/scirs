//! Advanced Visualization Features Demo
//!
//! This example demonstrates the advanced visualization capabilities of scirs2-ndimage,
//! including interactive HTML visualizations, export utilities, and comparison views.

use ndarray::{Array2, ArrayView2};
use scirs2_ndimage::{
    error::NdimageResult,
    filters::gaussian_filter,
    visualization::{
        advanced::{create_comparison_view, create_interactive_visualization},
        export::{save_plot, ExportConfig},
        plot_heatmap, plot_histogram, plot_surface, ColorMap, PlotConfig, ReportFormat,
    },
};
use statrs::statistics::Statistics;

#[allow(dead_code)]
fn main() -> NdimageResult<()> {
    println!("🎨 Advanced Visualization Features Demo");
    println!("=====================================");

    // Create sample data
    let image_size = 50;
    let originalimage = create_sampleimage(image_size);

    // Apply some processing
    let smoothedimage = gaussian_filter(&originalimage, 2.0, None, None)?;
    let noisyimage = add_noise(&originalimage);

    println!("\n1. Creating Interactive Visualization");
    println!("   📊 Generating interactive HTML with controls...");

    let interactive_html = create_interactive_visualization(
        &originalimage.view(),
        "Interactive Image Analysis Dashboard",
    )?;

    // Save the interactive visualization
    let export_config = ExportConfig {
        output_path: "examples/outputs/interactive_visualization.html".to_string(),
        include_metadata: true,
        ..ExportConfig::default()
    };

    save_plot(&interactive_html, &export_config)?;
    println!("   ✅ Saved to: {}", export_config.output_path);

    println!("\n2. Creating Image Comparison View");
    println!("   🔍 Generating side-by-side comparison...");

    let comparisonimages = vec![
        ("Original", originalimage.view()),
        ("Gaussian Smoothed", smoothedimage.view()),
        ("With Noise", noisyimage.view()),
    ];

    let comparison_html = create_comparison_view(&comparisonimages, "Image Processing Comparison")?;

    let comparison_config = ExportConfig {
        output_path: "examples/outputs/image_comparison.html".to_string(),
        include_metadata: true,
        ..ExportConfig::default()
    };

    save_plot(&comparison_html, &comparison_config)?;
    println!("   ✅ Saved to: {}", comparison_config.output_path);

    println!("\n3. Generating Multiple Plot Types");
    println!("   📈 Creating various visualization formats...");

    // Heatmap
    let heatmap_config = PlotConfig {
        title: "Data Heatmap".to_string(),
        colormap: ColorMap::Viridis,
        format: ReportFormat::Html,
        width: 600,
        height: 400,
        ..PlotConfig::default()
    };

    let heatmap_html = plot_heatmap(&originalimage.view(), &heatmap_config)?;
    let heatmap_export = ExportConfig {
        output_path: "examples/outputs/heatmap_visualization.html".to_string(),
        ..ExportConfig::default()
    };
    save_plot(&heatmap_html, &heatmap_export)?;
    println!("   🔥 Heatmap saved to: {}", heatmap_export.output_path);

    // Surface plot
    let surface_config = PlotConfig {
        title: "3D Surface Plot".to_string(),
        colormap: ColorMap::Plasma,
        format: ReportFormat::Html,
        width: 600,
        height: 400,
        ..PlotConfig::default()
    };

    let surface_html = plot_surface(&originalimage.view(), &surface_config)?;
    let surface_export = ExportConfig {
        output_path: "examples/outputs/surface_visualization.html".to_string(),
        ..ExportConfig::default()
    };
    save_plot(&surface_html, &surface_export)?;
    println!(
        "   🏔️  Surface plot saved to: {}",
        surface_export.output_path
    );

    // Histogram of flattened data
    let flat_data = originalimage.iter().cloned().collect::<Vec<_>>();
    let flat_array = ndarray::Array1::from_vec(flat_data);

    let histogram_config = PlotConfig {
        title: "Data Distribution Histogram".to_string(),
        format: ReportFormat::Html,
        num_bins: 30,
        ..PlotConfig::default()
    };

    let histogram_html = plot_histogram(&flat_array.view(), &histogram_config)?;
    let histogram_export = ExportConfig {
        output_path: "examples/outputs/histogram_visualization.html".to_string(),
        ..ExportConfig::default()
    };
    save_plot(&histogram_html, &histogram_export)?;
    println!("   📊 Histogram saved to: {}", histogram_export.output_path);

    println!("\n4. Color Map Demonstrations");
    println!("   🌈 Testing different color schemes...");

    let colormaps = vec![
        (ColorMap::Viridis, "viridis"),
        (ColorMap::Plasma, "plasma"),
        (ColorMap::Jet, "jet"),
        (ColorMap::Hot, "hot"),
        (ColorMap::Cool, "cool"),
        (ColorMap::Inferno, "inferno"),
    ];

    for (colormap, name) in colormaps {
        let config = PlotConfig {
            title: format!("{} Color Map Demo", name.to_uppercase()),
            colormap,
            format: ReportFormat::Html,
            width: 400,
            height: 300,
            ..PlotConfig::default()
        };

        let colormap_html = plot_heatmap(&originalimage.view(), &config)?;
        let colormap_export = ExportConfig {
            output_path: format!("examples/outputs/colormap_{}.html", name),
            ..ExportConfig::default()
        };
        save_plot(&colormap_html, &colormap_export)?;
        println!("   🎨 {} colormap saved", name);
    }

    println!("\n✨ Advanced Visualization Demo Complete!");
    println!("\n📁 All outputs saved to examples/outputs/");
    println!("   Open the HTML files in a web browser to view interactive visualizations");

    // Print summary statistics
    let stats = computeimage_stats(&originalimage.view());
    println!("\n📊 Sample Data Statistics:");
    println!("   Dimensions: {}×{}", image_size, image_size);
    println!("   Mean: {:.4}", stats.mean);
    println!("   Min: {:.4}, Max: {:.4}", stats.min, stats.max);
    println!("   Standard Deviation: {:.4}", stats.std_dev);

    Ok(())
}

#[allow(dead_code)]
fn create_sampleimage(size: usize) -> Array2<f64> {
    Array2::from_shape_fn((size, size), |(i, j)| {
        let x = i as f64 / size as f64;
        let y = j as f64 / size as f64;

        // Create an interesting pattern with multiple features
        let wave1 = (x * 10.0).sin() * (y * 10.0).cos();
        let wave2 = ((x - 0.5).powi(2) + (y - 0.5).powi(2)).sqrt() * 20.0;
        let gaussian = (-((x - 0.5).powi(2) + (y - 0.5).powi(2)) * 50.0).exp();

        wave1 * 0.3 + wave2.sin() * 0.4 + gaussian * 0.8
    })
}

#[allow(dead_code)]
fn add_noise(image: &Array2<f64>) -> Array2<f64> {
    use rand::Rng;
    let mut rng = rand::rng();
    image.mapv(|x| x + (rng.random::<f64>() - 0.5) * 0.1)
}

struct ImageStats {
    mean: f64,
    min: f64,
    max: f64,
    std_dev: f64,
}

#[allow(dead_code)]
fn computeimage_stats(image: &ArrayView2<f64>) -> ImageStats {
    let mean = image.mean().unwrap_or(0.0);
    let min = image.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let variance_array = image.mapv(|x| (x - mean).powi(2));
    let variance = variance_array.mean();
    let std_dev = variance.sqrt();

    ImageStats {
        mean,
        min,
        max,
        std_dev,
    }
}
