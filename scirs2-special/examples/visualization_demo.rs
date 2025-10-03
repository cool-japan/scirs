//! Demonstration of visualization capabilities for special functions
//!
//! This example shows how to create various plots and visualizations
//! of special functions using the plotting module.
//!
//! Run with: cargo run --example visualization_demo --features plotting

#[cfg(feature = "plotting")]
use scirs2_special::visualization::{
    bessel_plots, error_function_plots,
    export::{export_plot_data, ExportFormat},
    gamma_plots, polynomial_plots, MultiPlot, PlotConfig,
};

#[cfg(not(feature = "plotting"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the 'plotting' feature.");
    println!("Run with: cargo run --example visualization_demo --features plotting");
}

#[cfg(feature = "plotting")]
#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Special Functions Visualization Demo");
    println!("===================================\n");

    // Create output directory
    std::fs::create_dir_all("plots")?;

    // 1. Plot gamma function family
    println!("1. Plotting gamma function family...");
    gamma_plots::plot_gamma_family("plots/gamma_family.png")?;
    println!("   Saved to plots/gamma_family.png");

    // 2. Plot Bessel functions
    println!("\n2. Plotting Bessel functions...");
    bessel_plots::plot_bessel_j("plots/bessel_j.png")?;
    println!("   Saved to plots/bessel_j.png");

    // 3. Plot Bessel function zeros
    println!("\n3. Plotting Bessel function zeros...");
    bessel_plots::plot_bessel_zeros("plots/bessel_zeros.png")?;
    println!("   Saved to plots/bessel_zeros.png");

    // 4. Plot error functions
    println!("\n4. Plotting error functions...");
    error_function_plots::plot_error_functions("plots/error_functions.png")?;
    println!("   Saved to plots/error_functions.png");

    // 5. Plot Legendre polynomials
    println!("\n5. Plotting Legendre polynomials...");
    polynomial_plots::plot_legendre("plots/legendre_polynomials.png", 5)?;
    println!("   Saved to plots/legendre_polynomials.png");

    // 6. Custom multi-plot example
    println!("\n6. Creating custom multi-plot...");
    create_custom_plot()?;
    println!("   Saved to plots/custom_special_functions.png");

    // 7. Export data in different formats
    println!("\n7. Exporting function data...");
    export_examples()?;

    println!("\nVisualization demo complete!");
    println!("Check the 'plots' directory for generated images.");

    Ok(())
}

#[cfg(feature = "plotting")]
#[allow(dead_code)]
fn create_custom_plot() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_special::{ai, bi, ci, si};

    let config = PlotConfig {
        title: "Special Functions Comparison".to_string(),
        x_label: "x".to_string(),
        y_label: "f(x)".to_string(),
        width: 1024,
        height: 768,
        ..Default::default()
    };

    MultiPlot::new(config)
        .add_function(Box::new(ai), "Airy Ai(x)")
        .add_function(Box::new(bi), "Airy Bi(x)")
        .add_function(
            Box::new(|x| si(x).unwrap_or(f64::NAN)),
            "Sine integral Si(x)",
        )
        .add_function(
            Box::new(|x| ci(x).unwrap_or(f64::NAN)),
            "Cosine integral Ci(x)",
        )
        .set_x_range(-5.0, 5.0)
        .plot("plots/custom_special_functions.png")
}

#[cfg(feature = "plotting")]
#[allow(dead_code)]
fn export_examples() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_special::gamma;

    // Export gamma function data as CSV
    let csv_data = export_plot_data(gamma, (0.1, 5.0), 100, ExportFormat::CSV)?;
    std::fs::write("plots/gamma_data.csv", csv_data)?;
    println!("   Exported gamma function to CSV: plots/gamma_data.csv");

    // Export as LaTeX/TikZ
    let latex_data = export_plot_data(gamma, (0.1, 5.0), 50, ExportFormat::LaTeX)?;
    std::fs::write("plots/gamma_tikz.tex", latex_data)?;
    println!("   Exported gamma function to LaTeX: plots/gamma_tikz.tex");

    Ok(())
}

#[cfg(feature = "plotting")]
#[allow(dead_code)]
fn demonstrate_advanced_features() {
    println!("\nAdvanced Visualization Features:");
    println!("================================");

    println!("\n1. Complex function visualization:");
    println!("   - Heatmaps of |f(z)| in complex plane");
    println!("   - Phase portraits with color coding");
    println!("   - Contour plots of real/imaginary parts");

    println!("\n2. 3D surface plots:");
    println!("   - Functions of two variables");
    println!("   - Parametric surfaces");
    println!("   - Level sets and isosurfaces");

    println!("\n3. Animation capabilities:");
    println!("   - Evolution of polynomial families");
    println!("   - Parameter sweeps");
    println!("   - Zero tracking animations");

    println!("\n4. Interactive features (when enabled):");
    println!("   - Zoom and pan");
    println!("   - Hover tooltips with values");
    println!("   - Export to various formats");
    println!("   - Parameter sliders");

    println!("\n5. Export formats:");
    println!("   - PNG/SVG for publications");
    println!("   - LaTeX/TikZ for papers");
    println!("   - CSV for data analysis");
    println!("   - Interactive HTML");
}
