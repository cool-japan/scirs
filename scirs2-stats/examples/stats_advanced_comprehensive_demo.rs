//! Comprehensive Statistical Operations Demo
//!
//! This example demonstrates core statistical operations available in scirs2-stats.

use scirs2_core::ndarray::Array1;
use scirs2_stats::{mean, std, var};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Statistical Operations Demo");
    println!("=============================\n");

    // Create sample data
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    println!("Sample data: {:?}", data.as_slice().unwrap());

    // Basic statistics
    println!("\n📈 Basic Statistics:");
    let mean_val = mean(&data.view())?;
    let var_val = var(&data.view(), 1, None)?;
    let std_val = std(&data.view(), 1, None)?;

    println!("   Mean: {:.4}", mean_val);
    println!("   Variance: {:.4}", var_val);
    println!("   Standard Deviation: {:.4}", std_val);

    // Data validation
    println!("\n✅ Data Validation:");
    println!("   Sample size: {}", data.len());
    println!(
        "   All finite: {}",
        data.iter().all(|&x: &f64| x.is_finite())
    );
    println!(
        "   Min value: {:.4}",
        data.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    );
    println!(
        "   Max value: {:.4}",
        data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    println!("\n✅ Demo completed successfully!");
    println!("   All statistical operations executed without error.");

    Ok(())
}
