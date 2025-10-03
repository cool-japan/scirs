use scirs2_core::ndarray::{array, Array2};
use scirs2_stats::{partial_corr, partial_corrr};

#[allow(dead_code)]
fn main() {
    // Create sample data
    let x = array![10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0];
    let y = array![8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68];

    // Control variable
    let z1 = array![7.0, 5.0, 8.0, 7.0, 8.0, 7.0, 5.0, 3.0, 9.0, 4.0, 6.0];
    let z = Array2::from_shape_vec((11, 1), z1.to_vec()).unwrap();

    // Calculate partial correlation coefficient (without p-value)
    let pr = partial_corr(&x.view(), &y.view(), &z.view()).unwrap();
    println!("Partial correlation coefficient: {}", pr);

    // Calculate partial correlation with p-value
    let (pr, p_value) = partial_corrr(&x.view(), &y.view(), &z.view(), "two-sided").unwrap();
    println!("Partial correlation coefficient: {}", pr);
    println!("Two-sided p-value: {}", p_value);

    // Check if correlation is statistically significant
    println!(
        "Is correlation significant at alpha=0.05? {}",
        p_value < 0.05
    );

    // Try with multiple control variables
    let z2 = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let mut z_multi = Array2::<f64>::zeros((11, 2));
    for i in 0..11 {
        z_multi[[i, 0]] = z1[i];
        z_multi[[i, 1]] = z2[i];
    }

    let (pr_multi, p_value_multi) =
        partial_corrr(&x.view(), &y.view(), &z_multi.view(), "two-sided").unwrap();
    println!("\nPartial correlation with multiple controls: {}", pr_multi);
    println!("Two-sided p-value: {}", p_value_multi);

    // Try with one-sided tests
    let (_, p_greater) = partial_corrr(&x.view(), &y.view(), &z.view(), "greater").unwrap();
    let (_, p_less) = partial_corrr(&x.view(), &y.view(), &z.view(), "less").unwrap();
    println!("\nOne-sided tests:");
    println!("P-value (greater): {}", p_greater);
    println!("P-value (less): {}", p_less);
}
