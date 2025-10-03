use scirs2_core::ndarray::array;
use scirs2_stats::{point_biserial, point_biserialr};

#[allow(dead_code)]
fn main() {
    // Create binary and continuous data
    let binary = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let continuous = array![2.5, 4.5, 3.2, 5.1, 2.0, 4.8, 2.8, 5.5];

    // Calculate point-biserial correlation coefficient (without p-value)
    let rpb = point_biserial(&binary.view(), &continuous.view()).unwrap();
    println!("Point-biserial correlation coefficient: {}", rpb);

    // Calculate point-biserial correlation with p-value
    let (rpb, p_value) = point_biserialr(&binary.view(), &continuous.view(), "two-sided").unwrap();
    println!("Point-biserial correlation coefficient: {}", rpb);
    println!("Two-sided p-value: {}", p_value);

    // Check if correlation is statistically significant
    println!(
        "Is correlation significant at alpha=0.05? {}",
        p_value < 0.05
    );

    // Try with a weaker relationship
    let continuous_weak = array![2.5, 3.8, 3.2, 4.1, 2.9, 3.9, 2.8, 4.2];
    let (rpb_weak, p_value_weak) =
        point_biserialr(&binary.view(), &continuous_weak.view(), "two-sided").unwrap();
    println!("\nPoint-biserial with weaker relationship: {}", rpb_weak);
    println!("Two-sided p-value: {}", p_value_weak);

    // Try with one-sided tests
    let (_, p_greater) = point_biserialr(&binary.view(), &continuous.view(), "greater").unwrap();
    let (_, p_less) = point_biserialr(&binary.view(), &continuous.view(), "less").unwrap();
    println!("\nOne-sided tests:");
    println!("P-value (greater): {}", p_greater); // Should be small
    println!("P-value (less): {}", p_less); // Should be 1.0

    // Try with negative correlation
    let continuous_neg = array![4.5, 2.5, 5.1, 3.2, 4.8, 2.0, 5.5, 2.8];
    let (rpb_neg, p_value_neg) =
        point_biserialr(&binary.view(), &continuous_neg.view(), "two-sided").unwrap();
    println!("\nPoint-biserial with negative relationship: {}", rpb_neg);
    println!("Two-sided p-value: {}", p_value_neg);
}
