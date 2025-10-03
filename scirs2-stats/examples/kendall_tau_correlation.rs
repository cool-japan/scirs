use scirs2_core::ndarray::array;
use scirs2_stats::{kendall_tau, kendalltau};

#[allow(dead_code)]
fn main() {
    // Create sample data with a monotonic relationship
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![5.0, 4.0, 3.0, 2.0, 1.0];

    // Calculate Kendall tau correlation coefficient (without p-value)
    let tau = kendall_tau(&x.view(), &y.view(), "b").unwrap();
    println!("Kendall tau correlation coefficient: {}", tau);
    // Perfect negative ordinal association (tau should be -1.0)
    assert!((tau - (-1.0f64)).abs() < 1e-10f64);

    // Calculate Kendall tau correlation with p-value
    let (tau, p_value) = kendalltau(&x.view(), &y.view(), "b", "two-sided").unwrap();
    println!("Kendall tau correlation coefficient: {}", tau);
    println!("Two-sided p-value: {}", p_value);
    // Perfect negative ordinal association (tau should be -1.0)
    assert!((tau - (-1.0f64)).abs() < 1e-10f64);

    // Check if correlation is statistically significant
    println!(
        "Is correlation significant at alpha=0.05? {}",
        p_value < 0.05
    );

    // Try with a non-perfect correlation
    let y2 = array![5.0, 3.0, 4.0, 2.0, 1.0]; // Not perfectly ordered
    let (tau2, p_value2) = kendalltau(&x.view(), &y2.view(), "b", "two-sided").unwrap();
    println!("\nKendall tau with imperfect ranking: {}", tau2);
    println!("Two-sided p-value: {}", p_value2);

    // Try with no correlation
    let y3 = array![3.0, 1.0, 5.0, 2.0, 4.0]; // Random ordering
    let (tau3, p_value3) = kendalltau(&x.view(), &y3.view(), "b", "two-sided").unwrap();
    println!("\nKendall tau with random ordering: {}", tau3);
    println!("Two-sided p-value: {}", p_value3);

    // Try with tied values
    let x_ties = array![1.0, 2.0, 3.0, 3.0, 5.0]; // Tied values in x
    let y_ties = array![5.0, 4.0, 3.0, 3.0, 1.0]; // Tied values in y
    let (tau_ties, p_value_ties) =
        kendalltau(&x_ties.view(), &y_ties.view(), "b", "two-sided").unwrap();
    println!("\nKendall tau with tied values: {}", tau_ties);
    println!("Two-sided p-value: {}", p_value_ties);

    // Try with one-sided tests
    let (_, p_greater) = kendalltau(&x.view(), &y.view(), "b", "greater").unwrap();
    let (_, p_less) = kendalltau(&x.view(), &y.view(), "b", "less").unwrap();
    println!("\nOne-sided tests for negative correlation:");
    println!("P-value (greater): {}", p_greater); // Should be 1.0
    println!("P-value (less): {}", p_less); // Should be small

    // Try the tau-c method for rectangular tables
    let (tau_c, p_value_c) = kendalltau(&x.view(), &y.view(), "c", "two-sided").unwrap();
    println!("\nKendall tau-c correlation: {}", tau_c);
    println!("Two-sided p-value: {}", p_value_c);
}
