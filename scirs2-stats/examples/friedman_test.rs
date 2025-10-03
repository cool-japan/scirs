use scirs2_core::ndarray::array;
use scirs2_stats::friedman;

#[allow(dead_code)]
fn main() {
    println!("Friedman Test Example");
    println!("====================\n");

    // Example 1: Different treatments on the same subjects
    // This is a common use case for the Friedman test
    println!("Example 1: Different treatments on the same subjects");

    // Consider a study where 4 subjects are treated with 3 different medications
    // Each row is a subject, each column is a treatment (medication)
    let data = array![
        [7.0, 9.0, 8.0], // Subject 1's response to treatments A, B, C
        [6.0, 5.0, 7.0], // Subject 2's response to treatments A, B, C
        [9.0, 7.0, 6.0], // Subject 3's response to treatments A, B, C
        [8.0, 5.0, 6.0]  // Subject 4's response to treatments A, B, C
    ];

    // Perform the Friedman test
    let (chi2, p_value) = friedman(&data.view()).unwrap();

    println!("Data matrix (rows = subjects, columns = treatments):");
    for row in data.rows() {
        println!("{:?}", row);
    }
    println!("\nFriedman test results:");
    println!("Chi-square statistic: {:.4}", chi2);
    println!("P-value: {:.4}", p_value);
    println!(
        "At α = 0.05: {}",
        if p_value < 0.05 {
            "Significant differences detected"
        } else {
            "No significant differences"
        }
    );

    // Example 2: Repeated measurements with clear differences
    println!("\nExample 2: Measurements with clear differences");

    // This example shows data with a clear trend (increasing values in columns)
    let data2 = array![
        [1.0, 5.0, 9.0],
        [2.0, 6.0, 10.0],
        [3.0, 7.0, 11.0],
        [4.0, 8.0, 12.0]
    ];

    // Perform the Friedman test
    let (chi2, p_value) = friedman(&data2.view()).unwrap();

    println!("Data matrix (rows = subjects, columns = treatments):");
    for row in data2.rows() {
        println!("{:?}", row);
    }
    println!("\nFriedman test results:");
    println!("Chi-square statistic: {:.4}", chi2);
    println!("P-value: {:.4}", p_value);
    println!(
        "At α = 0.05: {}",
        if p_value < 0.05 {
            "Significant differences detected"
        } else {
            "No significant differences"
        }
    );

    // Example 3: Treatment comparison with ties
    println!("\nExample 3: Treatment comparison with ties");

    // This example contains ties within subjects (first two treatments tie for each subject)
    let data3 = array![
        [5.0, 5.0, 8.0],
        [7.0, 7.0, 10.0],
        [3.0, 3.0, 6.0],
        [9.0, 9.0, 12.0]
    ];

    // Perform the Friedman test
    let (chi2, p_value) = friedman(&data3.view()).unwrap();

    println!("Data matrix (rows = subjects, columns = treatments):");
    for row in data3.rows() {
        println!("{:?}", row);
    }
    println!("\nFriedman test results:");
    println!("Chi-square statistic: {:.4}", chi2);
    println!("P-value: {:.4}", p_value);
    println!(
        "At α = 0.05: {}",
        if p_value < 0.05 {
            "Significant differences detected"
        } else {
            "No significant differences"
        }
    );

    // Note: The Friedman test is often followed by post-hoc analysis to determine
    // which specific treatments differ when the overall test is significant.
    // This example only shows the omnibus test.
}
