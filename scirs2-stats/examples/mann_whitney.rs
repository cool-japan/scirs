use scirs2_core::ndarray::array;
use scirs2_stats::mann_whitney;

#[allow(dead_code)]
fn main() {
    println!("Mann-Whitney U Test Example");
    println!("==========================\n");

    // Create two independent samples
    let group1 = array![2.9, 3.0, 2.5, 2.6, 3.2, 2.8];
    let group2 = array![3.8, 3.7, 3.9, 4.0, 4.2];

    println!("Group 1: {:?}", group1);
    println!("Group 2: {:?}\n", group2);

    // Perform two-sided Mann-Whitney U test
    let (u_stat, p_value) = mann_whitney(&group1.view(), &group2.view(), "two-sided", true)
        .expect("Failed to perform Mann-Whitney U test");

    println!("Two-sided test:");
    println!("  U-statistic: {:.4}", u_stat);
    println!("  p-value: {:.4}", p_value);
    println!(
        "  Conclusion: {}\n",
        if p_value < 0.05 {
            "Reject null hypothesis (p < 0.05)"
        } else {
            "Fail to reject null hypothesis (p ≥ 0.05)"
        }
    );

    // Perform one-sided Mann-Whitney U test (less)
    let (u_stat, p_value) = mann_whitney(&group1.view(), &group2.view(), "less", true)
        .expect("Failed to perform Mann-Whitney U test");

    println!("One-sided test (group1 < group2):");
    println!("  U-statistic: {:.4}", u_stat);
    println!("  p-value: {:.4}", p_value);
    println!(
        "  Conclusion: {}\n",
        if p_value < 0.05 {
            "Reject null hypothesis (p < 0.05)"
        } else {
            "Fail to reject null hypothesis (p ≥ 0.05)"
        }
    );

    // Perform one-sided Mann-Whitney U test (greater)
    let (u_stat, p_value) = mann_whitney(&group1.view(), &group2.view(), "greater", true)
        .expect("Failed to perform Mann-Whitney U test");

    println!("One-sided test (group1 > group2):");
    println!("  U-statistic: {:.4}", u_stat);
    println!("  p-value: {:.4}", p_value);
    println!(
        "  Conclusion: {}\n",
        if p_value < 0.05 {
            "Reject null hypothesis (p < 0.05)"
        } else {
            "Fail to reject null hypothesis (p ≥ 0.05)"
        }
    );

    // Example with different sample sizes
    let group_a = array![10.5, 9.8, 12.7, 8.3, 11.2, 10.0, 9.5, 13.1];
    let group_b = array![15.6, 14.2, 13.8, 16.1, 13.5];

    println!("Example with different sample sizes:");
    println!("Group A: {:?}", group_a);
    println!("Group B: {:?}\n", group_b);

    let (u_stat, p_value) = mann_whitney(&group_a.view(), &group_b.view(), "two-sided", true)
        .expect("Failed to perform Mann-Whitney U test");

    println!("Two-sided test:");
    println!("  U-statistic: {:.4}", u_stat);
    println!("  p-value: {:.4}", p_value);
    println!(
        "  Conclusion: {}",
        if p_value < 0.05 {
            "Reject null hypothesis (p < 0.05)"
        } else {
            "Fail to reject null hypothesis (p ≥ 0.05)"
        }
    );

    // Test with ties
    let group_c = array![1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0];
    let group_d = array![2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0];

    println!("\nExample with ties:");
    println!("Group C: {:?}", group_c);
    println!("Group D: {:?}\n", group_d);

    let (u_stat, p_value) = mann_whitney(&group_c.view(), &group_d.view(), "two-sided", false)
        .expect("Failed to perform Mann-Whitney U test");

    println!("Two-sided test without continuity correction:");
    println!("  U-statistic: {:.4}", u_stat);
    println!("  p-value: {:.4}", p_value);
    println!(
        "  Conclusion: {}",
        if p_value < 0.05 {
            "Reject null hypothesis (p < 0.05)"
        } else {
            "Fail to reject null hypothesis (p ≥ 0.05)"
        }
    );

    // Same test with continuity correction
    let (u_stat, p_value) = mann_whitney(&group_c.view(), &group_d.view(), "two-sided", true)
        .expect("Failed to perform Mann-Whitney U test");

    println!("\nTwo-sided test with continuity correction:");
    println!("  U-statistic: {:.4}", u_stat);
    println!("  p-value: {:.4}", p_value);
    println!(
        "  Conclusion: {}",
        if p_value < 0.05 {
            "Reject null hypothesis (p < 0.05)"
        } else {
            "Fail to reject null hypothesis (p ≥ 0.05)"
        }
    );
}
