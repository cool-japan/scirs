use scirs2_core::ndarray::array;
use scirs2_stats::{
    tests::ttest::Alternative,
    tests::{enhanced_ttest_1samp, enhanced_ttest_ind, enhanced_ttest_rel, ttest_ind_from_stats},
    TTestResult,
};

#[allow(dead_code)]
fn main() {
    println!("Enhanced T-test Examples");
    println!("=======================");

    // Example 1: One-sample t-test
    println!("\n1. One-Sample T-Test");
    println!("-------------------");

    let data = array![5.1, 4.9, 6.2, 5.7, 5.5, 5.1, 5.2, 5.0];
    let null_mean = 5.0;

    println!(
        "Testing if the sample mean is significantly different from {}:",
        null_mean
    );
    println!("Sample data: {:?}", data);

    // Two-sided test
    let two_sided =
        enhanced_ttest_1samp(&data.view(), null_mean, Alternative::TwoSided, "omit").unwrap();
    print_ttest_result(&two_sided, "Two-sided test");

    // One-sided test (greater)
    let greater =
        enhanced_ttest_1samp(&data.view(), null_mean, Alternative::Greater, "omit").unwrap();
    print_ttest_result(&greater, "One-sided test (greater)");

    // One-sided test (less)
    let less = enhanced_ttest_1samp(&data.view(), null_mean, Alternative::Less, "omit").unwrap();
    print_ttest_result(&less, "One-sided test (less)");

    // Example 2: Independent two-sample t-test
    println!("\n2. Independent Two-Sample T-Test");
    println!("------------------------------");

    let group1 = array![5.1, 4.9, 6.2, 5.7, 5.5];
    let group2 = array![4.8, 5.2, 5.1, 4.7, 4.9];

    println!("Testing if two independent samples have different means:");
    println!("Group 1: {:?}", group1);
    println!("Group 2: {:?}", group2);

    // Equal variances
    let equal_var = enhanced_ttest_ind(
        &group1.view(),
        &group2.view(),
        true,
        Alternative::TwoSided,
        "omit",
    )
    .unwrap();
    print_ttest_result(&equal_var, "Equal variances (Student's t-test)");

    // Unequal variances (Welch's t-test)
    let welch = enhanced_ttest_ind(
        &group1.view(),
        &group2.view(),
        false,
        Alternative::TwoSided,
        "omit",
    )
    .unwrap();
    print_ttest_result(&welch, "Unequal variances (Welch's t-test)");

    // Example 3: Paired t-test
    println!("\n3. Paired T-Test");
    println!("--------------");

    let before = array![68.5, 70.2, 65.3, 72.1, 69.8];
    let after = array![67.2, 68.5, 66.1, 70.3, 68.7];

    println!("Testing if there's a significant difference between paired measurements:");
    println!("Before: {:?}", before);
    println!("After: {:?}", after);

    // Paired t-test (two-sided)
    let paired =
        enhanced_ttest_rel(&before.view(), &after.view(), Alternative::TwoSided, "omit").unwrap();
    print_ttest_result(&paired, "Paired t-test");

    // Example 4: T-test from statistics
    println!("\n4. T-Test from Descriptive Statistics");
    println!("----------------------------------");

    let mean1 = 5.48;
    let std1 = 0.49;
    let n1 = 5;
    let mean2 = 4.94;
    let std2 = 0.21;
    let n2 = 5;

    println!("Group 1: mean = {}, std = {}, n = {}", mean1, std1, n1);
    println!("Group 2: mean = {}, std = {}, n = {}", mean2, std2, n2);

    // T-test from statistics (two-sided, equal variances)
    let from_stats = ttest_ind_from_stats(
        mean1,
        std1,
        n1,
        mean2,
        std2,
        n2,
        true,
        Alternative::TwoSided,
    )
    .unwrap();
    print_ttest_result(&from_stats, "T-test from statistics");

    // Example 5: Handling NaN values
    println!("\n5. NaN Handling");
    println!("-------------");

    // Create data with a NaN value
    let mut data_with_nan = array![5.1, 4.9, 6.2, 5.7, 5.5];
    data_with_nan[2] = f64::NAN;

    println!("Data with NaN: {:?}", data_with_nan);

    // Omit NaN values
    let omit_result =
        enhanced_ttest_1samp(&data_with_nan.view(), 5.0, Alternative::TwoSided, "omit").unwrap();
    print_ttest_result(&omit_result, "Omit NaN values");

    // Try to report NaN handling error (this will fail, but we'll catch it)
    let raise_result =
        enhanced_ttest_1samp(&data_with_nan.view(), 5.0, Alternative::TwoSided, "raise");
    match raise_result {
        Ok(_) => println!("This should not happen - test should fail with NaN values"),
        Err(e) => println!("Raise NaN error: {}", e),
    }
}

#[allow(dead_code)]
fn print_ttest_result<F: std::fmt::Display + scirs2_core::numeric::Float>(
    result: &TTestResult<F>,
    title: &str,
) {
    let alt_str = match result.alternative {
        Alternative::TwoSided => "different from null",
        Alternative::Greater => "greater than null",
        Alternative::Less => "less than null",
    };

    println!("\n{}", title);
    println!("t-statistic: {:.4}", result.statistic);
    println!("p-value: {:.4}", result.pvalue);
    println!("degrees of freedom: {:.4}", result.df);
    println!("alternative: {}", alt_str);

    if let Some(info) = &result.info {
        println!("additional info: {}", info);
    }

    // Interpret the result
    let threshold = F::from(0.05).unwrap();
    if result.pvalue < threshold {
        println!("Conclusion: Reject the null hypothesis (p < 0.05)");
    } else {
        println!("Conclusion: Fail to reject the null hypothesis (p ≥ 0.05)");
    }
}
