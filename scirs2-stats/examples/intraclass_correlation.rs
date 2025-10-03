use scirs2_core::ndarray::array;
use scirs2_stats::icc;

#[allow(dead_code)]
fn main() {
    // Create sample data
    // Each row is a subject, each column is a rater/measurement
    let data = array![
        [9.0, 10.0, 8.0, 8.5],
        [7.5, 8.0, 7.0, 7.5],
        [6.0, 5.5, 6.5, 6.0],
        [5.0, 5.0, 4.5, 5.5],
        [8.0, 7.5, 8.0, 7.5],
        [7.0, 7.0, 6.5, 6.0],
        [10.0, 9.5, 9.0, 10.0],
        [6.5, 6.0, 6.0, 5.5],
        [4.0, 4.5, 3.5, 4.0],
        [8.5, 8.0, 8.5, 8.0],
    ];

    // Calculate ICC - type 1 (one-way random effects)
    let (icc1, conf_int1) = icc(&data.view(), 1, None).unwrap();
    println!("ICC(1) - One-way random effects model:");
    println!("ICC = {}", icc1);
    println!(
        "95% Confidence interval: [{}, {}]",
        conf_int1[0], conf_int1[1]
    );

    // Calculate ICC - type 2 (two-way random effects)
    let (icc2, conf_int2) = icc(&data.view(), 2, None).unwrap();
    println!("\nICC(2) - Two-way random effects model:");
    println!("ICC = {}", icc2);
    println!(
        "95% Confidence interval: [{}, {}]",
        conf_int2[0], conf_int2[1]
    );

    // Calculate ICC - type 3 (two-way mixed effects)
    let (icc3, conf_int3) = icc(&data.view(), 3, None).unwrap();
    println!("\nICC(3) - Two-way mixed effects model:");
    println!("ICC = {}", icc3);
    println!(
        "95% Confidence interval: [{}, {}]",
        conf_int3[0], conf_int3[1]
    );

    // Data with poor agreement
    let data_poor = array![
        [9.0, 5.0, 7.0, 3.0],
        [7.5, 2.0, 8.0, 5.5],
        [6.0, 8.5, 3.5, 7.0],
        [5.0, 9.0, 6.5, 2.5],
        [8.0, 4.0, 7.0, 9.5],
    ];

    let (icc_poor, conf_int_poor) = icc(&data_poor.view(), 2, None).unwrap();
    println!("\nICC(2) with poor agreement data:");
    println!("ICC = {}", icc_poor);
    println!(
        "95% Confidence interval: [{}, {}]",
        conf_int_poor[0], conf_int_poor[1]
    );
}
