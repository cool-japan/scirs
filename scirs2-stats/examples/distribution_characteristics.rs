use scirs2_core::ndarray::array;
use scirs2_stats::distribution_characteristics::{
    cross_entropy, entropy, kl_divergence, kurtosis_ci, mode, skewness_ci, ModeMethod,
};
use scirs2_stats::skew;

#[allow(dead_code)]
fn main() {
    // Create some sample data
    println!("Distribution Characteristics Example");
    println!("===================================");

    // Multimodal data for mode calculation (using integers since floats don't implement Hash/Eq)
    let multidata = array![1, 2, 2, 3, 3, 4, 5, 5, 6];

    // Find modes
    let unimodal_result = mode(&multidata.view(), ModeMethod::Unimodal).unwrap();
    let multimodal_result = mode(&multidata.view(), ModeMethod::MultiModal).unwrap();

    println!("\n1. Mode Analysis");
    println!("---------------");
    println!("Data: {:?}", multidata);
    println!(
        "Unimodal mode: {:?} (count: {})",
        unimodal_result.values[0], unimodal_result.counts[0]
    );
    print!("Multimodal modes: ");
    for (i, (&val, &count)) in multimodal_result
        .values
        .iter()
        .zip(multimodal_result.counts.iter())
        .enumerate()
    {
        if i > 0 {
            print!(", ");
        }
        print!("{} (count: {})", val, count);
    }
    println!();

    // Entropy calculation
    println!("\n2. Entropy Analysis");
    println!("-----------------");

    // Uniform distribution (maximum entropy)
    let uniform = array![1, 2, 3, 4, 5, 6];
    // Less uniform distribution (lower entropy)
    let less_uniform = array![1, 1, 1, 2, 3, 4];
    // Single value (zero entropy)
    let single_value = array![1, 1, 1, 1, 1];

    let entropy_uniform = entropy(&uniform.view(), Some(2.0)).unwrap();
    let entropy_less = entropy(&less_uniform.view(), Some(2.0)).unwrap();
    let entropy_single = entropy(&single_value.view(), Some(2.0)).unwrap();

    println!("Uniform data: {:?}", uniform);
    println!("Entropy (base 2): {:.6}", entropy_uniform);
    println!("Less uniform data: {:?}", less_uniform);
    println!("Entropy (base 2): {:.6}", entropy_less);
    println!("Single value data: {:?}", single_value);
    println!("Entropy (base 2): {:.6}", entropy_single);

    // KL divergence and cross-entropy examples
    println!("\n3. Probability Distribution Comparisons");
    println!("------------------------------------");

    // Create two probability distributions
    let p = array![0.5f64, 0.5];
    let q = array![0.9f64, 0.1];

    let kl_div = kl_divergence(&p.view(), &q.view()).unwrap();
    let kl_div_reverse = kl_divergence(&q.view(), &p.view()).unwrap();
    let cross_ent = cross_entropy(&p.view(), &q.view()).unwrap();

    println!("Distribution P: {:?}", p);
    println!("Distribution Q: {:?}", q);
    println!("KL divergence (P||Q): {:.6}", kl_div);
    println!("KL divergence (Q||P): {:.6}", kl_div_reverse);
    println!("Cross entropy H(P,Q): {:.6}", cross_ent);

    // Skewness and kurtosis with confidence intervals
    println!("\n4. Distribution Shape with Confidence Intervals");
    println!("-------------------------------------------");

    // Symmetric data
    let symmetric = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
    // Right-skewed data
    let right_skewed = array![1.0f64, 2.0, 2.5, 3.0, 10.0];
    // Left-skewed data
    let left_skewed = array![0.0f64, 7.0, 7.5, 8.0, 9.0];

    // Calculate direct skewness values
    let _skew_sym = skew(&symmetric.view(), false, None).unwrap();
    let _skew_right = skew(&right_skewed.view(), false, None).unwrap();
    let _skew_left = skew(&left_skewed.view(), false, None).unwrap();

    // Calculate skewness with confidence intervals
    let skew_ci_sym = skewness_ci(&symmetric.view(), false, None, Some(1000), Some(42)).unwrap();
    let skew_ci_right =
        skewness_ci(&right_skewed.view(), false, None, Some(1000), Some(42)).unwrap();
    let skew_ci_left = skewness_ci(&left_skewed.view(), false, None, Some(1000), Some(42)).unwrap();

    println!("Symmetric data: {:?}", symmetric);
    println!(
        "Skewness: {:.4} (95% CI: {:.4}, {:.4})",
        skew_ci_sym.estimate, skew_ci_sym.lower, skew_ci_sym.upper
    );

    println!("Right-skewed data: {:?}", right_skewed);
    println!(
        "Skewness: {:.4} (95% CI: {:.4}, {:.4})",
        skew_ci_right.estimate, skew_ci_right.lower, skew_ci_right.upper
    );

    println!("Left-skewed data: {:?}", left_skewed);
    println!(
        "Skewness: {:.4} (95% CI: {:.4}, {:.4})",
        skew_ci_left.estimate, skew_ci_left.lower, skew_ci_left.upper
    );

    // Kurtosis examples
    println!("\n5. Kurtosis Analysis");
    println!("-----------------");

    // Uniform-like distribution (platykurtic - negative excess kurtosis)
    let platykurtic = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    // Normal-like distribution (mesokurtic - near zero excess kurtosis)
    let mesokurtic = array![1.0f64, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0];
    // Peaked distribution (leptokurtic - positive excess kurtosis)
    let leptokurtic = array![3.0f64, 3.2, 3.4, 3.4, 3.5, 3.5, 3.6, 3.6, 3.8, 10.0];

    // Calculate kurtosis with confidence intervals
    let kurt_ci_platy =
        kurtosis_ci(&platykurtic.view(), true, false, None, Some(1000), Some(42)).unwrap();
    let kurt_ci_meso =
        kurtosis_ci(&mesokurtic.view(), true, false, None, Some(1000), Some(42)).unwrap();
    let kurt_ci_lepto =
        kurtosis_ci(&leptokurtic.view(), true, false, None, Some(1000), Some(42)).unwrap();

    println!("Platykurtic data (uniform-like, negative excess kurtosis):");
    println!(
        "Kurtosis: {:.4} (95% CI: {:.4}, {:.4})",
        kurt_ci_platy.estimate, kurt_ci_platy.lower, kurt_ci_platy.upper
    );

    println!("Mesokurtic data (normal-like, zero excess kurtosis):");
    println!(
        "Kurtosis: {:.4} (95% CI: {:.4}, {:.4})",
        kurt_ci_meso.estimate, kurt_ci_meso.lower, kurt_ci_meso.upper
    );

    println!("Leptokurtic data (peaked, positive excess kurtosis):");
    println!(
        "Kurtosis: {:.4} (95% CI: {:.4}, {:.4})",
        kurt_ci_lepto.estimate, kurt_ci_lepto.lower, kurt_ci_lepto.upper
    );
}
