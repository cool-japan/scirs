use scirs2_core::ndarray::array;
use scirs2_stats::mann_whitney;

#[allow(dead_code)]
fn main() {
    // Example data: ages at diagnosis for type II diabetes
    let males = array![19.0, 22.0, 16.0, 29.0, 24.0];
    let females = array![20.0, 11.0, 17.0, 12.0];

    println!("Male ages at diagnosis: {:?}", males);
    println!("Female ages at diagnosis: {:?}", females);
    println!();

    // Perform the Mann-Whitney U test with different parameters

    // Default: two-sided test with continuity correction
    let (u, p_value) = mann_whitney(&males.view(), &females.view(), "two-sided", true).unwrap();
    println!(
        "Two-sided Mann-Whitney U test (with continuity correction):\n  U = {}\n  p-value = {}",
        u, p_value
    );
    println!("  Significant at α = 0.05? {}\n", p_value < 0.05);

    // One-sided test (testing if females are diagnosed at younger ages)
    let (u, p_value) = mann_whitney(&females.view(), &males.view(), "less", true).unwrap();
    println!(
        "One-sided Mann-Whitney U test (females < males):\n  U = {}\n  p-value = {}",
        u, p_value
    );
    println!("  Significant at α = 0.05? {}\n", p_value < 0.05);

    // Without continuity correction
    let (u, p_value) = mann_whitney(&males.view(), &females.view(), "two-sided", false).unwrap();
    println!(
        "Two-sided Mann-Whitney U test (no continuity correction):\n  U = {}\n  p-value = {}",
        u, p_value
    );
    println!("  Significant at α = 0.05? {}", p_value < 0.05);

    // Another one-sided test (testing if males are diagnosed at older ages)
    let (u, p_value) = mann_whitney(&males.view(), &females.view(), "greater", true).unwrap();
    println!(
        "One-sided Mann-Whitney U test (males > females):\n  U = {}\n  p-value = {}",
        u, p_value
    );
    println!("  Significant at α = 0.05? {}", p_value < 0.05);
}
