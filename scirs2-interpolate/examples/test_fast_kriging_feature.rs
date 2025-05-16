fn main() {
    println!("Successfully imported fast_kriging module!");

    // Check if the linalg feature is enabled
    #[cfg(feature = "linalg")]
    println!("linalg feature is enabled - full functionality available");

    #[cfg(not(feature = "linalg"))]
    println!("linalg feature is not enabled - using fallback implementations");
}
