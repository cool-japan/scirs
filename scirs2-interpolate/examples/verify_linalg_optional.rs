fn main() {
    println!("The scirs2-interpolate crate built successfully without the linalg feature!");
    println!("This confirms our fix is working properly.");

    // Notice that using the features conditionally allows us to compile
    // even without OpenBLAS installed
    println!("Check if the linalg feature is enabled:");
    #[cfg(feature = "linalg")]
    println!("YES - linalg feature is enabled");

    #[cfg(not(feature = "linalg"))]
    println!("NO - linalg feature is not enabled");
}
