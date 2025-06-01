fn main() {
    // Link with BLAS libraries for ndarray operations
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=blas");
        println!("cargo:rustc-link-lib=openblas");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    } else if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=blas");
    }
}
