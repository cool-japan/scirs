fn main() {
    // Platform-specific BLAS/LAPACK linking
    #[cfg(target_os = "linux")]
    {
        // Link against system OpenBLAS on Linux
        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/openblas-pthread");
    }

    #[cfg(target_os = "macos")]
    {
        // macOS uses Accelerate framework - handled by .cargo/config.toml
        // No explicit linking needed here
    }
}
