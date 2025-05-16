fn main() {
    // Link pthread library for OpenBLAS static linking
    // This must be BEFORE openblas is linked
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=gfortran");
        println!("cargo:rustc-link-lib=gomp");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=pthread");
    }

    // Force static linking of pthread
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,--start-group");

    // Ensure proper link order
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-search=native=/lib/x86_64-linux-gnu");
}
