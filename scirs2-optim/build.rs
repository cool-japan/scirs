fn main() {
    // Link to OpenBLAS for doctests
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu/openblas-pthread");
}
