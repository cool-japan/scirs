use std::env;

fn main() {
    // Skip clippy checks for dependencies to avoid build failures
    if let Ok(val) = env::var("CARGO_PRIMARY_PACKAGE") {
        if val == "1" {
            // This is the primary package, run clippy
        } else {
            // This is a dependency, skip clippy
            println!("cargo:rustc-env=CLIPPY_DISABLE=1");
        }
    }
}
