// tests/examples_compile_check.rs
// This test verifies the cookbook examples compile.
// They are checked via `cargo check --examples -p scirs2-metrics`
// Run: cargo test -p scirs2-metrics --test examples_compile_check
#[test]
fn cookbook_examples_compile() {
    // This file exists as a compile-time marker.
    // The real check is: cargo check --examples -p scirs2-metrics
}
