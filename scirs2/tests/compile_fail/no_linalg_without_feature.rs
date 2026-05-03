// This file must NOT compile: `scirs2_linalg` is not re-exported as a top-level
// item from the `scirs2` facade crate. Only `scirs2::linalg` (behind the `linalg`
// feature) is part of the public API. Attempting to reference `scirs2::scirs2_linalg`
// is always a compile error regardless of features.
fn main() {
    // `scirs2_linalg` is not a module path inside the `scirs2` facade.
    let _ = scirs2::scirs2_linalg::LinalgError;
}
