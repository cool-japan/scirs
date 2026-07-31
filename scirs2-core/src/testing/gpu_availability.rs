// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Shared helpers for GPU/wgpu smoke tests that must run correctly whether or
//! not a real GPU adapter is present in the environment.
//!
//! Several `*_gpu_smoke.rs` test files across the workspace attempt a real GPU
//! operation and gracefully skip only when the failure is attributable to
//! adapter/hardware unavailability, panicking on anything else. Before this
//! module existed, each file re-implemented the same fragile
//! `msg.contains("adapter") || msg.contains("Adapter") || ...` check
//! independently. Consolidating it here means the pattern list only needs
//! updating in one place, and every skip goes through the same
//! [`print_gpu_skip`] call, so a CI log can be grepped for the exact marker
//! (`[GPU-UNAVAILABLE-SKIP]`) to distinguish "skipped: no GPU" from "passed"
//! without needing to inspect each test's own ad hoc wording.

/// Returns `true` if `error_message` indicates the failure was caused by
/// GPU/wgpu adapter or hardware unavailability rather than a real bug.
#[must_use]
pub fn is_gpu_unavailable_error(error_message: &str) -> bool {
    const PATTERNS: &[&str] = &["adapter", "Adapter", "GPU", "no suitable", "wgpu"];
    PATTERNS.iter().any(|p| error_message.contains(p))
}

/// Print a standardized, greppable message for a GPU smoke test that skipped
/// because no adapter/hardware was available. Callers should `return`
/// immediately afterward — this function does not skip anything itself, it
/// only standardizes the visible signal.
pub fn print_gpu_skip(test_name: &str, error_message: &str) {
    println!("[GPU-UNAVAILABLE-SKIP] {test_name}: {error_message}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_adapter_unavailable_messages() {
        assert!(is_gpu_unavailable_error(
            "wgpu: no suitable Adapter found for this backend"
        ));
        assert!(is_gpu_unavailable_error("failed to request GPU adapter"));
        assert!(is_gpu_unavailable_error("no wgpu adapter available"));
    }

    #[test]
    fn does_not_misclassify_unrelated_errors() {
        assert!(!is_gpu_unavailable_error("shape mismatch: expected [3, 3]"));
        assert!(!is_gpu_unavailable_error("division by zero"));
    }
}
