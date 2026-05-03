//! Compile-time check that the `wasm` feature re-exports `scirs2_wasm` correctly.
//!
//! When the `wasm` feature is enabled, `scirs2::wasm` must resolve to `scirs2_wasm`.
//! The real assertion here is compilation — if the `use` statement below fails to
//! compile, the feature wiring is broken.
//!
//! Run without the `wasm` feature (normal CI path):
//!   cargo nextest run -p scirs2
//!
//! Compile-check with the `wasm` feature (wasm32 target or native):
//!   cargo check -p scirs2 --features wasm

// When the `wasm` feature is active, importing the module must succeed.
// The `as _` suppresses unused-import warnings without needing a use site.
#[cfg(feature = "wasm")]
#[allow(unused_imports)]
use scirs2::wasm as _;

#[cfg(feature = "wasm")]
mod wasm_reexport {
    #[test]
    fn can_see_wasm_module() {
        // If this test file compiled, the re-export works.
        // The `use scirs2::wasm as _` at module level above is the real check.
    }
}

#[cfg(not(feature = "wasm"))]
mod no_wasm {
    #[test]
    fn wasm_feature_not_enabled() {
        // Normal CI path — wasm feature disabled; nothing to verify at runtime.
    }
}
