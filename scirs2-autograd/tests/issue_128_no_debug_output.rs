//! Regression test for issue #128: scirs2-autograd must not emit stray
//! debug `println!`/`eprintln!` on optimizer hot paths.
//!
//! The Adam/AdamW optimizer `compute` methods previously contained
//! unconditional `eprintln!` calls that flooded stderr on every training
//! step. This test statically guards the hot-path source files against any
//! reintroduced stdout/stderr write macros by scanning their source at
//! compile time via `include_str!`.

/// (display path, embedded source) pairs for the optimizer hot-path files.
const HOT_PATH_SOURCES: &[(&str, &str)] = &[
    (
        "src/tensor_ops/gradient_descent_ops/adam.rs",
        include_str!("../src/tensor_ops/gradient_descent_ops/adam.rs"),
    ),
    (
        "src/tensor_ops/gradient_descent_ops/adamw.rs",
        include_str!("../src/tensor_ops/gradient_descent_ops/adamw.rs"),
    ),
    (
        "src/optimizers/adam.rs",
        include_str!("../src/optimizers/adam.rs"),
    ),
    (
        "src/optimizers/adamw.rs",
        include_str!("../src/optimizers/adamw.rs"),
    ),
];

#[test]
fn test_issue_128_no_debug_prints_on_optimizer_hot_paths() {
    let macros = ["println!", "eprintln!", "print!", "eprint!"];
    for (path, src) in HOT_PATH_SOURCES {
        for (idx, raw) in src.lines().enumerate() {
            let line = raw.trim_start();
            // Skip comment and doc-comment lines (`//`, `///`, `//!`).
            if line.starts_with("//") {
                continue;
            }
            // Stop scanning at the test module: prints there are allowed.
            if line.starts_with("#[cfg(test)]") {
                break;
            }
            for m in macros {
                assert!(
                    !line.contains(m),
                    "issue #128 regression: stray `{m}` at {path}:{} \u{2014} \
                     library hot paths must not write to stdout/stderr",
                    idx + 1
                );
            }
        }
    }
}
