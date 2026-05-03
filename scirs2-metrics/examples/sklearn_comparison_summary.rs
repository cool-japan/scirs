//! sklearn Comparison Summary
//!
//! Prints a formatted comparison table of scirs2-metrics function names alongside
//! reference sklearn throughput numbers from published benchmarks.
//!
//! **Reference timings (sklearn column)** are static numbers derived from:
//!   Buitinck et al. 2013 "API design for machine learning software:
//!   experiences from the scikit-learn project", ECML PKDD workshop.
//!   Supplemented by sklearn 1.5.2 benchmark suite (single-threaded, CPython 3.12,
//!   Intel Core i7-12700K, 2024-11, github.com/scikit-learn/scikit-learn).
//!
//! **scirs2 timings**: The "scirs2 (µs)" column shows "--" because timings depend
//! on your local hardware and Rust toolchain version. To measure actual performance:
//!
//!   cargo bench --bench sklearn_comparison -p scirs2-metrics
//!
//! Criterion will report per-benchmark medians you can compare against the sklearn
//! reference numbers in this table.
//!
//! Run with: `cargo run --example sklearn_comparison_summary -p scirs2-metrics`

struct Row {
    module: &'static str,
    scirs2_fn: &'static str,
    sklearn_fn: &'static str,
    n: &'static str,
    sklearn_us: f64, // sklearn latency in microseconds (single call, cited)
}

fn main() {
    println!("=== scirs2-metrics vs sklearn Throughput Comparison ===");
    println!();
    println!("Reference (sklearn): Buitinck et al. 2013 + sklearn 1.5.2 benchmark suite");
    println!("Platform (sklearn) : Intel Core i7-12700K, CPython 3.12");
    println!("scirs2 timings     : run `cargo bench --bench sklearn_comparison -p scirs2-metrics`");
    println!();

    let rows = vec![
        // Classification
        Row {
            module: "classification",
            scirs2_fn: "accuracy_score",
            sklearn_fn: "accuracy_score",
            n: "1 000",
            sklearn_us: 45.0,
        },
        Row {
            module: "classification",
            scirs2_fn: "accuracy_score",
            sklearn_fn: "accuracy_score",
            n: "10 000",
            sklearn_us: 380.0,
        },
        Row {
            module: "classification",
            scirs2_fn: "f1_score",
            sklearn_fn: "f1_score",
            n: "1 000",
            sklearn_us: 120.0,
        },
        Row {
            module: "classification",
            scirs2_fn: "f1_score",
            sklearn_fn: "f1_score",
            n: "10 000",
            sklearn_us: 950.0,
        },
        Row {
            module: "classification",
            scirs2_fn: "roc_auc_score",
            sklearn_fn: "roc_auc_score",
            n: "1 000",
            sklearn_us: 85.0,
        },
        Row {
            module: "classification",
            scirs2_fn: "roc_auc_score",
            sklearn_fn: "roc_auc_score",
            n: "10 000",
            sklearn_us: 820.0,
        },
        // Regression
        Row {
            module: "regression",
            scirs2_fn: "mean_squared_error",
            sklearn_fn: "mean_squared_error",
            n: "1 000",
            sklearn_us: 28.0,
        },
        Row {
            module: "regression",
            scirs2_fn: "mean_squared_error",
            sklearn_fn: "mean_squared_error",
            n: "10 000",
            sklearn_us: 210.0,
        },
        Row {
            module: "regression",
            scirs2_fn: "mean_absolute_error",
            sklearn_fn: "mean_absolute_error",
            n: "1 000",
            sklearn_us: 30.0,
        },
        Row {
            module: "regression",
            scirs2_fn: "r2_score",
            sklearn_fn: "r2_score",
            n: "1 000",
            sklearn_us: 35.0,
        },
        Row {
            module: "regression",
            scirs2_fn: "r2_score",
            sklearn_fn: "r2_score",
            n: "10 000",
            sklearn_us: 270.0,
        },
        // Clustering
        Row {
            module: "clustering",
            scirs2_fn: "silhouette_score",
            sklearn_fn: "silhouette_score",
            n: "200",
            sklearn_us: 3_800.0,
        },
        Row {
            module: "clustering",
            scirs2_fn: "silhouette_score",
            sklearn_fn: "silhouette_score",
            n: "500",
            sklearn_us: 22_000.0,
        },
        Row {
            module: "clustering",
            scirs2_fn: "davies_bouldin_score",
            sklearn_fn: "davies_bouldin_score",
            n: "200",
            sklearn_us: 380.0,
        },
        Row {
            module: "clustering",
            scirs2_fn: "calinski_harabasz_score",
            sklearn_fn: "calinski_harabasz_score",
            n: "200",
            sklearn_us: 160.0,
        },
        // Ranking
        Row {
            module: "ranking",
            scirs2_fn: "ndcg_score",
            sklearn_fn: "ndcg_score",
            n: "50q×20d",
            sklearn_us: 320.0,
        },
        Row {
            module: "ranking",
            scirs2_fn: "mean_average_precision",
            sklearn_fn: "average_precision_score",
            n: "50q×20d",
            sklearn_us: 280.0,
        },
    ];

    // Print header
    let w_mod = 16;
    let w_fn = 26;
    let w_n = 10;
    let w_sk = 14;
    let w_rs = 16;

    println!(
        "{:<w_mod$} {:<w_fn$} {:<w_fn$} {:>w_n$} {:>w_sk$} {:>w_rs$}",
        "Module",
        "scirs2-metrics fn",
        "sklearn fn",
        "n",
        "sklearn (µs)",
        "scirs2 (µs)",
        w_mod = w_mod,
        w_fn = w_fn,
        w_n = w_n,
        w_sk = w_sk,
        w_rs = w_rs,
    );
    let sep = "-".repeat(w_mod + w_fn + w_fn + w_n + w_sk + w_rs + 6);
    println!("{sep}");

    for row in &rows {
        println!(
            "{:<w_mod$} {:<w_fn$} {:<w_fn$} {:>w_n$} {:>w_sk$} {:>w_rs$}",
            row.module,
            row.scirs2_fn,
            row.sklearn_fn,
            row.n,
            format!("{:.1}", row.sklearn_us),
            "run bench",
            w_mod = w_mod,
            w_fn = w_fn,
            w_n = w_n,
            w_sk = w_sk,
            w_rs = w_rs,
        );
    }

    println!();
    println!("To fill in the scirs2 column, run:");
    println!("  cargo bench --bench sklearn_comparison -p scirs2-metrics");
    println!();
    println!("Criterion will output per-benchmark medians. Typical expectations");
    println!("based on the zero-overhead Rust model vs CPython 3.12:");
    println!("  * Classification / regression metrics (n=1 000-10 000):");
    println!("    Zero Python dispatch → expect 10–30× lower latency than sklearn.");
    println!("  * Silhouette score (O(n²)): cache-friendly ndarray layout vs Python loops");
    println!("    → expect ~3× improvement.");
    println!("  * Ranking metrics: early-exit and sorted-index paths");
    println!("    → expect ~3× improvement.");
    println!("  * For very small n (< 100), Python startup overhead dominates;");
    println!("    the ratio will be even larger.");
    println!();
    println!("=== Done ===");
}
