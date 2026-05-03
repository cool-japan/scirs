//! Cookbook: Ranking Metrics
//!
//! This example demonstrates MAP, NDCG, MRR on synthetic ranking data.
//! Ranking metrics evaluate how well a system orders items by relevance.
//!
//! Run with: `cargo run --example cookbook_ranking -p scirs2-metrics`

use scirs2_core::ndarray::Array1;
use scirs2_metrics::ranking::{
    mean_average_precision, mean_reciprocal_rank, ndcg_score, precision_at_k, recall_at_k,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Cookbook: Ranking Metrics ===\n");

    // ─── 1. Synthetic ranking data ───────────────────────────────────────────
    // We simulate 4 queries to a search engine.
    // Each query has 8 candidate documents.
    // y_true[i][j] = 1.0 if document j is relevant for query i, 0.0 otherwise.
    // y_score[i][j] = model's relevance score for document j, query i.

    // Query 0: relevant docs at positions 0, 3 (0-indexed)
    // Query 1: relevant doc at position 2
    // Query 2: relevant docs at positions 1, 5
    // Query 3: relevant doc at position 7 (hardest — buried at the end)

    let y_true_vecs: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ];

    // Good ranker: high scores for relevant docs
    let y_score_good_vecs: Vec<Vec<f64>> = vec![
        vec![0.90, 0.20, 0.30, 0.85, 0.10, 0.15, 0.05, 0.25],
        vec![0.10, 0.30, 0.92, 0.20, 0.15, 0.10, 0.40, 0.05],
        vec![0.20, 0.88, 0.30, 0.10, 0.20, 0.82, 0.05, 0.15],
        vec![0.10, 0.20, 0.30, 0.15, 0.25, 0.20, 0.10, 0.80],
    ];

    // Mediocre ranker: relevant docs not always at the top
    let y_score_bad_vecs: Vec<Vec<f64>> = vec![
        vec![0.40, 0.50, 0.60, 0.35, 0.70, 0.80, 0.90, 0.20],
        vec![0.90, 0.70, 0.55, 0.80, 0.60, 0.40, 0.30, 0.20],
        vec![0.70, 0.40, 0.80, 0.90, 0.50, 0.30, 0.60, 0.20],
        vec![0.60, 0.70, 0.80, 0.90, 0.50, 0.40, 0.30, 0.10],
    ];

    let y_true: Vec<Array1<f64>> = y_true_vecs
        .iter()
        .map(|v| Array1::from(v.clone()))
        .collect();
    let y_score_good: Vec<Array1<f64>> = y_score_good_vecs
        .iter()
        .map(|v| Array1::from(v.clone()))
        .collect();
    let y_score_bad: Vec<Array1<f64>> = y_score_bad_vecs
        .iter()
        .map(|v| Array1::from(v.clone()))
        .collect();

    // ─── 2. Compute ranking metrics ──────────────────────────────────────────
    let map_good = mean_average_precision(&y_true, &y_score_good, None)?;
    let map_bad = mean_average_precision(&y_true, &y_score_bad, None)?;

    let map_k3_good = mean_average_precision(&y_true, &y_score_good, Some(3))?;
    let map_k3_bad = mean_average_precision(&y_true, &y_score_bad, Some(3))?;

    let ndcg_good = ndcg_score(&y_true, &y_score_good, None)?;
    let ndcg_bad = ndcg_score(&y_true, &y_score_bad, None)?;

    let mrr_good = mean_reciprocal_rank(&y_true, &y_score_good)?;
    let mrr_bad = mean_reciprocal_rank(&y_true, &y_score_bad)?;

    let p_at_3_good = precision_at_k(&y_true, &y_score_good, 3)?;
    let p_at_3_bad = precision_at_k(&y_true, &y_score_bad, 3)?;

    let r_at_5_good = recall_at_k(&y_true, &y_score_good, 5)?;
    let r_at_5_bad = recall_at_k(&y_true, &y_score_bad, 5)?;

    // ─── 3. Print results ────────────────────────────────────────────────────
    println!("Dataset: 4 queries, 8 candidates each");
    println!("Good ranker: relevant docs ranked near top");
    println!("Bad ranker:  relevant docs buried or out of order\n");

    println!(
        "{:<30} {:>12}  {:>12}",
        "Metric", "Good ranker", "Bad ranker"
    );
    println!("{}", "-".repeat(58));

    println!(
        "{:<30} {:>12.4}  {:>12.4}",
        "MAP (all k)", map_good, map_bad
    );
    println!(
        "{:<30} {:>12.4}  {:>12.4}",
        "MAP@3", map_k3_good, map_k3_bad
    );
    println!(
        "{:<30} {:>12.4}  {:>12.4}",
        "NDCG (all k)", ndcg_good, ndcg_bad
    );
    println!("{:<30} {:>12.4}  {:>12.4}", "MRR", mrr_good, mrr_bad);
    println!(
        "{:<30} {:>12.4}  {:>12.4}",
        "Precision@3", p_at_3_good, p_at_3_bad
    );
    println!(
        "{:<30} {:>12.4}  {:>12.4}",
        "Recall@5", r_at_5_good, r_at_5_bad
    );

    println!("\n--- When to use which metric ---");
    println!("  MAP       : multi-relevant results per query; averages over all thresholds");
    println!("  NDCG      : graded relevance (0/1/2/3) and position-discounted gain");
    println!("  MRR       : when only the first relevant result matters (Q&A, one-shot)");
    println!("  Precision@k : what fraction of top-k are relevant? (precision-oriented)");
    println!("  Recall@k    : what fraction of all relevant docs appear in top-k?");

    println!("\n=== Done ===");
    Ok(())
}
