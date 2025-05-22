/// Example demonstrating the use of external clustering metrics
/// to evaluate clustering algorithm performance.
use ndarray::{array, Array2};
use scirs2_core::error::{CoreError, CoreResult, ErrorContext};
use scirs2_metrics::clustering::{
    adjusted_mutual_info_score, adjusted_rand_index, fowlkes_mallows_score,
    homogeneity_completeness_v_measure, normalized_mutual_info_score,
};

fn main() -> CoreResult<()> {
    // Create a sample dataset with simulated clusters
    let _data = Array2::from_shape_vec(
        (20, 2),
        vec![
            // Cluster 1
            1.0, 2.0, 1.2, 1.8, 1.5, 2.1, 1.3, 2.3, 1.0, 2.5, // Cluster 2
            5.0, 6.0, 5.2, 6.2, 5.5, 5.8, 4.8, 6.3, 5.1, 5.9, // Cluster 3
            9.0, 2.0, 9.2, 2.2, 8.8, 1.8, 9.3, 2.4, 8.7, 2.3, // Cluster 4
            3.0, 8.0, 2.8, 8.2, 3.2, 7.8, 2.5, 8.5, 3.5, 7.5,
        ],
    )
    .unwrap();

    // Ground truth labels (4 clusters)
    let true_labels = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3];

    println!("=== External Clustering Metrics Example ===\n");
    println!("This example demonstrates evaluating clustering results using external metrics");
    println!("by comparing different clustering results to known ground truth labels.\n");

    // Perfect clustering (identical to ground truth)
    let perfect_clustering = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3];

    // Clustering with different labels but same structure
    let permuted_clustering = array![3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1];

    // Imperfect clustering - combining two clusters and splitting another
    let imperfect_clustering = array![
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // Combined cluster 2 into cluster 1
        2, 2, 3, 3, 3 // Split cluster 3 into two clusters
    ];

    // Random clustering (poor performance)
    let random_clustering = array![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];

    // Evaluate each clustering using all metrics
    println!("Scenario 1: Perfect clustering (identical to ground truth)");
    evaluate_clustering(&true_labels, &perfect_clustering)?;

    println!("\nScenario 2: Perfect clustering with permuted labels");
    evaluate_clustering(&true_labels, &permuted_clustering)?;

    println!("\nScenario 3: Imperfect clustering (merges and splits)");
    evaluate_clustering(&true_labels, &imperfect_clustering)?;

    println!("\nScenario 4: Random clustering");
    evaluate_clustering(&true_labels, &random_clustering)?;

    Ok(())
}

fn evaluate_clustering<T, U, S1, S2, D1, D2>(
    true_labels: &ndarray::ArrayBase<S1, D1>,
    pred_labels: &ndarray::ArrayBase<S2, D2>,
) -> CoreResult<()>
where
    T: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    U: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    S1: ndarray::Data<Elem = T>,
    S2: ndarray::Data<Elem = U>,
    D1: ndarray::Dimension,
    D2: ndarray::Dimension,
{
    // Calculate all metrics
    let ari = adjusted_rand_index(true_labels, pred_labels)
        .map_err(|e| CoreError::InvalidArgument(ErrorContext::new(e.to_string())))?;

    let nmi_arithmetic = normalized_mutual_info_score(true_labels, pred_labels, "arithmetic")
        .map_err(|e| CoreError::InvalidArgument(ErrorContext::new(e.to_string())))?;
    let nmi_geometric = normalized_mutual_info_score(true_labels, pred_labels, "geometric")
        .map_err(|e| CoreError::InvalidArgument(ErrorContext::new(e.to_string())))?;
    let nmi_min = normalized_mutual_info_score(true_labels, pred_labels, "min")
        .map_err(|e| CoreError::InvalidArgument(ErrorContext::new(e.to_string())))?;
    let nmi_max = normalized_mutual_info_score(true_labels, pred_labels, "max")
        .map_err(|e| CoreError::InvalidArgument(ErrorContext::new(e.to_string())))?;

    let ami = adjusted_mutual_info_score(true_labels, pred_labels, "arithmetic")
        .map_err(|e| CoreError::InvalidArgument(ErrorContext::new(e.to_string())))?;

    let (homogeneity, completeness, v_measure) =
        homogeneity_completeness_v_measure(true_labels, pred_labels, 1.0)
            .map_err(|e| CoreError::InvalidArgument(ErrorContext::new(e.to_string())))?;

    let fmi = fowlkes_mallows_score(true_labels, pred_labels)
        .map_err(|e| CoreError::InvalidArgument(ErrorContext::new(e.to_string())))?;

    // Print results
    println!("  Adjusted Rand Index: {:.4}", ari);

    println!("  Normalized Mutual Information:");
    println!("    - Arithmetic: {:.4}", nmi_arithmetic);
    println!("    - Geometric:  {:.4}", nmi_geometric);
    println!("    - Min:        {:.4}", nmi_min);
    println!("    - Max:        {:.4}", nmi_max);

    println!("  Adjusted Mutual Information: {:.4}", ami);

    println!("  Homogeneity: {:.4}", homogeneity);
    println!("  Completeness: {:.4}", completeness);
    println!("  V-measure: {:.4}", v_measure);

    println!("  Fowlkes-Mallows Index: {:.4}", fmi);

    Ok(())
}
