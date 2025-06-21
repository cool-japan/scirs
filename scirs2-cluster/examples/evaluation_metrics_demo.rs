use ndarray::{Array1, Array2};
use scirs2_cluster::metrics::{
    adjusted_rand_index, calinski_harabasz_score, davies_bouldin_score,
    homogeneity_completeness_v_measure, normalized_mutual_info, silhouette_score,
};
use scirs2_cluster::vq::{kmeans2, MinitMethod, MissingMethod};

fn main() {
    println!("Clustering Evaluation Metrics Demo");
    println!("{}", "=".repeat(50));

    // Generate synthetic data with known ground truth
    let (data, true_labels) = generate_data_with_ground_truth();
    println!(
        "Generated {} samples with 3 true classes\n",
        data.shape()[0]
    );

    // Test different numbers of clusters
    for k in 2..=5 {
        println!("Clustering with k={}", k);
        println!("{}", "-".repeat(30));

        // Perform clustering
        let (_, pred_labels) = kmeans2(
            data.view(),
            k,
            Some(300),
            None,
            Some(MinitMethod::PlusPlus),
            Some(MissingMethod::Warn),
            Some(true),
            Some(42),
        )
        .unwrap();
        let pred_labels_i32 = pred_labels.mapv(|x| x as i32);

        // Compute all evaluation metrics

        // 1. Silhouette Score (doesn't need ground truth)
        match silhouette_score(data.view(), pred_labels_i32.view()) {
            Ok(score) => println!("  Silhouette Score:      {:.3}", score),
            Err(_) => println!("  Silhouette Score:      N/A"),
        }

        // 2. Davies-Bouldin Score (doesn't need ground truth)
        match davies_bouldin_score(data.view(), pred_labels_i32.view()) {
            Ok(score) => println!("  Davies-Bouldin Score:  {:.3} (lower is better)", score),
            Err(_) => println!("  Davies-Bouldin Score:  N/A"),
        }

        // 3. Calinski-Harabasz Score (doesn't need ground truth)
        match calinski_harabasz_score(data.view(), pred_labels_i32.view()) {
            Ok(score) => println!("  Calinski-Harabasz:     {:.1} (higher is better)", score),
            Err(_) => println!("  Calinski-Harabasz:     N/A"),
        }

        // 4. Adjusted Rand Index (needs ground truth)
        match adjusted_rand_index::<f64>(true_labels.view(), pred_labels_i32.view()) {
            Ok(ari) => println!("  Adjusted Rand Index:   {:.3}", ari),
            Err(_) => println!("  Adjusted Rand Index:   N/A"),
        }

        // 5. Normalized Mutual Information (needs ground truth)
        match normalized_mutual_info::<f64>(
            true_labels.view(),
            pred_labels_i32.view(),
            "arithmetic",
        ) {
            Ok(nmi) => println!("  Normalized MI (arith): {:.3}", nmi),
            Err(_) => println!("  Normalized MI (arith): N/A"),
        }

        // 6. Homogeneity, Completeness, V-measure (needs ground truth)
        match homogeneity_completeness_v_measure::<f64>(true_labels.view(), pred_labels_i32.view())
        {
            Ok((h, c, v)) => {
                println!("  Homogeneity:          {:.3}", h);
                println!("  Completeness:         {:.3}", c);
                println!("  V-measure:            {:.3}", v);
            }
            Err(_) => println!("  H/C/V-measure:        N/A"),
        }

        println!();
    }

    // Demonstrate comparison between different clustering algorithms
    println!("\nComparing Different Clustering Results");
    println!("{}", "=".repeat(50));

    // Create two different clusterings
    let (_, clustering1) = kmeans2(
        data.view(),
        3,
        Some(300),
        None,
        Some(MinitMethod::PlusPlus),
        Some(MissingMethod::Warn),
        Some(true),
        Some(42),
    )
    .unwrap();

    let (_, clustering2) = kmeans2(
        data.view(),
        3,
        Some(300),
        None,
        Some(MinitMethod::PlusPlus),
        Some(MissingMethod::Warn),
        Some(true),
        Some(123),
    )
    .unwrap();

    let clustering1_i32 = clustering1.mapv(|x| x as i32);
    let clustering2_i32 = clustering2.mapv(|x| x as i32);

    println!("Comparing two K-means runs with different random seeds:");

    // Compare the two clusterings
    match adjusted_rand_index::<f64>(clustering1_i32.view(), clustering2_i32.view()) {
        Ok(ari) => println!("  ARI between runs:      {:.3}", ari),
        Err(_) => println!("  ARI between runs:      N/A"),
    }

    match normalized_mutual_info::<f64>(
        clustering1_i32.view(),
        clustering2_i32.view(),
        "arithmetic",
    ) {
        Ok(nmi) => println!("  NMI between runs:      {:.3}", nmi),
        Err(_) => println!("  NMI between runs:      N/A"),
    }

    // Compare with ground truth
    println!("\nComparison with ground truth:");
    match adjusted_rand_index::<f64>(true_labels.view(), clustering1_i32.view()) {
        Ok(ari) => println!("  Run 1 ARI:             {:.3}", ari),
        Err(_) => println!("  Run 1 ARI:             N/A"),
    }

    match adjusted_rand_index::<f64>(true_labels.view(), clustering2_i32.view()) {
        Ok(ari) => println!("  Run 2 ARI:             {:.3}", ari),
        Err(_) => println!("  Run 2 ARI:             N/A"),
    }
}

fn generate_data_with_ground_truth() -> (Array2<f64>, Array1<i32>) {
    let mut data = Vec::new();
    let mut labels = Vec::new();

    // Class 0: centered at (0, 0)
    for _ in 0..30 {
        let x = rand::random::<f64>() * 2.0 - 1.0;
        let y = rand::random::<f64>() * 2.0 - 1.0;
        data.push(x);
        data.push(y);
        labels.push(0);
    }

    // Class 1: centered at (5, 0)
    for _ in 0..25 {
        let x = 5.0 + rand::random::<f64>() * 2.0 - 1.0;
        let y = rand::random::<f64>() * 2.0 - 1.0;
        data.push(x);
        data.push(y);
        labels.push(1);
    }

    // Class 2: centered at (2.5, 4)
    for _ in 0..20 {
        let x = 2.5 + rand::random::<f64>() * 2.0 - 1.0;
        let y = 4.0 + rand::random::<f64>() * 2.0 - 1.0;
        data.push(x);
        data.push(y);
        labels.push(2);
    }

    let data_array = Array2::from_shape_vec((75, 2), data).unwrap();
    let labels_array = Array1::from_vec(labels);

    (data_array, labels_array)
}
