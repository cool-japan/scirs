use scirs2_cluster::{
    // Affinity Propagation
    affinity_propagation,
    // BIRCH
    birch,
    dbscan_clustering,
    // OPTICS
    density::optics::optics,
    // DBSCAN
    density::{dbscan, DistanceMetric},
    // GMM
    gaussian_mixture,
    // HDBSCAN
    hdbscan,
    // Hierarchical
    hierarchy::{fcluster, linkage, ClusterCriterion, LinkageMethod, Metric},
    // Mean Shift
    meanshift::{mean_shift, MeanShiftOptions},
    // Metrics
    metrics::silhouette_score,
    // Spectral
    spectral_clustering,
    // K-means
    vq::{kmeans2, MinitMethod, MissingMethod},
    AffinityMode,
    AffinityPropagationOptions,
    BirchOptions,
    CovarianceType,
    GMMOptions,
    HDBSCANOptions,
    SpectralClusteringOptions,
};
use scirs2_core::ndarray::{array, Array2};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comprehensive Clustering Algorithm Test");
    println!("======================================\n");

    // Create test dataset with 3 clear clusters
    let data = array![
        // Cluster 1
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.15, 0.15],
        // Cluster 2
        [3.0, 3.0],
        [3.1, 3.1],
        [3.2, 3.0],
        [3.0, 3.2],
        [3.15, 3.15],
        // Cluster 3
        [6.0, 0.0],
        [6.1, 0.1],
        [6.2, 0.0],
        [6.0, 0.2],
        [6.15, 0.15],
    ];

    println!("Test data: 3 clusters with 5 points each\n");

    // 1. K-means
    println!("1. K-means Clustering");
    let (_, kmeans_labels) = kmeans2(
        data.view(),
        3,
        Some(10), // iterations
        None,     // threshold
        Some(MinitMethod::Random),
        Some(MissingMethod::Warn),
        Some(true), // check_finite
        Some(42),   // random_seed
    )?;
    print_results(&kmeans_labels, &data);

    // 2. DBSCAN
    println!("\n2. DBSCAN Clustering");
    let dbscan_labels = dbscan(data.view(), 0.5, 2, Some(DistanceMetric::Euclidean))?;
    print_results(&dbscan_labels.mapv(|x| x as usize), &data);

    // 3. HDBSCAN
    println!("\n3. HDBSCAN Clustering");
    let hdbscan_opts = HDBSCANOptions {
        min_cluster_size: 2,
        minsamples: Some(2),
        ..Default::default()
    };
    let hdbscan_result = hdbscan(data.view(), Some(hdbscan_opts))?;

    // Try DBSCAN extraction if HDBSCAN doesn't find clusters
    if hdbscan_result.labels.iter().all(|&x| x == -1) {
        println!("   HDBSCAN found all noise, extracting DBSCAN with cut_distance=1.0");
        let dbscan_from_hdbscan = dbscan_clustering(&hdbscan_result, 1.0)?;
        print_results(&dbscan_from_hdbscan.mapv(|x| x as usize), &data);
    } else {
        print_results(&hdbscan_result.labels.mapv(|x| x as usize), &data);
    }

    // 4. OPTICS
    println!("\n4. OPTICS Clustering");
    let optics_result = optics(data.view(), 2, None, Some(DistanceMetric::Euclidean))?;
    let optics_labels =
        scirs2_cluster::density::optics::extract_dbscan_clustering(&optics_result, 0.5);
    print_results(&optics_labels.mapv(|x| x as usize), &data);

    // 5. Mean Shift
    println!("\n5. Mean Shift Clustering");
    let ms_opts = MeanShiftOptions {
        bandwidth: Some(1.0),
        ..Default::default()
    };
    let (_, ms_labels) = mean_shift(&data.view(), ms_opts)?;
    print_results(&ms_labels.mapv(|x| x as usize), &data);

    // 6. Gaussian Mixture Model
    println!("\n6. Gaussian Mixture Model");
    let gmm_opts = GMMOptions {
        n_components: 3,
        covariance_type: CovarianceType::Full,
        ..Default::default()
    };
    let gmm_labels = gaussian_mixture(data.view(), gmm_opts)?;
    print_results(&gmm_labels.mapv(|x| x as usize), &data);

    // 7. Hierarchical Clustering
    println!("\n7. Hierarchical Clustering");
    let linkage_result = linkage(data.view(), LinkageMethod::Complete, Metric::Euclidean)?;
    let hier_labels = fcluster(&linkage_result, 3, Some(ClusterCriterion::MaxClust))?;
    print_results(&hier_labels, &data);

    // 8. Spectral Clustering
    println!("\n8. Spectral Clustering");
    let spec_opts = SpectralClusteringOptions {
        affinity: AffinityMode::RBF,
        gamma: 1.0,
        ..Default::default()
    };
    let (_, spec_labels) = spectral_clustering(data.view(), 3, Some(spec_opts))?;
    print_results(&spec_labels, &data);

    // 9. BIRCH
    println!("\n9. BIRCH Clustering");
    let birch_opts = BirchOptions {
        threshold: 1.0,
        n_clusters: Some(3),
        ..Default::default()
    };
    let (_, birch_labels) = birch(data.view(), birch_opts)?;
    print_results(&birch_labels.mapv(|x| x as usize), &data);

    // 10. Affinity Propagation
    println!("\n10. Affinity Propagation");
    let ap_opts = AffinityPropagationOptions {
        damping: 0.7,
        preference: Some(-5.0),
        ..Default::default()
    };
    let (_, ap_labels) = affinity_propagation(data.view(), false, Some(ap_opts))?;
    print_results(&ap_labels.mapv(|x| x as usize), &data);

    Ok(())
}

#[allow(dead_code)]
fn print_results(labels: &scirs2_core::ndarray::Array1<usize>, data: &Array2<f64>) {
    // Count unique _labels
    let mut unique_labels = std::collections::HashSet::new();
    for &label in labels.iter() {
        unique_labels.insert(label);
    }

    let n_clusters = unique_labels.iter().filter(|&&l| l != usize::MAX).count();
    println!("   Clusters found: {}", n_clusters);

    // Calculate silhouette score if we have valid clusters
    if n_clusters > 1 {
        match silhouette_score(data.view(), labels.mapv(|x| x as i32).view()) {
            Ok(score) => println!("   Silhouette score: {:.3}", score),
            Err(_) => println!("   Silhouette score: N/A"),
        }
    }

    // Show label distribution
    let mut label_counts: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for &label in labels.iter() {
        *label_counts.entry(label).or_insert(0) += 1;
    }

    print!("   Label distribution: ");
    let mut sorted_labels: Vec<_> = label_counts.into_iter().collect();
    sorted_labels.sort_by_key(|&(label_, _count)| label_);

    for (i, (label, count)) in sorted_labels.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        if *label == usize::MAX {
            print!("Noise: {}", count);
        } else {
            print!("C{}: {}", label, count);
        }
    }
    println!();
}
