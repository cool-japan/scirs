//! Complex pattern generators demonstration
//!
//! This example demonstrates the use of advanced pattern generators for creating
//! non-linear datasets, complex clustering patterns, and hierarchical structures.

use scirs2_datasets::{
    make_anisotropic_blobs, make_circles, make_hierarchical_clusters, make_moons, make_spirals,
    make_swiss_roll,
};

fn main() {
    println!("=== Complex Pattern Generators Demonstration ===\n");

    // Demonstrate non-linear pattern generators
    println!("=== Non-Linear Pattern Generators ================");

    // Spiral dataset
    println!("1. Spiral Patterns:");
    let spirals = make_spirals(200, 3, 0.1, Some(42)).unwrap();
    println!(
        "   Generated {} spirals with {} samples",
        3,
        spirals.n_samples()
    );
    print_dataset_summary(&spirals, "Spirals");

    // Two moons dataset
    println!("\n2. Two Moons Pattern:");
    let moons = make_moons(300, 0.05, Some(42)).unwrap();
    print_dataset_summary(&moons, "Moons");

    // Concentric circles
    println!("\n3. Concentric Circles:");
    let circles = make_circles(250, 0.4, 0.03, Some(42)).unwrap();
    print_dataset_summary(&circles, "Circles");

    // Swiss roll manifold
    println!("\n4. Swiss Roll Manifold:");
    let swiss_roll = make_swiss_roll(400, 0.1, Some(42)).unwrap();
    print_dataset_summary(&swiss_roll, "Swiss Roll");
    println!("   3D manifold with intrinsic 2D structure");
    println!();

    // Demonstrate complex clustering patterns
    println!("=== Complex Clustering Patterns ==================");

    // Anisotropic (elongated) clusters
    println!("1. Anisotropic Clusters:");
    let aniso_clusters = make_anisotropic_blobs(300, 2, 4, 1.0, 5.0, Some(42)).unwrap();
    print_dataset_summary(&aniso_clusters, "Anisotropic Clusters");
    println!("   Elongated clusters with anisotropy factor 5.0");

    // Different anisotropy factors demonstration
    println!("\n   Anisotropy Factor Comparison:");
    for factor in [1.0, 2.0, 5.0, 10.0] {
        let _dataset = make_anisotropic_blobs(100, 2, 3, 1.0, factor, Some(42)).unwrap();
        println!("     Factor {:.1}: {} clusters", factor, 3);
    }

    // Hierarchical clusters
    println!("\n2. Hierarchical Clusters:");
    let hierarchical = make_hierarchical_clusters(240, 3, 3, 4, 3.0, 0.8, Some(42)).unwrap();
    print_dataset_summary(&hierarchical, "Hierarchical Clusters");
    println!("   3 main clusters, each with 4 sub-clusters");

    if let Some(_metadata) = hierarchical.metadata.get("sub_cluster_labels") {
        println!("   Sub-cluster structure preserved in metadata");
    }
    println!();

    // Demonstrate parameter effects
    println!("=== Parameter Effects Demonstration ==============");

    // Noise effect on spirals
    println!("1. Noise Effect on Spirals:");
    for noise in [0.0, 0.05, 0.1, 0.2] {
        let _spiral_data = make_spirals(100, 2, noise, Some(42)).unwrap();
        println!(
            "   Noise {:.2}: Clean separation = {}",
            noise,
            if noise < 0.1 { "High" } else { "Low" }
        );
    }

    // Factor effect on circles
    println!("\n2. Factor Effect on Concentric Circles:");
    for factor in [0.2, 0.4, 0.6, 0.8] {
        let _circle_data = make_circles(100, factor, 0.05, Some(42)).unwrap();
        println!("   Factor {:.1}: Inner/Outer ratio = {:.1}", factor, factor);
    }

    // Cluster complexity in hierarchical patterns
    println!("\n3. Hierarchical Cluster Complexity:");
    for (main, sub) in [(2, 2), (2, 4), (3, 3), (4, 2)] {
        let _hier_data = make_hierarchical_clusters(120, 2, main, sub, 2.0, 0.5, Some(42)).unwrap();
        println!(
            "   {} main × {} sub = {} total clusters",
            main,
            sub,
            main * sub
        );
    }
    println!();

    // Demonstrate use cases
    println!("=== Use Cases and Applications ====================");

    println!("1. **Non-linear Classification Testing**:");
    println!("   - Spirals: Test algorithms that can handle multiple interleaved classes");
    println!("   - Moons: Classic benchmark for non-linear separability");
    println!("   - Circles: Test radial basis function methods");

    println!("\n2. **Dimensionality Reduction Evaluation**:");
    println!("   - Swiss Roll: Test manifold learning algorithms (t-SNE, UMAP, Isomap)");
    println!("   - Preserves intrinsic 2D structure in 3D space");

    println!("\n3. **Clustering Algorithm Testing**:");
    println!("   - Anisotropic: Test algorithms robust to cluster shape variations");
    println!("   - Hierarchical: Test multi-level clustering methods");

    println!("\n4. **Robustness Testing**:");
    println!("   - Variable noise levels test algorithm stability");
    println!("   - Different cluster properties test generalization");
    println!();

    // Demonstrate advanced configurations
    println!("=== Advanced Configuration Examples ===============");

    println!("1. Multi-scale Spiral (Large dataset):");
    let large_spirals = make_spirals(2000, 4, 0.08, Some(42)).unwrap();
    print_dataset_summary(&large_spirals, "Large Spirals");

    println!("\n2. High-dimensional Anisotropic Clusters:");
    let hd_aniso = make_anisotropic_blobs(500, 10, 5, 1.5, 8.0, Some(42)).unwrap();
    print_dataset_summary(&hd_aniso, "High-D Anisotropic");

    println!("\n3. Deep Hierarchical Structure:");
    let deep_hier = make_hierarchical_clusters(300, 4, 2, 6, 4.0, 1.0, Some(42)).unwrap();
    print_dataset_summary(&deep_hier, "Deep Hierarchical");
    println!("   Deep structure: 2 main → 12 sub-clusters");
    println!();

    // Performance and memory considerations
    println!("=== Performance Guidelines =======================");
    println!("**Recommended dataset sizes:**");
    println!("- Development/Testing: 100-500 samples");
    println!("- Algorithm benchmarking: 1,000-5,000 samples");
    println!("- Performance testing: 10,000+ samples");

    println!("\n**Memory usage (approximate):**");
    println!("- Spirals (1000, 2D): ~16 KB");
    println!("- Swiss Roll (1000, 3D): ~24 KB");
    println!("- Hierarchical (1000, 5D): ~40 KB");

    println!("\n**Parameter tuning tips:**");
    println!("- Start with moderate noise (0.05-0.1)");
    println!("- Use anisotropy factors 2.0-10.0 for clear elongation");
    println!("- Keep sub-clusters ≤ 8 per main cluster for interpretability");
    println!();

    // Real-world applications
    println!("=== Real-World Applications =======================");
    println!("**Computer Vision:**");
    println!("- Spirals: Object boundary detection");
    println!("- Circles: Radial pattern recognition");

    println!("\n**Machine Learning Research:**");
    println!("- Benchmarking new clustering algorithms");
    println!("- Testing manifold learning methods");
    println!("- Evaluating non-linear classifiers");

    println!("\n**Data Science Education:**");
    println!("- Demonstrating algorithm limitations");
    println!("- Visualizing high-dimensional data challenges");
    println!("- Teaching feature engineering concepts");
    println!();

    println!("=== Complex Patterns Demo Complete ===============");
}

/// Print a concise summary of a dataset
fn print_dataset_summary(dataset: &scirs2_datasets::Dataset, name: &str) {
    let n_classes = if let Some(target) = &dataset.target {
        let unique_labels: std::collections::HashSet<_> =
            target.iter().map(|&x| x as i32).collect();
        unique_labels.len()
    } else {
        0
    };

    let class_info = if n_classes > 0 {
        format!(", {} classes", n_classes)
    } else {
        " (unsupervised)".to_string()
    };

    println!(
        "   {}: {} samples, {} features{}",
        name,
        dataset.n_samples(),
        dataset.n_features(),
        class_info
    );

    // Print first few data points for small datasets
    if dataset.n_samples() <= 10 && dataset.n_features() <= 3 {
        println!("   Sample points:");
        for i in 0..dataset.n_samples().min(3) {
            let point: Vec<f64> = (0..dataset.n_features())
                .map(|j| dataset.data[[i, j]])
                .collect();
            println!(
                "     [{:.3}, {:.3}{}]",
                point[0],
                point[1],
                if point.len() > 2 {
                    format!(", {:.3}", point[2])
                } else {
                    "".to_string()
                }
            );
        }
    }
}
