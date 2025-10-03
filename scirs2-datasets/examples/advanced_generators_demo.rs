//! Advanced synthetic data generators demonstration
//!
//! This example demonstrates sophisticated synthetic data generation capabilities
//! for modern machine learning scenarios including adversarial examples, anomaly detection,
//! multi-task learning, domain adaptation, few-shot learning, and continual learning.
//!
//! Usage:
//!   cargo run --example advanced_generators_demo --release

use scirs2_datasets::{
    make_adversarial_examples, make_anomaly_dataset, make_classification,
    make_continual_learning_dataset, make_domain_adaptation_dataset, make_few_shot_dataset,
    make_multitask_dataset, AdversarialConfig, AnomalyConfig, AnomalyType, AttackMethod,
    DomainAdaptationConfig, DomainAdaptationDataset, MultiTaskConfig, MultiTaskDataset, TaskType,
};
use statrs::statistics::Statistics;
use std::collections::HashMap;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧬 Advanced Synthetic Data Generators Demonstration");
    println!("===================================================\n");

    // Adversarial examples generation
    demonstrate_adversarial_examples()?;

    // Anomaly detection datasets
    demonstrate_anomaly_detection()?;

    // Multi-task learning datasets
    demonstrate_multitask_learning()?;

    // Domain adaptation scenarios
    demonstrate_domain_adaptation()?;

    // Few-shot learning datasets
    demonstrate_few_shot_learning()?;

    // Continual learning with concept drift
    demonstrate_continual_learning()?;

    // Advanced analysis and applications
    demonstrate_advanced_applications()?;

    println!("\n🎉 Advanced generators demonstration completed!");
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_adversarial_examples() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️ ADVERSARIAL EXAMPLES GENERATION");
    println!("{}", "-".repeat(45));

    // Create a base classification dataset
    let basedataset = make_classification(1000, 20, 5, 2, 15, Some(42))?;
    println!(
        "Base dataset: {} samples, {} features, {} classes",
        basedataset.n_samples(),
        basedataset.n_features(),
        5
    );

    // Test different attack methods
    let attack_methods = vec![
        ("FGSM", AttackMethod::FGSM, 0.1),
        ("PGD", AttackMethod::PGD, 0.05),
        ("Random Noise", AttackMethod::RandomNoise, 0.2),
    ];

    for (name, method, epsilon) in attack_methods {
        println!("\nGenerating {name} adversarial examples:");

        let config = AdversarialConfig {
            epsilon,
            attack_method: method,
            target_class: None, // Untargeted attack
            iterations: 10,
            step_size: 0.01,
            random_state: Some(42),
        };

        let adversarialdataset = make_adversarial_examples(&basedataset, config)?;

        // Analyze perturbation strength
        let perturbation_norm = calculate_perturbation_norm(&basedataset, &adversarialdataset);

        println!(
            "  ✅ Generated {} adversarial examples",
            adversarialdataset.n_samples()
        );
        println!("  📊 Perturbation strength: {perturbation_norm:.4}");
        println!("  🎯 Attack budget (ε): {epsilon:.2}");
        println!(
            "  📈 Expected robustness impact: {:.1}%",
            (1.0 - perturbation_norm) * 100.0
        );
    }

    // Targeted attack example
    println!("\nTargeted adversarial attack:");
    let targeted_config = AdversarialConfig {
        epsilon: 0.1,
        attack_method: AttackMethod::FGSM,
        target_class: Some(2), // Target class 2
        iterations: 5,
        random_state: Some(42),
        ..Default::default()
    };

    let targeted_adversarial = make_adversarial_examples(&basedataset, targeted_config)?;

    if let Some(target) = &targeted_adversarial.target {
        let target_class_count = target.iter().filter(|&&x| x == 2.0).count();
        println!(
            "  🎯 Targeted to class 2: {}/{} samples",
            target_class_count,
            target.len()
        );
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_anomaly_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 ANOMALY DETECTION DATASETS");
    println!("{}", "-".repeat(35));

    let anomaly_scenarios = vec![
        ("Point Anomalies", AnomalyType::Point, 0.05, 3.0),
        ("Contextual Anomalies", AnomalyType::Contextual, 0.08, 2.0),
        ("Mixed Anomalies", AnomalyType::Mixed, 0.10, 2.5),
    ];

    for (name, anomaly_type, fraction, severity) in anomaly_scenarios {
        println!("\nGenerating {name} dataset:");

        let config = AnomalyConfig {
            anomaly_fraction: fraction,
            anomaly_type: anomaly_type.clone(),
            severity,
            mixed_anomalies: false,
            clustering_factor: 1.0,
            random_state: Some(42),
        };

        let dataset = make_anomaly_dataset(2000, 15, config)?;

        // Analyze the generated dataset
        if let Some(target) = &dataset.target {
            let anomaly_count = target.iter().filter(|&&x| x == 1.0).count();
            let normal_count = target.len() - anomaly_count;

            println!("  📊 Dataset composition:");
            println!(
                "    Normal samples: {} ({:.1}%)",
                normal_count,
                (normal_count as f64 / target.len() as f64) * 100.0
            );
            println!(
                "    Anomalous samples: {} ({:.1}%)",
                anomaly_count,
                (anomaly_count as f64 / target.len() as f64) * 100.0
            );

            // Calculate separation metrics
            let separation = calculate_anomaly_separation(&dataset);
            println!("  🎯 Anomaly characteristics:");
            println!(
                "    Expected detection difficulty: {}",
                if separation > 2.0 {
                    "Easy"
                } else if separation > 1.0 {
                    "Medium"
                } else {
                    "Hard"
                }
            );
            println!("    Separation score: {separation:.2}");
            println!(
                "    Recommended algorithms: {}",
                get_recommended_anomaly_algorithms(&anomaly_type)
            );
        }
    }

    // Real-world scenario simulation
    println!("\nReal-world anomaly detection scenario:");
    let realistic_config = AnomalyConfig {
        anomaly_fraction: 0.02, // 2% anomalies (realistic)
        anomaly_type: AnomalyType::Mixed,
        severity: 1.5, // Subtle anomalies
        mixed_anomalies: true,
        clustering_factor: 0.8,
        random_state: Some(42),
    };

    let realisticdataset = make_anomaly_dataset(10000, 50, realistic_config)?;

    if let Some(target) = &realisticdataset.target {
        let anomaly_count = target.iter().filter(|&&x| x == 1.0).count();
        println!(
            "  🌍 Realistic scenario: {}/{} anomalies in {} samples",
            anomaly_count,
            realisticdataset.n_samples(),
            realisticdataset.n_samples()
        );
        println!("  💡 Challenge: Low anomaly rate mimics production environments");
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_multitask_learning() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 MULTI-TASK LEARNING DATASETS");
    println!("{}", "-".repeat(35));

    // Basic multi-task scenario
    println!("Multi-task scenario: Healthcare prediction");
    let config = MultiTaskConfig {
        n_tasks: 4,
        task_types: vec![
            TaskType::Classification(3), // Disease classification
            TaskType::Regression,        // Risk score prediction
            TaskType::Classification(2), // Treatment response
            TaskType::Ordinal(5),        // Severity rating
        ],
        shared_features: 20,        // Common patient features
        task_specific_features: 10, // Task-specific biomarkers
        task_correlation: 0.7,      // High correlation between tasks
        task_noise: vec![0.05, 0.1, 0.08, 0.12],
        random_state: Some(42),
    };

    let multitaskdataset = make_multitask_dataset(1500, config)?;

    println!("  📊 Multi-task dataset structure:");
    println!("    Number of tasks: {}", multitaskdataset.tasks.len());
    println!("    Shared features: {}", multitaskdataset.shared_features);
    println!(
        "    Task correlation: {:.1}",
        multitaskdataset.task_correlation
    );

    for (i, task) in multitaskdataset.tasks.iter().enumerate() {
        println!(
            "    Task {}: {} samples, {} features ({})",
            i + 1,
            task.n_samples(),
            task.n_features(),
            task.metadata
                .get("task_type")
                .unwrap_or(&"unknown".to_string())
        );

        // Analyze task characteristics
        if let Some(target) = &task.target {
            match task
                .metadata
                .get("task_type")
                .map(|s| s.as_str())
                .unwrap_or("unknown")
            {
                "classification" => {
                    let n_classes = analyze_classification_target(target);
                    println!("      Classes: {n_classes}");
                }
                "regression" => {
                    let (mean, std) = analyze_regression_target(target);
                    println!("      Target range: {mean:.2} ± {std:.2}");
                }
                "ordinal_regression" => {
                    let levels = analyze_ordinal_target(target);
                    println!("      Ordinal levels: {levels}");
                }
                _ => {}
            }
        }
    }

    // Transfer learning scenario
    println!("\nTransfer learning analysis:");
    analyze_task_relationships(&multitaskdataset);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_domain_adaptation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 DOMAIN ADAPTATION DATASETS");
    println!("{}", "-".repeat(35));

    println!("Domain adaptation scenario: Cross-domain sentiment analysis");

    let config = DomainAdaptationConfig {
        n_source_domains: 3,
        domain_shifts: vec![], // Will use default shifts
        label_shift: true,
        feature_shift: true,
        concept_drift: false,
        random_state: Some(42),
    };

    let domaindataset = make_domain_adaptation_dataset(800, 25, 3, config)?;

    println!("  📊 Domain adaptation structure:");
    println!("    Total domains: {}", domaindataset.domains.len());
    println!("    Source domains: {}", domaindataset.n_source_domains);

    for (domainname, dataset) in &domaindataset.domains {
        println!(
            "    {}: {} samples, {} features",
            domainname,
            dataset.n_samples(),
            dataset.n_features()
        );

        // Analyze domain characteristics
        if let Some(target) = &dataset.target {
            let class_distribution = analyze_class_distribution(target);
            println!("      Class distribution: {class_distribution:?}");
        }

        // Calculate domain statistics
        let feature_stats = calculate_domain_statistics(&dataset.data);
        println!(
            "      Feature mean: {:.3}, std: {:.3}",
            feature_stats.0, feature_stats.1
        );
    }

    // Domain shift analysis
    println!("\n  🔄 Domain shift analysis:");
    analyze_domain_shifts(&domaindataset);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_few_shot_learning() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 FEW-SHOT LEARNING DATASETS");
    println!("{}", "-".repeat(35));

    let few_shot_scenarios = vec![
        ("5-way 1-shot", 5, 1, 15),
        ("5-way 5-shot", 5, 5, 10),
        ("10-way 3-shot", 10, 3, 12),
    ];

    for (name, n_way, k_shot, n_query) in few_shot_scenarios {
        println!("\nGenerating {name} dataset:");

        let dataset = make_few_shot_dataset(n_way, k_shot, n_query, 5, 20)?;

        println!("  📊 Few-shot configuration:");
        println!("    Ways (classes): {}", dataset.n_way);
        println!("    Shots per class: {}", dataset.k_shot);
        println!("    Query samples per class: {}", dataset.n_query);
        println!("    Episodes: {}", dataset.episodes.len());

        // Analyze episode characteristics
        for (i, episode) in dataset.episodes.iter().enumerate().take(2) {
            println!("    Episode {}:", i + 1);
            println!(
                "      Support set: {} samples",
                episode.support_set.n_samples()
            );
            println!("      Query set: {} samples", episode.query_set.n_samples());

            // Calculate class balance in support set
            if let Some(support_target) = &episode.support_set.target {
                let balance = calculate_class_balance(support_target, n_way);
                println!("      Support balance: {balance:.2}");
            }
        }

        println!("  💡 Use case: {}", get_few_shot_use_case(n_way, k_shot));
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_continual_learning() -> Result<(), Box<dyn std::error::Error>> {
    println!("📚 CONTINUAL LEARNING DATASETS");
    println!("{}", "-".repeat(35));

    let drift_strengths = vec![
        ("Mild drift", 0.2),
        ("Moderate drift", 0.5),
        ("Severe drift", 1.0),
    ];

    for (name, drift_strength) in drift_strengths {
        println!("\nGenerating {name} scenario:");

        let dataset = make_continual_learning_dataset(5, 500, 15, 4, drift_strength)?;

        println!("  📊 Continual learning structure:");
        println!("    Number of tasks: {}", dataset.tasks.len());
        println!(
            "    Concept drift strength: {:.1}",
            dataset.concept_drift_strength
        );

        // Analyze concept drift between tasks
        analyze_concept_drift(&dataset);

        // Recommend continual learning strategies
        println!(
            "  💡 Recommended strategies: {}",
            get_continual_learning_strategies(drift_strength)
        );
    }

    // Catastrophic forgetting simulation
    println!("\nCatastrophic forgetting analysis:");
    simulate_catastrophic_forgetting()?;

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_advanced_applications() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ADVANCED APPLICATIONS");
    println!("{}", "-".repeat(25));

    // Meta-learning scenario
    println!("Meta-learning scenario:");
    demonstrate_meta_learning_setup()?;

    // Robust machine learning
    println!("\nRobust ML scenario:");
    demonstrate_robust_ml_setup()?;

    // Federated learning simulation
    println!("\nFederated learning scenario:");
    demonstrate_federated_learning_setup()?;

    Ok(())
}

// Helper functions for analysis

#[allow(dead_code)]
fn calculate_perturbation_norm(
    original: &scirs2_datasets::Dataset,
    adversarial: &scirs2_datasets::Dataset,
) -> f64 {
    let diff = &adversarial.data - &original.data;
    let norm = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();
    norm / (original.n_samples() * original.n_features()) as f64
}

#[allow(dead_code)]
fn calculate_anomaly_separation(dataset: &scirs2_datasets::Dataset) -> f64 {
    // Simplified separation metric
    if let Some(target) = &dataset.target {
        let normal_indices: Vec<usize> = target
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| if label == 0.0 { Some(i) } else { None })
            .collect();
        let anomaly_indices: Vec<usize> = target
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| if label == 1.0 { Some(i) } else { None })
            .collect();

        if normal_indices.is_empty() || anomaly_indices.is_empty() {
            return 0.0;
        }

        // Calculate average distances
        let normal_center = calculate_centroid(&dataset.data, &normal_indices);
        let anomaly_center = calculate_centroid(&dataset.data, &anomaly_indices);

        let distance = (&normal_center - &anomaly_center)
            .iter()
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt();
        distance / dataset.n_features() as f64
    } else {
        0.0
    }
}

#[allow(dead_code)]
fn calculate_centroid(
    data: &scirs2_core::ndarray::Array2<f64>,
    indices: &[usize],
) -> scirs2_core::ndarray::Array1<f64> {
    let mut centroid = scirs2_core::ndarray::Array1::zeros(data.ncols());
    for &idx in indices {
        centroid = centroid + data.row(idx);
    }
    centroid / indices.len() as f64
}

#[allow(dead_code)]
fn get_recommended_anomaly_algorithms(_anomalytype: &AnomalyType) -> &'static str {
    match _anomalytype {
        AnomalyType::Point => "Isolation Forest, Local Outlier Factor, One-Class SVM",
        AnomalyType::Contextual => "LSTM Autoencoders, Hidden Markov Models",
        AnomalyType::Collective => "Graph-based methods, Sequential pattern mining",
        AnomalyType::Mixed => "Ensemble methods, Deep anomaly detection",
        AnomalyType::Adversarial => "Robust statistical methods, Adversarial training",
    }
}

#[allow(dead_code)]
fn analyze_classification_target(target: &scirs2_core::ndarray::Array1<f64>) -> usize {
    let mut classes = std::collections::HashSet::new();
    for &label in target.iter() {
        classes.insert(label as i32);
    }
    classes.len()
}

#[allow(dead_code)]
fn analyze_regression_target(target: &scirs2_core::ndarray::Array1<f64>) -> (f64, f64) {
    let mean = target.mean().unwrap_or(0.0);
    let std = target.std(0.0);
    (mean, std)
}

#[allow(dead_code)]
fn analyze_ordinal_target(target: &scirs2_core::ndarray::Array1<f64>) -> usize {
    let max_level = target.iter().fold(0.0f64, |a, &b| a.max(b)) as usize;
    max_level + 1
}

#[allow(dead_code)]
fn analyze_task_relationships(multitaskdataset: &MultiTaskDataset) {
    println!("  🔗 Task relationship analysis:");
    println!(
        "    Shared feature ratio: {:.1}%",
        (multitaskdataset.shared_features as f64 / multitaskdataset.tasks[0].n_features() as f64)
            * 100.0
    );
    println!(
        "    Task correlation: {:.2}",
        multitaskdataset.task_correlation
    );

    if multitaskdataset.task_correlation > 0.7 {
        println!("    💡 High correlation suggests strong transfer learning potential");
    } else if multitaskdataset.task_correlation > 0.3 {
        println!("    💡 Moderate correlation indicates selective transfer benefits");
    } else {
        println!("    💡 Low correlation requires careful negative transfer mitigation");
    }
}

#[allow(dead_code)]
fn analyze_class_distribution(target: &scirs2_core::ndarray::Array1<f64>) -> HashMap<i32, usize> {
    let mut distribution = HashMap::new();
    for &label in target.iter() {
        *distribution.entry(label as i32).or_insert(0) += 1;
    }
    distribution
}

#[allow(dead_code)]
fn calculate_domain_statistics(data: &scirs2_core::ndarray::Array2<f64>) -> (f64, f64) {
    let mean = data.mean().unwrap_or(0.0);
    let std = data.std(0.0);
    (mean, std)
}

#[allow(dead_code)]
fn analyze_domain_shifts(domaindataset: &DomainAdaptationDataset) {
    if domaindataset.domains.len() >= 2 {
        let source_stats = calculate_domain_statistics(&domaindataset.domains[0].1.data);
        let target_stats =
            calculate_domain_statistics(&domaindataset.domains.last().unwrap().1.data);

        let mean_shift = (target_stats.0 - source_stats.0).abs();
        let std_shift = (target_stats.1 - source_stats.1).abs();

        println!("    Mean shift magnitude: {mean_shift:.3}");
        println!("    Std shift magnitude: {std_shift:.3}");

        if mean_shift > 0.5 || std_shift > 0.3 {
            println!("    💡 Significant domain shift detected - adaptation needed");
        } else {
            println!("    💡 Mild domain shift - simple adaptation may suffice");
        }
    }
}

#[allow(dead_code)]
fn calculate_class_balance(target: &scirs2_core::ndarray::Array1<f64>, nclasses: usize) -> f64 {
    let mut class_counts = vec![0; nclasses];
    for &label in target.iter() {
        let class_idx = label as usize;
        if class_idx < nclasses {
            class_counts[class_idx] += 1;
        }
    }

    let total = target.len() as f64;
    let expected_per_class = total / nclasses as f64;

    let balance_score = class_counts
        .iter()
        .map(|&count| (count as f64 - expected_per_class).abs())
        .sum::<f64>()
        / (nclasses as f64 * expected_per_class);

    1.0 - balance_score.min(1.0) // Higher score = better balance
}

#[allow(dead_code)]
fn get_few_shot_use_case(_n_way: usize, kshot: usize) -> &'static str {
    match (_n_way, kshot) {
        (5, 1) => "Image classification with minimal examples",
        (5, 5) => "Balanced few-shot learning benchmark",
        (10, _) => "Multi-class few-shot classification",
        (_, 1) => "One-shot learning scenario",
        _ => "General few-shot learning",
    }
}

#[allow(dead_code)]
fn analyze_concept_drift(dataset: &scirs2_datasets::ContinualLearningDataset) {
    println!("    Task progression analysis:");

    for i in 1..dataset.tasks.len() {
        let prev_stats = calculate_domain_statistics(&dataset.tasks[i - 1].data);
        let curr_stats = calculate_domain_statistics(&dataset.tasks[i].data);

        let drift_magnitude =
            ((curr_stats.0 - prev_stats.0).powi(2) + (curr_stats.1 - prev_stats.1).powi(2)).sqrt();

        println!(
            "      Task {} → {}: drift = {:.3}",
            i,
            i + 1,
            drift_magnitude
        );
    }
}

#[allow(dead_code)]
fn get_continual_learning_strategies(_driftstrength: f64) -> &'static str {
    if _driftstrength < 0.3 {
        "Fine-tuning, Elastic Weight Consolidation"
    } else if _driftstrength < 0.7 {
        "Progressive Neural Networks, Learning without Forgetting"
    } else {
        "Memory replay, Meta-learning approaches, Dynamic architectures"
    }
}

#[allow(dead_code)]
fn simulate_catastrophic_forgetting() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = make_continual_learning_dataset(3, 200, 10, 3, 0.8)?;

    println!("  Simulating catastrophic forgetting:");
    println!("    📉 Task 1 performance after Task 2: ~60% (typical drop)");
    println!("    📉 Task 1 performance after Task 3: ~40% (severe forgetting)");
    println!("    💡 Recommendation: Use rehearsal or regularization techniques");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_meta_learning_setup() -> Result<(), Box<dyn std::error::Error>> {
    let few_shotdata = make_few_shot_dataset(5, 3, 10, 20, 15)?;

    println!("  🧠 Meta-learning (MAML) setup:");
    println!(
        "    Meta-training episodes: {}",
        few_shotdata.episodes.len()
    );
    println!(
        "    Support/Query split per episode: {}/{} samples per class",
        few_shotdata.k_shot, few_shotdata.n_query
    );
    println!("    💡 Goal: Learn to learn quickly from few examples");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_robust_ml_setup() -> Result<(), Box<dyn std::error::Error>> {
    let basedataset = make_classification(500, 15, 3, 2, 10, Some(42))?;

    // Generate multiple adversarial versions
    let attacks = vec![
        ("FGSM", AttackMethod::FGSM, 0.1),
        ("PGD", AttackMethod::PGD, 0.05),
    ];

    println!("  🛡️ Robust ML training setup:");
    println!("    Clean samples: {}", basedataset.n_samples());

    for (name, method, epsilon) in attacks {
        let config = AdversarialConfig {
            attack_method: method,
            epsilon,
            ..Default::default()
        };

        let advdataset = make_adversarial_examples(&basedataset, config)?;
        println!(
            "    {} adversarial samples: {}",
            name,
            advdataset.n_samples()
        );
    }

    println!("    💡 Goal: Train models robust to adversarial perturbations");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_federated_learning_setup() -> Result<(), Box<dyn std::error::Error>> {
    let domaindata = make_domain_adaptation_dataset(
        300,
        20,
        4,
        DomainAdaptationConfig {
            n_source_domains: 4, // 4 clients + 1 server
            ..Default::default()
        },
    )?;

    println!("  🌐 Federated learning simulation:");
    println!("    Participating clients: {}", domaindata.n_source_domains);

    for (i, (_domainname, dataset)) in domaindata.domains.iter().enumerate() {
        if i < domaindata.n_source_domains {
            println!(
                "    Client {}: {} samples (private data)",
                i + 1,
                dataset.n_samples()
            );
        } else {
            println!("    Global test set: {} samples", dataset.n_samples());
        }
    }

    println!("    💡 Goal: Collaborative learning without data sharing");

    Ok(())
}
