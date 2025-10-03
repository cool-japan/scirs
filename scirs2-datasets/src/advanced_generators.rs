//! Advanced synthetic data generators
//!
//! This module provides sophisticated synthetic data generation capabilities
//! for complex scenarios including adversarial examples, anomaly detection,
//! multi-task learning, and domain adaptation.

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::Uniform;

/// Configuration for adversarial example generation
#[derive(Debug, Clone)]
pub struct AdversarialConfig {
    /// Perturbation strength (epsilon)
    pub epsilon: f64,
    /// Attack method
    pub attack_method: AttackMethod,
    /// Target class for targeted attacks
    pub target_class: Option<usize>,
    /// Number of attack iterations
    pub iterations: usize,
    /// Step size for iterative attacks
    pub step_size: f64,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

/// Adversarial attack methods
#[derive(Debug, Clone, PartialEq)]
pub enum AttackMethod {
    /// Fast Gradient Sign Method
    FGSM,
    /// Projected Gradient Descent
    PGD,
    /// Carlini & Wagner attack
    CW,
    /// DeepFool attack
    DeepFool,
    /// Random noise baseline
    RandomNoise,
}

impl Default for AdversarialConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            attack_method: AttackMethod::FGSM,
            target_class: None,
            iterations: 10,
            step_size: 0.01,
            random_state: None,
        }
    }
}

/// Configuration for anomaly detection datasets
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    /// Fraction of anomalous samples
    pub anomaly_fraction: f64,
    /// Type of anomalies to generate
    pub anomaly_type: AnomalyType,
    /// Severity of anomalies
    pub severity: f64,
    /// Whether to include multiple anomaly types
    pub mixed_anomalies: bool,
    /// Clustering factor for normal data
    pub clustering_factor: f64,
    /// Random seed
    pub random_state: Option<u64>,
}

/// Types of anomalies
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    /// Point anomalies (outliers)
    Point,
    /// Contextual anomalies
    Contextual,
    /// Collective anomalies
    Collective,
    /// Adversarial anomalies
    Adversarial,
    /// Mixed anomaly types
    Mixed,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            anomaly_fraction: 0.1,
            anomaly_type: AnomalyType::Point,
            severity: 2.0,
            mixed_anomalies: false,
            clustering_factor: 1.0,
            random_state: None,
        }
    }
}

/// Configuration for multi-task learning datasets
#[derive(Debug, Clone)]
pub struct MultiTaskConfig {
    /// Number of tasks
    pub n_tasks: usize,
    /// Task types (classification or regression)
    pub task_types: Vec<TaskType>,
    /// Shared feature dimensions
    pub shared_features: usize,
    /// Task-specific feature dimensions
    pub task_specific_features: usize,
    /// Correlation between tasks
    pub task_correlation: f64,
    /// Noise level for each task
    pub task_noise: Vec<f64>,
    /// Random seed
    pub random_state: Option<u64>,
}

/// Task types for multi-task learning
#[derive(Debug, Clone, PartialEq)]
pub enum TaskType {
    /// Classification task with specified number of classes
    Classification(usize),
    /// Regression task
    Regression,
    /// Ordinal regression
    Ordinal(usize),
}

impl Default for MultiTaskConfig {
    fn default() -> Self {
        Self {
            n_tasks: 3,
            task_types: vec![
                TaskType::Classification(3),
                TaskType::Regression,
                TaskType::Classification(5),
            ],
            shared_features: 10,
            task_specific_features: 5,
            task_correlation: 0.5,
            task_noise: vec![0.1, 0.1, 0.1],
            random_state: None,
        }
    }
}

/// Configuration for domain adaptation datasets
#[derive(Debug, Clone)]
pub struct DomainAdaptationConfig {
    /// Number of source domains
    pub n_source_domains: usize,
    /// Domain shift parameters
    pub domain_shifts: Vec<DomainShift>,
    /// Label shift (different class distributions)
    pub label_shift: bool,
    /// Feature shift (different feature distributions)
    pub feature_shift: bool,
    /// Concept drift over time
    pub concept_drift: bool,
    /// Random seed
    pub random_state: Option<u64>,
}

/// Domain shift types
#[derive(Debug, Clone)]
pub struct DomainShift {
    /// Shift in feature means
    pub mean_shift: Array1<f64>,
    /// Shift in feature covariances
    pub covariance_shift: Option<Array2<f64>>,
    /// Shift strength
    pub shift_strength: f64,
}

impl Default for DomainAdaptationConfig {
    fn default() -> Self {
        Self {
            n_source_domains: 2,
            domain_shifts: vec![],
            label_shift: true,
            feature_shift: true,
            concept_drift: false,
            random_state: None,
        }
    }
}

/// Advanced data generator
pub struct AdvancedGenerator {
    random_state: Option<u64>,
}

impl AdvancedGenerator {
    /// Create a new advanced generator
    pub fn new(_random_state: Option<u64>) -> Self {
        Self {
            random_state: _random_state,
        }
    }

    /// Generate adversarial examples
    pub fn make_adversarial_examples(
        &self,
        base_dataset: &Dataset,
        config: AdversarialConfig,
    ) -> Result<Dataset> {
        let n_samples = base_dataset.n_samples();
        let _n_features = base_dataset.n_features();

        println!(
            "Generating adversarial examples using {:?}",
            config.attack_method
        );

        // Create adversarial perturbations
        let perturbations = self.generate_perturbations(&base_dataset.data, &config)?;

        // Apply perturbations
        let adversarial_data = &base_dataset.data + &perturbations;

        // Clip to valid range if needed
        let clipped_data = adversarial_data.mapv(|x| x.clamp(-5.0, 5.0));

        // Create adversarial labels
        let adversarial_target = if let Some(target) = &base_dataset.target {
            match config.target_class {
                Some(target_class) => {
                    // Targeted attack - change labels to target class
                    Some(Array1::from_elem(n_samples, target_class as f64))
                }
                None => {
                    // Untargeted attack - keep original labels but mark as adversarial
                    Some(target.clone())
                }
            }
        } else {
            None
        };

        let mut metadata = base_dataset.metadata.clone();
        let _old_description = metadata.get("description").cloned().unwrap_or_default();
        let oldname = metadata.get("name").cloned().unwrap_or_default();

        metadata.insert(
            "description".to_string(),
            format!(
                "Adversarial examples generated using {:?}",
                config.attack_method
            ),
        );
        metadata.insert("name".to_string(), format!("{oldname} (Adversarial)"));

        Ok(Dataset {
            data: clipped_data,
            target: adversarial_target,
            targetnames: base_dataset.targetnames.clone(),
            featurenames: base_dataset.featurenames.clone(),
            feature_descriptions: base_dataset.feature_descriptions.clone(),
            description: base_dataset.description.clone(),
            metadata,
        })
    }

    /// Generate anomaly detection dataset
    pub fn make_anomaly_dataset(
        &self,
        n_samples: usize,
        n_features: usize,
        config: AnomalyConfig,
    ) -> Result<Dataset> {
        let n_anomalies = (n_samples as f64 * config.anomaly_fraction) as usize;
        let n_normal = n_samples - n_anomalies;

        println!("Generating anomaly dataset: {n_normal} normal, {n_anomalies} anomalous");

        // Generate normal data
        let normal_data =
            self.generate_normal_data(n_normal, n_features, config.clustering_factor)?;

        // Generate anomalous data
        let anomalous_data =
            self.generate_anomalous_data(n_anomalies, n_features, &normal_data, &config)?;

        // Combine data
        let mut combined_data = Array2::zeros((n_samples, n_features));
        combined_data
            .slice_mut(scirs2_core::ndarray::s![..n_normal, ..])
            .assign(&normal_data);
        combined_data
            .slice_mut(scirs2_core::ndarray::s![n_normal.., ..])
            .assign(&anomalous_data);

        // Create labels (0 = normal, 1 = anomaly)
        let mut target = Array1::zeros(n_samples);
        target
            .slice_mut(scirs2_core::ndarray::s![n_normal..])
            .fill(1.0);

        // Shuffle the data
        let shuffled_indices = self.generate_shuffle_indices(n_samples)?;
        let shuffled_data = self.shuffle_by_indices(&combined_data, &shuffled_indices);
        let shuffled_target = self.shuffle_array_by_indices(&target, &shuffled_indices);

        let metadata = crate::registry::DatasetMetadata {
            name: "Anomaly Detection Dataset".to_string(),
            description: format!(
                "Synthetic anomaly detection dataset with {:.1}% anomalies",
                config.anomaly_fraction * 100.0
            ),
            n_samples,
            n_features,
            task_type: "anomaly_detection".to_string(),
            targetnames: Some(vec!["normal".to_string(), "anomaly".to_string()]),
            ..Default::default()
        };

        Ok(Dataset::from_metadata(
            shuffled_data,
            Some(shuffled_target),
            metadata,
        ))
    }

    /// Generate multi-task learning dataset
    pub fn make_multitask_dataset(
        &self,
        n_samples: usize,
        config: MultiTaskConfig,
    ) -> Result<MultiTaskDataset> {
        let total_features =
            config.shared_features + config.task_specific_features * config.n_tasks;

        println!(
            "Generating multi-task dataset: {} tasks, {} samples, {} features",
            config.n_tasks, n_samples, total_features
        );

        // Generate shared features
        let shared_data = self.generate_shared_features(n_samples, config.shared_features)?;

        // Generate task-specific features and targets
        let mut task_datasets = Vec::new();

        for (task_id, task_type) in config.task_types.iter().enumerate() {
            let task_specific_data = self.generate_task_specific_features(
                n_samples,
                config.task_specific_features,
                task_id,
            )?;

            // Combine shared and task-specific features
            let task_data = self.combine_features(&shared_data, &task_specific_data);

            // Generate task target based on task type
            let task_target = self.generate_task_target(
                &task_data,
                task_type,
                config.task_correlation,
                config.task_noise.get(task_id).unwrap_or(&0.1),
            )?;

            let task_metadata = crate::registry::DatasetMetadata {
                name: format!("Task {task_id}"),
                description: format!("Multi-task learning task {task_id} ({task_type:?})"),
                n_samples,
                n_features: task_data.ncols(),
                task_type: match task_type {
                    TaskType::Classification(_) => "classification".to_string(),
                    TaskType::Regression => "regression".to_string(),
                    TaskType::Ordinal(_) => "ordinal_regression".to_string(),
                },
                ..Default::default()
            };

            task_datasets.push(Dataset::from_metadata(
                task_data,
                Some(task_target),
                task_metadata,
            ));
        }

        Ok(MultiTaskDataset {
            tasks: task_datasets,
            shared_features: config.shared_features,
            task_correlation: config.task_correlation,
        })
    }

    /// Generate domain adaptation dataset
    pub fn make_domain_adaptation_dataset(
        &self,
        n_samples_per_domain: usize,
        n_features: usize,
        n_classes: usize,
        config: DomainAdaptationConfig,
    ) -> Result<DomainAdaptationDataset> {
        let total_domains = config.n_source_domains + 1; // +1 for target _domain

        println!(
            "Generating _domain adaptation dataset: {total_domains} domains, {n_samples_per_domain} samples each"
        );

        let mut domain_datasets = Vec::new();

        // Generate source _domain (reference)
        let source_dataset =
            self.generate_base_domain_dataset(n_samples_per_domain, n_features, n_classes)?;

        domain_datasets.push(("source".to_string(), source_dataset.clone()));

        // Generate additional source domains with shifts
        for domain_id in 1..config.n_source_domains {
            let shift = if domain_id - 1 < config.domain_shifts.len() {
                &config.domain_shifts[domain_id - 1]
            } else {
                // Generate default shift
                &DomainShift {
                    mean_shift: Array1::from_elem(n_features, 0.5),
                    covariance_shift: None,
                    shift_strength: 1.0,
                }
            };

            let shifted_dataset = self.apply_domain_shift(&source_dataset, shift)?;
            domain_datasets.push((format!("source_{domain_id}"), shifted_dataset));
        }

        // Generate target _domain with different shift
        let target_shift = DomainShift {
            mean_shift: Array1::from_elem(n_features, 1.0),
            covariance_shift: None,
            shift_strength: 1.5,
        };

        let target_dataset = self.apply_domain_shift(&source_dataset, &target_shift)?;
        domain_datasets.push(("target".to_string(), target_dataset));

        Ok(DomainAdaptationDataset {
            domains: domain_datasets,
            n_source_domains: config.n_source_domains,
        })
    }

    /// Generate few-shot learning dataset
    pub fn make_few_shot_dataset(
        &self,
        n_way: usize,
        k_shot: usize,
        n_query: usize,
        n_episodes: usize,
        n_features: usize,
    ) -> Result<FewShotDataset> {
        println!(
            "Generating few-_shot dataset: {n_way}-_way {k_shot}-_shot, {n_episodes} _episodes"
        );

        let mut episodes = Vec::new();

        for episode_id in 0..n_episodes {
            let support_set = self.generate_support_set(n_way, k_shot, n_features, episode_id)?;
            let query_set =
                self.generate_query_set(n_way, n_query, n_features, &support_set, episode_id)?;

            episodes.push(FewShotEpisode {
                support_set,
                query_set,
                n_way,
                k_shot,
            });
        }

        Ok(FewShotDataset {
            episodes,
            n_way,
            k_shot,
            n_query,
        })
    }

    /// Generate continual learning dataset with concept drift
    pub fn make_continual_learning_dataset(
        &self,
        n_tasks: usize,
        n_samples_per_task: usize,
        n_features: usize,
        n_classes: usize,
        concept_drift_strength: f64,
    ) -> Result<ContinualLearningDataset> {
        println!("Generating continual learning dataset: {n_tasks} _tasks with concept drift");

        let mut task_datasets = Vec::new();
        let mut base_centers = self.generate_class_centers(n_classes, n_features)?;

        for task_id in 0..n_tasks {
            // Apply concept drift
            if task_id > 0 {
                let drift = Array2::from_shape_fn((n_classes, n_features), |_| {
                    thread_rng().random::<f64>() * concept_drift_strength
                });
                base_centers = base_centers + drift;
            }

            let task_dataset = self.generate_classification_from_centers(
                n_samples_per_task,
                &base_centers,
                1.0, // cluster_std
                task_id as u64,
            )?;

            let mut metadata = task_dataset.metadata.clone();
            metadata.insert(
                "name".to_string(),
                format!("Continual Learning Task {task_id}"),
            );
            metadata.insert(
                "description".to_string(),
                format!("Task {task_id} with concept drift _strength {concept_drift_strength:.2}"),
            );

            task_datasets.push(Dataset {
                data: task_dataset.data,
                target: task_dataset.target,
                targetnames: task_dataset.targetnames,
                featurenames: task_dataset.featurenames,
                feature_descriptions: task_dataset.feature_descriptions,
                description: task_dataset.description,
                metadata,
            });
        }

        Ok(ContinualLearningDataset {
            tasks: task_datasets,
            concept_drift_strength,
        })
    }

    // Private helper methods

    fn generate_perturbations(
        &self,
        data: &Array2<f64>,
        config: &AdversarialConfig,
    ) -> Result<Array2<f64>> {
        let (n_samples, n_features) = data.dim();

        match config.attack_method {
            AttackMethod::FGSM => {
                // Fast Gradient Sign Method
                let mut perturbations = Array2::zeros((n_samples, n_features));
                for i in 0..n_samples {
                    for j in 0..n_features {
                        let sign = if thread_rng().random::<f64>() > 0.5 {
                            1.0
                        } else {
                            -1.0
                        };
                        perturbations[[i, j]] = config.epsilon * sign;
                    }
                }
                Ok(perturbations)
            }
            AttackMethod::PGD => {
                // Projected Gradient Descent (simplified)
                let mut perturbations: Array2<f64> = Array2::zeros((n_samples, n_features));
                for _iter in 0..config.iterations {
                    for i in 0..n_samples {
                        for j in 0..n_features {
                            let gradient = thread_rng().random::<f64>() * 2.0 - 1.0; // Simulated gradient
                            perturbations[[i, j]] += config.step_size * gradient.signum();
                            // Clip to epsilon ball
                            perturbations[[i, j]] =
                                perturbations[[i, j]].clamp(-config.epsilon, config.epsilon);
                        }
                    }
                }
                Ok(perturbations)
            }
            AttackMethod::RandomNoise => {
                // Random noise baseline
                let perturbations = Array2::from_shape_fn((n_samples, n_features), |_| {
                    (thread_rng().random::<f64>() * 2.0 - 1.0) * config.epsilon
                });
                Ok(perturbations)
            }
            _ => {
                // For other methods, use random noise directly
                let mut perturbations = Array2::zeros(data.dim());
                for i in 0..data.nrows() {
                    for j in 0..data.ncols() {
                        let noise = thread_rng().random::<f64>() * 2.0 - 1.0;
                        perturbations[[i, j]] = config.epsilon * noise;
                    }
                }
                Ok(perturbations)
            }
        }
    }

    fn generate_normal_data(
        &self,
        n_samples: usize,
        n_features: usize,
        clustering_factor: f64,
    ) -> Result<Array2<f64>> {
        // Generate clustered normal data
        use crate::generators::make_blobs;
        let n_clusters = ((n_features as f64).sqrt() as usize).max(2);
        let dataset = make_blobs(
            n_samples,
            n_features,
            n_clusters,
            clustering_factor,
            self.random_state,
        )?;
        Ok(dataset.data)
    }

    fn generate_anomalous_data(
        &self,
        n_anomalies: usize,
        n_features: usize,
        normal_data: &Array2<f64>,
        config: &AnomalyConfig,
    ) -> Result<Array2<f64>> {
        use scirs2_core::random::Rng;
        let mut rng = thread_rng();

        match config.anomaly_type {
            AnomalyType::Point => {
                // Point _anomalies - outliers far from normal distribution
                let normal_mean = normal_data.mean_axis(Axis(0)).ok_or_else(|| {
                    DatasetsError::ComputationError(
                        "Failed to compute mean for normal data".to_string(),
                    )
                })?;
                let normal_std = normal_data.std_axis(Axis(0), 0.0);

                let mut anomalies = Array2::zeros((n_anomalies, n_features));
                for i in 0..n_anomalies {
                    for j in 0..n_features {
                        let direction = if rng.random::<f64>() > 0.5 { 1.0 } else { -1.0 };
                        anomalies[[i, j]] =
                            normal_mean[j] + direction * config.severity * normal_std[j];
                    }
                }
                Ok(anomalies)
            }
            AnomalyType::Contextual => {
                // Contextual _anomalies - normal values but in wrong context
                let mut anomalies: Array2<f64> = Array2::zeros((n_anomalies, n_features));
                for i in 0..n_anomalies {
                    // Pick a random normal sample and permute some _features
                    let base_idx = rng.sample(Uniform::new(0, normal_data.nrows()).unwrap());
                    let mut anomaly = normal_data.row(base_idx).to_owned();

                    // Permute random _features
                    let n_permute = (n_features as f64 * 0.3) as usize;
                    for _ in 0..n_permute {
                        let j = rng.sample(Uniform::new(0, n_features).unwrap());
                        let k = rng.sample(Uniform::new(0, n_features).unwrap());
                        let temp = anomaly[j];
                        anomaly[j] = anomaly[k];
                        anomaly[k] = temp;
                    }

                    anomalies.row_mut(i).assign(&anomaly);
                }
                Ok(anomalies)
            }
            _ => {
                // Default to point _anomalies implementation
                let normal_mean = normal_data.mean_axis(Axis(0)).ok_or_else(|| {
                    DatasetsError::ComputationError(
                        "Failed to compute mean for normal data".to_string(),
                    )
                })?;
                let normal_std = normal_data.std_axis(Axis(0), 0.0);

                let mut anomalies = Array2::zeros((n_anomalies, n_features));
                for i in 0..n_anomalies {
                    for j in 0..n_features {
                        let direction = if rng.random::<f64>() > 0.5 { 1.0 } else { -1.0 };
                        anomalies[[i, j]] =
                            normal_mean[j] + direction * config.severity * normal_std[j];
                    }
                }
                Ok(anomalies)
            }
        }
    }

    fn generate_shuffle_indices(&self, n_samples: usize) -> Result<Vec<usize>> {
        use scirs2_core::random::Rng;
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Simple shuffle using Fisher-Yates
        for i in (1..n_samples).rev() {
            let j = rng.sample(Uniform::new(0, i).unwrap());
            indices.swap(i, j);
        }

        Ok(indices)
    }

    fn shuffle_by_indices(&self, data: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let mut shuffled = Array2::zeros(data.dim());
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            shuffled.row_mut(new_idx).assign(&data.row(old_idx));
        }
        shuffled
    }

    fn shuffle_array_by_indices(&self, array: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
        let mut shuffled = Array1::zeros(array.len());
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            shuffled[new_idx] = array[old_idx];
        }
        shuffled
    }

    fn generate_shared_features(&self, n_samples: usize, n_features: usize) -> Result<Array2<f64>> {
        // Generate shared _features using multivariate normal distribution
        let data = Array2::from_shape_fn((n_samples, n_features), |_| {
            thread_rng().random::<f64>() * 2.0 - 1.0 // Standard normal approximation
        });
        Ok(data)
    }

    fn generate_task_specific_features(
        &self,
        n_samples: usize,
        n_features: usize,
        task_id: usize,
    ) -> Result<Array2<f64>> {
        // Generate task-specific _features with slight bias per task
        let task_bias = task_id as f64 * 0.1;
        let data = Array2::from_shape_fn((n_samples, n_features), |_| {
            thread_rng().random::<f64>() * 2.0 - 1.0 + task_bias
        });
        Ok(data)
    }

    fn combine_features(&self, shared: &Array2<f64>, task_specific: &Array2<f64>) -> Array2<f64> {
        let n_samples = shared.nrows();
        let total_features = shared.ncols() + task_specific.ncols();
        let mut combined = Array2::zeros((n_samples, total_features));

        combined
            .slice_mut(scirs2_core::ndarray::s![.., ..shared.ncols()])
            .assign(shared);
        combined
            .slice_mut(scirs2_core::ndarray::s![.., shared.ncols()..])
            .assign(task_specific);

        combined
    }

    fn generate_task_target(
        &self,
        data: &Array2<f64>,
        task_type: &TaskType,
        correlation: f64,
        noise: &f64,
    ) -> Result<Array1<f64>> {
        let n_samples = data.nrows();

        match task_type {
            TaskType::Classification(n_classes) => {
                // Generate classification target based on data
                let target = Array1::from_shape_fn(n_samples, |i| {
                    let feature_sum = data.row(i).sum();
                    let class = ((feature_sum * correlation).abs() as usize) % n_classes;
                    class as f64
                });
                Ok(target)
            }
            TaskType::Regression => {
                // Generate regression target as linear combination of features
                let target = Array1::from_shape_fn(n_samples, |i| {
                    let weighted_sum = data
                        .row(i)
                        .iter()
                        .enumerate()
                        .map(|(j, &x)| x * (j as f64 + 1.0) * correlation)
                        .sum::<f64>();
                    weighted_sum + thread_rng().random::<f64>() * noise
                });
                Ok(target)
            }
            TaskType::Ordinal(n_levels) => {
                // Generate ordinal target
                let target = Array1::from_shape_fn(n_samples, |i| {
                    let feature_sum = data.row(i).sum();
                    let level = ((feature_sum * correlation).abs() as usize) % n_levels;
                    level as f64
                });
                Ok(target)
            }
        }
    }

    fn generate_base_domain_dataset(
        &self,
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
    ) -> Result<Dataset> {
        use crate::generators::make_classification;
        make_classification(
            n_samples,
            n_features,
            n_classes,
            2,
            n_features / 2,
            self.random_state,
        )
    }

    fn apply_domain_shift(&self, base_dataset: &Dataset, shift: &DomainShift) -> Result<Dataset> {
        let shifted_data = &base_dataset.data + &shift.mean_shift;

        let mut metadata = base_dataset.metadata.clone();
        let old_description = metadata.get("description").cloned().unwrap_or_default();
        metadata.insert(
            "description".to_string(),
            format!("{old_description} (Domain Shifted)"),
        );

        Ok(Dataset {
            data: shifted_data,
            target: base_dataset.target.clone(),
            targetnames: base_dataset.targetnames.clone(),
            featurenames: base_dataset.featurenames.clone(),
            feature_descriptions: base_dataset.feature_descriptions.clone(),
            description: base_dataset.description.clone(),
            metadata,
        })
    }

    fn generate_support_set(
        &self,
        n_way: usize,
        k_shot: usize,
        n_features: usize,
        episode_id: usize,
    ) -> Result<Dataset> {
        let n_samples = n_way * k_shot;
        use crate::generators::make_classification;
        make_classification(
            n_samples,
            n_features,
            n_way,
            1,
            n_features / 2,
            Some(episode_id as u64),
        )
    }

    fn generate_query_set(
        &self,
        n_way: usize,
        n_query: usize,
        n_features: usize,
        _set: &Dataset,
        episode_id: usize,
    ) -> Result<Dataset> {
        let n_samples = n_way * n_query;
        use crate::generators::make_classification;
        make_classification(
            n_samples,
            n_features,
            n_way,
            1,
            n_features / 2,
            Some(episode_id as u64 + 1000),
        )
    }

    fn generate_class_centers(&self, n_classes: usize, n_features: usize) -> Result<Array2<f64>> {
        let centers = Array2::from_shape_fn((n_classes, n_features), |_| {
            thread_rng().random::<f64>() * 4.0 - 2.0
        });
        Ok(centers)
    }

    fn generate_classification_from_centers(
        &self,
        n_samples: usize,
        centers: &Array2<f64>,
        cluster_std: f64,
        seed: u64,
    ) -> Result<Dataset> {
        use crate::generators::make_blobs;
        make_blobs(
            n_samples,
            centers.ncols(),
            centers.nrows(),
            cluster_std,
            Some(seed),
        )
    }
}

/// Multi-task learning dataset container
#[derive(Debug)]
pub struct MultiTaskDataset {
    /// Individual task datasets
    pub tasks: Vec<Dataset>,
    /// Number of shared features
    pub shared_features: usize,
    /// Correlation between tasks
    pub task_correlation: f64,
}

/// Domain adaptation dataset container
#[derive(Debug)]
pub struct DomainAdaptationDataset {
    /// Datasets for each domain (name, dataset)
    pub domains: Vec<(String, Dataset)>,
    /// Number of source domains
    pub n_source_domains: usize,
}

/// Few-shot learning episode
#[derive(Debug)]
pub struct FewShotEpisode {
    /// Support set for learning
    pub support_set: Dataset,
    /// Query set for evaluation
    pub query_set: Dataset,
    /// Number of classes (ways)
    pub n_way: usize,
    /// Number of examples per class (shots)
    pub k_shot: usize,
}

/// Few-shot learning dataset
#[derive(Debug)]
pub struct FewShotDataset {
    /// Training/evaluation episodes
    pub episodes: Vec<FewShotEpisode>,
    /// Number of classes per episode
    pub n_way: usize,
    /// Number of shots per class
    pub k_shot: usize,
    /// Number of query samples per class
    pub n_query: usize,
}

/// Continual learning dataset
#[derive(Debug)]
pub struct ContinualLearningDataset {
    /// Sequential tasks
    pub tasks: Vec<Dataset>,
    /// Strength of concept drift between tasks
    pub concept_drift_strength: f64,
}

/// Convenience functions for advanced data generation
///
/// Generate adversarial examples from a base dataset
#[allow(dead_code)]
pub fn make_adversarial_examples(
    base_dataset: &Dataset,
    config: AdversarialConfig,
) -> Result<Dataset> {
    let generator = AdvancedGenerator::new(config.random_state);
    generator.make_adversarial_examples(base_dataset, config)
}

/// Generate anomaly detection dataset
#[allow(dead_code)]
pub fn make_anomaly_dataset(
    n_samples: usize,
    n_features: usize,
    config: AnomalyConfig,
) -> Result<Dataset> {
    let generator = AdvancedGenerator::new(config.random_state);
    generator.make_anomaly_dataset(n_samples, n_features, config)
}

/// Generate multi-task learning dataset
#[allow(dead_code)]
pub fn make_multitask_dataset(
    n_samples: usize,
    config: MultiTaskConfig,
) -> Result<MultiTaskDataset> {
    let generator = AdvancedGenerator::new(config.random_state);
    generator.make_multitask_dataset(n_samples, config)
}

/// Generate domain adaptation dataset
#[allow(dead_code)]
pub fn make_domain_adaptation_dataset(
    n_samples_per_domain: usize,
    n_features: usize,
    n_classes: usize,
    config: DomainAdaptationConfig,
) -> Result<DomainAdaptationDataset> {
    let generator = AdvancedGenerator::new(config.random_state);
    generator.make_domain_adaptation_dataset(n_samples_per_domain, n_features, n_classes, config)
}

/// Generate few-shot learning dataset
#[allow(dead_code)]
pub fn make_few_shot_dataset(
    n_way: usize,
    k_shot: usize,
    n_query: usize,
    n_episodes: usize,
    n_features: usize,
) -> Result<FewShotDataset> {
    let generator = AdvancedGenerator::new(Some(42));
    generator.make_few_shot_dataset(n_way, k_shot, n_query, n_episodes, n_features)
}

/// Generate continual learning dataset
#[allow(dead_code)]
pub fn make_continual_learning_dataset(
    n_tasks: usize,
    n_samples_per_task: usize,
    n_features: usize,
    n_classes: usize,
    concept_drift_strength: f64,
) -> Result<ContinualLearningDataset> {
    let generator = AdvancedGenerator::new(Some(42));
    generator.make_continual_learning_dataset(
        n_tasks,
        n_samples_per_task,
        n_features,
        n_classes,
        concept_drift_strength,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::make_classification;

    #[test]
    fn test_adversarial_config() {
        let config = AdversarialConfig::default();
        assert_eq!(config.epsilon, 0.1);
        assert_eq!(config.attack_method, AttackMethod::FGSM);
        assert_eq!(config.iterations, 10);
    }

    #[test]
    fn test_anomaly_dataset_generation() {
        let config = AnomalyConfig {
            anomaly_fraction: 0.2,
            anomaly_type: AnomalyType::Point,
            severity: 2.0,
            ..Default::default()
        };

        let dataset = make_anomaly_dataset(100, 10, config).unwrap();

        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 10);
        assert!(dataset.target.is_some());

        // Check that we have both normal and anomalous samples
        let target = dataset.target.unwrap();
        let anomalies = target.iter().filter(|&&x| x == 1.0).count();
        assert!(anomalies > 0);
        assert!(anomalies < 100);
    }

    #[test]
    fn test_multitask_dataset_generation() {
        let config = MultiTaskConfig {
            n_tasks: 2,
            task_types: vec![TaskType::Classification(3), TaskType::Regression],
            shared_features: 5,
            task_specific_features: 3,
            ..Default::default()
        };

        let dataset = make_multitask_dataset(50, config).unwrap();

        assert_eq!(dataset.tasks.len(), 2);
        assert_eq!(dataset.shared_features, 5);

        for task in &dataset.tasks {
            assert_eq!(task.n_samples(), 50);
            assert!(task.target.is_some());
        }
    }

    #[test]
    fn test_adversarial_examples_generation() {
        let base_dataset = make_classification(100, 10, 3, 2, 8, Some(42)).unwrap();
        let config = AdversarialConfig {
            epsilon: 0.1,
            attack_method: AttackMethod::FGSM,
            ..Default::default()
        };

        let adversarial_dataset = make_adversarial_examples(&base_dataset, config).unwrap();

        assert_eq!(adversarial_dataset.n_samples(), base_dataset.n_samples());
        assert_eq!(adversarial_dataset.n_features(), base_dataset.n_features());

        // Check that the data has been perturbed (mean difference can cancel; use max absolute diff)
        let diff = &adversarial_dataset.data - &base_dataset.data;
        let mut max_abs = 0.0_f64;
        for v in diff.iter() {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        assert!(
            max_abs > 1e-9,
            "Adversarial perturbation appears to be zero (max abs diff = {max_abs})"
        );
    }

    #[test]
    fn test_few_shot_dataset() {
        let dataset = make_few_shot_dataset(5, 3, 10, 2, 20).unwrap();

        assert_eq!(dataset.n_way, 5);
        assert_eq!(dataset.k_shot, 3);
        assert_eq!(dataset.n_query, 10);
        assert_eq!(dataset.episodes.len(), 2);

        for episode in &dataset.episodes {
            assert_eq!(episode.n_way, 5);
            assert_eq!(episode.k_shot, 3);
            assert_eq!(episode.support_set.n_samples(), 5 * 3); // n_way * k_shot
            assert_eq!(episode.query_set.n_samples(), 5 * 10); // n_way * n_query
        }
    }

    #[test]
    fn test_continual_learning_dataset() {
        let dataset = make_continual_learning_dataset(3, 100, 10, 5, 0.5).unwrap();

        assert_eq!(dataset.tasks.len(), 3);
        assert_eq!(dataset.concept_drift_strength, 0.5);

        for task in &dataset.tasks {
            assert_eq!(task.n_samples(), 100);
            assert_eq!(task.n_features(), 10);
            assert!(task.target.is_some());
        }
    }
}
