//! Machine Learning evaluation metrics module for SciRS2
//!
//! This module provides functions for evaluating machine learning models
//! including classification, regression, clustering, and ranking metrics, as well as
//! model evaluation utilities like cross-validation and train-test split.
//!
//! # Classification Metrics
//!
//! Classification metrics evaluate the performance of classification models:
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::classification::{accuracy_score, precision_score, f1_score};
//!
//! let y_true = array![0, 1, 2, 0, 1, 2];
//! let y_pred = array![0, 2, 1, 0, 0, 2];
//!
//! let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
//! let precision = precision_score(&y_true, &y_pred, 1).unwrap();
//! let f1 = f1_score(&y_true, &y_pred, 1).unwrap();
//! ```
//!
//! ## One-vs-One Classification Metrics
//!
//! One-vs-One metrics are useful for evaluating multi-class classification problems by
//! considering each pair of classes separately:
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::classification::one_vs_one::{one_vs_one_accuracy, weighted_one_vs_one_f1_score};
//!
//! let y_true = array![0, 1, 2, 0, 1, 2];
//! let y_pred = array![0, 2, 1, 0, 0, 2];
//!
//! let ovo_acc = one_vs_one_accuracy(&y_true, &y_pred).unwrap();
//! let weighted_f1 = weighted_one_vs_one_f1_score(&y_true, &y_pred).unwrap();
//! ```
//!
//! # Regression Metrics
//!
//! Regression metrics evaluate the performance of regression models:
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::regression::{mean_squared_error, r2_score};
//!
//! let y_true = array![3.0, -0.5, 2.0, 7.0];
//! let y_pred = array![2.5, 0.0, 2.0, 8.0];
//!
//! let mse = mean_squared_error(&y_true, &y_pred).unwrap();
//! let r2 = r2_score(&y_true, &y_pred).unwrap();
//! ```
//!
//! # Clustering Metrics
//!
//! Clustering metrics evaluate the performance of clustering algorithms:
//!
//! ```
//! use ndarray::{array, Array2};
//! use scirs2_metrics::clustering::silhouette_score;
//!
//! // Create a small dataset with 2 clusters
//! let X = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0,
//!     1.5, 1.8,
//!     1.2, 2.2,
//!     5.0, 6.0,
//!     5.2, 5.8,
//!     5.5, 6.2,
//! ]).unwrap();
//!
//! let labels = array![0, 0, 0, 1, 1, 1];
//!
//! let score = silhouette_score(&X, &labels, "euclidean").unwrap();
//! ```
//!
//! # Ranking Metrics
//!
//! Ranking metrics evaluate the performance of ranking and recommendation models:
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::ranking::{
//!     mean_reciprocal_rank, ndcg_score, mean_average_precision,
//!     precision_at_k, recall_at_k, map_at_k, click_through_rate
//! };
//! use scirs2_metrics::ranking::label::{
//!     coverage_error, label_ranking_loss, label_ranking_average_precision_score
//! };
//!
//! // Example: search engine results where each array is a different query
//! // Values indicate whether a result is relevant (1.0) or not (0.0)
//! let y_true = vec![
//!     array![0.0, 1.0, 0.0, 0.0, 0.0],  // First query: second result is relevant
//!     array![0.0, 0.0, 0.0, 1.0, 0.0],  // Second query: fourth result is relevant
//! ];
//! let y_score = vec![
//!     array![0.1, 0.9, 0.2, 0.3, 0.4],  // Scores for first query
//!     array![0.5, 0.6, 0.7, 0.9, 0.8],  // Scores for second query
//! ];
//!
//! // Basic ranking metrics
//! let mrr = mean_reciprocal_rank(&y_true, &y_score).unwrap();
//! let ndcg = ndcg_score(&y_true, &y_score, Some(5)).unwrap();
//! let map = mean_average_precision(&y_true, &y_score, None).unwrap();
//! let precision = precision_at_k(&y_true, &y_score, 3).unwrap();
//! let recall = recall_at_k(&y_true, &y_score, 3).unwrap();
//!
//! // Advanced metrics
//! let map_k = map_at_k(&y_true, &y_score, 3).unwrap();
//! let ctr = click_through_rate(&y_true, &y_score, 3).unwrap();
//! ```
//!
//! ## Rank Correlation Metrics
//!
//! For evaluating correlation between rankings:
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::ranking::{kendalls_tau, spearmans_rho};
//!
//! // Compare two different ranking methods
//! let ranking_a = array![1.0, 2.0, 3.0, 4.0, 5.0];
//! let ranking_b = array![1.5, 2.5, 3.0, 3.5, 5.0];
//!
//! // Measure rank correlation
//! let tau = kendalls_tau(&ranking_a, &ranking_b).unwrap();
//! let rho = spearmans_rho(&ranking_a, &ranking_b).unwrap();
//! ```
//!
//! ## Label Ranking Metrics
//!
//! For multi-label ranking problems:
//!
//! ```
//! use ndarray::Array2;
//! use scirs2_metrics::ranking::label::{
//!     coverage_error_multiple, label_ranking_loss, label_ranking_average_precision_score
//! };
//!
//! // Multi-label data: 3 samples, 5 labels
//! let y_true = Array2::from_shape_vec((3, 5), vec![
//!     1.0, 0.0, 1.0, 0.0, 0.0,  // Sample 1: labels 0 and 2 are relevant
//!     0.0, 0.0, 1.0, 1.0, 0.0,  // Sample 2: labels 2 and 3 are relevant
//!     0.0, 1.0, 1.0, 0.0, 1.0,  // Sample 3: labels 1, 2, and 4 are relevant
//! ]).unwrap();
//!
//! // Predicted scores for each label
//! let y_score = Array2::from_shape_vec((3, 5), vec![
//!     0.9, 0.2, 0.8, 0.3, 0.1,  // Scores for sample 1
//!     0.2, 0.3, 0.9, 0.7, 0.1,  // Scores for sample 2
//!     0.1, 0.9, 0.8, 0.2, 0.7,  // Scores for sample 3
//! ]).unwrap();
//!
//! // Coverage error measures how far we need to go down the list to cover all true labels
//! let coverage = coverage_error_multiple(&y_true, &y_score).unwrap();
//!
//! // Label ranking loss counts incorrectly ordered label pairs
//! let loss = label_ranking_loss(&y_true, &y_score).unwrap();
//!
//! // Label ranking average precision measures precision at each relevant position
//! let precision = label_ranking_average_precision_score(&y_true, &y_score).unwrap();
//! ```
//!
//! # Anomaly Detection Metrics
//!
//! Metrics for evaluating anomaly detection systems:
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::anomaly::{
//!     detection_accuracy, false_alarm_rate, miss_detection_rate,
//!     anomaly_auc_score, anomaly_average_precision_score
//! };
//!
//! // Ground truth (1 for anomalies, 0 for normal points)
//! let y_true = array![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
//!
//! // Predicted labels
//! let y_pred = array![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
//!
//! // Anomaly scores (higher means more anomalous)
//! let y_score = array![0.1, 0.2, 0.9, 0.7, 0.8, 0.3, 0.6, 0.95, 0.2, 0.1];
//!
//! // Detection accuracy
//! let accuracy = detection_accuracy(&y_true, &y_pred).unwrap();
//!
//! // False alarm rate (Type I error)
//! let far = false_alarm_rate(&y_true, &y_pred).unwrap();
//!
//! // Miss detection rate (Type II error)
//! let mdr = miss_detection_rate(&y_true, &y_pred).unwrap();
//!
//! // AUC for anomaly detection
//! let auc = anomaly_auc_score(&y_true, &y_score).unwrap();
//!
//! // Average precision score
//! let ap = anomaly_average_precision_score(&y_true, &y_score).unwrap();
//! ```
//!
//! ## Distribution Metrics
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::anomaly::{
//!     kl_divergence, js_divergence, wasserstein_distance, maximum_mean_discrepancy
//! };
//!
//! // Two probability distributions
//! let p = array![0.2, 0.5, 0.3];
//! let q = array![0.3, 0.4, 0.3];
//!
//! // Compute KL divergence
//! let kl = kl_divergence(&p, &q).unwrap();
//!
//! // Jensen-Shannon divergence
//! let js = js_divergence(&p, &q).unwrap();
//!
//! // Wasserstein distance (1D)
//! let samples_p = array![1.0, 2.0, 3.0, 4.0, 5.0];
//! let samples_q = array![1.5, 2.5, 3.5, 4.5, 5.5];
//! let w_dist = wasserstein_distance(&samples_p, &samples_q).unwrap();
//!
//! // Maximum Mean Discrepancy (MMD)
//! let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = array![1.2, 2.1, 3.0, 4.1, 5.2];
//! let mmd = maximum_mean_discrepancy(&x, &y, None).unwrap();
//! ```
//!
//! # Fairness and Bias Metrics
//!
//! Metrics for evaluating fairness and bias in machine learning models:
//!
//! ```
//! use ndarray::{array, Array2};
//! use scirs2_metrics::fairness::{
//!     demographic_parity_difference, equalized_odds_difference, equal_opportunity_difference,
//!     disparate_impact, consistency_score
//! };
//! use scirs2_metrics::fairness::bias_detection::{
//!     slice_analysis, subgroup_performance, intersectional_fairness
//! };
//! use scirs2_metrics::classification::accuracy_score;
//!
//! // Example: binary predictions for two protected groups
//! // y_true: ground truth labels (0 or 1)
//! let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
//! // y_pred: predicted labels (0 or 1)
//! let y_pred = array![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
//! // protected_group: binary array indicating protected group membership (1 for protected group, 0 otherwise)
//! let protected_group = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
//!
//! // Compute demographic parity difference
//! // A value of 0 indicates perfect demographic parity
//! let dp_diff = demographic_parity_difference(&y_pred, &protected_group).unwrap();
//!
//! // Compute equalized odds difference
//! // A value of 0 indicates that the false positive and true positive rates are
//! // the same for both groups
//! let eod_diff = equalized_odds_difference(&y_true, &y_pred, &protected_group).unwrap();
//!
//! // Compute equal opportunity difference
//! // A value of 0 indicates equal true positive rates across groups
//! let eo_diff = equal_opportunity_difference(&y_true, &y_pred, &protected_group).unwrap();
//!
//! // Calculate disparate impact
//! // A value of 1.0 indicates perfect fairness; less than 0.8 or greater than 1.25
//! // is often considered problematic
//! let di = disparate_impact(&y_pred, &protected_group).unwrap();
//!
//! // Comprehensive bias detection
//! // Create a dataset with multiple demographic attributes
//! let features = Array2::from_shape_vec((8, 3), vec![
//!     // Feature columns: age, gender(0=male, 1=female), income_level(0=low, 1=medium, 2=high)
//!     30.0, 0.0, 1.0,
//!     25.0, 0.0, 0.0,
//!     35.0, 1.0, 2.0,
//!     28.0, 1.0, 1.0,
//!     45.0, 0.0, 2.0,
//!     42.0, 0.0, 1.0,
//!     33.0, 1.0, 0.0,
//!     50.0, 1.0, 2.0,
//! ]).unwrap();
//!
//! let ground_truth = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
//! let predictions = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
//!
//! // Analyze model performance across different data slices
//! let slice_results = slice_analysis(
//!     &features,
//!     &[1, 2],  // Use gender and income level columns for slicing
//!     &ground_truth,
//!     &predictions,
//!     |y_t, y_p| {
//!         // Convert Vec<f64> to Array1<f64> for accuracy_score
//!         let y_t_array = ndarray::Array::from_vec(y_t.to_vec());
//!         let y_p_array = ndarray::Array::from_vec(y_p.to_vec());
//!         accuracy_score(&y_t_array, &y_p_array).unwrap_or(0.0)
//!     }
//! ).unwrap();
//!
//! // Analyze performance for intersectional groups
//! let protected_attrs = Array2::from_shape_vec((8, 2), vec![
//!     // gender, income_level (simplified to binary: 0=low, 1=high)
//!     0.0, 1.0,
//!     0.0, 0.0,
//!     1.0, 1.0,
//!     1.0, 1.0,
//!     0.0, 1.0,
//!     0.0, 1.0,
//!     1.0, 0.0,
//!     1.0, 1.0,
//! ]).unwrap();
//!
//! let attr_names = vec!["gender".to_string(), "income".to_string()];
//!
//! // Analyze fairness metrics across intersectional groups
//! let fairness_metrics = intersectional_fairness(
//!     &ground_truth,
//!     &predictions,
//!     &protected_attrs,
//!     &attr_names
//! ).unwrap();
//!
//! // Evaluate model performance across different demographic subgroups
//! let performance_metrics = subgroup_performance(
//!     &ground_truth,
//!     &predictions,
//!     &protected_attrs,
//!     &attr_names,
//!     |y_t, y_p| {
//!         // Convert Vec<f64> to Array1<f64> for accuracy_score
//!         let y_t_array = ndarray::Array::from_vec(y_t.to_vec());
//!         let y_p_array = ndarray::Array::from_vec(y_p.to_vec());
//!         accuracy_score(&y_t_array, &y_p_array).unwrap_or(0.0)
//!     }
//! ).unwrap();
//! ```
//!
//! # Model Evaluation Utilities
//!
//! Utilities for model evaluation like cross-validation:
//!
//! ```
//! use ndarray::{Array, Ix1};
//! use scirs2_metrics::evaluation::train_test_split;
//!
//! let x = Array::<f64, _>::linspace(0., 9., 10).into_shape(Ix1(10)).unwrap();
//! let y = &x * 2.;
//!
//! let (train_arrays, test_arrays) = train_test_split(&[&x, &y], 0.3, Some(42)).unwrap();
//! ```
//!
//! # Optimization and Performance
//!
//! Optimized metrics computation for better performance and memory efficiency:
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::optimization::parallel::ParallelConfig;
//! use scirs2_metrics::optimization::memory::{ChunkedMetrics, StreamingMetric};
//! use scirs2_metrics::optimization::numeric::StableMetrics;
//! use scirs2_metrics::error::{Result, MetricsError};
//! use scirs2_metrics::classification::{accuracy_score, precision_score};
//!
//! // Example data
//! let y_true = array![0, 1, 2, 0, 1, 2];
//! let y_pred = array![0, 2, 1, 0, 0, 2];
//!
//! // Compute metrics with parallel configuration
//! let config = ParallelConfig {
//!     parallel_enabled: true,
//!     min_chunk_size: 1000,
//!     num_threads: None,
//! };
//!
//! // Define metrics functions to compute
//! // Note: We need to specify concrete types for these closures
//! let metrics: Vec<Box<dyn Fn(&ndarray::ArrayBase<ndarray::OwnedRepr<i32>, ndarray::Dim<[usize; 1]>>,
//!                            &ndarray::ArrayBase<ndarray::OwnedRepr<i32>, ndarray::Dim<[usize; 1]>>)
//!                            -> Result<f64> + Send + Sync>> = vec![
//!     Box::new(|y_t, y_p| accuracy_score(y_t, y_p)),
//!     Box::new(|y_t, y_p| precision_score(y_t, y_p, 1)),
//! ];
//!
//! // Use chunked metrics for memory efficiency
//! let chunked = ChunkedMetrics::new()
//!     .with_chunk_size(1000)
//!     .with_parallel_config(config.clone());
//!
//! // Example of a streaming metric for incremental computation
//! struct StreamingMeanAbsoluteError;
//!
//! impl StreamingMetric<f64> for StreamingMeanAbsoluteError {
//!     type State = (f64, usize); // Running sum and count
//!     
//!     fn init_state(&self) -> Self::State {
//!         (0.0, 0)
//!     }
//!     
//!     fn update_state(&self, state: &mut Self::State, batch_true: &[f64], batch_pred: &[f64]) -> Result<()> {
//!         for (y_t, y_p) in batch_true.iter().zip(batch_pred.iter()) {
//!             state.0 += (y_t - y_p).abs();
//!             state.1 += 1;
//!         }
//!         Ok(())
//!     }
//!     
//!     fn finalize(&self, state: &Self::State) -> Result<f64> {
//!         if state.1 == 0 {
//!             return Err(MetricsError::DivisionByZero);
//!         }
//!         Ok(state.0 / state.1 as f64)
//!     }
//! }
//!
//! // Numerically stable computations
//! let stable = StableMetrics::<f64>::default();
//! let p = vec![0.5, 0.5, 0.0];
//! let q = vec![0.25, 0.25, 0.5];
//! let kl = stable.kl_divergence(&p, &q).unwrap();
//! let js = stable.js_divergence(&p, &q).unwrap();
//!
//! // Compute additional stable metrics
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let mean = stable.stable_mean(&data).unwrap();
//! let variance = stable.stable_variance(&data, 1).unwrap(); // Sample variance
//! let std_dev = stable.stable_std(&data, 1).unwrap(); // Sample standard deviation
//! ```
//!
//! # Visualization
//!
//! Visualization utilities for metrics results:
//!
//! ```
//! use ndarray::{array, Array2};
//! use scirs2_metrics::classification::confusion_matrix;
//! use scirs2_metrics::classification::curves::{roc_curve, precision_recall_curve, calibration_curve};
//! use scirs2_metrics::visualization::{
//!     MetricVisualizer, VisualizationData, VisualizationMetadata, PlotType,
//!     confusion_matrix::confusion_matrix_visualization,
//!     roc_curve::roc_curve_visualization,
//!     precision_recall::precision_recall_visualization,
//!     calibration::calibration_visualization,
//!     learning_curve::learning_curve_visualization,
//!     interactive::interactive_roc_curve_visualization
//! };
//!
//! // Example: Confusion matrix visualization
//! let y_true = array![0, 1, 2, 0, 1, 2];
//! let y_pred = array![0, 2, 1, 0, 0, 2];
//!
//! let (cm, classes) = confusion_matrix(&y_true, &y_pred, None).unwrap();
//! let labels = vec!["Class 0".to_string(), "Class 1".to_string(), "Class 2".to_string()];
//!
//! // Convert to f64 for visualization
//! let cm_f64 = cm.mapv(|x| x as f64);
//! let cm_viz = confusion_matrix_visualization(cm_f64, Some(labels), false);
//!
//! // Get data and metadata for visualization
//! let viz_data = cm_viz.prepare_data().unwrap();
//! let viz_metadata = cm_viz.get_metadata();
//!
//! // Example: ROC curve visualization
//! let y_true_binary = array![0, 1, 1, 0, 1, 0];
//! let y_score = array![0.1, 0.8, 0.7, 0.2, 0.9, 0.3];
//!
//! let (fpr, tpr, thresholds) = roc_curve(&y_true_binary, &y_score).unwrap();
//! let auc = 0.83; // Example AUC value
//!
//! let roc_viz = roc_curve_visualization(fpr.to_vec(), tpr.to_vec(), Some(thresholds.to_vec()), Some(auc));
//!
//! // Example: Interactive ROC curve visualization with threshold adjustment
//! let interactive_roc_viz = interactive_roc_curve_visualization(
//!     fpr.to_vec(), tpr.to_vec(), Some(thresholds.to_vec()), Some(auc));
//!
//! // Example: Precision-Recall curve visualization
//! let (precision, recall, pr_thresholds) = precision_recall_curve(&y_true_binary, &y_score).unwrap();
//! let ap = 0.75; // Example average precision
//!
//! let pr_viz = precision_recall_visualization(precision.to_vec(), recall.to_vec(), Some(pr_thresholds.to_vec()), Some(ap));
//!
//! // Example: Calibration curve visualization
//! let (prob_true, prob_pred, counts) = calibration_curve(&y_true_binary, &y_score, Some(5)).unwrap();
//!
//! let cal_viz = calibration_visualization(prob_true.to_vec(), prob_pred.to_vec(), 5, "uniform".to_string());
//!
//! // Example: Learning curve visualization
//! let train_sizes = vec![10, 30, 50, 100, 200];
//! let train_scores = vec![
//!     vec![0.6, 0.62, 0.64],  // 10 samples
//!     vec![0.7, 0.72, 0.74],  // 30 samples
//!     vec![0.75, 0.77, 0.79], // 50 samples
//!     vec![0.8, 0.82, 0.84],  // 100 samples
//!     vec![0.85, 0.87, 0.89], // 200 samples
//! ];
//! let val_scores = vec![
//!     vec![0.5, 0.52, 0.54],  // 10 samples
//!     vec![0.6, 0.62, 0.64],  // 30 samples
//!     vec![0.65, 0.67, 0.69], // 50 samples
//!     vec![0.7, 0.72, 0.74],  // 100 samples
//!     vec![0.75, 0.77, 0.79], // 200 samples
//! ];
//!
//! let lc_viz = learning_curve_visualization(train_sizes, train_scores, val_scores, "Accuracy").unwrap();
//! ```
//!
//! ## Interactive Visualizations
//!
//! The library also provides interactive visualizations that allow for dynamic exploration
//! of metrics via web interfaces:
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::classification::curves::roc_curve;
//! use scirs2_metrics::visualization::{
//!     helpers, InteractiveOptions,
//!     backends::{default_interactive_backend, PlotlyInteractiveBackendInterface},
//! };
//!
//! // Create binary classification data
//! let y_true = array![0, 0, 0, 0, 1, 1, 1, 1];
//! let y_score = array![0.1, 0.2, 0.4, 0.6, 0.5, 0.7, 0.8, 0.9];
//!
//! // Compute ROC curve
//! let (fpr, tpr, thresholds) = roc_curve(&y_true, &y_score).unwrap();
//!
//! // Interactive ROC curve with threshold adjustment
//! let interactive_options = InteractiveOptions {
//!     width: 900,
//!     height: 600,
//!     show_threshold_slider: true,
//!     show_metric_values: true,
//!     show_confusion_matrix: true,
//!     custom_layout: std::collections::HashMap::new(),
//! };
//!
//! // Create interactive ROC curve visualization
//! let viz = helpers::visualize_interactive_roc_curve(
//!     fpr.view(),
//!     tpr.view(),
//!     Some(thresholds.view()),
//!     Some(0.94), // AUC value
//!     Some(interactive_options),
//! );
//!
//! // Note: In a real application, you would save this to an HTML file with:
//! // let viz_data = viz.prepare_data().unwrap();
//! // let viz_metadata = viz.get_metadata();
//! // let backend = default_interactive_backend();
//! // backend.save_interactive_roc(&viz_data, &viz_metadata, &Default::default(), "interactive_roc.html");
//! ```
//!
//! # Metric Serialization
//!
//! Utilities for saving, loading, and comparing metric results:
//!
//! ```
//! use std::collections::HashMap;
//! use scirs2_metrics::serialization::{
//!     MetricResult, MetricMetadata, MetricCollection, SerializationFormat,
//!     create_metric_result,
//!     comparison::compare_collections
//! };
//!
//! // Create metric results
//! let accuracy_metadata = MetricMetadata {
//!     dataset_id: Some("test_dataset".to_string()),
//!     model_id: Some("model_v1".to_string()),
//!     parameters: Some({
//!         let mut params = HashMap::new();
//!         params.insert("normalize".to_string(), "true".to_string());
//!         params
//!     }),
//!     additional_metadata: None,
//! };
//!
//! let accuracy = create_metric_result(
//!     "accuracy",
//!     0.85,
//!     None,
//!     Some(accuracy_metadata),
//! );
//!
//! let f1_score = create_metric_result(
//!     "f1_score",
//!     0.82,
//!     Some({
//!         let mut values = HashMap::new();
//!         values.insert("precision".to_string(), 0.80);
//!         values.insert("recall".to_string(), 0.84);
//!         values
//!     }),
//!     None,
//! );
//!
//! // Create a metric collection
//! let mut collection1 = MetricCollection::new(
//!     "Model Evaluation - v1",
//!     Some("Evaluation results for model version 1"),
//! );
//!
//! collection1.add_metric(accuracy);
//! collection1.add_metric(f1_score);
//!
//! // Create another collection for comparison
//! let mut collection2 = MetricCollection::new(
//!     "Model Evaluation - v2",
//!     Some("Evaluation results for model version 2"),
//! );
//!
//! let accuracy_v2 = create_metric_result("accuracy", 0.87, None, None);
//! let f1_score_v2 = create_metric_result("f1_score", 0.84, None, None);
//!
//! collection2.add_metric(accuracy_v2);
//! collection2.add_metric(f1_score_v2);
//!
//! // Compare collections
//! let comparison = compare_collections(&collection1, &collection2, Some(0.01));
//!
//! // Save collection to a file (in-memory example)
//! // collection1.save("metrics_v1.json", SerializationFormat::Json).unwrap();
//! // collection2.save("metrics_v2.json", SerializationFormat::Json).unwrap();
//!
//! // Load collection from a file (in-memory example)
//! // let loaded = MetricCollection::load("metrics_v1.json", SerializationFormat::Json).unwrap();
//! ```

#![allow(
    unused_imports,
    dead_code,
    unexpected_cfgs,
    clippy::clone_on_copy,
    clippy::needless_range_loop,
    clippy::map_entry,
    clippy::len_zero
)]
//#![warn(missing_docs)]

pub mod anomaly;
pub mod bayesian;
pub mod classification;
pub mod clustering;
pub mod custom;
pub mod dashboard;
pub mod domains;
pub mod error;
pub mod evaluation;
pub mod fairness;

// Integration modules with conditional compilation
#[cfg(any(feature = "neural_common", feature = "optim_integration"))]
pub mod integration;

pub mod optimization;
pub mod ranking;
pub mod regression;
pub mod selection;
pub mod serialization;
pub mod sklearn_compat;
pub mod streaming;
pub mod visualization;
