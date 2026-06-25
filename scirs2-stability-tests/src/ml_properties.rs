//! Property-based validation utilities for ML algorithms.
//!
//! Provides generic test harnesses that any ML library can use to verify
//! that algorithm implementations satisfy fundamental mathematical properties.
//! These validators are independent of any specific ML framework -- they operate
//! on plain slices and vectors.

/// Properties that clustering algorithms should satisfy.
pub mod clustering {
    /// Verify that every point is assigned to exactly one cluster
    /// and all labels are in the range `[0, n_clusters)`.
    pub fn validate_partition(labels: &[usize], n_clusters: usize) -> bool {
        if labels.is_empty() {
            return n_clusters == 0;
        }
        labels.iter().all(|&l| l < n_clusters)
    }

    /// Verify that every cluster index in `[0, n_clusters)` has at least one assigned point.
    pub fn validate_non_empty_clusters(labels: &[usize], n_clusters: usize) -> bool {
        if n_clusters == 0 {
            return labels.is_empty();
        }
        let mut counts = vec![0usize; n_clusters];
        for &l in labels {
            if l >= n_clusters {
                return false;
            }
            counts[l] += 1;
        }
        counts.iter().all(|&c| c > 0)
    }

    /// Verify the elbow property: inertia (sum of squared distances to centroids)
    /// should be non-increasing as the number of clusters k increases.
    ///
    /// `inertias` is a slice of `(k, inertia)` pairs, assumed sorted by k.
    /// A small tolerance `tol` allows for floating-point noise.
    pub fn validate_elbow_property(inertias: &[(usize, f64)], tol: f64) -> bool {
        if inertias.len() < 2 {
            return true;
        }
        for w in inertias.windows(2) {
            // When k increases, inertia should not increase (within tolerance)
            if w[0].0 < w[1].0 && w[1].1 > w[0].1 + tol {
                return false;
            }
        }
        true
    }

    /// Verify that two runs produce identical cluster assignments.
    /// (For deterministic algorithms with the same seed/input.)
    pub fn validate_determinism(labels1: &[usize], labels2: &[usize]) -> bool {
        if labels1.len() != labels2.len() {
            return false;
        }
        labels1.iter().zip(labels2.iter()).all(|(a, b)| a == b)
    }

    /// Verify that cluster labels are contiguous starting from 0.
    /// That is, every integer in `[0, max_label]` appears at least once.
    pub fn validate_contiguous_labels(labels: &[usize]) -> bool {
        if labels.is_empty() {
            return true;
        }
        let max_label = match labels.iter().max() {
            Some(&m) => m,
            None => return true,
        };
        let mut seen = vec![false; max_label + 1];
        for &l in labels {
            seen[l] = true;
        }
        seen.iter().all(|&s| s)
    }

    /// Validate that within-cluster distances are smaller than between-cluster
    /// distances on average. Returns the ratio (within / between); values < 1
    /// indicate well-separated clusters.
    pub fn compute_cluster_separation_ratio(points: &[[f64; 2]], labels: &[usize]) -> Option<f64> {
        if points.len() != labels.len() || points.is_empty() {
            return None;
        }

        let mut within_sum = 0.0;
        let mut within_count = 0u64;
        let mut between_sum = 0.0;
        let mut between_count = 0u64;

        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dx = points[i][0] - points[j][0];
                let dy = points[i][1] - points[j][1];
                let dist = (dx * dx + dy * dy).sqrt();
                if labels[i] == labels[j] {
                    within_sum += dist;
                    within_count += 1;
                } else {
                    between_sum += dist;
                    between_count += 1;
                }
            }
        }

        if within_count == 0 || between_count == 0 {
            return None;
        }

        let within_avg = within_sum / within_count as f64;
        let between_avg = between_sum / between_count as f64;
        if between_avg.abs() < f64::EPSILON {
            return None;
        }
        Some(within_avg / between_avg)
    }
}

/// Properties that regression algorithms should satisfy.
pub mod regression {
    /// Verify that residuals sum to approximately zero.
    /// This holds for OLS with an intercept term.
    pub fn validate_residual_sum_zero(residuals: &[f64], tol: f64) -> bool {
        if residuals.is_empty() {
            return true;
        }
        let sum: f64 = residuals.iter().sum();
        sum.abs() <= tol
    }

    /// Verify that R^2 is in `[0, 1]` for in-sample predictions.
    /// (Out-of-sample R^2 can be negative, so this checks in-sample only.)
    pub fn validate_r_squared_range(r2: f64) -> bool {
        (-f64::EPSILON..=1.0 + f64::EPSILON).contains(&r2)
    }

    /// Verify that the model perfectly interpolates the training data
    /// when `n_features >= n_samples` (under-determined system).
    pub fn validate_interpolation(y_true: &[f64], y_pred: &[f64], tol: f64) -> bool {
        if y_true.len() != y_pred.len() {
            return false;
        }
        y_true
            .iter()
            .zip(y_pred.iter())
            .all(|(t, p)| (t - p).abs() <= tol)
    }

    /// Verify that R^2 is non-decreasing as more features are added
    /// (on training data). `r2_values` is `(n_features, r2)` sorted by n_features.
    pub fn validate_monotone_r2(r2_values: &[(usize, f64)], tol: f64) -> bool {
        if r2_values.len() < 2 {
            return true;
        }
        for w in r2_values.windows(2) {
            if w[0].0 < w[1].0 && w[1].1 < w[0].1 - tol {
                return false;
            }
        }
        true
    }

    /// Compute R^2 from true and predicted values.
    pub fn compute_r_squared(y_true: &[f64], y_pred: &[f64]) -> Option<f64> {
        if y_true.len() != y_pred.len() || y_true.is_empty() {
            return None;
        }
        let n = y_true.len() as f64;
        let mean = y_true.iter().sum::<f64>() / n;
        let ss_tot: f64 = y_true.iter().map(|y| (y - mean) * (y - mean)).sum();
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p) * (t - p))
            .sum();

        if ss_tot.abs() < f64::EPSILON {
            return None; // Constant y -- R^2 undefined
        }
        Some(1.0 - ss_res / ss_tot)
    }

    /// Validate that predictions on a scaled input produce correspondingly
    /// scaled outputs (for linear models). Checks approximate linearity.
    pub fn validate_linear_scaling(
        y_pred_original: &[f64],
        y_pred_scaled: &[f64],
        scale_factor: f64,
        tol: f64,
    ) -> bool {
        if y_pred_original.len() != y_pred_scaled.len() {
            return false;
        }
        y_pred_original
            .iter()
            .zip(y_pred_scaled.iter())
            .all(|(orig, scaled)| (orig * scale_factor - scaled).abs() <= tol)
    }
}

/// Properties that classification algorithms should satisfy.
pub mod classification {
    /// Verify that all predictions are members of the valid label set.
    pub fn validate_label_set(predictions: &[usize], valid_labels: &[usize]) -> bool {
        predictions.iter().all(|p| valid_labels.contains(p))
    }

    /// Verify that accuracy is in `[0, 1]`.
    pub fn validate_accuracy_range(accuracy: f64) -> bool {
        (-f64::EPSILON..=1.0 + f64::EPSILON).contains(&accuracy)
    }

    /// Compute accuracy from true and predicted labels.
    pub fn compute_accuracy(y_true: &[usize], y_pred: &[usize]) -> Option<f64> {
        if y_true.len() != y_pred.len() || y_true.is_empty() {
            return None;
        }
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| t == p)
            .count();
        Some(correct as f64 / y_true.len() as f64)
    }

    /// Verify that the confusion matrix row sums match the true class counts.
    /// `cm` is a `n_classes x n_classes` matrix where `cm[i][j]` = count of
    /// true class `i` predicted as class `j`.
    pub fn validate_confusion_matrix(cm: &[Vec<usize>], true_labels: &[usize]) -> bool {
        if cm.is_empty() {
            return true;
        }
        let n_classes = cm.len();

        // Verify square matrix
        if cm.iter().any(|row| row.len() != n_classes) {
            return false;
        }

        // Count true class occurrences
        let mut class_counts = vec![0usize; n_classes];
        for &l in true_labels {
            if l >= n_classes {
                return false;
            }
            class_counts[l] += 1;
        }

        // Check row sums match class counts
        for (i, row) in cm.iter().enumerate() {
            let row_sum: usize = row.iter().sum();
            if row_sum != class_counts[i] {
                return false;
            }
        }
        true
    }

    /// Verify that predicted probabilities lie on the probability simplex:
    /// each row sums to 1 and all entries are non-negative.
    pub fn validate_probability_simplex(probs: &[Vec<f64>], tol: f64) -> bool {
        probs.iter().all(|row| {
            if row.is_empty() {
                return false;
            }
            let all_non_negative = row.iter().all(|&p| p >= -tol);
            let sum: f64 = row.iter().sum();
            all_non_negative && (sum - 1.0).abs() <= tol
        })
    }

    /// Compute precision, recall, and F1 for a binary classification problem.
    /// Returns `(precision, recall, f1)` or `None` if undefined.
    pub fn compute_precision_recall_f1(
        y_true: &[bool],
        y_pred: &[bool],
    ) -> Option<(f64, f64, f64)> {
        if y_true.len() != y_pred.len() || y_true.is_empty() {
            return None;
        }

        let mut tp = 0u64;
        let mut fp = 0u64;
        let mut r#fn = 0u64;

        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            match (t, p) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => r#fn += 1,
                _ => {}
            }
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            return None;
        };

        let recall = if tp + r#fn > 0 {
            tp as f64 / (tp + r#fn) as f64
        } else {
            return None;
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Some((precision, recall, f1))
    }
}

/// Properties that dimensionality reduction algorithms should satisfy.
pub mod dim_reduction {
    /// Verify that the output has the correct shape.
    pub fn validate_output_shape(
        output: &[Vec<f64>],
        n_samples: usize,
        n_components: usize,
    ) -> bool {
        output.len() == n_samples && output.iter().all(|row| row.len() == n_components)
    }

    /// Verify that PCA components are orthogonal to each other.
    /// `components` is a list of component vectors (rows of the projection matrix).
    pub fn validate_orthogonal_components(components: &[Vec<f64>], tol: f64) -> bool {
        if components.len() < 2 {
            return true;
        }
        let d = components[0].len();
        if components.iter().any(|c| c.len() != d) {
            return false;
        }
        for i in 0..components.len() {
            for j in (i + 1)..components.len() {
                let dot: f64 = components[i]
                    .iter()
                    .zip(components[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                if dot.abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Verify that PCA components are unit-length.
    pub fn validate_unit_components(components: &[Vec<f64>], tol: f64) -> bool {
        components.iter().all(|c| {
            let norm_sq: f64 = c.iter().map(|x| x * x).sum();
            (norm_sq.sqrt() - 1.0).abs() <= tol
        })
    }

    /// Verify that explained variance ratios are non-negative and sum to at most 1.
    pub fn validate_explained_variance(ratios: &[f64], tol: f64) -> bool {
        let all_non_negative = ratios.iter().all(|&r| r >= -tol);
        let sum: f64 = ratios.iter().sum();
        all_non_negative && sum <= 1.0 + tol
    }

    /// Compute the Spearman-like distance correlation between high-dimensional
    /// and low-dimensional pairwise distances. Returns a value in `[-1, 1]`
    /// where values close to 1 indicate good distance preservation.
    pub fn validate_distance_correlation(d_high: &[f64], d_low: &[f64]) -> Option<f64> {
        if d_high.len() != d_low.len() || d_high.is_empty() {
            return None;
        }

        let n = d_high.len() as f64;
        let mean_h = d_high.iter().sum::<f64>() / n;
        let mean_l = d_low.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_h = 0.0;
        let mut var_l = 0.0;

        for i in 0..d_high.len() {
            let dh = d_high[i] - mean_h;
            let dl = d_low[i] - mean_l;
            cov += dh * dl;
            var_h += dh * dh;
            var_l += dl * dl;
        }

        let denom = (var_h * var_l).sqrt();
        if denom.abs() < f64::EPSILON {
            return None;
        }
        Some(cov / denom)
    }

    /// Verify that the number of components does not exceed the number of
    /// features or the number of samples (PCA constraint).
    pub fn validate_component_count(
        n_samples: usize,
        n_features: usize,
        n_components: usize,
    ) -> bool {
        n_components <= n_samples.min(n_features)
    }
}

/// Properties that tree-based methods should satisfy.
pub mod trees {
    /// Verify that the number of leaves respects the max-depth constraint.
    /// A binary tree of depth `d` has at most `2^d` leaves.
    pub fn validate_max_depth_constraint(n_leaves: usize, max_depth: usize) -> bool {
        if max_depth >= 64 {
            // Avoid overflow
            return true;
        }
        n_leaves <= (1usize << max_depth)
    }

    /// Verify that feature importances sum to approximately 1.
    pub fn validate_feature_importance_sum(importances: &[f64], tol: f64) -> bool {
        if importances.is_empty() {
            return true;
        }
        let all_non_negative = importances.iter().all(|&v| v >= -tol);
        let sum: f64 = importances.iter().sum();
        all_non_negative && (sum - 1.0).abs() <= tol
    }

    /// Verify that feature importances are all non-negative.
    pub fn validate_feature_importance_non_negative(importances: &[f64]) -> bool {
        importances.iter().all(|&v| v >= 0.0)
    }

    /// Verify the monotone property of ensembles: adding more estimators
    /// should not decrease training accuracy. `accuracies` is
    /// `(n_estimators, accuracy)` sorted by n_estimators.
    pub fn validate_ensemble_monotone(accuracies: &[(usize, f64)], tol: f64) -> bool {
        if accuracies.len() < 2 {
            return true;
        }
        for w in accuracies.windows(2) {
            if w[0].0 < w[1].0 && w[1].1 < w[0].1 - tol {
                return false;
            }
        }
        true
    }

    /// Verify that a decision tree partitions the feature space:
    /// every sample should reach exactly one leaf.
    pub fn validate_single_leaf_per_sample(leaf_assignments: &[usize]) -> bool {
        // Every sample has exactly one leaf (by definition of a tree traversal),
        // so we just check that no assignment is a sentinel "unassigned" value.
        // Using usize::MAX as sentinel.
        leaf_assignments.iter().all(|&l| l != usize::MAX)
    }
}

/// Generic ML metric validators.
pub mod metrics {
    /// Verify that a metric value is in `[0, 1]`.
    pub fn validate_metric_range(value: f64) -> bool {
        (-f64::EPSILON..=1.0 + f64::EPSILON).contains(&value)
    }

    /// Verify the F1 formula: `F1 = 2*P*R / (P+R)` when `P+R > 0`.
    pub fn validate_f1_formula(precision: f64, recall: f64, f1: f64, tol: f64) -> bool {
        if precision + recall < f64::EPSILON {
            return f1.abs() < tol;
        }
        let expected = 2.0 * precision * recall / (precision + recall);
        (f1 - expected).abs() <= tol
    }

    /// Verify that AUC-ROC is in `[0, 1]`.
    pub fn validate_auc_range(auc: f64) -> bool {
        (-f64::EPSILON..=1.0 + f64::EPSILON).contains(&auc)
    }

    /// Verify that the silhouette score is in `[-1, 1]`.
    pub fn validate_silhouette_range(score: f64) -> bool {
        (-1.0 - f64::EPSILON..=1.0 + f64::EPSILON).contains(&score)
    }

    /// Verify that the Matthews Correlation Coefficient is in `[-1, 1]`.
    pub fn validate_mcc_range(mcc: f64) -> bool {
        (-1.0 - f64::EPSILON..=1.0 + f64::EPSILON).contains(&mcc)
    }

    /// Verify that log loss is non-negative.
    pub fn validate_log_loss_non_negative(loss: f64) -> bool {
        loss >= -f64::EPSILON
    }

    /// Compute the mean absolute error.
    pub fn compute_mae(y_true: &[f64], y_pred: &[f64]) -> Option<f64> {
        if y_true.len() != y_pred.len() || y_true.is_empty() {
            return None;
        }
        let sum: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).abs())
            .sum();
        Some(sum / y_true.len() as f64)
    }

    /// Compute the mean squared error.
    pub fn compute_mse(y_true: &[f64], y_pred: &[f64]) -> Option<f64> {
        if y_true.len() != y_pred.len() || y_true.is_empty() {
            return None;
        }
        let sum: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p) * (t - p))
            .sum();
        Some(sum / y_true.len() as f64)
    }

    /// Verify metric symmetry: `metric(a, b) == metric(b, a)` within tolerance.
    pub fn validate_metric_symmetry(metric_ab: f64, metric_ba: f64, tol: f64) -> bool {
        (metric_ab - metric_ba).abs() <= tol
    }
}
