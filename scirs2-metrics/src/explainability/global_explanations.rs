//! Global explanation methods for understanding overall model behavior

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::numeric::Float;
use statrs::statistics::Statistics;
use std::collections::HashMap;

/// Global explainer for model-level insights
pub struct GlobalExplainer<F: Float> {
    /// Number of samples for global analysis
    pub n_samples: usize,
    /// Bootstrap samples for stability analysis
    pub n_bootstrap: usize,
    /// Random seed
    pub random_seed: Option<u64>,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum> Default
    for GlobalExplainer<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum> GlobalExplainer<F> {
    /// Create new global explainer
    pub fn new() -> Self {
        Self {
            n_samples: 1000,
            n_bootstrap: 100,
            random_seed: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set number of samples
    pub fn with_samples(mut self, n: usize) -> Self {
        self.n_samples = n;
        self
    }

    /// Set number of bootstrap samples
    pub fn with_bootstrap(mut self, n: usize) -> Self {
        self.n_bootstrap = n;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Compute global feature importance
    pub fn global_feature_importance<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
        feature_names: &[String],
    ) -> Result<GlobalExplanation<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Compute feature importance using multiple methods
        let permutation_importance =
            self.compute_global_permutation_importance(model, xdata, feature_names)?;
        let variance_importance =
            self.compute_variance_based_importance(model, xdata, feature_names)?;
        let interaction_effects = self.compute_interaction_effects(model, xdata, feature_names)?;

        // Compute model complexity metrics
        let complexity_metrics = self.compute_model_complexity(model, xdata)?;

        // Stability analysis
        let stability_metrics = self.compute_stability_metrics(model, xdata, feature_names)?;

        Ok(GlobalExplanation {
            feature_importance: permutation_importance,
            variance_importance,
            interaction_effects,
            complexity_metrics,
            stability_metrics,
            sample_size: xdata.nrows(),
        })
    }

    /// Compute partial dependence plots data
    pub fn partial_dependence<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
        featureidx: usize,
        n_grid_points: usize,
    ) -> Result<PartialDependencePlot<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        if featureidx >= xdata.ncols() {
            return Err(MetricsError::InvalidInput(
                "Feature index out of bounds".to_string(),
            ));
        }

        let feature_column = xdata.column(featureidx);
        let min_val = feature_column.iter().cloned().fold(F::infinity(), F::min);
        let max_val = feature_column
            .iter()
            .cloned()
            .fold(F::neg_infinity(), F::max);

        let mut grid_values = Vec::new();
        let mut pd_values = Vec::new();

        for i in 0..n_grid_points {
            let t = F::from(i).unwrap() / F::from(n_grid_points - 1).unwrap();
            let grid_val = min_val + t * (max_val - min_val);
            grid_values.push(grid_val);

            // Create modified dataset with feature set to grid value
            let mut x_modified = xdata.clone();
            for j in 0..x_modified.nrows() {
                x_modified[[j, featureidx]] = grid_val;
            }

            let predictions = model(&x_modified.view());
            let mean_prediction = predictions.mean().unwrap_or(F::zero());
            pd_values.push(mean_prediction);
        }

        Ok(PartialDependencePlot {
            featureidx,
            grid_values,
            pd_values,
            ice_curves: None, // Could be computed separately
        })
    }

    /// Compute individual conditional expectation (ICE) curves
    pub fn ice_curves<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
        featureidx: usize,
        n_grid_points: usize,
        max_instances: Option<usize>,
    ) -> Result<Vec<ICECurve<F>>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        if featureidx >= xdata.ncols() {
            return Err(MetricsError::InvalidInput(
                "Feature index out of bounds".to_string(),
            ));
        }

        let feature_column = xdata.column(featureidx);
        let min_val = feature_column.iter().cloned().fold(F::infinity(), F::min);
        let max_val = feature_column
            .iter()
            .cloned()
            .fold(F::neg_infinity(), F::max);

        let n_instances = max_instances.unwrap_or(xdata.nrows()).min(xdata.nrows());
        let mut ice_curves = Vec::new();

        for instance_idx in 0..n_instances {
            let mut grid_values = Vec::new();
            let mut predictions = Vec::new();

            for i in 0..n_grid_points {
                let t = F::from(i).unwrap() / F::from(n_grid_points - 1).unwrap();
                let grid_val = min_val + t * (max_val - min_val);
                grid_values.push(grid_val);

                // Create instance with modified feature value
                let mut instance = xdata.row(instance_idx).to_owned();
                instance[featureidx] = grid_val;

                let prediction = model(&instance.insert_axis(Axis(0)).view())[0];
                predictions.push(prediction);
            }

            ice_curves.push(ICECurve {
                instance_idx,
                grid_values: grid_values.clone(),
                predictions,
            });
        }

        Ok(ice_curves)
    }

    /// Analyze feature interactions using functional ANOVA
    pub fn feature_interactions<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
        feature_names: &[String],
        max_interaction_order: usize,
    ) -> Result<InteractionAnalysis<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_features = xdata.ncols();
        let mut pairwiseinteractions = HashMap::new();
        let mut higher_order_interactions = HashMap::new();

        // Compute pairwise interactions
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let interaction_strength = self.compute_pairwise_interaction(model, xdata, i, j)?;
                let pair_name = format!(
                    "{}_{}",
                    feature_names.get(i).unwrap_or(&i.to_string()),
                    feature_names.get(j).unwrap_or(&j.to_string())
                );
                pairwiseinteractions.insert(pair_name, interaction_strength);
            }
        }

        // Compute higher-_order interactions if requested
        if max_interaction_order > 2 {
            for _order in 3..=max_interaction_order.min(n_features) {
                let interactions =
                    self.compute_higher_order_interactions(model, xdata, feature_names, _order)?;
                higher_order_interactions.insert(_order, interactions);
            }
        }

        let total_interaction_strength =
            self.compute_total_interaction_strength(&pairwiseinteractions);

        Ok(InteractionAnalysis {
            pairwiseinteractions,
            higher_order_interactions,
            total_interaction_strength,
        })
    }

    /// Compute global model behavior summary
    pub fn model_behavior_summary<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
        feature_names: &[String],
    ) -> Result<ModelBehaviorSummary<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let predictions = model(&xdata.view());

        // Basic prediction statistics
        let mean_prediction = predictions.mean().unwrap_or(F::zero());
        let prediction_variance = self.compute_variance(&predictions)?;
        let prediction_range = (
            predictions.iter().cloned().fold(F::infinity(), F::min),
            predictions.iter().cloned().fold(F::neg_infinity(), F::max),
        );

        // Feature sensitivity analysis
        let feature_sensitivity = self.compute_feature_sensitivity(model, xdata, feature_names)?;

        // Model linearity assessment
        let linearity_score = self.assess_model_linearity(model, xdata)?;

        // Prediction confidence/uncertainty
        let prediction_uncertainty = self.compute_prediction_uncertainty(model, xdata)?;

        Ok(ModelBehaviorSummary {
            mean_prediction,
            prediction_variance,
            prediction_range,
            feature_sensitivity,
            linearity_score,
            prediction_uncertainty,
            sample_coverage: self.compute_sample_coverage(xdata)?,
        })
    }

    // Helper methods

    fn compute_global_permutation_importance<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let baseline_predictions = model(&xdata.view());
        let baseline_variance = self.compute_variance(&baseline_predictions)?;

        let mut importance_scores = HashMap::new();

        for (i, feature_name) in feature_names.iter().enumerate() {
            if i >= xdata.ncols() {
                continue;
            }

            let mut x_permuted = xdata.clone();
            self.permute_column(&mut x_permuted, i)?;

            let permuted_predictions = model(&x_permuted.view());
            let permuted_variance = self.compute_variance(&permuted_predictions)?;

            let importance = (baseline_variance - permuted_variance) / baseline_variance;
            importance_scores.insert(feature_name.clone(), importance);
        }

        Ok(importance_scores)
    }

    fn compute_variance_based_importance<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let total_variance = self.compute_total_variance(model, xdata)?;
        let mut importance_scores = HashMap::new();

        for (i, feature_name) in feature_names.iter().enumerate() {
            if i >= xdata.ncols() {
                continue;
            }

            let feature_variance = self.compute_feature_variance(model, xdata, i)?;
            let importance = feature_variance / total_variance;
            importance_scores.insert(feature_name.clone(), importance);
        }

        Ok(importance_scores)
    }

    fn compute_interaction_effects<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let mut interaction_effects = HashMap::new();
        let n_features = xdata.ncols();

        // Sample a subset for efficiency
        let _sample_size = self.n_samples.min(xdata.nrows());

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let interaction = self.compute_pairwise_interaction(model, xdata, i, j)?;
                let pair_name = format!(
                    "{}_{}",
                    feature_names.get(i).unwrap_or(&i.to_string()),
                    feature_names.get(j).unwrap_or(&j.to_string())
                );
                interaction_effects.insert(pair_name, interaction);
            }
        }

        Ok(interaction_effects)
    }

    fn compute_model_complexity<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
    ) -> Result<ModelComplexityMetrics<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let predictions = model(&xdata.view());

        // Effective degrees of freedom (simplified approximation)
        let effective_dof = self.estimate_effective_dof(model, xdata)?;

        // Model smoothness (local variation)
        let smoothness = self.compute_model_smoothness(model, xdata)?;

        // Prediction diversity
        let prediction_diversity = self.compute_variance(&predictions)?;

        Ok(ModelComplexityMetrics {
            effective_degrees_of_freedom: effective_dof,
            smoothness,
            prediction_diversity,
        })
    }

    fn compute_stability_metrics<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
        feature_names: &[String],
    ) -> Result<StabilityMetrics<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let mut bootstrap_importances = Vec::new();

        // Bootstrap sampling for stability
        for _ in 0..self.n_bootstrap {
            let bootstrap_indices = self.generate_bootstrap_indices(xdata.nrows())?;
            let bootstrap_data = self.sample_by_indices(xdata, &bootstrap_indices)?;

            let importance =
                self.compute_global_permutation_importance(model, &bootstrap_data, feature_names)?;
            bootstrap_importances.push(importance);
        }

        // Compute stability statistics
        let importance_stability =
            self.compute_importance_stability(&bootstrap_importances, feature_names)?;
        let ranking_stability =
            self.compute_ranking_stability(&bootstrap_importances, feature_names)?;

        Ok(StabilityMetrics {
            importance_stability,
            ranking_stability,
            bootstrap_samples: self.n_bootstrap,
        })
    }

    fn compute_pairwise_interaction<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
        feature_i: usize,
        feature_j: usize,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Simplified interaction computation using partial dependence
        let sample_size = self.n_samples.min(xdata.nrows());

        let mut interaction_sum = F::zero();
        let mut count = 0;

        for idx in 0..sample_size {
            let instance = xdata.row(idx);

            // Get baseline prediction
            let baseline = model(&instance.insert_axis(Axis(0)).view())[0];

            // Vary both features
            let val_i = instance[feature_i];
            let val_j = instance[feature_j];

            let mut modified = instance.to_owned();

            // Perturb feature _i
            modified[feature_i] = val_i + self.get_feature_std(xdata, feature_i)?;
            let pred_i = model(&modified.clone().insert_axis(Axis(0)).view())[0];

            // Perturb feature _j
            modified[feature_i] = val_i;
            modified[feature_j] = val_j + self.get_feature_std(xdata, feature_j)?;
            let pred_j = model(&modified.clone().insert_axis(Axis(0)).view())[0];

            // Perturb both features
            modified[feature_i] = val_i + self.get_feature_std(xdata, feature_i)?;
            let pred_ij = model(&modified.clone().insert_axis(Axis(0)).view())[0];

            // Interaction effect
            let interaction = pred_ij - pred_i - pred_j + baseline;
            interaction_sum = interaction_sum + interaction.abs();
            count += 1;
        }

        Ok(if count > 0 {
            interaction_sum / F::from(count).unwrap()
        } else {
            F::zero()
        })
    }

    fn compute_higher_order_interactions<M>(
        &self,
        model: &M,
        _data: &Array2<F>,
        _feature_names: &[String],
        _order: usize,
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Simplified higher-_order interaction computation
        // In practice, this would use ANOVA decomposition or other methods
        Ok(HashMap::new())
    }

    fn compute_total_interaction_strength(&self, pairwiseinteractions: &HashMap<String, F>) -> F {
        pairwiseinteractions.values().cloned().sum()
    }

    fn compute_feature_sensitivity<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let mut sensitivity_scores = HashMap::new();

        for (i, feature_name) in feature_names.iter().enumerate() {
            if i >= xdata.ncols() {
                continue;
            }

            let sensitivity = self.compute_single_feature_sensitivity(model, xdata, i)?;
            sensitivity_scores.insert(feature_name.clone(), sensitivity);
        }

        Ok(sensitivity_scores)
    }

    fn compute_single_feature_sensitivity<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
        featureidx: usize,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let feature_std = self.get_feature_std(xdata, featureidx)?;
        let sample_size = self.n_samples.min(xdata.nrows());

        let mut sensitivity_sum = F::zero();

        for i in 0..sample_size {
            let instance = xdata.row(i);
            let baseline_pred = model(&instance.insert_axis(Axis(0)).view())[0];

            let mut perturbed = instance.to_owned();
            perturbed[featureidx] = perturbed[featureidx] + feature_std;
            let perturbed_pred = model(&perturbed.insert_axis(Axis(0)).view())[0];

            sensitivity_sum = sensitivity_sum + (perturbed_pred - baseline_pred).abs();
        }

        Ok(sensitivity_sum / F::from(sample_size).unwrap())
    }

    fn assess_model_linearity<M>(&self, model: &M, xdata: &Array2<F>) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Simplified linearity assessment using second-order differences
        let _predictions = model(&xdata.view());
        let mut nonlinearity_score = F::zero();
        let sample_size = self.n_samples.min(xdata.nrows());

        for i in 0..sample_size {
            let instance = xdata.row(i);

            for j in 0..xdata.ncols() {
                let step = self.get_feature_std(xdata, j)? / F::from(10).unwrap();

                let mut left = instance.to_owned();
                left[j] = left[j] - step;
                let pred_left = model(&left.insert_axis(Axis(0)).view())[0];

                let pred_center = model(&instance.insert_axis(Axis(0)).view())[0];

                let mut right = instance.to_owned();
                right[j] = right[j] + step;
                let pred_right = model(&right.insert_axis(Axis(0)).view())[0];

                // Second derivative approximation
                let second_deriv = pred_right - F::from(2).unwrap() * pred_center + pred_left;
                nonlinearity_score = nonlinearity_score + second_deriv.abs();
            }
        }

        let linearity_score =
            F::one() / (F::one() + nonlinearity_score / F::from(sample_size).unwrap());
        Ok(linearity_score)
    }

    fn compute_prediction_uncertainty<M>(&self, model: &M, xdata: &Array2<F>) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Simplified uncertainty estimation using prediction variance
        let predictions = model(&xdata.view());
        self.compute_variance(&predictions)
    }

    fn compute_sample_coverage(&self, xdata: &Array2<F>) -> Result<F> {
        // Simplified coverage metric based on _data distribution
        let n_features = xdata.ncols();
        let mut coverage_score = F::zero();

        for j in 0..n_features {
            let column = xdata.column(j);
            let range = column.iter().cloned().fold(F::neg_infinity(), F::max)
                - column.iter().cloned().fold(F::infinity(), F::min);
            let std_dev = self.compute_column_std(xdata, j)?;

            // Normalized range-to-std ratio
            let feature_coverage = if std_dev > F::zero() {
                (range / std_dev).min(F::from(10).unwrap()) / F::from(10).unwrap()
            } else {
                F::zero()
            };
            coverage_score = coverage_score + feature_coverage;
        }

        Ok(coverage_score / F::from(n_features).unwrap())
    }

    // Utility methods

    fn compute_variance(&self, data: &Array1<F>) -> Result<F> {
        let mean = data.mean().unwrap_or(F::zero());
        let variance =
            data.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>() / F::from(data.len()).unwrap();
        Ok(variance)
    }

    fn get_feature_std(&self, xdata: &Array2<F>, featureidx: usize) -> Result<F> {
        self.compute_column_std(xdata, featureidx)
    }

    fn compute_column_std(&self, xdata: &Array2<F>, colidx: usize) -> Result<F> {
        let column = xdata.column(colidx);
        let mean = column.mean().unwrap_or(F::zero());
        let variance = column.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>()
            / F::from(column.len()).unwrap();
        Ok(variance.sqrt())
    }

    fn permute_column(&self, data: &mut Array2<F>, colidx: usize) -> Result<()> {
        let mut column_values: Vec<F> = data.column(colidx).to_vec();

        // Simple shuffle
        for i in (1..column_values.len()).rev() {
            let j = (self.random_seed.unwrap_or(0) as usize + i) % (i + 1);
            column_values.swap(i, j);
        }

        for (i, &value) in column_values.iter().enumerate() {
            data[[i, colidx]] = value;
        }

        Ok(())
    }

    fn compute_total_variance<M>(&self, model: &M, xdata: &Array2<F>) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let predictions = model(&xdata.view());
        self.compute_variance(&predictions)
    }

    fn compute_feature_variance<M>(
        &self,
        model: &M,
        xdata: &Array2<F>,
        featureidx: usize,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Compute variance explained by this feature
        let mut x_baseline = xdata.clone();
        let feature_mean = xdata.column(featureidx).mean().unwrap_or(F::zero());

        // Set feature to its mean value
        for i in 0..x_baseline.nrows() {
            x_baseline[[i, featureidx]] = feature_mean;
        }

        let baseline_predictions = model(&x_baseline.view());
        let original_predictions = model(&xdata.view());

        let baseline_variance = self.compute_variance(&baseline_predictions)?;
        let original_variance = self.compute_variance(&original_predictions)?;

        Ok(original_variance - baseline_variance)
    }

    fn estimate_effective_dof<M>(&self, model: &M, xdata: &Array2<F>) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Simplified effective degrees of freedom estimation
        // In practice, this would use more sophisticated methods
        Ok(F::from(xdata.ncols()).unwrap())
    }

    fn compute_model_smoothness<M>(&self, model: &M, xdata: &Array2<F>) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let predictions = model(&xdata.view());

        // Compute local variation as a measure of smoothness
        let mut variation_sum = F::zero();
        let sample_size = self.n_samples.min(xdata.nrows());

        for i in 0..(sample_size - 1) {
            let diff = (predictions[i + 1] - predictions[i]).abs();
            variation_sum = variation_sum + diff;
        }

        let smoothness = F::one() / (F::one() + variation_sum / F::from(sample_size - 1).unwrap());
        Ok(smoothness)
    }

    fn generate_bootstrap_indices(&self, nsamples: usize) -> Result<Vec<usize>> {
        let mut indices = Vec::with_capacity(nsamples);
        for i in 0..nsamples {
            let idx = (self.random_seed.unwrap_or(0) as usize + i) % nsamples;
            indices.push(idx);
        }
        Ok(indices)
    }

    fn sample_by_indices(&self, data: &Array2<F>, indices: &[usize]) -> Result<Array2<F>> {
        let mut sampled = Array2::zeros((indices.len(), data.ncols()));

        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..data.ncols() {
                sampled[[i, j]] = data[[idx, j]];
            }
        }

        Ok(sampled)
    }

    fn compute_importance_stability(
        &self,
        bootstrap_importances: &[HashMap<String, F>],
        feature_names: &[String],
    ) -> Result<HashMap<String, F>> {
        let mut stability_scores = HashMap::new();

        for feature_name in feature_names {
            let mut importance_values = Vec::new();

            for importance_map in bootstrap_importances {
                if let Some(&value) = importance_map.get(feature_name) {
                    importance_values.push(value);
                }
            }

            if !importance_values.is_empty() {
                let stability = self.compute_coefficient_of_variation(&importance_values)?;
                stability_scores.insert(feature_name.clone(), F::one() - stability);
            }
        }

        Ok(stability_scores)
    }

    fn compute_ranking_stability(
        &self,
        bootstrap_importances: &[HashMap<String, F>],
        feature_names: &[String],
    ) -> Result<F> {
        if bootstrap_importances.len() < 2 {
            return Ok(F::one());
        }

        let mut rank_correlations = Vec::new();

        for i in 0..bootstrap_importances.len() {
            for j in (i + 1)..bootstrap_importances.len() {
                let correlation = self.compute_rank_correlation(
                    &bootstrap_importances[i],
                    &bootstrap_importances[j],
                    feature_names,
                )?;
                rank_correlations.push(correlation);
            }
        }

        let avg_correlation = rank_correlations.iter().cloned().sum::<F>()
            / F::from(rank_correlations.len()).unwrap();

        Ok(avg_correlation)
    }

    fn compute_coefficient_of_variation(&self, values: &[F]) -> Result<F> {
        if values.is_empty() {
            return Ok(F::zero());
        }

        let mean = values.iter().cloned().sum::<F>() / F::from(values.len()).unwrap();
        let variance = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>()
            / F::from(values.len()).unwrap();

        let std_dev = variance.sqrt();

        if mean != F::zero() {
            Ok(std_dev / mean.abs())
        } else {
            Ok(F::zero())
        }
    }

    fn compute_rank_correlation(
        &self,
        importance_a: &HashMap<String, F>,
        importance_b: &HashMap<String, F>,
        feature_names: &[String],
    ) -> Result<F> {
        let mut ranks_a = Vec::new();
        let mut ranks_b = Vec::new();

        for feature_name in feature_names {
            if let (Some(&val_a), Some(&val_b)) = (
                importance_a.get(feature_name),
                importance_b.get(feature_name),
            ) {
                ranks_a.push(val_a);
                ranks_b.push(val_b);
            }
        }

        if ranks_a.len() < 2 {
            return Ok(F::one());
        }

        // Compute Spearman correlation (simplified)
        self.compute_correlation(&ranks_a, &ranks_b)
    }

    fn compute_correlation(&self, x: &[F], y: &[F]) -> Result<F> {
        if x.len() != y.len() || x.is_empty() {
            return Ok(F::zero());
        }

        let n = F::from(x.len()).unwrap();
        let mean_x = x.iter().cloned().sum::<F>() / n;
        let mean_y = y.iter().cloned().sum::<F>() / n;

        let numerator: F = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: F = x.iter().map(|&xi| (xi - mean_x) * (xi - mean_x)).sum();
        let sum_sq_y: F = y.iter().map(|&yi| (yi - mean_y) * (yi - mean_y)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator == F::zero() {
            Ok(F::zero())
        } else {
            Ok(numerator / denominator)
        }
    }
}

/// Global explanation result
#[derive(Debug, Clone)]
pub struct GlobalExplanation<F: Float> {
    /// Global feature importance scores
    pub feature_importance: HashMap<String, F>,
    /// Variance-based importance
    pub variance_importance: HashMap<String, F>,
    /// Feature interaction effects
    pub interaction_effects: HashMap<String, F>,
    /// Model complexity metrics
    pub complexity_metrics: ModelComplexityMetrics<F>,
    /// Stability analysis results
    pub stability_metrics: StabilityMetrics<F>,
    /// Sample size used for analysis
    pub sample_size: usize,
}

/// Partial dependence plot data
#[derive(Debug, Clone)]
pub struct PartialDependencePlot<F: Float> {
    /// Feature index
    pub featureidx: usize,
    /// Grid values for the feature
    pub grid_values: Vec<F>,
    /// Partial dependence values
    pub pd_values: Vec<F>,
    /// Individual conditional expectation curves (optional)
    pub ice_curves: Option<Vec<ICECurve<F>>>,
}

/// Individual conditional expectation curve
#[derive(Debug, Clone)]
pub struct ICECurve<F: Float> {
    /// Instance index
    pub instance_idx: usize,
    /// Grid values
    pub grid_values: Vec<F>,
    /// Predictions for each grid value
    pub predictions: Vec<F>,
}

/// Feature interaction analysis results
#[derive(Debug, Clone)]
pub struct InteractionAnalysis<F: Float> {
    /// Pairwise interaction strengths
    pub pairwiseinteractions: HashMap<String, F>,
    /// Higher-order interactions by order
    pub higher_order_interactions: HashMap<usize, HashMap<String, F>>,
    /// Total interaction strength
    pub total_interaction_strength: F,
}

/// Model behavior summary
#[derive(Debug, Clone)]
pub struct ModelBehaviorSummary<F: Float> {
    /// Mean prediction across dataset
    pub mean_prediction: F,
    /// Variance in predictions
    pub prediction_variance: F,
    /// Prediction range (min, max)
    pub prediction_range: (F, F),
    /// Feature sensitivity scores
    pub feature_sensitivity: HashMap<String, F>,
    /// Model linearity score (0 = highly nonlinear, 1 = linear)
    pub linearity_score: F,
    /// Prediction uncertainty measure
    pub prediction_uncertainty: F,
    /// Sample space coverage measure
    pub sample_coverage: F,
}

/// Model complexity metrics
#[derive(Debug, Clone)]
pub struct ModelComplexityMetrics<F: Float> {
    /// Effective degrees of freedom
    pub effective_degrees_of_freedom: F,
    /// Model smoothness score
    pub smoothness: F,
    /// Prediction diversity
    pub prediction_diversity: F,
}

/// Stability analysis metrics
#[derive(Debug, Clone)]
pub struct StabilityMetrics<F: Float> {
    /// Per-feature importance stability
    pub importance_stability: HashMap<String, F>,
    /// Overall ranking stability
    pub ranking_stability: F,
    /// Number of bootstrap samples used
    pub bootstrap_samples: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Mock model for testing
    fn mock_model(x: &ArrayView2<f64>) -> Array1<f64> {
        // Simple linear model with interaction: prediction = x1 + x2 + x1*x2
        x.map_axis(Axis(1), |row| {
            if row.len() >= 2 {
                row[0] + row[1] + row[0] * row[1]
            } else {
                row.sum()
            }
        })
    }

    #[test]
    fn test_global_explainer_creation() {
        let explainer = GlobalExplainer::<f64>::new()
            .with_samples(500)
            .with_bootstrap(50)
            .with_seed(42);

        assert_eq!(explainer.n_samples, 500);
        assert_eq!(explainer.n_bootstrap, 50);
        assert_eq!(explainer.random_seed, Some(42));
    }

    #[test]
    fn test_partial_dependence() {
        let explainer = GlobalExplainer::<f64>::new().with_seed(42);
        let xdata = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let pd_plot = explainer
            .partial_dependence(&mock_model, &xdata, 0, 5)
            .unwrap();

        assert_eq!(pd_plot.featureidx, 0);
        assert_eq!(pd_plot.grid_values.len(), 5);
        assert_eq!(pd_plot.pd_values.len(), 5);
    }

    #[test]
    fn test_ice_curves() {
        let explainer = GlobalExplainer::<f64>::new().with_seed(42);
        let xdata = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let ice_curves = explainer
            .ice_curves(&mock_model, &xdata, 0, 3, Some(2))
            .unwrap();

        assert_eq!(ice_curves.len(), 2); // max_instances = 2
        assert_eq!(ice_curves[0].grid_values.len(), 3);
        assert_eq!(ice_curves[0].predictions.len(), 3);
    }

    #[test]
    fn test_feature_interactions() {
        let explainer = GlobalExplainer::<f64>::new().with_seed(42);
        let xdata = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let feature_names = vec!["f1".to_string(), "f2".to_string()];

        let interactions = explainer
            .feature_interactions(&mock_model, &xdata, &feature_names, 2)
            .unwrap();

        assert_eq!(interactions.pairwiseinteractions.len(), 1); // Only one pair for 2 features
        assert!(interactions.pairwiseinteractions.contains_key("f1_f2"));
    }

    #[test]
    fn test_model_behavior_summary() {
        let explainer = GlobalExplainer::<f64>::new().with_seed(42);
        let xdata = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let feature_names = vec!["f1".to_string(), "f2".to_string()];

        let summary = explainer
            .model_behavior_summary(&mock_model, &xdata, &feature_names)
            .unwrap();

        assert!(summary.mean_prediction > 0.0);
        assert!(summary.prediction_variance >= 0.0);
        assert!(summary.linearity_score >= 0.0 && summary.linearity_score <= 1.0);
        assert_eq!(summary.feature_sensitivity.len(), 2);
    }

    #[test]
    fn test_variance_computation() {
        let explainer = GlobalExplainer::<f64>::new();
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let variance = explainer.compute_variance(&data).unwrap();
        assert!((variance - 2.0).abs() < 1e-10); // Variance of 1,2,3,4,5 is 2.0
    }

    #[test]
    fn test_correlation_computation() {
        let explainer = GlobalExplainer::<f64>::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation

        let correlation = explainer.compute_correlation(&x, &y).unwrap();
        assert!((correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_interaction() {
        let explainer = GlobalExplainer::<f64>::new().with_seed(42);
        let xdata = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let interaction = explainer
            .compute_pairwise_interaction(&mock_model, &xdata, 0, 1)
            .unwrap();

        // Should detect interaction since our mock model has x1*x2 term
        assert!(interaction > 0.0);
    }
}
