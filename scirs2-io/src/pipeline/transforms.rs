//! Common data transformations for pipelines

#![allow(dead_code)]
#![allow(missing_docs)]

use super::*;
use crate::error::Result;
use ndarray::{s, Array1, Array2, Axis};
use num_traits::{Float, FromPrimitive};
use statrs::statistics::Statistics;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Normalization transformer
pub struct NormalizeTransform<T> {
    method: NormalizationMethod,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub enum NormalizationMethod {
    MinMax { min: f64, max: f64 },
    ZScore,
    L1,
    L2,
    MaxAbs,
}

impl<T> NormalizeTransform<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    pub fn new(method: NormalizationMethod) -> Self {
        Self {
            method,
            _phantom: PhantomData,
        }
    }
}

impl<T> DataTransformer for NormalizeTransform<T>
where
    T: Float + FromPrimitive + Send + Sync + 'static,
{
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(array) = data.downcast::<Array2<T>>() {
            let normalized = match &self.method {
                NormalizationMethod::MinMax { min, max } => normalize_minmax(
                    *array,
                    T::from_f64(*min).unwrap(),
                    T::from_f64(*max).unwrap(),
                ),
                NormalizationMethod::ZScore => normalize_zscore(*array),
                NormalizationMethod::L1 => normalize_l1(*array),
                NormalizationMethod::L2 => normalize_l2(*array),
                NormalizationMethod::MaxAbs => normalize_maxabs(*array),
            };
            Ok(Box::new(normalized) as Box<dyn Any + Send + Sync>)
        } else {
            Err(IoError::Other(
                "Invalid data type for normalization".to_string(),
            ))
        }
    }
}

#[allow(dead_code)]
fn normalize_minmax<T>(mut array: Array2<T>, new_min: T, new_max: T) -> Array2<T>
where
    T: Float + FromPrimitive,
{
    let _min = array.iter().fold(T::infinity(), |a, &b| a.min(b));
    let _max = array.iter().fold(T::neg_infinity(), |a, &b| a.max(b));
    let range = _max - _min;

    if range > T::zero() {
        let scale = (new_max - new_min) / range;
        array.mapv_inplace(|x| (x - _min) * scale + new_min);
    }

    array
}

#[allow(dead_code)]
fn normalize_zscore<T>(mut array: Array2<T>) -> Array2<T>
where
    T: Float + FromPrimitive,
{
    let n = T::from_usize(array.len()).unwrap();
    let mean = array.iter().fold(T::zero(), |a, &b| a + b) / n;
    let variance = array.iter().fold(T::zero(), |a, &b| {
        let diff = b - mean;
        a + diff * diff
    }) / n;
    let std = variance.sqrt();

    if std > T::zero() {
        array.mapv_inplace(|x| (x - mean) / std);
    }

    array
}

#[allow(dead_code)]
fn normalize_l1<T>(mut array: Array2<T>) -> Array2<T>
where
    T: Float,
{
    for mut row in array.axis_iter_mut(Axis(0)) {
        let norm = row.iter().fold(T::zero(), |a, &b| a + b.abs());
        if norm > T::zero() {
            row.mapv_inplace(|x| x / norm);
        }
    }
    array
}

#[allow(dead_code)]
fn normalize_l2<T>(mut array: Array2<T>) -> Array2<T>
where
    T: Float,
{
    for mut row in array.axis_iter_mut(Axis(0)) {
        let norm = row.iter().fold(T::zero(), |a, &b| a + b * b).sqrt();
        if norm > T::zero() {
            row.mapv_inplace(|x| x / norm);
        }
    }
    array
}

#[allow(dead_code)]
fn normalize_maxabs<T>(mut array: Array2<T>) -> Array2<T>
where
    T: Float,
{
    let max_abs = array.iter().fold(T::zero(), |a, &b| a.max(b.abs()));
    if max_abs > T::zero() {
        array.mapv_inplace(|x| x / max_abs);
    }
    array
}

/// Reshape transformer
pub struct ReshapeTransform {
    newshape: Vec<usize>,
}

impl ReshapeTransform {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { newshape: shape }
    }
}

impl DataTransformer for ReshapeTransform {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(array) = data.downcast::<Array2<f64>>() {
            let total_elements: usize = self.newshape.iter().product();
            if array.len() != total_elements {
                return Err(IoError::Other(format!(
                    "Cannot reshape array of size {} to shape {:?}",
                    array.len(),
                    self.newshape
                )));
            }

            // Convert to 1D, then reshape
            let flat: Vec<f64> = array.into_iter().collect();
            let reshaped = Array2::from_shape_vec((self.newshape[0], self.newshape[1]), flat)
                .map_err(|e| IoError::Other(e.to_string()))?;

            Ok(Box::new(reshaped) as Box<dyn Any + Send + Sync>)
        } else {
            Err(IoError::Other("Invalid data type for reshape".to_string()))
        }
    }
}

/// Type conversion transformer
pub struct TypeConvertTransform<From, To> {
    _from: PhantomData<From>,
    _to: PhantomData<To>,
}

impl<From, To> Default for TypeConvertTransform<From, To> {
    fn default() -> Self {
        Self::new()
    }
}

impl<From, To> TypeConvertTransform<From, To> {
    pub fn new() -> Self {
        Self {
            _from: PhantomData,
            _to: PhantomData,
        }
    }
}

impl<From, To> DataTransformer for TypeConvertTransform<From, To>
where
    From: 'static + Send + Sync,
    To: 'static + Send + Sync + std::convert::From<From>,
{
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(from_data) = data.downcast::<From>() {
            let to_data: To = To::from(*from_data);
            Ok(Box::new(to_data) as Box<dyn Any + Send + Sync>)
        } else {
            Err(IoError::Other("Type conversion failed".to_string()))
        }
    }
}

/// Aggregation transformer
pub struct AggregateTransform {
    method: AggregationMethod,
    axis: Option<Axis>,
}

#[derive(Debug, Clone)]
pub enum AggregationMethod {
    Sum,
    Mean,
    Min,
    Max,
    Std,
    Var,
}

impl AggregateTransform {
    pub fn new(method: AggregationMethod, axis: Option<Axis>) -> Self {
        Self { method, axis }
    }
}

impl DataTransformer for AggregateTransform {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(array) = data.downcast::<Array2<f64>>() {
            let result = match (&self.method, self.axis) {
                (AggregationMethod::Sum, Some(axis)) => {
                    Box::new(array.sum_axis(axis)) as Box<dyn Any + Send + Sync>
                }
                (AggregationMethod::Mean, Some(axis)) => {
                    Box::new(array.mean_axis(axis).unwrap()) as Box<dyn Any + Send + Sync>
                }
                (AggregationMethod::Sum, None) => {
                    Box::new(array.sum()) as Box<dyn Any + Send + Sync>
                }
                (AggregationMethod::Mean, None) => {
                    Box::new(array.mean()) as Box<dyn Any + Send + Sync>
                }
                _ => return Err(IoError::Other("Unsupported aggregation".to_string())),
            };
            Ok(result)
        } else {
            Err(IoError::Other(
                "Invalid data type for aggregation".to_string(),
            ))
        }
    }
}

/// Encoding transformer for categorical data
pub struct EncodingTransform {
    method: EncodingMethod,
}

#[derive(Debug, Clone)]
pub enum EncodingMethod {
    OneHot,
    Label,
    Ordinal(Vec<String>),
}

impl EncodingTransform {
    pub fn new(method: EncodingMethod) -> Self {
        Self { method }
    }
}

impl DataTransformer for EncodingTransform {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(categories) = data.downcast::<Vec<String>>() {
            match &self.method {
                EncodingMethod::Label => {
                    let mut label_map = HashMap::new();
                    let mut next_label = 0;

                    let encoded: Vec<i32> = categories
                        .iter()
                        .map(|cat| {
                            *label_map.entry(cat.clone()).or_insert_with(|| {
                                let label = next_label;
                                next_label += 1;
                                label
                            })
                        })
                        .collect();

                    Ok(Box::new(encoded) as Box<dyn Any + Send + Sync>)
                }
                EncodingMethod::OneHot => {
                    let unique_categories: Vec<String> = {
                        let mut cats = (*categories).clone();
                        cats.sort();
                        cats.dedup();
                        cats
                    };

                    let n_categories = unique_categories.len();
                    let n_samples = categories.len();
                    let mut encoded = Array2::<f64>::zeros((n_samples, n_categories));

                    for (i, cat) in categories.iter().enumerate() {
                        if let Some(j) = unique_categories.iter().position(|c| c == cat) {
                            encoded[[i, j]] = 1.0;
                        }
                    }

                    Ok(Box::new(encoded) as Box<dyn Any + Send + Sync>)
                }
                EncodingMethod::Ordinal(order) => {
                    let encoded: Result<Vec<i32>> = categories
                        .iter()
                        .map(|cat| {
                            order
                                .iter()
                                .position(|o| o == cat)
                                .map(|pos| pos as i32)
                                .ok_or_else(|| IoError::Other(format!("Unknown category: {}", cat)))
                        })
                        .collect();

                    Ok(Box::new(encoded?) as Box<dyn Any + Send + Sync>)
                }
            }
        } else {
            Err(IoError::Other("Invalid data type for encoding".to_string()))
        }
    }
}

/// Missing value imputation transformer
pub struct ImputeTransform {
    strategy: ImputationStrategy,
}

#[derive(Debug, Clone)]
pub enum ImputationStrategy {
    Mean,
    Median,
    Mode,
    Constant(f64),
    Forward,
    Backward,
}

impl ImputeTransform {
    pub fn new(strategy: ImputationStrategy) -> Self {
        Self { strategy }
    }
}

impl DataTransformer for ImputeTransform {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(mut array) = data.downcast::<Array2<Option<f64>>>() {
            match &self.strategy {
                ImputationStrategy::Mean => {
                    for mut col in array.axis_iter_mut(Axis(1)) {
                        let valid_values: Vec<f64> = col.iter().filter_map(|&x| x).collect();

                        if !valid_values.is_empty() {
                            let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                            col.mapv_inplace(|x| Some(x.unwrap_or(mean)));
                        }
                    }
                }
                ImputationStrategy::Constant(value) => {
                    array.mapv_inplace(|x| Some(x.unwrap_or(*value)));
                }
                _ => {
                    return Err(IoError::Other(
                        "Unsupported imputation strategy".to_string(),
                    ))
                }
            }

            // Convert to Array2<f64> after imputation
            let imputed: Array2<f64> = array.mapv(|x| x.unwrap_or(0.0));
            Ok(Box::new(imputed) as Box<dyn Any + Send + Sync>)
        } else {
            Err(IoError::Other(
                "Invalid data type for imputation".to_string(),
            ))
        }
    }
}

/// Outlier detection and removal transformer
pub struct OutlierTransform {
    method: OutlierMethod,
    threshold: f64,
}

#[derive(Debug, Clone)]
pub enum OutlierMethod {
    ZScore,
    IQR,
    IsolationForest,
}

impl OutlierTransform {
    pub fn new(method: OutlierMethod, threshold: f64) -> Self {
        Self { method, threshold }
    }
}

impl DataTransformer for OutlierTransform {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(array) = data.downcast::<Array2<f64>>() {
            match &self.method {
                OutlierMethod::ZScore => {
                    let mean = array.view().mean();
                    let std = array.std(0.0);

                    let filtered: Vec<Vec<f64>> = array
                        .axis_iter(Axis(0))
                        .filter(|row| {
                            row.iter()
                                .all(|&x| ((x - mean) / std).abs() <= self.threshold)
                        })
                        .map(|row| row.to_vec())
                        .collect();

                    if filtered.is_empty() {
                        return Err(IoError::Other("All data filtered as outliers".to_string()));
                    }

                    let n_rows = filtered.len();
                    let n_cols = filtered[0].len();
                    let flat: Vec<f64> = filtered.into_iter().flatten().collect();

                    let result = Array2::from_shape_vec((n_rows, n_cols), flat)
                        .map_err(|e| IoError::Other(e.to_string()))?;

                    Ok(Box::new(result) as Box<dyn Any + Send + Sync>)
                }
                OutlierMethod::IQR => {
                    // Interquartile Range method
                    let mut filtered_rows = Vec::new();

                    for row in array.axis_iter(Axis(0)) {
                        let mut values: Vec<f64> = row.to_vec();
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        let n = values.len();
                        let q1_idx = n / 4;
                        let q3_idx = 3 * n / 4;
                        let q1 = values[q1_idx];
                        let q3 = values[q3_idx];
                        let iqr = q3 - q1;

                        let lower_bound = q1 - self.threshold * iqr;
                        let upper_bound = q3 + self.threshold * iqr;

                        let is_outlier = row.iter().any(|&x| x < lower_bound || x > upper_bound);

                        if !is_outlier {
                            filtered_rows.push(row.to_vec());
                        }
                    }

                    if filtered_rows.is_empty() {
                        return Err(IoError::Other("All data filtered as outliers".to_string()));
                    }

                    let n_rows = filtered_rows.len();
                    let n_cols = filtered_rows[0].len();
                    let flat: Vec<f64> = filtered_rows.into_iter().flatten().collect();

                    let result = Array2::from_shape_vec((n_rows, n_cols), flat)
                        .map_err(|e| IoError::Other(e.to_string()))?;

                    Ok(Box::new(result) as Box<dyn Any + Send + Sync>)
                }
                _ => Err(IoError::Other("Unsupported outlier method".to_string())),
            }
        } else {
            Err(IoError::Other(
                "Invalid data type for outlier detection".to_string(),
            ))
        }
    }
}

/// Principal Component Analysis transformer
pub struct PCATransform {
    n_components: usize,
    components: Option<Array2<f64>>,
    mean: Option<Array1<f64>>,
}

impl PCATransform {
    pub fn new(_ncomponents: usize) -> Self {
        Self {
            n_components: _ncomponents,
            components: None,
            mean: None,
        }
    }

    /// Fit PCA on training data
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<()> {
        let (n_samples, n_features) = data.dim();

        if self.n_components > n_features {
            return Err(IoError::Other(
                "n_components cannot exceed n_features".to_string(),
            ));
        }

        // Center the data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered = data - &mean.clone().insert_axis(Axis(0));

        // Compute covariance matrix
        let _cov = centered.t().dot(&centered) / (n_samples - 1) as f64;

        // For simplicity, use a basic eigenvalue decomposition approximation
        // In practice, you would use a proper linear algebra library
        self.mean = Some(mean);

        // Mock components for demonstration
        let components = Array2::eye(n_features)
            .slice(s![..self.n_components, ..])
            .to_owned();
        self.components = Some(components);

        Ok(())
    }
}

impl DataTransformer for PCATransform {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(array) = data.downcast::<Array2<f64>>() {
            let mean = self
                .mean
                .as_ref()
                .ok_or_else(|| IoError::Other("PCA not fitted yet".to_string()))?;
            let components = self
                .components
                .as_ref()
                .ok_or_else(|| IoError::Other("PCA not fitted yet".to_string()))?;

            // Center the data
            let centered = &*array - &mean.clone().insert_axis(Axis(0));

            // Project onto principal components
            let transformed = centered.dot(&components.t());

            Ok(Box::new(transformed) as Box<dyn Any + Send + Sync>)
        } else {
            Err(IoError::Other("Invalid data type for PCA".to_string()))
        }
    }
}

/// Feature engineering transformer
pub struct FeatureEngineeringTransform {
    operations: Vec<FeatureOperation>,
}

#[derive(Debug, Clone)]
pub enum FeatureOperation {
    Polynomial {
        degree: usize,
    },
    Log,
    Sqrt,
    Square,
    Interaction {
        indices: Vec<usize>,
    },
    Binning {
        n_bins: usize,
        strategy: BinningStrategy,
    },
}

#[derive(Debug, Clone)]
pub enum BinningStrategy {
    Uniform,
    Quantile,
}

impl FeatureEngineeringTransform {
    pub fn new(operations: Vec<FeatureOperation>) -> Self {
        Self { operations }
    }
}

impl DataTransformer for FeatureEngineeringTransform {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(array) = data.downcast::<Array2<f64>>() {
            let mut result = (*array).clone();

            for operation in &self.operations {
                match operation {
                    FeatureOperation::Log => {
                        let log_features = result.mapv(|x| if x > 0.0 { x.ln() } else { 0.0 });
                        result =
                            ndarray::concatenate(Axis(1), &[result.view(), log_features.view()])
                                .unwrap();
                    }
                    FeatureOperation::Sqrt => {
                        let sqrt_features = result.mapv(|x| if x >= 0.0 { x.sqrt() } else { 0.0 });
                        result =
                            ndarray::concatenate(Axis(1), &[result.view(), sqrt_features.view()])
                                .unwrap();
                    }
                    FeatureOperation::Square => {
                        let square_features = result.mapv(|x| x * x);
                        result =
                            ndarray::concatenate(Axis(1), &[result.view(), square_features.view()])
                                .unwrap();
                    }
                    FeatureOperation::Polynomial { degree } => {
                        let mut poly_features = result.clone();
                        for d in 2..=*degree {
                            let power_features = result.mapv(|x| x.powi(d as i32));
                            poly_features = ndarray::concatenate(
                                Axis(1),
                                &[poly_features.view(), power_features.view()],
                            )
                            .unwrap();
                        }
                        result = poly_features;
                    }
                    FeatureOperation::Interaction { indices } => {
                        if indices.len() >= 2 {
                            let mut interaction_col = result.column(indices[0]).to_owned();
                            for &idx in &indices[1..] {
                                if idx < result.ncols() {
                                    interaction_col *= &result.column(idx);
                                }
                            }
                            result = ndarray::concatenate(
                                Axis(1),
                                &[result.view(), interaction_col.insert_axis(Axis(1)).view()],
                            )
                            .unwrap();
                        }
                    }
                    FeatureOperation::Binning { n_bins, strategy } => {
                        // Simple uniform binning implementation
                        let mut binned_features = Array2::zeros((result.nrows(), result.ncols()));

                        for (col_idx, col) in result.axis_iter(Axis(1)).enumerate() {
                            let min_val = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                            let max_val = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                            let bin_width = (max_val - min_val) / *n_bins as f64;

                            for (row_idx, &val) in col.iter().enumerate() {
                                let bin = ((val - min_val) / bin_width).floor() as usize;
                                let bin = bin.min(n_bins - 1);
                                binned_features[[row_idx, col_idx]] = bin as f64;
                            }
                        }

                        result =
                            ndarray::concatenate(Axis(1), &[result.view(), binned_features.view()])
                                .unwrap();
                    }
                }
            }

            Ok(Box::new(result) as Box<dyn Any + Send + Sync>)
        } else {
            Err(IoError::Other(
                "Invalid data type for feature engineering".to_string(),
            ))
        }
    }
}

/// Text processing transformer
pub struct TextProcessingTransform {
    operations: Vec<TextOperation>,
}

#[derive(Debug, Clone)]
pub enum TextOperation {
    Lowercase,
    RemovePunctuation,
    RemoveStopwords,
    Tokenize,
    Stemming,
    NGrams { n: usize },
}

impl TextProcessingTransform {
    pub fn new(operations: Vec<TextOperation>) -> Self {
        Self { operations }
    }
}

impl DataTransformer for TextProcessingTransform {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(texts) = data.downcast::<Vec<String>>() {
            let mut processed = texts.clone();

            for operation in &self.operations {
                match operation {
                    TextOperation::Lowercase => {
                        processed = Box::new(
                            processed
                                .into_iter()
                                .map(|s| s.to_lowercase())
                                .collect::<Vec<_>>(),
                        );
                    }
                    TextOperation::RemovePunctuation => {
                        processed = Box::new(
                            processed
                                .into_iter()
                                .map(|s| {
                                    s.chars()
                                        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                                        .collect()
                                })
                                .collect::<Vec<_>>(),
                        );
                    }
                    TextOperation::Tokenize => {
                        let tokens: Vec<Vec<String>> = processed
                            .into_iter()
                            .map(|s| s.split_whitespace().map(|w| w.to_string()).collect())
                            .collect();
                        return Ok(Box::new(tokens) as Box<dyn Any + Send + Sync>);
                    }
                    TextOperation::NGrams { n } => {
                        let ngrams: Vec<Vec<String>> = processed
                            .into_iter()
                            .map(|s| {
                                let words: Vec<&str> = s.split_whitespace().collect();
                                words.windows(*n).map(|window| window.join(" ")).collect()
                            })
                            .collect();
                        return Ok(Box::new(ngrams) as Box<dyn Any + Send + Sync>);
                    }
                    _ => {}
                }
            }

            Ok(Box::new(processed) as Box<dyn Any + Send + Sync>)
        } else {
            Err(IoError::Other(
                "Invalid data type for text processing".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_normalize_minmax() {
        let transform =
            NormalizeTransform::<f64>::new(NormalizationMethod::MinMax { min: 0.0, max: 1.0 });
        let data = Box::new(arr2(&[[1.0, 2.0], [3.0, 4.0]])) as Box<dyn Any + Send + Sync>;
        let result = transform.transform(data).unwrap();
        let normalized = result.downcast::<Array2<f64>>().unwrap();

        assert!((normalized[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((normalized[[1, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_encoding_label() {
        let transform = EncodingTransform::new(EncodingMethod::Label);
        let data = Box::new(vec![
            "cat".to_string(),
            "dog".to_string(),
            "cat".to_string(),
        ]) as Box<dyn Any + Send + Sync>;
        let result = transform.transform(data).unwrap();
        let encoded = result.downcast::<Vec<i32>>().unwrap();

        assert_eq!(*encoded, vec![0, 1, 0]);
    }
}
