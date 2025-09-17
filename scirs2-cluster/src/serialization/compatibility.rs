//! Cross-platform model compatibility utilities
//!
//! This module provides utilities for converting between different model formats
//! and maintaining compatibility with popular machine learning libraries.

use crate::error::{ClusteringError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::models::*;

/// Create a scikit-learn compatible parameter grid
pub fn create_sklearn_param_grid(
    algorithm: &str,
    param_ranges: HashMap<String, Vec<Value>>,
) -> Result<HashMap<String, Vec<Value>>> {
    match algorithm {
        "kmeans" => {
            let mut grid = HashMap::new();
            if let Some(n_clusters) = param_ranges.get("n_clusters") {
                grid.insert("n_clusters".to_string(), n_clusters.clone());
            }
            if let Some(init) = param_ranges.get("init") {
                grid.insert("init".to_string(), init.clone());
            }
            Ok(grid)
        }
        "dbscan" => {
            let mut grid = HashMap::new();
            if let Some(eps) = param_ranges.get("eps") {
                grid.insert("eps".to_string(), eps.clone());
            }
            if let Some(min_samples) = param_ranges.get("min_samples") {
                grid.insert("min_samples".to_string(), min_samples.clone());
            }
            Ok(grid)
        }
        _ => Err(ClusteringError::InvalidInput(format!(
            "Unsupported algorithm for sklearn parameter grid: {}",
            algorithm
        ))),
    }
}

/// Convert from joblib format (simplified)
pub fn from_joblib_format(data: Vec<u8>) -> Result<Value> {
    // This is a simplified implementation
    // Real joblib support would require proper pickle deserialization
    serde_json::from_slice(&data)
        .map_err(|e| ClusteringError::InvalidInput(format!("Failed to parse joblib format: {}", e)))
}

/// Convert from numpy format (simplified)
pub fn from_numpy_format(data: Vec<u8>) -> Result<ndarray::Array2<f64>> {
    // This is a simplified implementation
    // Real numpy support would require proper .npy file parsing
    let json_data: Value = serde_json::from_slice(&data).map_err(|e| {
        ClusteringError::InvalidInput(format!("Failed to parse numpy format: {}", e))
    })?;

    if let Value::Array(array) = json_data {
        let mut flat_data = Vec::new();
        let mut nrows = 0;
        let mut ncols = 0;

        if let Some(Value::Array(first_row)) = array.first() {
            ncols = first_row.len();
        }
        nrows = array.len();

        for row in array {
            if let Value::Array(row_values) = row {
                for val in row_values {
                    if let Value::Number(num) = val {
                        flat_data.push(num.as_f64().unwrap_or(0.0));
                    }
                }
            }
        }

        ndarray::Array2::from_shape_vec((nrows, ncols), flat_data).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to create array from numpy data: {}", e))
        })
    } else {
        Err(ClusteringError::InvalidInput(
            "Invalid numpy format".to_string(),
        ))
    }
}

/// Convert from sklearn format
pub fn from_sklearn_format(data: Value) -> Result<Value> {
    // sklearn models are typically stored as dictionaries
    Ok(data)
}

/// Generate sklearn model summary
pub fn generate_sklearn_model_summary(model_type: &str, model_data: &Value) -> Result<String> {
    match model_type {
        "KMeans" => {
            let summary = serde_json::json!({
                "model_type": "KMeans",
                "n_clusters": model_data.get("n_clusters").unwrap_or(&Value::Null),
                "inertia": model_data.get("inertia_").unwrap_or(&Value::Null),
                "n_iter": model_data.get("n_iter_").unwrap_or(&Value::Null)
            });
            Ok(serde_json::to_string_pretty(&summary)?)
        }
        "DBSCAN" => {
            let summary = serde_json::json!({
                "model_type": "DBSCAN",
                "eps": model_data.get("eps").unwrap_or(&Value::Null),
                "min_samples": model_data.get("min_samples").unwrap_or(&Value::Null)
            });
            Ok(serde_json::to_string_pretty(&summary)?)
        }
        _ => Err(ClusteringError::InvalidInput(format!(
            "Unsupported sklearn model type: {}",
            model_type
        ))),
    }
}

/// Convert to Arrow schema format
pub fn to_arrow_schema<T: ClusteringModel>(model: &T) -> Result<Value> {
    let schema = serde_json::json!({
        "type": "struct",
        "fields": [
            {
                "name": "cluster_id",
                "type": {
                    "name": "int",
                    "bitWidth": 32
                },
                "nullable": false
            },
            {
                "name": "features",
                "type": {
                    "name": "list",
                    "valueType": {
                        "name": "floatingpoint",
                        "precision": "DOUBLE"
                    }
                },
                "nullable": false
            }
        ]
    });
    Ok(schema)
}

/// Convert to HuggingFace model card format
pub fn to_huggingface_card<T: ClusteringModel>(model: &T) -> Result<String> {
    let summary = model.summary()?;
    let card = format!(
        r#"
---
tags:
- clustering
- unsupervised-learning
- scirs2-cluster
library_name: scirs2-cluster
model_summary: {}
---

# Clustering Model

This is a clustering model trained using scirs2-cluster.

## Model Details

{}

## Usage

```rust
use scirs2_cluster::serialization::SerializableModel;

// Load the model
let model = Model::load_from_file("model.json")?;

// Use for prediction
let predictions = model.predict(data.view())?;
```
"#,
        serde_json::to_string_pretty(&summary)?,
        serde_json::to_string_pretty(&summary)?
    );

    Ok(card)
}

/// Convert to joblib format (simplified)
pub fn to_joblib_format<T: ClusteringModel>(model: &T) -> Result<Vec<u8>> {
    // This is a simplified implementation
    let summary = model.summary()?;
    Ok(serde_json::to_vec(&summary)?)
}

/// Convert to MLflow format
pub fn to_mlflow_format<T: ClusteringModel>(model: &T) -> Result<Value> {
    let summary = model.summary()?;
    Ok(serde_json::json!({
        "artifact_path": "model",
        "flavors": {
            "scirs2_cluster": {
                "model_type": "clustering",
                "scirs2_version": env!("CARGO_PKG_VERSION"),
                "data": summary
            }
        },
        "model_uuid": uuid::Uuid::new_v4().to_string(),
        "run_id": "unknown",
        "utc_time_created": chrono::Utc::now().to_rfc3339()
    }))
}

/// Convert to numpy format (simplified)
pub fn to_numpy_format(data: &ndarray::Array2<f64>) -> Result<Vec<u8>> {
    // This is a simplified implementation
    // Real numpy format would require proper .npy file generation
    let shape = data.shape();
    let numpy_data = serde_json::json!({
        "shape": shape,
        "data": data.as_slice().unwrap_or(&[])
    });
    Ok(serde_json::to_vec(&numpy_data)?)
}

/// Convert to ONNX metadata format
pub fn to_onnx_metadata<T: ClusteringModel>(model: &T) -> Result<Value> {
    let summary = model.summary()?;
    Ok(serde_json::json!({
        "ir_version": 7,
        "producer_name": "scirs2-cluster",
        "producer_version": env!("CARGO_PKG_VERSION"),
        "model_version": 1,
        "doc_string": "Clustering model exported from scirs2-cluster",
        "metadata_props": {
            "model_summary": summary
        }
    }))
}

/// Convert to pandas clustering report
pub fn to_pandas_clustering_report<T: ClusteringModel>(model: &T) -> Result<Value> {
    let summary = model.summary()?;
    Ok(serde_json::json!({
        "model_type": "clustering",
        "n_clusters": model.n_clusters(),
        "summary": summary,
        "pandas_version": "1.0.0",
        "created_at": chrono::Utc::now().to_rfc3339()
    }))
}

/// Convert to pandas format
pub fn to_pandas_format<T: ClusteringModel>(model: &T) -> Result<Value> {
    to_pandas_clustering_report(model)
}

/// Convert to pickle-like format (simplified)
pub fn to_pickle_like_format<T: ClusteringModel>(model: &T) -> Result<Vec<u8>> {
    // This is a simplified implementation
    let summary = model.summary()?;
    Ok(serde_json::to_vec(&summary)?)
}

/// Convert to PyTorch checkpoint format
pub fn to_pytorch_checkpoint<T: ClusteringModel>(model: &T) -> Result<Value> {
    let summary = model.summary()?;
    Ok(serde_json::json!({
        "model_state_dict": summary,
        "optimizer_state_dict": {},
        "epoch": 1,
        "loss": 0.0,
        "pytorch_version": "1.10.0",
        "scirs2_cluster_version": env!("CARGO_PKG_VERSION")
    }))
}

/// Convert to R format
pub fn to_r_format<T: ClusteringModel>(model: &T) -> Result<Value> {
    let summary = model.summary()?;
    Ok(serde_json::json!({
        "class": "clustering_model",
        "data": summary,
        "r_version": "4.0.0",
        "created_by": "scirs2-cluster"
    }))
}

/// Convert to SciPy dendrogram format
pub fn to_scipy_dendrogram_format(linkage_matrix: &ndarray::Array2<f64>) -> Result<Value> {
    Ok(serde_json::json!({
        "linkage": linkage_matrix.as_slice().unwrap_or(&[]),
        "format": "scipy_dendrogram",
        "shape": linkage_matrix.shape()
    }))
}

/// Convert to SciPy linkage format
pub fn to_scipy_linkage_format(linkage_matrix: &ndarray::Array2<f64>) -> Result<Value> {
    Ok(serde_json::json!({
        "linkage_matrix": linkage_matrix.as_slice().unwrap_or(&[]),
        "shape": linkage_matrix.shape(),
        "method": "ward",
        "metric": "euclidean"
    }))
}

/// Convert to sklearn clustering result format
pub fn to_sklearn_clustering_result<T: ClusteringModel>(model: &T) -> Result<Value> {
    let summary = model.summary()?;
    Ok(serde_json::json!({
        "labels_": [],
        "n_clusters_": model.n_clusters(),
        "model_summary": summary,
        "_sklearn_version": "1.0.0"
    }))
}

/// Convert to sklearn format
pub fn to_sklearn_format<T: ClusteringModel>(model: &T) -> Result<Value> {
    to_sklearn_clustering_result(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_create_sklearn_param_grid() {
        let mut params = HashMap::new();
        params.insert(
            "n_clusters".to_string(),
            vec![serde_json::json!(2), serde_json::json!(3)],
        );

        let grid = create_sklearn_param_grid("kmeans", params).unwrap();
        assert!(grid.contains_key("n_clusters"));
    }

    #[test]
    fn test_to_numpy_format() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = to_numpy_format(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_to_scipy_linkage_format() {
        let linkage = Array2::from_shape_vec((1, 3), vec![0.0, 1.0, 0.5]).unwrap();
        let result = to_scipy_linkage_format(&linkage);
        assert!(result.is_ok());
    }
}
