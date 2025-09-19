//! Dataset registry system for managing dataset metadata and locations

use crate::cache::RegistryEntry;
use crate::error::{DatasetsError, Result};
use std::collections::HashMap;

/// Dataset metadata information
#[derive(Debug, Clone, Default)]
pub struct DatasetMetadata {
    /// Name of the dataset
    pub name: String,
    /// Description of the dataset
    pub description: String,
    /// Number of samples in the dataset
    pub n_samples: usize,
    /// Number of features in the dataset
    pub n_features: usize,
    /// Whether this is a classification or regression dataset
    pub task_type: String,
    /// Optional target names for classification problems
    pub targetnames: Option<Vec<String>>,
    /// Optional feature names
    pub featurenames: Option<Vec<String>>,
    /// Optional download URL
    pub url: Option<String>,
    /// Optional checksum for verification
    pub checksum: Option<String>,
}

/// Global dataset registry containing metadata for downloadable datasets
pub struct DatasetRegistry {
    /// Map from dataset name to registry entry
    entries: HashMap<String, RegistryEntry>,
}

impl Default for DatasetRegistry {
    fn default() -> Self {
        let mut registry = Self::new();
        registry.populate_default_datasets();
        registry
    }
}

impl DatasetRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register a new dataset with the given name and metadata
    pub fn register(&mut self, name: String, entry: RegistryEntry) {
        self.entries.insert(name, entry);
    }

    /// Get a registry entry by name
    pub fn get(&self, name: &str) -> Option<&RegistryEntry> {
        self.entries.get(name)
    }

    /// List all available dataset names
    pub fn list_datasets(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Check if a dataset is registered
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Get metadata for a dataset
    pub fn get_metadata(&self, name: &str) -> Result<DatasetMetadata> {
        match name {
            "iris" => Ok(DatasetMetadata {
                name: "Iris".to_string(),
                description: "Classic iris flower dataset for classification".to_string(),
                n_samples: 150,
                n_features: 4,
                task_type: "classification".to_string(),
                targetnames: Some(vec![
                    "setosa".to_string(),
                    "versicolor".to_string(),
                    "virginica".to_string(),
                ]),
                featurenames: Some(vec![
                    "sepal_length".to_string(),
                    "sepal_width".to_string(),
                    "petal_length".to_string(),
                    "petal_width".to_string(),
                ]),
                url: None,
                checksum: None,
            }),
            "boston" => Ok(DatasetMetadata {
                name: "Boston Housing".to_string(),
                description: "Boston housing prices dataset for regression".to_string(),
                n_samples: 506,
                n_features: 13,
                task_type: "regression".to_string(),
                targetnames: None,
                featurenames: None,
                url: None,
                checksum: None,
            }),
            "digits" => Ok(DatasetMetadata {
                name: "Digits".to_string(),
                description: "Hand-written digits dataset for image classification".to_string(),
                n_samples: 1797,
                n_features: 64,
                task_type: "classification".to_string(),
                targetnames: Some(vec![
                    "0".to_string(),
                    "1".to_string(),
                    "2".to_string(),
                    "3".to_string(),
                    "4".to_string(),
                    "5".to_string(),
                    "6".to_string(),
                    "7".to_string(),
                    "8".to_string(),
                    "9".to_string(),
                ]),
                featurenames: None,
                url: None,
                checksum: None,
            }),
            "wine" => Ok(DatasetMetadata {
                name: "Wine".to_string(),
                description: "Wine recognition dataset for classification".to_string(),
                n_samples: 178,
                n_features: 13,
                task_type: "classification".to_string(),
                targetnames: Some(vec![
                    "class_0".to_string(),
                    "class_1".to_string(),
                    "class_2".to_string(),
                ]),
                featurenames: None,
                url: None,
                checksum: None,
            }),
            "breast_cancer" => Ok(DatasetMetadata {
                name: "Breast Cancer".to_string(),
                description: "Breast cancer wisconsin dataset for classification".to_string(),
                n_samples: 569,
                n_features: 30,
                task_type: "classification".to_string(),
                targetnames: Some(vec!["malignant".to_string(), "benign".to_string()]),
                featurenames: None,
                url: None,
                checksum: None,
            }),
            "diabetes" => Ok(DatasetMetadata {
                name: "Diabetes".to_string(),
                description: "Diabetes dataset for regression".to_string(),
                n_samples: 442,
                n_features: 10,
                task_type: "regression".to_string(),
                targetnames: None,
                featurenames: None,
                url: None,
                checksum: None,
            }),
            _ => Err(DatasetsError::Other(format!("Unknown dataset: {name}"))),
        }
    }

    /// Populate the registry with default datasets
    ///
    /// This includes both local sample datasets and references to potential remote datasets.
    /// Local datasets use verified SHA256 hashes computed from actual files.
    fn populate_default_datasets(&mut self) {
        // Local sample datasets (with verified SHA256 hashes)
        self.register(
            "example".to_string(),
            RegistryEntry {
                url: "file://data/example.csv",
                sha256: "c51c3ff2e8a5db28b1baed809a2ba29f29643e5a26ad476448eb3889996173d6",
            },
        );

        self.register(
            "sample_data".to_string(),
            RegistryEntry {
                url: "file://examples/sample_data.csv",
                sha256: "59cceb2c80692ee2c1c3b607335d1feb983ceed24214d1ffc2eace9f3ce5ab47",
            },
        );

        // Built-in toy datasets (no hash needed as they're embedded in code)
        self.register_toy_dataset("iris", "Classic iris flower dataset for classification");
        self.register_toy_dataset("boston", "Boston housing prices dataset for regression");
        self.register_toy_dataset(
            "digits",
            "Hand-written digits dataset for image classification",
        );
        self.register_toy_dataset("wine", "Wine recognition dataset for classification");
        self.register_toy_dataset(
            "breast_cancer",
            "Breast cancer wisconsin dataset for classification",
        );
        self.register_toy_dataset("diabetes", "Diabetes dataset for regression");

        // Future remote datasets (commented out until available)
        /*
        self.register(
            "california_housing".to_string(),
            RegistryEntry {
                url: "https://raw.githubusercontent.com/cool-japan/scirs-datasets/main/california_housing.csv",
                sha256: "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
            },
        );

        self.register(
            "electrocardiogram".to_string(),
            RegistryEntry {
                url: "https://raw.githubusercontent.com/cool-japan/scirs-datasets/main/electrocardiogram.json",
                sha256: "def0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd",
            },
        );

        self.register(
            "stock_market".to_string(),
            RegistryEntry {
                url: "https://raw.githubusercontent.com/cool-japan/scirs-datasets/main/stock_market.json",
                sha256: "456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef012345",
            },
        );

        self.register(
            "weather".to_string(),
            RegistryEntry {
                url: "https://raw.githubusercontent.com/cool-japan/scirs-datasets/main/weather.json",
                sha256: "789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            },
        );
        */
    }

    /// Register a toy dataset (built-in datasets don't need URLs or hashes)
    fn register_toy_dataset(&mut self, name: &str, _description: &str) {
        let url = match name {
            "iris" => "builtin://iris",
            "boston" => "builtin://boston",
            "digits" => "builtin://digits",
            "wine" => "builtin://wine",
            "breast_cancer" => "builtin://breast_cancer",
            "diabetes" => "builtin://diabetes",
            _ => "builtin://unknown",
        };

        self.register(
            name.to_string(),
            RegistryEntry {
                url,
                sha256: "builtin", // Special marker for built-in datasets
            },
        );
    }
}

/// Get the global dataset registry
#[allow(dead_code)]
pub fn get_registry() -> DatasetRegistry {
    DatasetRegistry::default()
}

/// Load a dataset by name from the registry
#[cfg(feature = "download")]
#[allow(dead_code)]
pub fn load_dataset_byname(name: &str, forcedownload: bool) -> Result<crate::utils::Dataset> {
    let registry = get_registry();

    if let Some(entry) = registry.get(name) {
        // Handle different URL schemes
        if entry.url.starts_with("builtin://") {
            // Built-in toy datasets
            match name {
                "iris" => crate::toy::load_iris(),
                "boston" => crate::toy::load_boston(),
                "digits" => crate::toy::load_digits(),
                "wine" => crate::sample::load_wine(false),
                "breast_cancer" => crate::toy::load_breast_cancer(),
                "diabetes" => crate::toy::load_diabetes(),
                _ => Err(DatasetsError::Other(format!(
                    "Built-in dataset '{}' not implemented",
                    name
                ))),
            }
        } else if entry.url.starts_with("file://") {
            // Local file datasets
            load_local_dataset(name, &entry.url[7..], entry.sha256) // Remove "file://" prefix
        } else if entry.url.starts_with("http") {
            // Remote datasets (when available)
            match name {
                "california_housing" => crate::sample::load_california_housing(forcedownload),
                "electrocardiogram" => crate::time_series::electrocardiogram(),
                "stock_market" => crate::time_series::stock_market(false),
                "weather" => crate::time_series::weather(None),
                _ => Err(DatasetsError::Other(format!(
                    "Remote dataset '{}' not yet implemented for loading",
                    name
                ))),
            }
        } else {
            Err(DatasetsError::Other(format!(
                "Unsupported URL scheme for dataset '{}': {}",
                name, entry.url
            )))
        }
    } else {
        Err(DatasetsError::Other(format!(
            "Unknown dataset: '{}'. Available datasets: {:?}",
            name,
            registry.list_datasets()
        )))
    }
}

/// Load a local dataset file
#[cfg(feature = "download")]
#[allow(dead_code)]
fn load_local_dataset(
    name: &str,
    relativepath: &str,
    expected_sha256: &str,
) -> Result<crate::utils::Dataset> {
    use crate::loaders::{load_csv, CsvConfig};
    use std::path::Path;

    // Build absolute path from workspace root
    let workspace_root = env!("CARGO_MANIFEST_DIR");
    let filepath = Path::new(workspace_root).join(relativepath);

    if !filepath.exists() {
        return Err(DatasetsError::Other(format!(
            "Local dataset file not found: {}",
            filepath.display()
        )));
    }

    // Verify SHA256 hash
    if expected_sha256 != "builtin" {
        if let Ok(actual_hash) = crate::cache::sha256_hash_file(&filepath) {
            if actual_hash != expected_sha256 {
                return Err(DatasetsError::Other(format!(
                    "Hash verification failed for dataset '{}'. Expected: {}, Got: {}",
                    name, expected_sha256, actual_hash
                )));
            }
        }
    }

    // Load the CSV file
    let config = CsvConfig::default().with_header(true);
    let mut dataset = load_csv(&filepath, config)?;

    // Add metadata
    dataset = dataset.with_description(format!("Local dataset: {}", name));

    Ok(dataset)
}

#[cfg(not(feature = "download"))]
/// Load a dataset by name from the registry (stub for when download feature is disabled)
#[allow(dead_code)]
pub fn load_dataset_byname(_name: &str, _forcedownload: bool) -> Result<crate::utils::Dataset> {
    Err(DatasetsError::Other(
        "Download feature is not enabled. Recompile with --features _download".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = DatasetRegistry::new();
        assert!(registry.entries.is_empty());
    }

    #[test]
    fn test_registry_default() {
        let registry = DatasetRegistry::default();
        assert!(!registry.entries.is_empty());

        // Test local datasets
        assert!(registry.contains("example"));
        assert!(registry.contains("sample_data"));

        // Test built-in toy datasets
        assert!(registry.contains("iris"));
        assert!(registry.contains("boston"));
        assert!(registry.contains("wine"));
        assert!(registry.contains("digits"));
        assert!(registry.contains("breast_cancer"));
        assert!(registry.contains("diabetes"));
    }

    #[test]
    fn test_registry_operations() {
        let mut registry = DatasetRegistry::new();

        let entry = RegistryEntry {
            url: "https://example.com/test.csv",
            sha256: "abcd1234",
        };

        registry.register("test_dataset".to_string(), entry);

        assert!(registry.contains("test_dataset"));
        assert!(!registry.contains("nonexistent"));

        let retrieved = registry.get("test_dataset").unwrap();
        assert_eq!(retrieved.url, "https://example.com/test.csv");
        assert_eq!(retrieved.sha256, "abcd1234");

        let datasets = registry.list_datasets();
        assert_eq!(datasets.len(), 1);
        assert!(datasets.contains(&"test_dataset".to_string()));
    }

    #[test]
    fn test_get_registry() {
        let registry = get_registry();
        assert!(!registry.list_datasets().is_empty());
    }

    #[test]
    fn test_registry_url_schemes() {
        let registry = DatasetRegistry::default();

        // Test built-in datasets have builtin:// URLs
        if let Some(iris_entry) = registry.get("iris") {
            assert_eq!(iris_entry.url, "builtin://iris");
            assert_eq!(iris_entry.sha256, "builtin");
        }

        // Test local datasets have file:// URLs
        if let Some(example_entry) = registry.get("example") {
            assert_eq!(example_entry.url, "file://data/example.csv");
            assert_eq!(
                example_entry.sha256,
                "c51c3ff2e8a5db28b1baed809a2ba29f29643e5a26ad476448eb3889996173d6"
            );
        }
    }

    #[test]
    fn test_dataset_count() {
        let registry = DatasetRegistry::default();
        let datasets = registry.list_datasets();

        // Should have 2 local datasets + 6 built-in toy datasets = 8 total
        assert_eq!(datasets.len(), 8);

        // Verify all expected datasets are present
        let expected_datasets = vec![
            "example",
            "sample_data", // local
            "iris",
            "boston",
            "digits",
            "wine",
            "breast_cancer",
            "diabetes", // built-in
        ];

        for expected in expected_datasets {
            assert!(
                datasets.contains(&expected.to_string()),
                "Dataset '{expected}' not found in registry"
            );
        }
    }
}
