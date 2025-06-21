//! Dataset registry system for managing dataset metadata and locations

use crate::cache::RegistryEntry;
use crate::error::{DatasetsError, Result};
use std::collections::HashMap;

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

    /// Populate the registry with default datasets
    ///
    /// Note: The SHA256 hashes below are placeholder values and must be updated
    /// with actual hashes when the dataset files are available in the repository.
    fn populate_default_datasets(&mut self) {
        // Real-world datasets
        self.register(
            "california_housing".to_string(),
            RegistryEntry {
                url: "https://raw.githubusercontent.com/cool-japan/scirs-datasets/main/california_housing.csv",
                sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", // Placeholder - update with actual hash
            },
        );

        self.register(
            "wine".to_string(),
            RegistryEntry {
                url: "https://raw.githubusercontent.com/cool-japan/scirs-datasets/main/wine.csv",
                sha256: "d4e1c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b856", // Placeholder - update with actual hash
            },
        );

        // Time series datasets
        self.register(
            "electrocardiogram".to_string(),
            RegistryEntry {
                url: "https://raw.githubusercontent.com/cool-japan/scirs-datasets/main/electrocardiogram.json",
                sha256: "a1b2c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b857", // Placeholder - update with actual hash
            },
        );

        self.register(
            "stock_market".to_string(),
            RegistryEntry {
                url: "https://raw.githubusercontent.com/cool-japan/scirs-datasets/main/stock_market.json",
                sha256: "f5e6c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b858", // Placeholder - update with actual hash
            },
        );

        self.register(
            "weather".to_string(),
            RegistryEntry {
                url:
                    "https://raw.githubusercontent.com/cool-japan/scirs-datasets/main/weather.json",
                sha256: "b7c8c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b859", // Placeholder - update with actual hash
            },
        );
    }
}

/// Get the global dataset registry
pub fn get_registry() -> DatasetRegistry {
    DatasetRegistry::default()
}

/// Load a dataset by name from the registry
#[cfg(feature = "download")]
pub fn load_dataset_by_name(name: &str, force_download: bool) -> Result<crate::utils::Dataset> {
    let registry = get_registry();

    match name {
        "california_housing" => crate::sample::load_california_housing(force_download),
        "wine" => crate::sample::load_wine(force_download),
        "electrocardiogram" => crate::time_series::electrocardiogram(),
        "stock_market" => crate::time_series::stock_market(false), // Default to raw prices
        "weather" => crate::time_series::weather(None),            // Default to all features
        _ => {
            if registry.contains(name) {
                Err(DatasetsError::Other(format!(
                    "Dataset '{}' is registered but not yet implemented for loading",
                    name
                )))
            } else {
                Err(DatasetsError::Other(format!(
                    "Unknown dataset: '{}'. Available datasets: {:?}",
                    name,
                    registry.list_datasets()
                )))
            }
        }
    }
}

#[cfg(not(feature = "download"))]
/// Load a dataset by name from the registry (stub for when download feature is disabled)
pub fn load_dataset_by_name(_name: &str, _force_download: bool) -> Result<crate::utils::Dataset> {
    Err(DatasetsError::Other(
        "Download feature is not enabled. Recompile with --features download".to_string(),
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
        assert!(registry.contains("california_housing"));
        assert!(registry.contains("wine"));
        assert!(registry.contains("electrocardiogram"));
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
}
