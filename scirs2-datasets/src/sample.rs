//! Sample datasets for testing and demonstration
//!
//! This module provides larger, real-world datasets that can be downloaded
//! and loaded for testing and demonstration purposes.

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;

#[cfg(feature = "download")]
use crate::cache::download_data;
#[cfg(feature = "download")]
use crate::loaders;

/// URL for dataset resources
#[allow(dead_code)]
const DATASET_BASE_URL: &str = "https://raw.githubusercontent.com/cool-japan/scirs-datasets/main/";

/// Load the California Housing dataset
#[cfg(feature = "download")]
pub fn load_california_housing(force_download: bool) -> Result<Dataset> {
    let url = format!("{}/california_housing.csv", DATASET_BASE_URL);

    // Download or load from cache
    let data = download_data(&url, force_download)?;

    // Create a temporary file
    use std::io::Write;
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("scirs2_california_housing.csv");

    let mut temp_file = std::fs::File::create(&temp_path).map_err(DatasetsError::IoError)?;

    temp_file.write_all(&data).map_err(DatasetsError::IoError)?;

    // Load from the temporary file (using CSV loader)
    let mut dataset = loaders::load_csv(&temp_path, true, Some(8))?;

    // Add metadata
    let feature_names = vec![
        "MedInc".to_string(),
        "HouseAge".to_string(),
        "AveRooms".to_string(),
        "AveBedrms".to_string(),
        "Population".to_string(),
        "AveOccup".to_string(),
        "Latitude".to_string(),
        "Longitude".to_string(),
    ];

    let description = "California Housing dataset
    
The data was derived from the 1990 U.S. census, using one row per census block group.
A block group is the smallest geographical unit for which the U.S. Census Bureau 
publishes sample data.

Features:
- MedInc: median income in block group
- HouseAge: median house age in block group
- AveRooms: average number of rooms per household
- AveBedrms: average number of bedrooms per household
- Population: block group population
- AveOccup: average number of household members
- Latitude: block group latitude
- Longitude: block group longitude

Target: Median house value for California districts, expressed in hundreds of thousands of dollars.

This dataset is useful for regression tasks."
        .to_string();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_description(description);

    // Remove the temporary file
    std::fs::remove_file(temp_path).ok();

    Ok(dataset)
}

// Stub when download feature is not enabled
#[cfg(not(feature = "download"))]
/// Loads the California Housing dataset
///
/// This is a stub implementation when the download feature is not enabled.
/// It returns an error informing the user to enable the download feature.
///
/// # Arguments
///
/// * `_force_download` - If true, force a new download instead of using cache
///
/// # Returns
///
/// * An error indicating that the download feature is not enabled
pub fn load_california_housing(_force_download: bool) -> Result<Dataset> {
    Err(DatasetsError::Other(
        "Download feature is not enabled. Recompile with --features download".to_string(),
    ))
}

/// Load the Wine dataset
#[cfg(feature = "download")]
pub fn load_wine(force_download: bool) -> Result<Dataset> {
    let url = format!("{}/wine.csv", DATASET_BASE_URL);

    // Download or load from cache
    let data = download_data(&url, force_download)?;

    // Create a temporary file
    use std::io::Write;
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("scirs2_wine.csv");

    let mut temp_file = std::fs::File::create(&temp_path).map_err(DatasetsError::IoError)?;

    temp_file.write_all(&data).map_err(DatasetsError::IoError)?;

    // Load from the temporary file (using CSV loader)
    let mut dataset = loaders::load_csv(&temp_path, true, Some(0))?;

    // Add metadata
    let feature_names = vec![
        "alcohol".to_string(),
        "malic_acid".to_string(),
        "ash".to_string(),
        "alcalinity_of_ash".to_string(),
        "magnesium".to_string(),
        "total_phenols".to_string(),
        "flavanoids".to_string(),
        "nonflavanoid_phenols".to_string(),
        "proanthocyanins".to_string(),
        "color_intensity".to_string(),
        "hue".to_string(),
        "od280_od315_of_diluted_wines".to_string(),
        "proline".to_string(),
    ];

    let target_names = vec![
        "class_0".to_string(),
        "class_1".to_string(),
        "class_2".to_string(),
    ];

    let description = "Wine Recognition dataset
    
The data is the results of a chemical analysis of wines grown in the same region in Italy
but derived from three different cultivars. The analysis determined the quantities of
13 constituents found in each of the three types of wines.

Features: Various chemical properties of the wine

Target: Class of wine (0, 1, or 2)

This dataset is useful for classification tasks."
        .to_string();

    dataset = dataset
        .with_feature_names(feature_names)
        .with_target_names(target_names)
        .with_description(description);

    // Remove the temporary file
    std::fs::remove_file(temp_path).ok();

    Ok(dataset)
}

// Stub when download feature is not enabled
#[cfg(not(feature = "download"))]
/// Loads the Wine dataset
///
/// This is a stub implementation when the download feature is not enabled.
/// It returns an error informing the user to enable the download feature.
///
/// # Arguments
///
/// * `_force_download` - If true, force a new download instead of using cache
///
/// # Returns
///
/// * An error indicating that the download feature is not enabled
pub fn load_wine(_force_download: bool) -> Result<Dataset> {
    Err(DatasetsError::Other(
        "Download feature is not enabled. Recompile with --features download".to_string(),
    ))
}

/// Sample data fetcher - retrieves a list of available datasets
#[cfg(feature = "download")]
pub fn get_available_datasets() -> Result<Vec<String>> {
    let url = format!("{}/datasets_index.txt", DATASET_BASE_URL);

    // Download or load from cache
    let data = download_data(&url, true)?;

    // Parse the list of datasets
    let content = String::from_utf8(data).map_err(|e| {
        DatasetsError::InvalidFormat(format!("Failed to parse datasets index: {}", e))
    })?;

    let datasets = content
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    Ok(datasets)
}

// Stub when download feature is not enabled
#[cfg(not(feature = "download"))]
/// Retrieves a list of available datasets
///
/// This is a stub implementation when the download feature is not enabled.
/// It returns an error informing the user to enable the download feature.
///
/// # Returns
///
/// * An error indicating that the download feature is not enabled
pub fn get_available_datasets() -> Result<Vec<String>> {
    Err(DatasetsError::Other(
        "Download feature is not enabled. Recompile with --features download".to_string(),
    ))
}
