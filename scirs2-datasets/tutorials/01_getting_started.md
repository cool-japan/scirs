# Getting Started with SciRS2 Datasets

This tutorial introduces you to the core functionality of the SciRS2 datasets module, which provides comprehensive dataset loading utilities similar to scikit-learn's datasets module.

## Overview

The `scirs2-datasets` module offers:

- **Toy datasets**: Classical machine learning datasets (Iris, Boston, etc.)
- **Data generators**: Create synthetic datasets for various ML tasks
- **Dataset utilities**: Cross-validation, train/test splitting, normalization
- **Efficient caching**: Smart caching system for downloaded datasets
- **Registry system**: Centralized metadata management

## Installation and Setup

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-datasets = "0.1.0-beta.2"
```

## Your First Dataset

Let's start by loading the classic Iris dataset:

```rust
use scirs2_datasets::{load_iris, Dataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the Iris dataset
    let iris = load_iris()?;
    
    // Basic dataset information
    println!("Dataset name: {}", iris.metadata.description);
    println!("Samples: {}", iris.n_samples());
    println!("Features: {}", iris.n_features());
    println!("Target shape: {:?}", iris.target.as_ref().map(|t| t.shape()));
    
    // Feature names (if available)
    if let Some(feature_names) = iris.feature_names() {
        println!("Feature names: {:?}", feature_names);
    }
    
    // Target names (for classification datasets)
    if let Some(target_names) = iris.target_names() {
        println!("Target names: {:?}", target_names);
    }
    
    Ok(())
}
```

## Understanding the Dataset Structure

All datasets in SciRS2 use the `Dataset` struct:

```rust
pub struct Dataset {
    pub data: Array2<f64>,           // Feature matrix (n_samples × n_features)
    pub target: Option<Array1<f64>>, // Target values (n_samples,)
    pub metadata: DatasetMetadata,   // Metadata information
}
```

Key methods:
- `n_samples()`: Number of samples/rows
- `n_features()`: Number of features/columns  
- `feature_names()`: Feature column names
- `target_names()`: Target class names (for classification)

## Working with Different Dataset Types

### Classification Datasets

```rust
use scirs2_datasets::{load_iris, load_wine, load_breast_cancer};

// Multi-class classification
let iris = load_iris()?;
println!("Iris: {} classes", iris.target_names().map_or(0, |t| t.len()));

// Wine classification
let wine = load_wine()?;
println!("Wine: {} samples, {} features", wine.n_samples(), wine.n_features());

// Binary classification
let cancer = load_breast_cancer()?;
println!("Breast cancer: {} samples", cancer.n_samples());
```

### Regression Datasets

```rust
use scirs2_datasets::{load_boston, load_diabetes};

// Boston housing prices
let boston = load_boston()?;
println!("Boston housing: {} samples", boston.n_samples());

// Diabetes progression
let diabetes = load_diabetes()?;
println!("Diabetes: {} features", diabetes.n_features());
```

### Computer Vision Datasets

```rust
use scirs2_datasets::load_digits;

// Handwritten digits (8x8 images)
let digits = load_digits()?;
println!("Digits: {} samples, {} pixels per image", 
         digits.n_samples(), digits.n_features());

// The data contains flattened 8x8 = 64 pixel values
assert_eq!(digits.n_features(), 64);
```

## Data Exploration

```rust
use scirs2_datasets::load_iris;

let iris = load_iris()?;

// Basic statistics
let data = &iris.data;
println!("Data shape: {:?}", data.shape());

// Feature statistics (example for first feature)
let first_feature = data.column(0);
let mean = first_feature.mean().unwrap();
let std_dev = first_feature.std(0.0);
println!("First feature - Mean: {:.2}, Std: {:.2}", mean, std_dev);

// Target distribution
if let Some(target) = &iris.target {
    let mut class_counts = std::collections::HashMap::new();
    for &class in target.iter() {
        *class_counts.entry(class as i32).or_insert(0) += 1;
    }
    println!("Class distribution: {:?}", class_counts);
}
```

## Error Handling

The datasets module uses a comprehensive error handling system:

```rust
use scirs2_datasets::{load_iris, error::DatasetsError};

match load_iris() {
    Ok(dataset) => {
        println!("Successfully loaded {} samples", dataset.n_samples());
    }
    Err(DatasetsError::Io(e)) => {
        eprintln!("IO error: {}", e);
    }
    Err(DatasetsError::Parse(msg)) => {
        eprintln!("Parse error: {}", msg);
    }
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

## Next Steps

Now that you understand the basics, explore these topics:

1. **[Data Generation Tutorial](02_data_generation.md)** - Create synthetic datasets
2. **[Cross-Validation Tutorial](03_cross_validation.md)** - Split data for training/testing
3. **[Dataset Utilities Tutorial](04_dataset_utilities.md)** - Advanced dataset manipulation
4. **[Custom Datasets Tutorial](05_custom_datasets.md)** - Load your own data
5. **[Performance Optimization](06_performance.md)** - Optimize for large datasets

## Common Patterns

### Quick Dataset Loading and Splitting

```rust
use scirs2_datasets::{load_iris, utils::train_test_split};

let iris = load_iris()?;
let (train, test) = train_test_split(&iris, 0.2, Some(42))?;

println!("Training: {} samples", train.n_samples());
println!("Testing: {} samples", test.n_samples());
```

### Dataset Normalization

```rust
use scirs2_datasets::{load_boston, utils::normalize};

let boston = load_boston()?;
let mut data = boston.data.clone();
normalize(&mut data);  // In-place normalization
```

### Batch Processing

```rust
use scirs2_datasets::load_digits;

let digits = load_digits()?;
let batch_size = 32;

// Process in batches
for chunk in digits.data.axis_chunks_iter(ndarray::Axis(0), batch_size) {
    println!("Processing batch of {} samples", chunk.nrows());
    // Your batch processing logic here
}
```

This tutorial covered the fundamentals of working with datasets in SciRS2. The module provides a powerful, efficient, and user-friendly interface for all your dataset needs in scientific computing and machine learning applications.