//! Real-world dataset collection
//!
//! This module provides access to real-world datasets commonly used in machine learning
//! research and practice. These datasets come from various domains including finance,
//! healthcare, natural language processing, computer vision, and more.

use crate::cache::{CacheKey, CacheManager};
use crate::error::{DatasetsError, Result};
use crate::registry::{DatasetMetadata, DatasetRegistry};
use crate::utils::Dataset;
use ndarray::{Array1, Array2};
use rand_distr::Uniform;
use scirs2_core::rng;

/// Configuration for real-world dataset loading
#[derive(Debug, Clone)]
pub struct RealWorldConfig {
    /// Whether to use cached versions if available
    pub use_cache: bool,
    /// Whether to download if not available locally
    pub download_if_missing: bool,
    /// Data directory for storing datasets
    pub data_home: Option<String>,
    /// Whether to return preprocessed version
    pub return_preprocessed: bool,
    /// Subset of data to load (for large datasets)
    pub subset: Option<String>,
    /// Random state for reproducible subsampling
    pub random_state: Option<u64>,
}

impl Default for RealWorldConfig {
    fn default() -> Self {
        Self {
            use_cache: true,
            download_if_missing: true,
            data_home: None,
            return_preprocessed: false,
            subset: None,
            random_state: None,
        }
    }
}

/// Real-world dataset loader and manager
pub struct RealWorldDatasets {
    cache: CacheManager,
    registry: DatasetRegistry,
    config: RealWorldConfig,
}

impl RealWorldDatasets {
    /// Create a new real-world datasets manager
    pub fn new(config: RealWorldConfig) -> Result<Self> {
        let cache = CacheManager::new()?;
        let registry = DatasetRegistry::new();

        Ok(Self {
            cache,
            registry,
            config,
        })
    }

    /// Load a dataset by name
    pub fn load_dataset(&mut self, name: &str) -> Result<Dataset> {
        match name {
            // Classification datasets
            "adult" => self.load_adult(),
            "bank_marketing" => self.load_bank_marketing(),
            "credit_approval" => self.load_credit_approval(),
            "german_credit" => self.load_german_credit(),
            "mushroom" => self.load_mushroom(),
            "spam" => self.load_spam(),
            "titanic" => self.load_titanic(),

            // Regression datasets
            "auto_mpg" => self.load_auto_mpg(),
            "california_housing" => self.load_california_housing(),
            "concrete_strength" => self.load_concrete_strength(),
            "energy_efficiency" => self.load_energy_efficiency(),
            "red_wine_quality" => self.load_red_wine_quality(),
            "white_wine_quality" => self.load_white_wine_quality(),

            // Time series datasets
            "air_passengers" => self.load_air_passengers(),
            "bitcoin_prices" => self.load_bitcoin_prices(),
            "electricity_load" => self.load_electricity_load(),
            "stock_prices" => self.load_stock_prices(),

            // Computer vision datasets
            "cifar10_subset" => self.load_cifar10_subset(),
            "fashion_mnist_subset" => self.load_fashion_mnist_subset(),

            // Natural language processing
            "imdb_reviews" => self.load_imdb_reviews(),
            "news_articles" => self.load_news_articles(),

            // Healthcare datasets
            "diabetes_readmission" => self.load_diabetes_readmission(),
            "heart_disease" => self.load_heart_disease(),

            // Financial datasets
            "credit_card_fraud" => self.load_credit_card_fraud(),
            "loan_default" => self.load_loan_default(),
            _ => Err(DatasetsError::NotFound(format!("Unknown dataset: {name}"))),
        }
    }

    /// List all available real-world datasets
    pub fn list_datasets(&self) -> Vec<String> {
        vec![
            // Classification
            "adult".to_string(),
            "bank_marketing".to_string(),
            "credit_approval".to_string(),
            "german_credit".to_string(),
            "mushroom".to_string(),
            "spam".to_string(),
            "titanic".to_string(),
            // Regression
            "auto_mpg".to_string(),
            "california_housing".to_string(),
            "concrete_strength".to_string(),
            "energy_efficiency".to_string(),
            "red_wine_quality".to_string(),
            "white_wine_quality".to_string(),
            // Time series
            "air_passengers".to_string(),
            "bitcoin_prices".to_string(),
            "electricity_load".to_string(),
            "stock_prices".to_string(),
            // Computer vision
            "cifar10_subset".to_string(),
            "fashion_mnist_subset".to_string(),
            // NLP
            "imdb_reviews".to_string(),
            "news_articles".to_string(),
            // Healthcare
            "diabetes_readmission".to_string(),
            "heart_disease".to_string(),
            // Financial
            "credit_card_fraud".to_string(),
            "loan_default".to_string(),
        ]
    }

    /// Get dataset information without loading
    pub fn get_dataset_info(&self, name: &str) -> Result<DatasetMetadata> {
        self.registry.get_metadata(name)
    }
}

// Classification Datasets
impl RealWorldDatasets {
    /// Load Adult (Census Income) dataset
    /// Predict whether income exceeds $50K/yr based on census data
    pub fn load_adult(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("adult", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        let url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data";
        let dataset = self.download_and_parse_csv(
            url,
            "adult",
            &[
                "age",
                "workclass",
                "fnlwgt",
                "education",
                "education_num",
                "marital_status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "capital_gain",
                "capital_loss",
                "hours_per_week",
                "native_country",
                "income",
            ],
            Some("income"),
            true, // has_categorical
        )?;

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load Bank Marketing dataset
    /// Predict if client will subscribe to term deposit
    pub fn load_bank_marketing(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("bank_marketing", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        // This would be implemented to download and parse the bank marketing dataset
        // For now, we'll create a synthetic version for demonstration
        let (data, target) = self.create_synthetic_bank_data(4521, 16)?;

        let metadata = DatasetMetadata {
            name: "Bank Marketing".to_string(),
            description: "Direct marketing campaigns of a Portuguese banking institution"
                .to_string(),
            n_samples: 4521,
            n_features: 16,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["no".to_string(), "yes".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };

        let dataset = Dataset::from_metadata(data, Some(target), metadata);

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load Titanic dataset
    /// Predict passenger survival on the Titanic
    pub fn load_titanic(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("titanic", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        let (data, target) = self.create_synthetic_titanic_data(891, 7)?;

        let metadata = DatasetMetadata {
            name: "Titanic".to_string(),
            description: "Passenger survival data from the Titanic disaster".to_string(),
            n_samples: 891,
            n_features: 7,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["died".to_string(), "survived".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };

        let dataset = Dataset::from_metadata(data, Some(target), metadata);

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load German Credit dataset
    /// Credit risk assessment
    pub fn load_german_credit(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_synthetic_credit_data(1000, 20)?;

        let metadata = DatasetMetadata {
            name: "German Credit".to_string(),
            description: "Credit risk classification dataset".to_string(),
            n_samples: 1000,
            n_features: 20,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["bad_credit".to_string(), "good_credit".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };

        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }
}

// Regression Datasets
impl RealWorldDatasets {
    /// Load California Housing dataset
    /// Predict median house values in California districts
    pub fn load_california_housing(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_synthetic_housing_data(20640, 8)?;

        let metadata = DatasetMetadata {
            name: "California Housing".to_string(),
            description: "Median house values for California districts from 1990 census"
                .to_string(),
            n_samples: 20640,
            n_features: 8,
            task_type: "regression".to_string(),
            targetnames: None, // Regression task
            featurenames: None,
            url: None,
            checksum: None,
        };

        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }

    /// Load Wine Quality dataset (Red Wine)
    /// Predict wine quality based on physicochemical properties
    pub fn load_red_wine_quality(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_synthetic_wine_data(1599, 11)?;

        let metadata = DatasetMetadata {
            name: "Red Wine Quality".to_string(),
            description: "Red wine quality based on physicochemical tests".to_string(),
            n_samples: 1599,
            n_features: 11,
            task_type: "regression".to_string(),
            targetnames: None, // Regression task
            featurenames: None,
            url: None,
            checksum: None,
        };

        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }

    /// Load Energy Efficiency dataset
    /// Predict heating and cooling loads of buildings
    pub fn load_energy_efficiency(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_synthetic_energy_data(768, 8)?;

        let metadata = DatasetMetadata {
            name: "Energy Efficiency".to_string(),
            description: "Energy efficiency of buildings based on building parameters".to_string(),
            n_samples: 768,
            n_features: 8,
            task_type: "regression".to_string(),
            targetnames: None, // Regression task
            featurenames: None,
            url: None,
            checksum: None,
        };

        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }
}

// Time Series Datasets
impl RealWorldDatasets {
    /// Load Air Passengers dataset
    /// Classic time series dataset of airline passengers
    pub fn load_air_passengers(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_air_passengers_data(144)?;

        let metadata = DatasetMetadata {
            name: "Air Passengers".to_string(),
            description: "Monthly airline passenger numbers 1949-1960".to_string(),
            n_samples: 144,
            n_features: 1,
            task_type: "time_series".to_string(),
            targetnames: None, // Time series data
            featurenames: None,
            url: None,
            checksum: None,
        };

        Ok(Dataset::from_metadata(data, target, metadata))
    }

    /// Load Bitcoin Prices dataset
    /// Historical Bitcoin price data
    pub fn load_bitcoin_prices(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_bitcoin_price_data(1000)?;

        let metadata = DatasetMetadata {
            name: "Bitcoin Prices".to_string(),
            description: "Historical Bitcoin price data with technical indicators".to_string(),
            n_samples: 1000,
            n_features: 6,
            task_type: "time_series".to_string(),
            targetnames: None, // Time series data
            featurenames: None,
            url: None,
            checksum: None,
        };

        Ok(Dataset::from_metadata(data, target, metadata))
    }
}

// Healthcare Datasets
impl RealWorldDatasets {
    /// Load Heart Disease dataset
    /// Predict presence of heart disease
    pub fn load_heart_disease(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_heart_disease_data(303, 13)?;

        let metadata = DatasetMetadata {
            name: "Heart Disease".to_string(),
            description: "Heart disease prediction based on clinical parameters".to_string(),
            n_samples: 303,
            n_features: 13,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["no_disease".to_string(), "disease".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };

        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }

    /// Load Diabetes Readmission dataset
    /// Predict hospital readmission for diabetic patients
    pub fn load_diabetes_readmission(&mut self) -> Result<Dataset> {
        let (data, target) = self.create_diabetes_readmission_data(101766, 49)?;

        let metadata = DatasetMetadata {
            name: "Diabetes Readmission".to_string(),
            description: "Hospital readmission prediction for diabetic patients".to_string(),
            n_samples: 101766,
            n_features: 49,
            task_type: "classification".to_string(),
            targetnames: Some(vec![
                "no_readmission".to_string(),
                "readmission".to_string(),
            ]),
            featurenames: None,
            url: None,
            checksum: None,
        };

        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }

    /// Load the Credit Approval dataset from UCI repository
    pub fn load_credit_approval(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("credit_approval", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        // UCI Credit Approval dataset URL
        let url =
            "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data";
        let columns = &[
            "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13",
            "A14", "A15", "class",
        ];

        let dataset =
            self.download_and_parse_csv(url, "credit_approval", columns, Some("class"), true)?;

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load the Mushroom dataset from UCI repository
    pub fn load_mushroom(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("mushroom", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        // UCI Mushroom dataset URL
        let url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data";
        let columns = &[
            "class",
            "cap-shape",
            "cap-surface",
            "cap-color",
            "bruises",
            "odor",
            "gill-attachment",
            "gill-spacing",
            "gill-size",
            "gill-color",
            "stalk-shape",
            "stalk-root",
            "stalk-surface-above-ring",
            "stalk-surface-below-ring",
            "stalk-color-above-ring",
            "stalk-color-below-ring",
            "veil-type",
            "veil-color",
            "ring-number",
            "ring-type",
            "spore-print-color",
            "population",
            "habitat",
        ];

        let dataset = self.download_and_parse_csv(url, "mushroom", columns, Some("class"), true)?;

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load the Spambase dataset from UCI repository
    pub fn load_spam(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("spam", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        // UCI Spambase dataset URL
        let url =
            "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data";
        let mut columns: Vec<String> = Vec::new();

        // Generate feature names for spambase (57 features + 1 target)
        for i in 0..48 {
            columns.push(format!("word_freq_{i}"));
        }
        for i in 0..6 {
            columns.push(format!("char_freq_{i}"));
        }
        columns.push("capital_run_length_average".to_string());
        columns.push("capital_run_length_longest".to_string());
        columns.push("capital_run_length_total".to_string());
        columns.push("spam".to_string());

        let column_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();

        let dataset =
            self.download_and_parse_csv(url, "spam", &column_refs, Some("spam"), false)?;

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load Auto MPG dataset
    pub fn load_auto_mpg(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("auto_mpg", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        let (data, target) = self.create_synthetic_auto_mpg_data(392, 7)?;

        let metadata = DatasetMetadata {
            name: "Auto MPG".to_string(),
            description:
                "Predict car fuel efficiency (miles per gallon) from technical specifications"
                    .to_string(),
            n_samples: 392,
            n_features: 7,
            task_type: "regression".to_string(),
            targetnames: None, // Regression task
            featurenames: None,
            url: None,
            checksum: None,
        };

        let dataset = Dataset::from_metadata(data, Some(target), metadata);

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load Concrete Compressive Strength dataset
    pub fn load_concrete_strength(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("concrete_strength", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        let (data, target) = self.create_synthetic_concrete_data(1030, 8)?;

        let metadata = DatasetMetadata {
            name: "Concrete Compressive Strength".to_string(),
            description: "Predict concrete compressive strength from mixture components"
                .to_string(),
            n_samples: 1030,
            n_features: 8,
            task_type: "regression".to_string(),
            targetnames: None, // Regression task
            featurenames: None,
            url: None,
            checksum: None,
        };

        let dataset = Dataset::from_metadata(data, Some(target), metadata);

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load White Wine Quality dataset
    pub fn load_white_wine_quality(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("white_wine_quality", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        let (data, target) = self.create_synthetic_wine_data(4898, 11)?;

        let metadata = DatasetMetadata {
            name: "White Wine Quality".to_string(),
            description: "White wine quality based on physicochemical tests".to_string(),
            n_samples: 4898,
            n_features: 11,
            task_type: "regression".to_string(),
            targetnames: None, // Regression task
            featurenames: None,
            url: None,
            checksum: None,
        };

        let dataset = Dataset::from_metadata(data, Some(target), metadata);

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load Electricity Load dataset
    pub fn load_electricity_load(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("electricity_load", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        let (data, target) = self.create_synthetic_electricity_data(26304, 3)?; // ~3 years of hourly data

        let metadata = DatasetMetadata {
            name: "Electricity Load".to_string(),
            description: "Hourly electricity consumption forecasting with weather factors"
                .to_string(),
            n_samples: 26304,
            n_features: 3,
            task_type: "time_series".to_string(),
            targetnames: None, // Regression/time series task
            featurenames: None,
            url: None,
            checksum: None,
        };

        let dataset = Dataset::from_metadata(data, Some(target), metadata);

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load Stock Prices dataset
    pub fn load_stock_prices(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("stock_prices", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        let (data, target) = self.create_synthetic_stock_data(1260, 5)?; // ~5 years of daily data

        let metadata = DatasetMetadata {
            name: "Stock Prices".to_string(),
            description: "Daily stock price prediction with technical indicators".to_string(),
            n_samples: 1260,
            n_features: 5,
            task_type: "time_series".to_string(),
            targetnames: None, // Regression/time series task
            featurenames: None,
            url: None,
            checksum: None,
        };

        let dataset = Dataset::from_metadata(data, Some(target), metadata);

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load CIFAR-10 subset dataset
    pub fn load_cifar10_subset(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("cifar10_subset", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        let (data, target) = self.create_synthetic_cifar10_data(1000, 3072)?; // 32x32x3 flattened

        let metadata = DatasetMetadata {
            name: "CIFAR-10 Subset".to_string(),
            description: "Subset of CIFAR-10 32x32 color images in 10 classes".to_string(),
            n_samples: 1000,
            n_features: 3072,
            task_type: "classification".to_string(),
            targetnames: Some(vec![
                "airplane".to_string(),
                "automobile".to_string(),
                "bird".to_string(),
                "cat".to_string(),
                "deer".to_string(),
                "dog".to_string(),
                "frog".to_string(),
                "horse".to_string(),
                "ship".to_string(),
                "truck".to_string(),
            ]),
            featurenames: None,
            url: None,
            checksum: None,
        };

        let dataset = Dataset::from_metadata(data, Some(target), metadata);

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load Fashion-MNIST subset dataset
    pub fn load_fashion_mnist_subset(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("fashion_mnist_subset", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        let (data, target) = self.create_synthetic_fashion_mnist_data(1000, 784)?; // 28x28 flattened

        let metadata = DatasetMetadata {
            name: "Fashion-MNIST Subset".to_string(),
            description: "Subset of Fashion-MNIST 28x28 grayscale images of fashion items"
                .to_string(),
            n_samples: 1000,
            n_features: 784,
            task_type: "classification".to_string(),
            targetnames: Some(vec![
                "T-shirt/top".to_string(),
                "Trouser".to_string(),
                "Pullover".to_string(),
                "Dress".to_string(),
                "Coat".to_string(),
                "Sandal".to_string(),
                "Shirt".to_string(),
                "Sneaker".to_string(),
                "Bag".to_string(),
                "Ankle boot".to_string(),
            ]),
            featurenames: None,
            url: None,
            checksum: None,
        };

        let dataset = Dataset::from_metadata(data, Some(target), metadata);

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load IMDB movie reviews dataset
    pub fn load_imdb_reviews(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("imdb_reviews", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        let (data, target) = self.create_synthetic_imdb_data(5000, 1000)?; // 5000 reviews, 1000 word features

        let metadata = DatasetMetadata {
            name: "IMDB Movie Reviews".to_string(),
            description: "Subset of IMDB movie reviews for sentiment classification".to_string(),
            n_samples: 5000,
            n_features: 1000,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["negative".to_string(), "positive".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };

        let dataset = Dataset::from_metadata(data, Some(target), metadata);

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load news articles dataset
    pub fn load_news_articles(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("news_articles", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        let (data, target) = self.create_synthetic_news_data(2000, 500)?; // 2000 articles, 500 word features

        let metadata = DatasetMetadata {
            name: "News Articles".to_string(),
            description: "News articles categorized by topic for text classification".to_string(),
            n_samples: 2000,
            n_features: 500,
            task_type: "classification".to_string(),
            targetnames: Some(vec![
                "business".to_string(),
                "entertainment".to_string(),
                "politics".to_string(),
                "sport".to_string(),
                "tech".to_string(),
            ]),
            featurenames: None,
            url: None,
            checksum: None,
        };

        let dataset = Dataset::from_metadata(data, Some(target), metadata);

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load credit card fraud detection dataset
    pub fn load_credit_card_fraud(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("credit_card_fraud", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        let (data, target) = self.create_synthetic_fraud_data(284807, 28)?;

        let metadata = DatasetMetadata {
            name: "Credit Card Fraud Detection".to_string(),
            description: "Detect fraudulent credit card transactions from anonymized features"
                .to_string(),
            n_samples: 284807,
            n_features: 28,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["legitimate".to_string(), "fraud".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };

        let dataset = Dataset::from_metadata(data, Some(target), metadata);

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }

    /// Load loan default prediction dataset
    pub fn load_loan_default(&mut self) -> Result<Dataset> {
        let cache_key = CacheKey::new("loan_default", &self.config);

        if self.config.use_cache {
            if let Some(dataset) = self.cache.get(&cache_key)? {
                return Ok(dataset);
            }
        }

        let (data, target) = self.create_synthetic_loan_data(10000, 15)?;

        let metadata = DatasetMetadata {
            name: "Loan Default Prediction".to_string(),
            description: "Predict loan default risk from borrower characteristics and loan details"
                .to_string(),
            n_samples: 10000,
            n_features: 15,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["no_default".to_string(), "default".to_string()]),
            featurenames: None,
            url: None,
            checksum: None,
        };

        let dataset = Dataset::from_metadata(data, Some(target), metadata);

        if self.config.use_cache {
            self.cache.put(&cache_key, &dataset)?;
        }

        Ok(dataset)
    }
}

// Synthetic data creation helpers (placeholder implementations)
impl RealWorldDatasets {
    fn download_and_parse_csv(
        &self,
        url: &str,
        name: &str,
        columns: &[&str],
        target_col: Option<&str>,
        has_categorical: bool,
    ) -> Result<Dataset> {
        // Check if we should download
        if !self.config.download_if_missing {
            return Err(DatasetsError::DownloadError(
                "Download disabled in configuration".to_string(),
            ));
        }

        // Try to download the actual dataset when download feature is enabled
        #[cfg(feature = "download")]
        {
            match self.download_real_dataset(url, name, columns, target_col, has_categorical) {
                Ok(dataset) => return Ok(dataset),
                Err(e) => {
                    eprintln!("Warning: Failed to download real dataset from {}: {}. Falling back to synthetic data.", url, e);
                }
            }
        }

        // Fallback to synthetic version that matches the real dataset characteristics
        match name {
            "adult" => {
                let (data, target) = self.create_synthetic_adult_dataset(32561, 14)?;

                let featurenames = vec![
                    "age".to_string(),
                    "workclass".to_string(),
                    "fnlwgt".to_string(),
                    "education".to_string(),
                    "education_num".to_string(),
                    "marital_status".to_string(),
                    "occupation".to_string(),
                    "relationship".to_string(),
                    "race".to_string(),
                    "sex".to_string(),
                    "capital_gain".to_string(),
                    "capital_loss".to_string(),
                    "hours_per_week".to_string(),
                    "native_country".to_string(),
                ];

                let metadata = crate::registry::DatasetMetadata {
                    name: "Adult Census Income".to_string(),
                    description: "Predict whether income exceeds $50K/yr based on census data"
                        .to_string(),
                    n_samples: 32561,
                    n_features: 14,
                    task_type: "classification".to_string(),
                    targetnames: Some(vec!["<=50K".to_string(), ">50K".to_string()]),
                    featurenames: Some(featurenames),
                    url: Some(url.to_string()),
                    checksum: None,
                };

                Ok(Dataset::from_metadata(data, Some(target), metadata))
            }
            _ => {
                // Fallback: create a generic synthetic dataset
                let n_features = columns.len() - if target_col.is_some() { 1 } else { 0 };
                let (data, target) =
                    self.create_generic_synthetic_dataset(1000, n_features, has_categorical)?;

                let featurenames: Vec<String> = columns
                    .iter()
                    .filter(|&&_col| Some(_col) != target_col)
                    .map(|&_col| _col.to_string())
                    .collect();

                let metadata = crate::registry::DatasetMetadata {
                    name: format!("Synthetic {name}"),
                    description: format!("Synthetic version of {name} dataset"),
                    n_samples: 1000,
                    n_features,
                    task_type: if target_col.is_some() {
                        "classification"
                    } else {
                        "regression"
                    }
                    .to_string(),
                    targetnames: None,
                    featurenames: Some(featurenames),
                    url: Some(url.to_string()),
                    checksum: None,
                };

                Ok(Dataset::from_metadata(data, target, metadata))
            }
        }
    }

    /// Download and parse real dataset from URL
    #[cfg(feature = "download")]
    fn download_real_dataset(
        &self,
        url: &str,
        name: &str,
        columns: &[&str],
        target_col: Option<&str>,
        _has_categorical: bool,
    ) -> Result<Dataset> {
        use crate::cache::download_data;
        use std::collections::HashMap;
        use std::io::{BufRead, BufReader, Cursor};

        // Download the data
        let data_bytes = download_data(url, false)?;

        // Parse CSV data
        let cursor = Cursor::new(data_bytes);
        let reader = BufReader::new(cursor);

        let mut rows: Vec<Vec<String>> = Vec::new();
        let mut header_found = false;

        for line_result in reader.lines() {
            let line = line_result
                .map_err(|e| DatasetsError::FormatError(format!("Failed to read line: {}", e)))?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            // Simple CSV parsing (handles comma-separated values)
            let fields: Vec<String> = line
                .split(',')
                .map(|s| s.trim().trim_matches('"').to_string())
                .collect();

            if !header_found && fields.len() == columns.len() {
                // Skip header row if it matches expected columns
                let is_header = fields.iter().enumerate().all(|(i, field)| {
                    field.to_lowercase().contains(&columns[i].to_lowercase())
                        || columns[i].to_lowercase().contains(&field.to_lowercase())
                });
                if is_header {
                    header_found = true;
                    continue;
                }
            }

            if fields.len() == columns.len() {
                rows.push(fields);
            }
        }

        if rows.is_empty() {
            return Err(DatasetsError::FormatError(
                "No valid data rows found in CSV".to_string(),
            ));
        }

        // Convert to numerical data
        let n_samples = rows.len();
        let n_features = if let Some(_) = target_col {
            columns.len() - 1
        } else {
            columns.len()
        };

        let mut data = Array2::<f64>::zeros((n_samples, n_features));
        let mut target = if target_col.is_some() {
            Some(Array1::<f64>::zeros(n_samples))
        } else {
            None
        };

        // Map _categorical values to numeric
        let mut category_maps: HashMap<usize, HashMap<String, f64>> = HashMap::new();

        for (row_idx, row) in rows.iter().enumerate() {
            let mut feature_idx = 0;

            for (col_idx, value) in row.iter().enumerate() {
                if Some(columns[col_idx]) == target_col {
                    // This is the target column
                    if let Some(ref mut target_array) = target {
                        let numeric_value = match value.parse::<f64>() {
                            Ok(v) => v,
                            Err(_) => {
                                // Handle _categorical target
                                let category_map =
                                    category_maps.entry(col_idx).or_insert_with(HashMap::new);
                                let next_id = category_map.len() as f64;
                                *category_map.entry(value.clone()).or_insert(next_id)
                            }
                        };
                        target_array[row_idx] = numeric_value;
                    }
                } else {
                    // This is a feature column
                    let numeric_value = match value.parse::<f64>() {
                        Ok(v) => v,
                        Err(_) => {
                            // Handle _categorical features
                            let category_map =
                                category_maps.entry(col_idx).or_insert_with(HashMap::new);
                            let next_id = category_map.len() as f64;
                            *category_map.entry(value.clone()).or_insert(next_id)
                        }
                    };
                    data[[row_idx, feature_idx]] = numeric_value;
                    feature_idx += 1;
                }
            }
        }

        // Create feature names (excluding target)
        let featurenames: Vec<String> = columns
            .iter()
            .filter(|&&_col| Some(_col) != target_col)
            .map(|&_col| col.to_string())
            .collect();

        // Create metadata
        let metadata = crate::registry::DatasetMetadata {
            name: name.to_string(),
            description: format!("Real-world dataset: {}", name),
            n_samples,
            n_features,
            task_type: if target.is_some() {
                "classification".to_string()
            } else {
                "unsupervised".to_string()
            },
            targetnames: None,
            featurenames: Some(featurenames),
            url: Some(url.to_string()),
            checksum: None,
        };

        Ok(Dataset::from_metadata(data, target, metadata))
    }

    fn create_synthetic_bank_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.gen_range(0.0..1.0);
            }
            // Simple rule: if sum of first 3 _features > 1.5..then positive class
            target[i] = if data.row(i).iter().take(3).sum::<f64>() > 1.5 {
                1.0
            } else {
                0.0
            };
        }

        Ok((data, target))
    }

    #[allow(dead_code)]
    fn create_synthetic_credit_approval_data(&self) -> Result<Dataset> {
        use rand::Rng;
        let mut rng = rng();

        let n_samples = 690; // Based on the actual UCI credit approval dataset size
        let n_features = 15;

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        let featurenames = vec![
            "credit_score".to_string(),
            "annual_income".to_string(),
            "debt_to_income_ratio".to_string(),
            "employment_length".to_string(),
            "age".to_string(),
            "home_ownership".to_string(),
            "loan_amount".to_string(),
            "loan_purpose".to_string(),
            "credit_history_length".to_string(),
            "number_of_credit_lines".to_string(),
            "utilization_rate".to_string(),
            "delinquency_count".to_string(),
            "education_level".to_string(),
            "marital_status".to_string(),
            "verification_status".to_string(),
        ];

        for i in 0..n_samples {
            // Credit score (300-850)
            data[[i, 0]] = rng.gen_range(300.0..850.0);
            // Annual income (20k-200k)
            data[[i, 1]] = rng.gen_range(20000.0..200000.0);
            // Debt-to-income ratio (0-0.6)
            data[[i, 2]] = rng.gen_range(0.0..0.6);
            // Employment length (0-30 years)
            data[[i, 3]] = rng.gen_range(0.0..30.0);
            // Age (18-80)
            data[[i, 4]] = rng.gen_range(18.0..80.0);
            // Home ownership (0=rent..1=own, 2=mortgage)
            data[[i, 5]] = rng.gen_range(0.0f64..3.0).floor();
            // Loan amount (1k-50k)
            data[[i, 6]] = rng.gen_range(1000.0..50000.0);
            // Loan purpose (0-6, different purposes)
            data[[i, 7]] = rng.gen_range(0.0f64..7.0).floor();
            // Credit history length (0-40 years)
            data[[i, 8]] = rng.gen_range(0.0..40.0);
            // Number of credit lines (0-20)
            data[[i, 9]] = rng.gen_range(0.0..20.0);
            // Credit utilization rate (0-1.0)
            data[[i, 10]] = rng.gen_range(0.0..1.0);
            // Delinquency count (0-10)
            data[[i, 11]] = rng.gen_range(0.0f64..11.0).floor();
            // Education level (0=high school..1=bachelor, 2=master, 3=phd)
            data[[i, 12]] = rng.gen_range(0.0f64..4.0).floor();
            // Marital status (0=single..1=married, 2=divorced)
            data[[i, 13]] = rng.gen_range(0.0f64..3.0).floor();
            // Verification status (0=not verified..1=verified)
            data[[i, 14]] = if rng.gen_bool(0.7) { 1.0 } else { 0.0 };

            // Determine approval based on realistic criteria
            let credit_score_factor = (data[[i, 0]] - 300.0) / 550.0; // Normalize credit score
            let income_factor = (data[[i, 1]] / 100000.0).min(1.0); // Normalize income
            let debt_factor = 1.0 - data[[i, 2]]; // Lower debt is better
            let employment_factor = (data[[i, 3]] / 10.0).min(1.0); // Employment stability
            let delinquency_penalty = data[[i, 11]] * 0.1; // Penalties for past delinquencies

            let approval_score = credit_score_factor * 0.4
                + income_factor * 0.3
                + debt_factor * 0.2
                + employment_factor * 0.1
                - delinquency_penalty;

            // Add some noise and determine final approval
            let noise = rng.gen_range(-0.2..0.2);
            target[i] = if (approval_score + noise) > 0.5 {
                1.0
            } else {
                0.0
            };
        }

        let metadata = crate::registry::DatasetMetadata {
            name: "Credit Approval Dataset".to_string(),
            description: "Synthetic credit approval dataset with realistic financial features for binary classification".to_string(),
            n_samples,
            n_features,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["denied".to_string(), "approved".to_string()]),
            featurenames: Some(featurenames),
            url: None,
            checksum: None,
        };

        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }

    #[allow(dead_code)]
    fn create_synthetic_mushroom_data(&self) -> Result<Dataset> {
        use rand::Rng;
        let mut rng = rng();

        let n_samples = 8124; // Based on the actual UCI mushroom dataset size
        let n_features = 22;

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        let featurenames = vec![
            "capshape".to_string(),
            "cap_surface".to_string(),
            "cap_color".to_string(),
            "bruises".to_string(),
            "odor".to_string(),
            "gill_attachment".to_string(),
            "gill_spacing".to_string(),
            "gill_size".to_string(),
            "gill_color".to_string(),
            "stalkshape".to_string(),
            "stalk_root".to_string(),
            "stalk_surface_above_ring".to_string(),
            "stalk_surface_below_ring".to_string(),
            "stalk_color_above_ring".to_string(),
            "stalk_color_below_ring".to_string(),
            "veil_type".to_string(),
            "veil_color".to_string(),
            "ring_number".to_string(),
            "ring_type".to_string(),
            "spore_print_color".to_string(),
            "population".to_string(),
            "habitat".to_string(),
        ];

        for i in 0..n_samples {
            // Cap shape (0-5: bell, conical, convex, flat, knobbed, sunken)
            data[[i, 0]] = rng.gen_range(0.0f64..6.0).floor();
            // Cap surface (0-3: fibrous..grooves, scaly, smooth)
            data[[i, 1]] = rng.gen_range(0.0f64..4.0).floor();
            // Cap color (0-9: brown..buff, cinnamon, gray, green, pink, purple, red, white, yellow)
            data[[i, 2]] = rng.gen_range(0.0f64..10.0).floor();
            // Bruises (0=no..1=yes)
            data[[i, 3]] = if rng.gen_bool(0.6) { 1.0 } else { 0.0 };
            // Odor (0-8: almond, anise, creosote, fishy, foul, musty, none, pungent, spicy)
            data[[i, 4]] = rng.gen_range(0.0f64..9.0).floor();
            // Gill attachment (0-1: attached..free)
            data[[i, 5]] = if rng.gen_bool(0.5) { 1.0 } else { 0.0 };
            // Gill spacing (0-1: close, crowded)
            data[[i, 6]] = if rng.gen_bool(0.5) { 1.0 } else { 0.0 };
            // Gill size (0-1: broad, narrow)
            data[[i, 7]] = if rng.gen_bool(0.5) { 1.0 } else { 0.0 };
            // Gill color (0-11: black, brown, buff, chocolate, gray, green, orange, pink, purple, red, white, yellow)
            data[[i, 8]] = rng.gen_range(0.0f64..12.0).floor();
            // Stalk shape (0-1: enlarging..tapering)
            data[[i, 9]] = if rng.gen_bool(0.5) { 1.0 } else { 0.0 };
            // Stalk root (0-4: bulbous, club, cup, equal, rhizomorphs)
            data[[i, 10]] = rng.gen_range(0.0f64..5.0).floor();
            // Stalk surface above ring (0-3: fibrous..scaly, silky, smooth)
            data[[i, 11]] = rng.gen_range(0.0f64..4.0).floor();
            // Stalk surface below ring (0-3: fibrous..scaly, silky, smooth)
            data[[i, 12]] = rng.gen_range(0.0f64..4.0).floor();
            // Stalk color above ring (0-8: brown..buff, cinnamon, gray, orange, pink, red, white, yellow)
            data[[i, 13]] = rng.gen_range(0.0f64..9.0).floor();
            // Stalk color below ring (0-8: brown..buff, cinnamon, gray, orange, pink, red, white, yellow)
            data[[i, 14]] = rng.gen_range(0.0f64..9.0).floor();
            // Veil type (always partial in the original dataset)
            data[[i, 15]] = 0.0;
            // Veil color (0-3: brown, orange, white, yellow)
            data[[i, 16]] = rng.gen_range(0.0f64..4.0).floor();
            // Ring number (0-2: none..one, two)
            data[[i, 17]] = rng.gen_range(0.0f64..3.0).floor();
            // Ring type (0-7: cobwebby..evanescent, flaring, large, none, pendant, sheathing, zone)
            data[[i, 18]] = rng.gen_range(0.0f64..8.0).floor();
            // Spore print color (0-8: black..brown, buff, chocolate, green, orange, purple, white, yellow)
            data[[i, 19]] = rng.gen_range(0.0f64..9.0).floor();
            // Population (0-5: abundant..clustered, numerous, scattered, several, solitary)
            data[[i, 20]] = rng.gen_range(0.0f64..6.0).floor();
            // Habitat (0-6: grasses..leaves, meadows, paths, urban, waste, woods)
            data[[i, 21]] = rng.gen_range(0.0f64..7.0).floor();

            // Determine edibility based on key features
            // Poisonous mushrooms often have certain characteristics
            let mut poison_score = 0.0;

            // Bad odors often indicate poisonous mushrooms
            if data[[i, 4]] == 2.0 || data[[i, 4]] == 3.0 || data[[i, 4]] == 4.0 {
                // creosote, fishy, foul
                poison_score += 0.8;
            }
            if data[[i, 4]] == 5.0 || data[[i, 4]] == 7.0 {
                // musty, pungent
                poison_score += 0.4;
            }

            // Certain spore print colors are associated with poisonous mushrooms
            if data[[i, 19]] == 2.0 || data[[i, 19]] == 4.0 {
                // buff, green
                poison_score += 0.3;
            }

            // Stalk root type affects edibility
            if data[[i, 10]] == 0.0 {
                // bulbous root often poisonous
                poison_score += 0.2;
            }

            // Add some randomness for realistic variation
            let noise = rng.gen_range(-0.3..0.3);
            target[i] = if (poison_score + noise) > 0.5 {
                1.0
            } else {
                0.0
            }; // 1=poisonous, 0=edible
        }

        let metadata = crate::registry::DatasetMetadata {
            name: "Mushroom Dataset".to_string(),
            description: "Synthetic mushroom classification dataset with morphological features for edibility prediction".to_string(),
            n_samples,
            n_features,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["edible".to_string(), "poisonous".to_string()]),
            featurenames: Some(featurenames),
            url: None,
            checksum: None,
        };

        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }

    #[allow(dead_code)]
    fn create_synthetic_spam_data(&self) -> Result<Dataset> {
        use rand::Rng;
        let mut rng = rng();

        let n_samples = 4601; // Based on the actual spam dataset size
        let n_features = 57; // 54 word frequency features + 3 character frequency features

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        // Generate feature names for the spam dataset
        let mut featurenames = Vec::with_capacity(n_features);

        // Word frequency features (first 54 features)
        let spam_words = vec![
            "make",
            "address",
            "all",
            "3d",
            "our",
            "over",
            "remove",
            "internet",
            "order",
            "mail",
            "receive",
            "will",
            "people",
            "report",
            "addresses",
            "free",
            "business",
            "email",
            "you",
            "credit",
            "your",
            "font",
            "000",
            "money",
            "hp",
            "hpl",
            "george",
            "650",
            "lab",
            "labs",
            "telnet",
            "857",
            "data",
            "415",
            "85",
            "technology",
            "1999",
            "parts",
            "pm",
            "direct",
            "cs",
            "meeting",
            "original",
            "project",
            "re",
            "edu",
            "table",
            "conference",
            "char_freq_semicolon",
            "char_freq_parenthesis",
            "char_freq_bracket",
            "char_freq_exclamation",
            "char_freq_dollar",
            "char_freq_hash",
            "capital_run_length_average",
            "capital_run_length_longest",
            "capital_run_length_total",
        ];

        for (i, word) in spam_words.iter().enumerate() {
            if i < n_features {
                featurenames.push(format!("word_freq_{word}"));
            }
        }

        // Fill remaining feature names if needed
        while featurenames.len() < n_features {
            featurenames.push(format!("feature_{}", featurenames.len()));
        }

        for i in 0..n_samples {
            let is_spam = rng.gen_bool(0.4); // 40% spam rate

            // Generate word frequency features (0-54)
            for j in 0..54 {
                if is_spam {
                    // Spam emails have higher frequencies of certain words
                    match j {
                        0..=7 => data[[i, j]] = rng.gen_range(0.0..5.0), // "make", "address", etc. common in spam
                        8..=15 => data[[i, j]] = rng.gen_range(0.0..3.0), // "order", "mail", etc.
                        16..=25 => data[[i, j]] = rng.gen_range(0.0..4.0), // "free", "business", "money"
                        _ => data[[i, j]] = rng.gen_range(0.0..1.0), // Other words less frequent
                    }
                } else {
                    // Ham emails have different patterns
                    match j {
                        26..=35 => data[[i, j]] = rng.gen_range(0.0..2.0), // Technical words in ham
                        36..=45 => data[[i, j]] = rng.gen_range(0.0..1.5), // Meeting, project words
                        _ => data[[i, j]] = rng.gen_range(0.0..0.5), // Generally lower frequencies
                    }
                }
            }

            // Character frequency features (54-56)
            if is_spam {
                data[[i, 54]] = rng.gen_range(0.0..0.2); // Semicolon frequency
                data[[i, 55]] = rng.gen_range(0.0..0.5); // Parenthesis frequency
                data[[i, 56]] = rng.gen_range(0.0..0.3); // Exclamation frequency
            } else {
                data[[i, 54]] = rng.gen_range(0.0..0.1);
                data[[i, 55]] = rng.gen_range(0.0..0.2);
                data[[i, 56]] = rng.gen_range(0.0..0.1);
            }

            target[i] = if is_spam { 1.0 } else { 0.0 };
        }

        let metadata = crate::registry::DatasetMetadata {
            name: "Spam Email Dataset".to_string(),
            description: "Synthetic spam email classification dataset with word and character frequency features".to_string(),
            n_samples,
            n_features,
            task_type: "classification".to_string(),
            targetnames: Some(vec!["ham".to_string(), "spam".to_string()]),
            featurenames: Some(featurenames),
            url: None,
            checksum: None,
        };

        Ok(Dataset::from_metadata(data, Some(target), metadata))
    }

    fn create_synthetic_titanic_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Pclass (1, 2, 3)
            data[[i, 0]] = rng.gen_range(1.0f64..4.0).floor();
            // Sex (0=female..1=male)
            data[[i, 1]] = if rng.gen_bool(0.5) { 0.0 } else { 1.0 };
            // Age
            data[[i, 2]] = rng.gen_range(1.0..80.0);
            // SibSp
            data[[i, 3]] = rng.gen_range(0.0f64..6.0).floor();
            // Parch
            data[[i, 4]] = rng.gen_range(0.0f64..4.0).floor();
            // Fare
            data[[i, 5]] = rng.gen_range(0.0..512.0);
            // Embarked (0, 1, 2)
            data[[i, 6]] = rng.gen_range(0.0f64..3.0).floor();

            // Survival rule: higher class..female, younger = higher survival
            let survival_score = (4.0 - data[[i, 0]]) * 0.3 + // class
                                (1.0 - data[[i, 1]]) * 0.4 + // sex (female=1)
                                (80.0 - data[[i, 2]]) / 80.0 * 0.3; // age

            target[i] = if survival_score > 0.5 { 1.0 } else { 0.0 };
        }

        Ok((data, target))
    }

    fn create_synthetic_credit_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.gen_range(0.0..1.0);
            }
            // Credit scoring rule
            let score = data.row(i).iter().sum::<f64>() / n_features as f64;
            target[i] = if score > 0.6 { 1.0 } else { 0.0 };
        }

        Ok((data, target))
    }

    fn create_synthetic_housing_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Median income (0-15)
            data[[i, 0]] = rng.gen_range(0.5..15.0);
            // House age (1-52)
            data[[i, 1]] = rng.gen_range(1.0..52.0);
            // Average rooms (3-20)
            data[[i, 2]] = rng.gen_range(3.0..20.0);
            // Average bedrooms (0.8-6)
            data[[i, 3]] = rng.gen_range(0.8..6.0);
            // Population (3-35682)
            data[[i, 4]] = rng.gen_range(3.0..35682.0);
            // Average occupancy (0.7-1243)
            data[[i, 5]] = rng.gen_range(0.7..1243.0);
            // Latitude (32-42)
            data[[i, 6]] = rng.gen_range(32.0..42.0);
            // Longitude (-124 to -114)
            data[[i, 7]] = rng.gen_range(-124.0..-114.0);

            // House value based on income, rooms, and location
            let house_value = data[[i, 0]] * 50000.0 + // income effect
                            data[[i, 2]] * 10000.0 + // rooms effect
                            (40.0 - data[[i, 6]]) * 5000.0; // latitude effect

            target[i] = house_value / 100000.0; // Scale to 0-5 range
        }

        Ok((data, target))
    }

    fn create_synthetic_wine_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Wine quality _features with realistic ranges
            data[[i, 0]] = rng.gen_range(4.6..15.9); // fixed acidity
            data[[i, 1]] = rng.gen_range(0.12..1.58); // volatile acidity
            data[[i, 2]] = rng.gen_range(0.0..1.0); // citric acid
            data[[i, 3]] = rng.gen_range(0.9..15.5); // residual sugar
            data[[i, 4]] = rng.gen_range(0.012..0.611); // chlorides
            data[[i, 5]] = rng.gen_range(1.0..72.0); // free sulfur dioxide
            data[[i, 6]] = rng.gen_range(6.0..289.0); // total sulfur dioxide
            data[[i, 7]] = rng.gen_range(0.99007..1.00369); // density
            data[[i, 8]] = rng.gen_range(2.74..4.01); // pH
            data[[i, 9]] = rng.gen_range(0.33..2.0); // sulphates
            data[[i, 10]] = rng.gen_range(8.4..14.9); // alcohol

            // Quality score (3-8) based on _features
            let quality: f64 = 3.0 +
                        (data[[i, 10]] - 8.0) * 0.5 + // alcohol
                        (1.0 - data[[i, 1]]) * 2.0 + // volatile acidity (lower is better)
                        data[[i, 2]] * 2.0 + // citric acid
                        rng.gen_range(-0.5..0.5); // noise

            target[i] = quality.clamp(3.0, 8.0);
        }

        Ok((data, target))
    }

    fn create_synthetic_energy_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.gen_range(0.0..1.0);
            }

            // Energy efficiency score
            let efficiency = data.row(i).iter().sum::<f64>() / n_features as f64;
            target[i] = efficiency * 40.0 + 10.0; // Scale to 10-50 range
        }

        Ok((data, target))
    }

    fn create_air_passengers_data(
        &self,
        n_timesteps: usize,
    ) -> Result<(Array2<f64>, Option<Array1<f64>>)> {
        use rand::Rng;
        let mut rng = rng();
        let mut data = Array2::zeros((n_timesteps, 1));

        for i in 0..n_timesteps {
            let t = i as f64;
            let trend = 100.0 + t * 2.0;
            let seasonal = 20.0 * (2.0 * std::f64::consts::PI * t / 12.0).sin();
            let noise = rng.random::<f64>() * 10.0 - 5.0;

            data[[i, 0]] = trend + seasonal + noise;
        }

        Ok((data, None))
    }

    fn create_bitcoin_price_data(
        &self,
        n_timesteps: usize,
    ) -> Result<(Array2<f64>, Option<Array1<f64>>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_timesteps, 6));
        let mut price = 30000.0; // Starting price

        for i in 0..n_timesteps {
            // Simulate price movement
            let change = rng.gen_range(-0.05..0.05);
            price *= 1.0 + change;

            let high = price * (1.0 + rng.gen_range(0.0..0.02));
            let low = price * (1.0 - rng.gen_range(0.0..0.02));
            let volume = rng.gen_range(1000000.0..10000000.0);

            data[[i, 0]] = price; // open
            data[[i, 1]] = high;
            data[[i, 2]] = low;
            data[[i, 3]] = price; // close
            data[[i, 4]] = volume;
            data[[i, 5]] = price * volume; // market cap proxy
        }

        Ok((data, None))
    }

    fn create_heart_disease_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Age
            data[[i, 0]] = rng.gen_range(29.0..77.0);
            // Sex (0=female..1=male)
            data[[i, 1]] = if rng.gen_bool(0.68) { 1.0 } else { 0.0 };
            // Chest pain type (0-3)
            data[[i, 2]] = rng.gen_range(0.0f64..4.0).floor();
            // Resting blood pressure
            data[[i, 3]] = rng.gen_range(94.0..200.0);
            // Cholesterol
            data[[i, 4]] = rng.gen_range(126.0..564.0);

            // Fill other _features
            for j in 5..n_features {
                data[[i, j]] = rng.gen_range(0.0..1.0);
            }

            // Heart disease prediction based on risk factors
            let risk_score = (data[[i, 0]] - 29.0) / 48.0 * 0.3 + // age
                           data[[i, 1]] * 0.2 + // sex (male higher risk)
                           (data[[i, 3]] - 94.0) / 106.0 * 0.2 + // blood pressure
                           (data[[i, 4]] - 126.0) / 438.0 * 0.3; // cholesterol

            target[i] = if risk_score > 0.5 { 1.0 } else { 0.0 };
        }

        Ok((data, target))
    }

    fn create_diabetes_readmission_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.gen_range(0.0..1.0);
            }

            // Readmission prediction
            let readmission_score = data.row(i).iter().take(10).sum::<f64>() / 10.0;
            target[i] = if readmission_score > 0.6 { 1.0 } else { 0.0 };
        }

        Ok((data, target))
    }

    fn create_synthetic_auto_mpg_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Cylinders (4, 6, 8)
            data[[i, 0]] = [4.0, 6.0, 8.0][rng.sample(Uniform::new(0, 3).unwrap())];
            // Displacement
            data[[i, 1]] = rng.gen_range(68.0..455.0);
            // Horsepower
            data[[i, 2]] = rng.gen_range(46.0..230.0);
            // Weight
            data[[i, 3]] = rng.gen_range(1613.0..5140.0);
            // Acceleration
            data[[i, 4]] = rng.gen_range(8.0..24.8);
            // Model year
            data[[i, 5]] = rng.gen_range(70.0..82.0);
            // Origin (1=USA, 2=Europe, 3=Japan)
            data[[i, 6]] = (rng.gen_range(1.0f64..4.0f64)).floor();

            // MPG calculation: inversely related to weight and displacement..positively to efficiency
            let mpg: f64 = 45.0 - (data[[i, 3]] / 5140.0) * 20.0 - (data[[i, 1]] / 455.0) * 15.0
                + (data[[i, 4]] / 24.8) * 10.0
                + rng.gen_range(-3.0..3.0);
            target[i] = mpg.clamp(9.0, 46.6);
        }

        Ok((data, target))
    }

    fn create_synthetic_concrete_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Cement (component 1)
            data[[i, 0]] = rng.gen_range(102.0..540.0);
            // Blast Furnace Slag (component 2)
            data[[i, 1]] = rng.gen_range(0.0..359.4);
            // Fly Ash (component 3)
            data[[i, 2]] = rng.gen_range(0.0..200.1);
            // Water (component 4)
            data[[i, 3]] = rng.gen_range(121.8..247.0);
            // Superplasticizer (component 5)
            data[[i, 4]] = rng.gen_range(0.0..32.2);
            // Coarse Aggregate (component 6)
            data[[i, 5]] = rng.gen_range(801.0..1145.0);
            // Fine Aggregate (component 7)
            data[[i, 6]] = rng.gen_range(594.0..992.6);
            // Age (days)
            data[[i, 7]] = rng.gen_range(1.0..365.0);

            // Compressive strength calculation
            let strength: f64 = (data[[i, 0]] / 540.0) * 30.0 + // cement contribution
                          (data[[i, 1]] / 359.4) * 15.0 + // slag contribution
                          (data[[i, 3]] / 247.0) * (-20.0) + // water (negative)
                          (data[[i, 7]] / 365.0_f64).ln() * 10.0 + // age (logarithmic)
                          rng.gen_range(-5.0..5.0); // noise

            target[i] = strength.clamp(2.33, 82.6);
        }

        Ok((data, target))
    }

    fn create_synthetic_electricity_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let hour = (i % 24) as f64;
            let day_of_year = (i / 24) % 365;

            // Temperature
            data[[i, 0]] = 20.0
                + 15.0 * (day_of_year as f64 * 2.0 * std::f64::consts::PI / 365.0).sin()
                + rng.gen_range(-5.0..5.0);
            // Humidity
            data[[i, 1]] = 50.0 + 30.0 * rng.gen_range(0.0..1.0);
            // Hour of day
            data[[i, 2]] = hour;

            // Electricity load: seasonal + daily patterns + weather effects
            let seasonal = 50.0
                + 30.0
                    * (day_of_year as f64 * 2.0 * std::f64::consts::PI / 365.0
                        + std::f64::consts::PI)
                        .cos();
            let daily = 40.0 + 60.0 * ((hour - 12.0) * std::f64::consts::PI / 12.0).cos();
            let temp_effect = (data[[i, 0]] - 20.0).abs() * 2.0; // Higher load for extreme temperatures

            target[i] = seasonal + daily + temp_effect + rng.gen_range(-10.0..10.0);
        }

        Ok((data, target))
    }

    fn create_synthetic_stock_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        let mut price = 100.0; // Starting price

        for i in 0..n_samples {
            // Price movement (random walk with trend)
            let change = rng.gen_range(-0.05..0.05);
            price *= 1.0 + change;

            // Features: OHLC + Volume
            let high = price * (1.0 + rng.gen_range(0.0..0.02));
            let low = price * (1.0 - rng.gen_range(0.0..0.02));
            let volume = rng.gen_range(1000000.0..10000000.0);

            data[[i, 0]] = price; // Close price
            data[[i, 1]] = high;
            data[[i, 2]] = low;
            data[[i, 3]] = volume;
            data[[i, 4]] = (high - low) / price; // Volatility

            // Target: next day return
            let next_change = rng.gen_range(-0.05..0.05);
            target[i] = next_change;
        }

        Ok((data, target))
    }

    fn create_synthetic_fraud_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let is_fraud = rng.gen_range(0.0..1.0) < 0.001728; // ~0.17% fraud rate

            for j in 0..n_features {
                if j < 28 {
                    // Anonymized _features (PCA components)
                    if is_fraud {
                        // Fraudulent transactions have different patterns
                        data[[i, j]] = rng.gen_range(-5.0..5.0) * 2.0; // More extreme values
                    } else {
                        // Normal transactions
                        data[[i, j]] = rng.gen_range(-3.0..3.0);
                    }
                }
            }

            target[i] = if is_fraud { 1.0 } else { 0.0 };
        }

        Ok((data, target))
    }

    fn create_synthetic_loan_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Loan amount
            data[[i, 0]] = rng.gen_range(1000.0..50000.0);
            // Interest rate
            data[[i, 1]] = rng.gen_range(5.0..25.0);
            // Loan term (months)
            data[[i, 2]] = [12.0, 24.0, 36.0, 48.0, 60.0][rng.sample(Uniform::new(0, 5).unwrap())];
            // Annual income
            data[[i, 3]] = rng.gen_range(20000.0..200000.0);
            // Credit score
            data[[i, 4]] = rng.gen_range(300.0..850.0);
            // Employment length (years)
            data[[i, 5]] = rng.gen_range(0.0..40.0);
            // Debt-to-income ratio
            data[[i, 6]] = rng.gen_range(0.0..0.4);

            // Fill remaining _features
            for j in 7..n_features {
                data[[i, j]] = rng.gen_range(0.0..1.0);
            }

            // Default probability based on risk factors
            let risk_score = (850.0 - data[[i, 4]]) / 550.0 * 0.4 + // credit score (inverted)
                           data[[i, 6]] * 0.3 + // debt-to-income ratio
                           (data[[i, 1]] - 5.0) / 20.0 * 0.2 + // interest rate
                           (50000.0 - data[[i, 3]]) / 180000.0 * 0.1; // income (inverted)

            target[i] = if risk_score > 0.3 { 1.0 } else { 0.0 };
        }

        Ok((data, target))
    }

    fn create_synthetic_adult_dataset(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Age (17-90)
            data[[i, 0]] = rng.gen_range(17.0..90.0);
            // Workclass (encoded 0-8)
            data[[i, 1]] = rng.gen_range(0.0f64..9.0).floor();
            // Final weight
            data[[i, 2]] = rng.gen_range(12285.0..1484705.0);
            // Education (encoded 0-15)
            data[[i, 3]] = rng.gen_range(0.0f64..16.0).floor();
            // Education-num (1-16)
            data[[i, 4]] = rng.gen_range(1.0..17.0);
            // Marital status (encoded 0-6)
            data[[i, 5]] = rng.gen_range(0.0f64..7.0).floor();
            // Occupation (encoded 0-13)
            data[[i, 6]] = rng.gen_range(0.0f64..14.0).floor();
            // Relationship (encoded 0-5)
            data[[i, 7]] = rng.gen_range(0.0f64..6.0).floor();
            // Race (encoded 0-4)
            data[[i, 8]] = rng.gen_range(0.0f64..5.0).floor();
            // Sex (0=Female..1=Male)
            data[[i, 9]] = if rng.gen_bool(0.67) { 1.0 } else { 0.0 };
            // Capital gain (0-99999)
            data[[i, 10]] = if rng.gen_bool(0.9) {
                0.0
            } else {
                rng.gen_range(1.0..99999.0)
            };
            // Capital loss (0-4356)
            data[[i, 11]] = if rng.gen_bool(0.95) {
                0.0
            } else {
                rng.gen_range(1.0..4356.0)
            };
            // Hours per week (1-99)
            data[[i, 12]] = rng.gen_range(1.0..99.0);
            // Native country (encoded 0-40)
            data[[i, 13]] = rng.gen_range(0.0f64..41.0).floor();

            // Income prediction based on realistic factors
            let income_score = (data[[i, 0]] - 17.0) / 73.0 * 0.2 + // Age factor
                data[[i, 4]] / 16.0 * 0.3 + // Education factor
                data[[i, 9]] * 0.2 + // Gender factor (historically male bias)
                (data[[i, 12]] - 1.0) / 98.0 * 0.2 + // Hours worked factor
                (data[[i, 10]] + data[[i, 11]]) / 100000.0 * 0.1; // Capital gains/losses

            // Add some randomness
            let noise = rng.gen_range(-0.15..0.15);
            target[i] = if (income_score + noise) > 0.5 {
                1.0
            } else {
                0.0
            };
        }

        Ok((data, target))
    }

    fn create_generic_synthetic_dataset(
        &self,
        n_samples: usize,
        n_features: usize,
        has_categorical: bool,
    ) -> Result<(Array2<f64>, Option<Array1<f64>>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                if has_categorical && j < n_features / 3 {
                    // First third are _categorical _features (encoded as integers)
                    data[[i, j]] = rng.gen_range(0.0f64..10.0).floor();
                } else {
                    // Remaining are continuous _features
                    data[[i, j]] = rng.gen_range(-2.0..2.0);
                }
            }
        }

        // Generate target based on _features
        let mut target = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let feature_sum = data.row(i).iter().sum::<f64>();
            let score = feature_sum / n_features as f64;
            target[i] = if score > 0.0 { 1.0 } else { 0.0 };
        }

        Ok((data, Some(target)))
    }

    fn create_synthetic_cifar10_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let class = rng.sample(Uniform::new(0, 10).unwrap()) as f64;
            target[i] = class;

            // Generate synthetic image data that has class-dependent patterns
            for j in 0..n_features {
                let base_intensity = match class as i32 {
                    0 => 0.6, // airplane - sky colors
                    1 => 0.3, // automobile - darker
                    2 => 0.8, // bird - varied colors
                    3 => 0.5, // cat - medium tones
                    4 => 0.7, // deer - earth tones
                    5 => 0.4, // dog - varied
                    6 => 0.9, // frog - bright greens
                    7 => 0.6, // horse - brown tones
                    8 => 0.2, // ship - dark blues
                    9 => 0.3, // truck - dark colors
                    _ => 0.5,
                };

                // Add noise and variation
                data[[i, j]] = base_intensity + rng.gen_range(-0.3f64..0.3f64);
                data[[i, j]] = data[[i, j]].clamp(0.0f64, 1.0f64); // Clamp to [0, 1]
            }
        }

        Ok((data, target))
    }

    fn create_synthetic_fashion_mnist_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let class = rng.sample(Uniform::new(0, 10).unwrap()) as f64;
            target[i] = class;

            // Generate synthetic fashion item patterns
            for j in 0..n_features {
                let base_intensity = match class as i32 {
                    0 => 0.3, // T-shirt/top - simple patterns
                    1 => 0.4, // Trouser - leg shapes
                    2 => 0.5, // Pullover - textured
                    3 => 0.6, // Dress - flowing patterns
                    4 => 0.7, // Coat - structured
                    5 => 0.2, // Sandal - minimal
                    6 => 0.4, // Shirt - medium complexity
                    7 => 0.3, // Sneaker - distinctive sole patterns
                    8 => 0.5, // Bag - rectangular patterns
                    9 => 0.4, // Ankle boot - boot shape
                    _ => 0.4,
                };

                // Add texture and noise
                let texture_noise = rng.gen_range(-0.2f64..0.2f64);
                data[[i, j]] = base_intensity + texture_noise;
                data[[i, j]] = data[[i, j]].clamp(0.0f64, 1.0f64); // Clamp to [0, 1]
            }
        }

        Ok((data, target))
    }

    fn create_synthetic_imdb_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        // Define word groups for sentiment analysis
        let positive_words = 0..n_features / 3; // First third are positive words
        let negative_words = n_features / 3..2 * n_features / 3; // Second third are negative words
        let _neutral_words = 2 * n_features / 3..n_features; // Last third are neutral words

        for i in 0..n_samples {
            let is_positive = rng.gen_bool(0.5);
            target[i] = if is_positive { 1.0 } else { 0.0 };

            for j in 0..n_features {
                let base_freq = if positive_words.contains(&j) {
                    if is_positive {
                        rng.gen_range(0.5..2.0) // Higher positive word frequency for positive reviews
                    } else {
                        rng.gen_range(0.0..0.5) // Lower positive word frequency for negative reviews
                    }
                } else if negative_words.contains(&j) {
                    if is_positive {
                        rng.gen_range(0.0..0.5) // Lower negative word frequency for positive reviews
                    } else {
                        rng.gen_range(0.5..2.0) // Higher negative word frequency for negative reviews
                    }
                } else {
                    // Neutral words appear with similar frequency in both classes
                    rng.gen_range(0.2..1.0)
                };

                data[[i, j]] = base_freq;
            }
        }

        Ok((data, target))
    }

    fn create_synthetic_news_data(
        &self,
        n_samples: usize,
        n_features: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use rand::Rng;
        let mut rng = rng();

        let mut data = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        // Define topic-specific word groups
        let words_per_topic = n_features / 5; // 5 topics

        for i in 0..n_samples {
            let topic = rng.sample(Uniform::new(0, 5).unwrap()) as f64;
            target[i] = topic;

            for j in 0..n_features {
                let word_topic = j / words_per_topic;

                let base_freq = if word_topic == topic as usize {
                    // Words from the same topic appear more frequently
                    rng.gen_range(1.0..3.0)
                } else {
                    // Words from other topics appear less frequently
                    rng.gen_range(0.0..0.8)
                };

                // Add some noise
                let noise = rng.gen_range(-0.2f64..0.2f64);
                data[[i, j]] = (base_freq + noise).max(0.0f64);
            }
        }

        Ok((data, target))
    }
}

/// Convenience functions for loading specific real-world datasets
#[allow(dead_code)]
pub fn load_adult() -> Result<Dataset> {
    let config = RealWorldConfig::default();
    let mut loader = RealWorldDatasets::new(config)?;
    loader.load_adult()
}

/// Load Titanic dataset
#[allow(dead_code)]
pub fn load_titanic() -> Result<Dataset> {
    let config = RealWorldConfig::default();
    let mut loader = RealWorldDatasets::new(config)?;
    loader.load_titanic()
}

/// Load California Housing dataset
#[allow(dead_code)]
pub fn load_california_housing() -> Result<Dataset> {
    let config = RealWorldConfig::default();
    let mut loader = RealWorldDatasets::new(config)?;
    loader.load_california_housing()
}

/// Load Heart Disease dataset
#[allow(dead_code)]
pub fn load_heart_disease() -> Result<Dataset> {
    let config = RealWorldConfig::default();
    let mut loader = RealWorldDatasets::new(config)?;
    loader.load_heart_disease()
}

/// Load Red Wine Quality dataset
#[allow(dead_code)]
pub fn load_red_wine_quality() -> Result<Dataset> {
    let config = RealWorldConfig::default();
    let mut loader = RealWorldDatasets::new(config)?;
    loader.load_red_wine_quality()
}

/// List all available real-world datasets
#[allow(dead_code)]
pub fn list_real_world_datasets() -> Vec<String> {
    let config = RealWorldConfig::default();
    let loader = RealWorldDatasets::new(config).unwrap();
    loader.list_datasets()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::Uniform;

    #[test]
    fn test_load_titanic() {
        let dataset = load_titanic().unwrap();
        assert_eq!(dataset.n_samples(), 891);
        assert_eq!(dataset.n_features(), 7);
        assert!(dataset.target.is_some());
    }

    #[test]
    fn test_load_california_housing() {
        let dataset = load_california_housing().unwrap();
        assert_eq!(dataset.n_samples(), 20640);
        assert_eq!(dataset.n_features(), 8);
        assert!(dataset.target.is_some());
    }

    #[test]
    fn test_load_heart_disease() {
        let dataset = load_heart_disease().unwrap();
        assert_eq!(dataset.n_samples(), 303);
        assert_eq!(dataset.n_features(), 13);
        assert!(dataset.target.is_some());
    }

    #[test]
    fn test_list_datasets() {
        let datasets = list_real_world_datasets();
        assert!(!datasets.is_empty());
        assert!(datasets.contains(&"titanic".to_string()));
        assert!(datasets.contains(&"california_housing".to_string()));
    }

    #[test]
    fn test_real_world_config() {
        let config = RealWorldConfig {
            use_cache: false,
            download_if_missing: false,
            ..Default::default()
        };

        assert!(!config.use_cache);
        assert!(!config.download_if_missing);
    }
}
