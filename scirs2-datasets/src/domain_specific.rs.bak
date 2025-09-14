//! Domain-specific datasets for scientific research
//!
//! This module provides specialized datasets for various scientific domains:
//! - Astronomy and astrophysics
//! - Genomics and bioinformatics
//! - Climate science and meteorology
//! - Materials science
//! - Finance and economics
//! - Computer vision and image processing
//! - Natural language processing

use std::collections::HashMap;

use ndarray::{Array1, Array2};
use rand::{rng, Rng};
use rand_distr::Uniform;
use serde::{Deserialize, Serialize};

use crate::cache::DatasetCache;
use crate::error::{DatasetsError, Result};
use crate::external::ExternalClient;
use crate::utils::Dataset;

/// Configuration for domain-specific dataset loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainConfig {
    /// Base URL for dataset repository
    pub base_url: Option<String>,
    /// API key for authenticated access
    pub api_key: Option<String>,
    /// Data format preferences
    pub preferred_formats: Vec<String>,
    /// Quality filters
    pub quality_filters: QualityFilters,
}

impl Default for DomainConfig {
    fn default() -> Self {
        Self {
            base_url: None,
            api_key: None,
            preferred_formats: vec!["csv".to_string(), "fits".to_string(), "hdf5".to_string()],
            quality_filters: QualityFilters::default(),
        }
    }
}

/// Quality filters for dataset selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityFilters {
    /// Minimum number of samples
    pub min_samples: Option<usize>,
    /// Maximum missing data percentage
    pub max_missing_percent: Option<f64>,
    /// Required data completeness
    pub min_completeness: Option<f64>,
    /// Minimum publication year
    pub min_year: Option<u32>,
}

impl Default for QualityFilters {
    fn default() -> Self {
        Self {
            min_samples: Some(100),
            max_missing_percent: Some(0.1),
            min_completeness: Some(0.9),
            min_year: Some(2000),
        }
    }
}

/// Astronomy and astrophysics datasets
pub mod astronomy {
    use super::*;

    /// Stellar classification and properties
    pub struct StellarDatasets {
        #[allow(dead_code)]
        client: ExternalClient,
        #[allow(dead_code)]
        cache: DatasetCache,
    }

    impl StellarDatasets {
        /// Create a new stellar datasets client
        pub fn new() -> Result<Self> {
            let cachedir = dirs::cache_dir()
                .ok_or_else(|| {
                    DatasetsError::Other("Could not determine cache directory".to_string())
                })?
                .join("scirs2-datasets");
            Ok(Self {
                client: ExternalClient::new()?,
                cache: DatasetCache::new(cachedir),
            })
        }

        /// Load Hipparcos stellar catalog data
        pub fn load_hipparcos_catalog(&self) -> Result<Dataset> {
            self.load_synthetic_stellar_data("hipparcos", 118218)
        }

        /// Load Gaia DR3 stellar data (synthetic for demonstration)
        pub fn load_gaia_dr3_sample(&self) -> Result<Dataset> {
            self.load_synthetic_stellar_data("gaia_dr3", 50000)
        }

        /// Load exoplanet catalog
        pub fn load_exoplanet_catalog(&self) -> Result<Dataset> {
            self.load_synthetic_exoplanet_data(5000)
        }

        /// Load supernova photometry data
        pub fn load_supernova_photometry(&self) -> Result<Dataset> {
            self.load_synthetic_supernova_data(1000)
        }

        fn load_synthetic_stellar_data(&self, catalog: &str, nstars: usize) -> Result<Dataset> {
            use rand_distr::{Distribution, Normal};

            let mut rng = rng();

            // Generate synthetic stellar parameters
            let mut data = Vec::with_capacity(nstars * 8);
            let mut spectral_classes = Vec::with_capacity(nstars);

            // Distributions for stellar parameters
            let ra_dist = rand_distr::Uniform::new(0.0, 360.0).unwrap();
            let dec_dist = rand_distr::Uniform::new(-90.0, 90.0).unwrap();
            let magnitude_dist = Normal::new(8.0, 3.0).unwrap();
            let color_dist = Normal::new(0.5, 0.3).unwrap();
            let parallax_dist = Normal::new(10.0, 5.0).unwrap();
            let proper_motion_dist = Normal::new(0.0, 50.0).unwrap();
            let radial_velocity_dist = Normal::new(0.0, 30.0).unwrap();

            for _ in 0..nstars {
                // Right ascension (degrees)
                data.push(ra_dist.sample(&mut rng));
                // Declination (degrees)
                data.push(dec_dist.sample(&mut rng));
                // Apparent magnitude
                data.push(magnitude_dist.sample(&mut rng));
                // Color index (B-V)
                data.push(color_dist.sample(&mut rng));
                // Parallax (mas)
                data.push((parallax_dist.sample(&mut rng) as f64).max(0.1f64));
                // Proper motion RA (mas/yr)
                data.push(proper_motion_dist.sample(&mut rng));
                // Proper motion Dec (mas/yr)
                data.push(proper_motion_dist.sample(&mut rng));
                // Radial velocity (km/s)
                data.push(radial_velocity_dist.sample(&mut rng));

                // Assign spectral class based on color
                let color = data[data.len() - 5];
                let spectral_class = match color {
                    c if c < -0.3 => 0, // O
                    c if c < -0.1 => 1, // B
                    c if c < 0.2 => 2,  // A
                    c if c < 0.5 => 3,  // F
                    c if c < 0.8 => 4,  // G
                    c if c < 1.2 => 5,  // K
                    _ => 6,             // M
                };
                spectral_classes.push(spectral_class as f64);
            }

            let data_array = Array2::from_shape_vec((nstars, 8), data)
                .map_err(|e| DatasetsError::FormatError(e.to_string()))?;

            let target = Array1::from_vec(spectral_classes);

            Ok(Dataset {
                data: data_array,
                target: Some(target),
                featurenames: Some(vec![
                    "ra".to_string(),
                    "dec".to_string(),
                    "magnitude".to_string(),
                    "color_bv".to_string(),
                    "parallax".to_string(),
                    "pm_ra".to_string(),
                    "pm_dec".to_string(),
                    "radial_velocity".to_string(),
                ]),
                targetnames: Some(vec![
                    "O".to_string(),
                    "B".to_string(),
                    "A".to_string(),
                    "F".to_string(),
                    "G".to_string(),
                    "K".to_string(),
                    "M".to_string(),
                ]),
                feature_descriptions: Some(vec![
                    "Right Ascension (degrees)".to_string(),
                    "Declination (degrees)".to_string(),
                    "Apparent magnitude (visual)".to_string(),
                    "B-V color index".to_string(),
                    "Parallax (arcseconds)".to_string(),
                    "Proper motion RA (mas/year)".to_string(),
                    "Proper motion Dec (mas/year)".to_string(),
                    "Radial velocity (km/s)".to_string(),
                ]),
                description: Some(format!(
                    "Synthetic {catalog} stellar catalog with {nstars} _stars"
                )),
                metadata: std::collections::HashMap::new(),
            })
        }

        fn load_synthetic_exoplanet_data(&self, nplanets: usize) -> Result<Dataset> {
            use rand_distr::{Distribution, LogNormal, Normal};

            let mut rng = rng();

            // Generate synthetic exoplanet parameters
            let mut data = Vec::with_capacity(nplanets * 6);
            let mut planet_types = Vec::with_capacity(nplanets);

            // Distributions for planetary parameters
            let period_dist = LogNormal::new(1.0, 1.5).unwrap();
            let radius_dist = LogNormal::new(0.0, 0.8).unwrap();
            let mass_dist = LogNormal::new(1.0, 1.2).unwrap();
            let stellar_mass_dist = Normal::new(1.0, 0.3).unwrap();
            let stellar_temp_dist = Normal::new(5800.0, 1000.0).unwrap();
            let metallicity_dist = Normal::new(0.0, 0.3).unwrap();

            for _ in 0..nplanets {
                // Orbital period (days)
                data.push(period_dist.sample(&mut rng));
                // Planet radius (Earth radii)
                data.push(radius_dist.sample(&mut rng));
                // Planet mass (Earth masses)
                data.push(mass_dist.sample(&mut rng));
                // Stellar mass (Solar masses)
                data.push((stellar_mass_dist.sample(&mut rng) as f64).max(0.1f64));
                // Stellar temperature (K)
                data.push(stellar_temp_dist.sample(&mut rng));
                // Stellar metallicity [Fe/H]
                data.push(metallicity_dist.sample(&mut rng));

                // Classify planet type based on radius
                let radius = data[data.len() - 5];
                let planet_type = match radius {
                    r if r < 1.25 => 0, // Rocky
                    r if r < 2.0 => 1,  // Super-Earth
                    r if r < 4.0 => 2,  // Sub-Neptune
                    r if r < 11.0 => 3, // Neptune
                    _ => 4,             // Jupiter
                };
                planet_types.push(planet_type as f64);
            }

            let data_array = Array2::from_shape_vec((nplanets, 6), data)
                .map_err(|e| DatasetsError::FormatError(e.to_string()))?;

            let target = Array1::from_vec(planet_types);

            Ok(Dataset {
                data: data_array,
                target: Some(target),
                featurenames: Some(vec![
                    "period".to_string(),
                    "radius".to_string(),
                    "mass".to_string(),
                    "stellar_mass".to_string(),
                    "stellar_temp".to_string(),
                    "metallicity".to_string(),
                ]),
                targetnames: Some(vec![
                    "Rocky".to_string(),
                    "Super-Earth".to_string(),
                    "Sub-Neptune".to_string(),
                    "Neptune".to_string(),
                    "Jupiter".to_string(),
                ]),
                feature_descriptions: Some(vec![
                    "Orbital period (days)".to_string(),
                    "Planet radius (Earth radii)".to_string(),
                    "Planet mass (Earth masses)".to_string(),
                    "Stellar mass (Solar masses)".to_string(),
                    "Stellar temperature (K)".to_string(),
                    "Stellar metallicity [Fe/H]".to_string(),
                ]),
                description: Some(format!(
                    "Synthetic exoplanet catalog with {nplanets} _planets"
                )),
                metadata: std::collections::HashMap::new(),
            })
        }

        fn load_synthetic_supernova_data(&self, nsupernovae: usize) -> Result<Dataset> {
            use rand_distr::{Distribution, Normal};

            let mut rng = rng();

            // Generate synthetic supernova light curve features
            let mut data = Vec::with_capacity(nsupernovae * 10);
            let mut sn_types = Vec::with_capacity(nsupernovae);

            // Different supernova types have different characteristics
            let _type_probs = [0.7, 0.15, 0.10, 0.05]; // Ia, Ib/c, II-P, II-L

            for _ in 0..nsupernovae {
                let sn_type = rng.sample(Uniform::new(0, 4).unwrap());

                let (peak_mag, decline_rate, color_evolution, host_mass) = match sn_type {
                    0 => (-19.3, 1.1, 0.2, 10.5), // Type Ia
                    1 => (-18.5, 1.8, 0.5, 9.8),  // Type Ib/c
                    2 => (-16.8, 0.8, 0.3, 9.2),  // Type II-P
                    _ => (-17.5, 1.2, 0.4, 9.0),  // Type II-L
                };

                // Add noise to base parameters
                let peak_noise = Normal::new(0.0, 0.3).unwrap();
                let decline_noise = Normal::new(0.0, 0.2).unwrap();
                let color_noise = Normal::new(0.0, 0.1).unwrap();
                let host_noise = Normal::new(0.0, 0.5).unwrap();

                // Peak absolute magnitude
                data.push(peak_mag + peak_noise.sample(&mut rng));
                // Decline rate (mag/15 days)
                data.push(decline_rate + decline_noise.sample(&mut rng));
                // Color at maximum
                data.push(color_evolution + color_noise.sample(&mut rng));
                // Host galaxy mass (log M_sun)
                data.push(host_mass + host_noise.sample(&mut rng));
                // Redshift
                data.push(rng.gen_range(0.01..0.3));
                // Duration (days)
                data.push(rng.gen_range(20.0..200.0));
                // Stretch factor
                data.push(rng.gen_range(0.7..1.3));
                // Color excess E(B-V)
                data.push(rng.gen_range(0.0..0.5));
                // Discovery magnitude
                data.push(rng.gen_range(15.0..22.0));
                // Galactic latitude
                data.push(rng.gen_range(-90.0..90.0));

                sn_types.push(sn_type as f64);
            }

            let data_array = Array2::from_shape_vec((nsupernovae, 10), data)
                .map_err(|e| DatasetsError::FormatError(e.to_string()))?;

            let target = Array1::from_vec(sn_types);

            Ok(Dataset {
                data: data_array,
                target: Some(target),
                featurenames: Some(vec![
                    "peak_magnitude".to_string(),
                    "decline_rate".to_string(),
                    "color_max".to_string(),
                    "host_mass".to_string(),
                    "redshift".to_string(),
                    "duration".to_string(),
                    "stretch".to_string(),
                    "color_excess".to_string(),
                    "discovery_mag".to_string(),
                    "galactic_lat".to_string(),
                ]),
                targetnames: Some(vec![
                    "Type Ia".to_string(),
                    "Type Ib/c".to_string(),
                    "Type II-P".to_string(),
                    "Type II-L".to_string(),
                ]),
                feature_descriptions: Some(vec![
                    "Peak apparent magnitude".to_string(),
                    "Magnitude decline rate (mag/day)".to_string(),
                    "Maximum color index".to_string(),
                    "Host galaxy stellar mass (log10 M_sun)".to_string(),
                    "Cosmological redshift".to_string(),
                    "Light curve duration (days)".to_string(),
                    "Light curve stretch factor".to_string(),
                    "Host galaxy color excess E(B-V)".to_string(),
                    "Discovery magnitude".to_string(),
                    "Galactic latitude (degrees)".to_string(),
                ]),
                description: Some(format!(
                    "Synthetic supernova catalog with {nsupernovae} events"
                )),
                metadata: std::collections::HashMap::new(),
            })
        }
    }
}

/// Genomics and bioinformatics datasets
pub mod genomics {
    use super::*;

    /// Genomic sequence and expression datasets
    pub struct GenomicsDatasets {
        #[allow(dead_code)]
        client: ExternalClient,
        #[allow(dead_code)]
        cache: DatasetCache,
    }

    impl GenomicsDatasets {
        /// Create a new genomics datasets client
        pub fn new() -> Result<Self> {
            let cachedir = dirs::cache_dir()
                .ok_or_else(|| {
                    DatasetsError::Other("Could not determine cache directory".to_string())
                })?
                .join("scirs2-datasets");
            Ok(Self {
                client: ExternalClient::new()?,
                cache: DatasetCache::new(cachedir),
            })
        }

        /// Load synthetic gene expression data
        pub fn load_gene_expression(&self, n_samples: usize, ngenes: usize) -> Result<Dataset> {
            use rand_distr::{Distribution, LogNormal, Normal};

            let mut rng = rng();

            // Generate synthetic gene expression matrix
            let mut data = Vec::with_capacity(n_samples * ngenes);
            let mut phenotypes = Vec::with_capacity(n_samples);

            // Different expression patterns for different conditions
            let condition_effects = [1.0, 2.5, 0.4, 1.8, 0.7]; // Log-fold changes

            for sample_idx in 0..n_samples {
                let condition = sample_idx % condition_effects.len();
                let base_effect = condition_effects[condition];

                for gene_idx in 0..ngenes {
                    // Base expression level
                    let base_expr = LogNormal::new(5.0, 2.0).unwrap().sample(&mut rng);

                    // Condition-specific modulation
                    let gene_effect = if gene_idx < ngenes / 10 {
                        // 10% of _genes are differentially expressed
                        base_effect
                    } else {
                        1.0
                    };

                    // Add noise
                    let noise = Normal::new(1.0, 0.2).unwrap().sample(&mut rng);

                    let expression: f64 = base_expr * gene_effect * noise;
                    data.push(expression.ln()); // Log-transform
                }

                phenotypes.push(condition as f64);
            }

            let data_array = Array2::from_shape_vec((n_samples, ngenes), data)
                .map_err(|e| DatasetsError::FormatError(e.to_string()))?;

            let target = Array1::from_vec(phenotypes);

            // Generate gene names
            let featurenames: Vec<String> = (0..ngenes).map(|i| format!("GENE_{i:06}")).collect();

            Ok(Dataset {
                data: data_array,
                target: Some(target),
                featurenames: Some(featurenames.clone()),
                targetnames: Some(vec![
                    "Control".to_string(),
                    "Treatment_A".to_string(),
                    "Treatment_B".to_string(),
                    "Disease_X".to_string(),
                    "Disease_Y".to_string(),
                ]),
                feature_descriptions: Some(
                    featurenames
                        .iter()
                        .map(|name| format!("Expression level of {name}"))
                        .collect(),
                ),
                description: Some(format!(
                    "Synthetic gene expression data: {n_samples} _samples × {ngenes} _genes"
                )),
                metadata: std::collections::HashMap::new(),
            })
        }

        /// Load synthetic DNA sequence features
        pub fn load_dnasequences(
            &self,
            nsequences: usize,
            sequence_length: usize,
        ) -> Result<Dataset> {
            let mut rng = rng();
            let nucleotides = ['A', 'T', 'G', 'C'];

            let mut sequences = Vec::new();
            let mut sequence_types = Vec::with_capacity(nsequences);

            for seq_idx in 0..nsequences {
                let mut sequence = String::with_capacity(sequence_length);

                // Generate sequence with some patterns
                let seq_type = seq_idx % 3; // 3 different types

                for _pos in 0..sequence_length {
                    let nucleotide = match seq_type {
                        0 => {
                            // GC-rich sequences
                            if rng.random::<f64>() < 0.6 {
                                if rng.random::<f64>() < 0.5 {
                                    'G'
                                } else {
                                    'C'
                                }
                            } else if rng.random::<f64>() < 0.5 {
                                'A'
                            } else {
                                'T'
                            }
                        }
                        1 => {
                            // AT-rich sequences
                            if rng.random::<f64>() < 0.6 {
                                if rng.random::<f64>() < 0.5 {
                                    'A'
                                } else {
                                    'T'
                                }
                            } else if rng.random::<f64>() < 0.5 {
                                'G'
                            } else {
                                'C'
                            }
                        }
                        _ => {
                            // Random sequences
                            nucleotides[rng.sample(Uniform::new(0, 4).unwrap())]
                        }
                    };

                    sequence.push(nucleotide);
                }

                sequences.push(sequence);
                sequence_types.push(seq_type as f64);
            }

            // Convert sequences to k-mer features (k=3)
            let mut data = Vec::new();
            let k = 3;
            let kmers = Self::generate_kmers(k);

            for sequence in &sequences {
                let kmer_counts = Self::count_kmers(sequence, k, &kmers);
                data.extend(kmer_counts);
            }

            let n_features = 4_usize.pow(k as u32); // 4^k possible k-mers
            let data_array = Array2::from_shape_vec((nsequences, n_features), data)
                .map_err(|e| DatasetsError::FormatError(e.to_string()))?;

            let target = Array1::from_vec(sequence_types);

            Ok(Dataset {
                data: data_array,
                target: Some(target),
                featurenames: Some(kmers.clone()),
                targetnames: Some(vec![
                    "GC-rich".to_string(),
                    "AT-rich".to_string(),
                    "Random".to_string(),
                ]),
                feature_descriptions: Some(
                    kmers
                        .iter()
                        .map(|kmer| format!("Frequency of {k}-mer: {kmer}"))
                        .collect(),
                ),
                description: Some(format!(
                    "DNA sequences: {nsequences} seqs × {k}-mer features"
                )),
                metadata: std::collections::HashMap::new(),
            })
        }

        fn generate_kmers(k: usize) -> Vec<String> {
            let nucleotides = vec!['A', 'T', 'G', 'C'];
            let mut kmers = Vec::new();

            fn generate_recursive(
                current: String,
                remaining: usize,
                nucleotides: &[char],
                kmers: &mut Vec<String>,
            ) {
                if remaining == 0 {
                    kmers.push(current);
                    return;
                }

                for &nucleotide in nucleotides {
                    let mut new_current = current.clone();
                    new_current.push(nucleotide);
                    generate_recursive(new_current, remaining - 1, nucleotides, kmers);
                }
            }

            generate_recursive(String::new(), k, &nucleotides, &mut kmers);
            kmers
        }

        fn count_kmers(sequence: &str, k: usize, kmers: &[String]) -> Vec<f64> {
            let mut counts = vec![0.0; kmers.len()];
            let kmer_to_idx: HashMap<&str, usize> = kmers
                .iter()
                .enumerate()
                .map(|(i, k)| (k.as_str(), i))
                .collect();

            for i in 0..=sequence.len().saturating_sub(k) {
                let kmer = &sequence[i..i + k];
                if let Some(&idx) = kmer_to_idx.get(kmer) {
                    counts[idx] += 1.0;
                }
            }

            // Normalize by sequence length
            let total: f64 = counts.iter().sum();
            if total > 0.0 {
                for count in &mut counts {
                    *count /= total;
                }
            }

            counts
        }
    }
}

/// Climate science and meteorology datasets
pub mod climate {
    use super::*;

    /// Climate and weather datasets
    pub struct ClimateDatasets {
        #[allow(dead_code)]
        client: ExternalClient,
        #[allow(dead_code)]
        cache: DatasetCache,
    }

    impl ClimateDatasets {
        /// Create a new climate datasets client
        pub fn new() -> Result<Self> {
            let cachedir = dirs::cache_dir()
                .ok_or_else(|| {
                    DatasetsError::Other("Could not determine cache directory".to_string())
                })?
                .join("scirs2-datasets");
            Ok(Self {
                client: ExternalClient::new()?,
                cache: DatasetCache::new(cachedir),
            })
        }

        /// Load synthetic temperature time series data
        pub fn load_temperature_timeseries(
            &self,
            n_stations: usize,
            n_years: usize,
        ) -> Result<Dataset> {
            use rand_distr::{Distribution, Normal};

            let mut rng = rng();
            let days_per_year = 365;
            let total_days = n_years * days_per_year;

            let mut data = Vec::with_capacity(n_stations * 8); // 8 climate features per station
            let mut climate_zones = Vec::with_capacity(n_stations);

            for station_idx in 0..n_stations {
                // Assign climate zone
                let zone = station_idx % 5; // 5 climate zones
                climate_zones.push(zone as f64);

                // Base parameters for different climate zones
                let (base_temp, temp_amplitude, annual_precip, humidity) = match zone {
                    0 => (25.0, 5.0, 2000.0, 80.0),  // Tropical
                    1 => (15.0, 15.0, 800.0, 60.0),  // Temperate
                    2 => (-5.0, 20.0, 400.0, 70.0),  // Continental
                    3 => (5.0, 8.0, 200.0, 40.0),    // Desert
                    _ => (-10.0, 25.0, 300.0, 75.0), // Arctic
                };

                // Simulate temperature time series and derive statistics
                let mut temperatures = Vec::with_capacity(total_days);
                let mut precipitation = Vec::with_capacity(total_days);

                for day in 0..total_days {
                    let year_progress = (day % days_per_year) as f64 / days_per_year as f64;
                    let seasonal_temp = base_temp
                        + temp_amplitude * (year_progress * 2.0 * std::f64::consts::PI).cos();

                    // Add daily noise
                    let temp_noise = Normal::new(0.0, 2.0).unwrap();
                    let temp = seasonal_temp + temp_noise.sample(&mut rng);
                    temperatures.push(temp);

                    // Precipitation (more in summer for some zones)
                    let seasonal_precip_factor = match zone {
                        0 => {
                            1.0 + 0.3
                                * (year_progress * 2.0 * std::f64::consts::PI
                                    + std::f64::consts::PI)
                                    .cos()
                        }
                        1 => 1.0 + 0.2 * (year_progress * 2.0 * std::f64::consts::PI).sin(),
                        _ => 1.0,
                    };

                    let precip = if rng.random::<f64>() < 0.3 {
                        // 30% chance of precipitation
                        rng.gen_range(0.0..20.0) * seasonal_precip_factor
                    } else {
                        0.0
                    };
                    precipitation.push(precip);
                }

                // Calculate summary statistics
                let mean_temp = temperatures.iter().sum::<f64>() / temperatures.len() as f64;
                let max_temp = temperatures
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let min_temp = temperatures.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let temp_range = max_temp - min_temp;

                let total_precip = precipitation.iter().sum::<f64>();
                let precip_days = precipitation.iter().filter(|&&p| p > 0.0).count() as f64;

                // Generate additional climate variables
                let avg_humidity = humidity + Normal::new(0.0, 5.0).unwrap().sample(&mut rng);
                let wind_speed = rng.gen_range(2.0..15.0);

                data.extend(vec![
                    mean_temp,
                    temp_range,
                    total_precip,
                    precip_days,
                    avg_humidity,
                    wind_speed,
                    base_temp,             // Latitude proxy
                    annual_precip / 365.0, // Average daily precipitation
                ]);
            }

            let data_array = Array2::from_shape_vec((n_stations, 8), data)
                .map_err(|e| DatasetsError::FormatError(e.to_string()))?;

            let target = Array1::from_vec(climate_zones);

            Ok(Dataset {
                data: data_array,
                target: Some(target),
                featurenames: Some(vec![
                    "mean_temperature".to_string(),
                    "temperature_range".to_string(),
                    "annual_precipitation".to_string(),
                    "precipitation_days".to_string(),
                    "avg_humidity".to_string(),
                    "avg_wind_speed".to_string(),
                    "latitude_proxy".to_string(),
                    "daily_precip_avg".to_string(),
                ]),
                targetnames: Some(vec![
                    "Tropical".to_string(),
                    "Temperate".to_string(),
                    "Continental".to_string(),
                    "Desert".to_string(),
                    "Arctic".to_string(),
                ]),
                feature_descriptions: Some(vec![
                    "Mean annual temperature (°C)".to_string(),
                    "Temperature range (max-min, °C)".to_string(),
                    "Total annual precipitation (mm)".to_string(),
                    "Number of precipitation days per year".to_string(),
                    "Average humidity (%)".to_string(),
                    "Average wind speed (m/s)".to_string(),
                    "Latitude proxy (normalized)".to_string(),
                    "Average daily precipitation (mm/day)".to_string(),
                ]),
                description: Some(format!(
                    "Climate data: {n_stations} _stations × {n_years} _years"
                )),
                metadata: std::collections::HashMap::new(),
            })
        }

        /// Load atmospheric chemistry data
        pub fn load_atmospheric_chemistry(&self, nmeasurements: usize) -> Result<Dataset> {
            use rand_distr::{Distribution, LogNormal, Normal};

            let mut rng = rng();

            let mut data = Vec::with_capacity(nmeasurements * 12);
            let mut air_quality_index = Vec::with_capacity(nmeasurements);

            for _ in 0..nmeasurements {
                // Generate correlated atmospheric _measurements
                let base_pollution = rng.gen_range(0.0..1.0);

                // Major pollutants (concentrations in µg/m³)
                let pm25: f64 = LogNormal::new(2.0 + base_pollution, 0.5)
                    .unwrap()
                    .sample(&mut rng);
                let pm10 = pm25 * rng.gen_range(1.5..2.5);
                let no2 = LogNormal::new(3.0 + base_pollution * 0.5, 0.3)
                    .unwrap()
                    .sample(&mut rng);
                let so2 = LogNormal::new(1.0 + base_pollution * 0.3, 0.4)
                    .unwrap()
                    .sample(&mut rng);
                let o3 = LogNormal::new(4.0 - base_pollution * 0.2, 0.2)
                    .unwrap()
                    .sample(&mut rng);
                let co = LogNormal::new(0.5 + base_pollution * 0.4, 0.3)
                    .unwrap()
                    .sample(&mut rng);

                // Meteorological factors
                let temperature = Normal::new(20.0, 10.0).unwrap().sample(&mut rng);
                let humidity = rng.gen_range(30.0..90.0);
                let wind_speed = rng.gen_range(0.5..12.0);
                let pressure = Normal::new(1013.0, 15.0).unwrap().sample(&mut rng);

                // Derived _measurements
                let visibility = (50.0 - pm25.ln() * 5.0).max(1.0);
                let uv_index = rng.gen_range(0.0..12.0);

                data.extend(vec![
                    pm25,
                    pm10,
                    no2,
                    so2,
                    o3,
                    co,
                    temperature,
                    humidity,
                    wind_speed,
                    pressure,
                    visibility,
                    uv_index,
                ]);

                // Calculate air quality index
                let aqi = Self::calculate_aqi(pm25, pm10, no2, so2, o3, co);
                air_quality_index.push(aqi);
            }

            let data_array = Array2::from_shape_vec((nmeasurements, 12), data)
                .map_err(|e| DatasetsError::FormatError(e.to_string()))?;

            let target = Array1::from_vec(air_quality_index);

            Ok(Dataset {
                data: data_array,
                target: Some(target),
                featurenames: Some(vec![
                    "pm2_5".to_string(),
                    "pm10".to_string(),
                    "no2".to_string(),
                    "so2".to_string(),
                    "o3".to_string(),
                    "co".to_string(),
                    "temperature".to_string(),
                    "humidity".to_string(),
                    "wind_speed".to_string(),
                    "pressure".to_string(),
                    "visibility".to_string(),
                    "uv_index".to_string(),
                ]),
                targetnames: None,
                feature_descriptions: Some(vec![
                    "PM2.5 concentration (µg/m³)".to_string(),
                    "PM10 concentration (µg/m³)".to_string(),
                    "NO2 concentration (µg/m³)".to_string(),
                    "SO2 concentration (µg/m³)".to_string(),
                    "O3 concentration (µg/m³)".to_string(),
                    "CO concentration (µg/m³)".to_string(),
                    "Temperature (°C)".to_string(),
                    "Relative humidity (%)".to_string(),
                    "Wind speed (m/s)".to_string(),
                    "Atmospheric pressure (hPa)".to_string(),
                    "Visibility (km)".to_string(),
                    "UV index".to_string(),
                ]),
                description: Some(format!(
                    "Atmospheric chemistry _measurements: {nmeasurements} samples"
                )),
                metadata: std::collections::HashMap::new(),
            })
        }

        #[allow(clippy::too_many_arguments)]
        fn calculate_aqi(pm25: f64, pm10: f64, no2: f64, so2: f64, o3: f64, co: f64) -> f64 {
            // Simplified AQI calculation
            let pm25_aqi = (pm25 / 35.0 * 100.0).min(300.0);
            let pm10_aqi = (pm10 / 150.0 * 100.0).min(300.0);
            let no2_aqi = (no2 / 100.0 * 100.0).min(300.0);
            let so2_aqi = (so2 / 75.0 * 100.0).min(300.0);
            let o3_aqi = (o3 / 120.0 * 100.0).min(300.0);
            let co_aqi = (co / 9.0 * 100.0).min(300.0);

            // Return the maximum AQI (worst pollutant determines overall AQI)
            [pm25_aqi, pm10_aqi, no2_aqi, so2_aqi, o3_aqi, co_aqi]
                .iter()
                .fold(0.0f64, |a, &b| a.max(b))
        }
    }
}

/// Convenience functions for loading domain-specific datasets
pub mod convenience {
    use super::astronomy::StellarDatasets;
    use super::climate::ClimateDatasets;
    use super::genomics::GenomicsDatasets;
    use super::*;

    /// Load a stellar classification dataset
    pub fn load_stellar_classification() -> Result<Dataset> {
        let datasets = StellarDatasets::new()?;
        datasets.load_hipparcos_catalog()
    }

    /// Load an exoplanet dataset
    pub fn load_exoplanets() -> Result<Dataset> {
        let datasets = StellarDatasets::new()?;
        datasets.load_exoplanet_catalog()
    }

    /// Load a gene expression dataset
    pub fn load_gene_expression(
        n_samples: Option<usize>,
        ngenes: Option<usize>,
    ) -> Result<Dataset> {
        let datasets = GenomicsDatasets::new()?;
        datasets.load_gene_expression(n_samples.unwrap_or(200), ngenes.unwrap_or(1000))
    }

    /// Load a climate dataset
    pub fn load_climate_data(
        _n_stations: Option<usize>,
        n_years: Option<usize>,
    ) -> Result<Dataset> {
        let datasets = ClimateDatasets::new()?;
        datasets.load_temperature_timeseries(_n_stations.unwrap_or(100), n_years.unwrap_or(10))
    }

    /// Load atmospheric chemistry data
    pub fn load_atmospheric_chemistry(_nmeasurements: Option<usize>) -> Result<Dataset> {
        let datasets = ClimateDatasets::new()?;
        datasets.load_atmospheric_chemistry(_nmeasurements.unwrap_or(1000))
    }

    /// List all available domain-specific datasets
    pub fn list_domain_datasets() -> Vec<(&'static str, &'static str)> {
        vec![
            ("astronomy", "stellar_classification"),
            ("astronomy", "exoplanets"),
            ("astronomy", "supernovae"),
            ("astronomy", "gaia_dr3"),
            ("genomics", "gene_expression"),
            ("genomics", "dnasequences"),
            ("climate", "temperature_timeseries"),
            ("climate", "atmospheric_chemistry"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::convenience::*;
    use rand_distr::Uniform;

    #[test]
    fn test_load_stellar_classification() {
        let dataset = load_stellar_classification().unwrap();
        assert!(dataset.n_samples() > 1000);
        assert_eq!(dataset.n_features(), 8);
        assert!(dataset.target.is_some());
    }

    #[test]
    fn test_load_exoplanets() {
        let dataset = load_exoplanets().unwrap();
        assert!(dataset.n_samples() > 100);
        assert_eq!(dataset.n_features(), 6);
        assert!(dataset.target.is_some());
    }

    #[test]
    fn test_load_gene_expression() {
        let dataset = load_gene_expression(Some(50), Some(100)).unwrap();
        assert_eq!(dataset.n_samples(), 50);
        assert_eq!(dataset.n_features(), 100);
        assert!(dataset.target.is_some());
    }

    #[test]
    fn test_load_climate_data() {
        let dataset = load_climate_data(Some(20), Some(5)).unwrap();
        assert_eq!(dataset.n_samples(), 20);
        assert_eq!(dataset.n_features(), 8);
        assert!(dataset.target.is_some());
    }

    #[test]
    fn test_load_atmospheric_chemistry() {
        let dataset = load_atmospheric_chemistry(Some(100)).unwrap();
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 12);
        assert!(dataset.target.is_some());
    }

    #[test]
    fn test_list_domain_datasets() {
        let datasets = list_domain_datasets();
        assert!(!datasets.is_empty());
        assert!(datasets.iter().any(|(domain_, _)| *domain_ == "astronomy"));
        assert!(datasets.iter().any(|(domain_, _)| *domain_ == "genomics"));
        assert!(datasets.iter().any(|(domain_, _)| *domain_ == "climate"));
    }
}
