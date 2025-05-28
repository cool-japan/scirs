//! HDF5 file format example
//!
//! This example demonstrates how to use the HDF5 module to create and read
//! HDF5 files with groups, datasets, and attributes.

use ndarray::{array, Array2};
use scirs2_io::hdf5::{
    create_hdf5_with_structure, read_hdf5, write_hdf5, AttributeValue, CompressionOptions,
    DatasetOptions, FileMode, HDF5File,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== HDF5 File Format Example ===\n");

    // Example 1: Simple dataset writing
    simple_dataset_example()?;

    // Example 2: Structured file with groups and attributes
    structured_file_example()?;

    // Example 3: Working with compression
    compression_example()?;

    // Example 4: Scientific data example
    scientific_data_example()?;

    Ok(())
}

fn simple_dataset_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Simple Dataset Example");
    println!("-------------------------");

    // Create some datasets
    let mut datasets = HashMap::new();

    // 2D array
    let temperature = array![
        [20.5, 21.0, 21.5, 22.0],
        [21.0, 21.5, 22.0, 22.5],
        [21.5, 22.0, 22.5, 23.0]
    ];
    datasets.insert("temperature".to_string(), temperature.into_dyn());

    // 1D array
    let time = array![0.0, 0.5, 1.0, 1.5, 2.0];
    datasets.insert("time".to_string(), time.into_dyn());

    // Write to HDF5 file
    write_hdf5("simple_data.h5", datasets)?;
    println!("✓ Created simple_data.h5 with temperature and time datasets");

    // Read back the data
    let root = read_hdf5("simple_data.h5")?;
    println!("✓ Read simple_data.h5");
    println!("  Datasets: {:?}", root.datasets.keys().collect::<Vec<_>>());

    println!();
    Ok(())
}

fn structured_file_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Structured File Example");
    println!("--------------------------");

    create_hdf5_with_structure("structured_data.h5", |file| {
        // Add file-level attributes to root first
        {
            let root = file.root_mut();
            root.set_attribute("file_version", AttributeValue::String("1.0.0".to_string()));
            root.set_attribute(
                "created_by",
                AttributeValue::String("SciRS2 HDF5 Module".to_string()),
            );
        }

        // Create experiment group structure
        {
            let root = file.root_mut();
            let experiment = root.create_group("experiment");
            experiment.set_attribute(
                "name",
                AttributeValue::String("Temperature Study".to_string()),
            );
            experiment.set_attribute("date", AttributeValue::String("2024-01-15".to_string()));
            experiment.set_attribute("operator", AttributeValue::String("Dr. Smith".to_string()));

            // Create conditions subgroup
            let conditions = experiment.create_group("conditions");
            conditions.set_attribute("pressure", AttributeValue::Float(101.325)); // kPa
            conditions.set_attribute("humidity", AttributeValue::Float(45.0)); // %
        }

        // Add measurement data
        let measurements = array![
            [20.1, 20.3, 20.2, 20.4],
            [20.5, 20.4, 20.6, 20.5],
            [20.8, 20.7, 20.9, 20.8]
        ];
        file.create_dataset_from_array("experiment/measurements", &measurements, None)?;

        // Add calibration data
        let calibration = array![1.0, 1.001, 0.999, 1.002];
        file.create_dataset_from_array("calibration/factors", &calibration, None)?;

        // Create analysis group
        {
            let root = file.root_mut();
            let analysis = root.create_group("analysis");
            analysis.set_attribute("method", AttributeValue::String("Statistical".to_string()));
            analysis.set_attribute("confidence", AttributeValue::Float(0.95));
        }

        // Add analysis results
        let mean_values = array![20.25, 20.5, 20.8];
        let std_values = array![0.129, 0.082, 0.082];

        file.create_dataset_from_array("analysis/mean", &mean_values, None)?;
        file.create_dataset_from_array("analysis/std", &std_values, None)?;

        Ok(())
    })?;

    println!("✓ Created structured_data.h5 with hierarchical structure:");
    println!("  /");
    println!("  ├── experiment/");
    println!("  │   ├── conditions/");
    println!("  │   └── measurements");
    println!("  ├── calibration/");
    println!("  │   └── factors");
    println!("  └── analysis/");
    println!("      ├── mean");
    println!("      └── std");

    println!();
    Ok(())
}

fn compression_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Compression Example");
    println!("----------------------");

    let mut file = HDF5File::create("compressed_data.h5")?;

    // Create large dataset
    let large_data: Array2<f64> = Array2::from_shape_fn((1000, 500), |(i, j)| {
        (i as f64).sin() * (j as f64).cos() + 0.1 * rand::random::<f64>()
    });

    // Set up compression options
    let mut compression = CompressionOptions::default();
    compression.gzip = Some(6); // Compression level 6
    compression.shuffle = true; // Enable shuffle filter for better compression

    let mut options = DatasetOptions::default();
    options.compression = compression;
    options.chunk_size = Some(vec![100, 50]); // Chunk the data
    options.fletcher32 = true; // Enable checksum

    // Create compressed dataset
    file.create_dataset_from_array("compressed/data", &large_data, Some(options.clone()))?;

    // Create uncompressed dataset for comparison
    file.create_dataset_from_array("uncompressed/data", &large_data, None)?;

    // Add metadata after creating datasets
    {
        let root = file.root_mut();
        let compressed_group = root.get_group_mut("compressed").unwrap();
        compressed_group.set_attribute("compression", AttributeValue::String("gzip-6".to_string()));
        compressed_group.set_attribute(
            "filters",
            AttributeValue::StringArray(vec![
                "shuffle".to_string(),
                "gzip".to_string(),
                "fletcher32".to_string(),
            ]),
        );
    }

    file.write()?;
    file.close()?;

    println!("✓ Created compressed_data.h5 with compressed and uncompressed datasets");
    println!("  Dataset shape: 1000x500");
    println!("  Compression: GZIP level 6 with shuffle filter");
    println!("  Chunking: 100x50");

    println!();
    Ok(())
}

fn scientific_data_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Scientific Data Example");
    println!("--------------------------");

    create_hdf5_with_structure("scientific_data.h5", |file| {
        // Set root metadata
        {
            let root = file.root_mut();
            root.set_attribute(
                "project",
                AttributeValue::String("Climate Analysis".to_string()),
            );
            root.set_attribute("version", AttributeValue::String("2.0".to_string()));
        }

        // Create climate data group
        {
            let root = file.root_mut();
            let climate = root.create_group("climate_data");
            climate.set_attribute(
                "region",
                AttributeValue::String("North Pacific".to_string()),
            );
            climate.set_attribute(
                "coordinates",
                AttributeValue::FloatArray(vec![45.0, -125.0]),
            ); // lat, lon

            // Time series metadata
            let months = array![
                "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            ];
            let years = (2014..2024).collect::<Vec<i32>>();

            climate.set_attribute(
                "months",
                AttributeValue::StringArray(months.iter().map(|&s| s.to_string()).collect()),
            );
            climate.set_attribute(
                "years",
                AttributeValue::IntegerArray(years.iter().map(|&y| y as i64).collect()),
            );
            climate.set_attribute(
                "temperature_units",
                AttributeValue::String("Celsius".to_string()),
            );
            climate.set_attribute(
                "precipitation_units",
                AttributeValue::String("mm".to_string()),
            );
        }

        // Temperature data (12 months x 10 years)
        let temp_data: Array2<f64> = Array2::from_shape_fn((12, 10), |(month, year)| {
            15.0 + 10.0 * (2.0 * std::f64::consts::PI * month as f64 / 12.0).sin()
                + 0.5 * year as f64
                + rand::random::<f64>()
        });

        file.create_dataset_from_array("climate_data/temperature", &temp_data, None)?;

        // Precipitation data
        let precip_data: Array2<f64> = Array2::from_shape_fn((12, 10), |(month, year)| {
            50.0 + 30.0 * (2.0 * std::f64::consts::PI * (month + 6) as f64 / 12.0).sin()
                + 2.0 * year as f64
                + 5.0 * rand::random::<f64>()
        });

        file.create_dataset_from_array("climate_data/precipitation", &precip_data, None)?;

        // Model parameters group
        {
            let root = file.root_mut();
            let model = root.create_group("model_parameters");
            model.set_attribute(
                "model_name",
                AttributeValue::String("ClimatePredict v3.2".to_string()),
            );
            model.set_attribute("grid_resolution", AttributeValue::Float(0.25));
            // degrees
        }

        // Model coefficients
        let coefficients = array![0.98, 1.02, 0.95, 1.05, 0.99];
        file.create_dataset_from_array("model_parameters/coefficients", &coefficients, None)?;

        // Quality control flags
        let qc_flags: Array2<i64> =
            Array2::from_shape_fn(
                (12, 10),
                |(_, _)| if rand::random::<f64>() > 0.95 { 1 } else { 0 },
            );

        // Convert to ArrayD<f64> for our API
        let qc_flags_f64 = qc_flags.mapv(|x| x as f64);
        file.create_dataset_from_array("quality_control/flags", &qc_flags_f64, None)?;

        Ok(())
    })?;

    println!("✓ Created scientific_data.h5 with climate analysis data:");
    println!("  - Temperature data: 12 months x 10 years");
    println!("  - Precipitation data: 12 months x 10 years");
    println!("  - Model parameters and coefficients");
    println!("  - Quality control flags");

    // Read and analyze the data
    println!("\nReading back scientific data...");
    let file = HDF5File::open("scientific_data.h5", FileMode::ReadOnly)?;

    let temp_data = file.read_dataset("climate_data/temperature")?;
    let precip_data = file.read_dataset("climate_data/precipitation")?;

    println!("Temperature data shape: {:?}", temp_data.shape());
    println!("Mean temperature: {:.2}°C", temp_data.mean().unwrap());
    println!("Precipitation data shape: {:?}", precip_data.shape());
    println!("Mean precipitation: {:.2} mm", precip_data.mean().unwrap());

    println!();
    Ok(())
}

// Add dependency on rand for example data generation
use rand;
