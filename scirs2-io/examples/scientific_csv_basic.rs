use scirs2_core::ndarray::{Array1, Array2};
use scirs2_io::csv::{read_csv_numeric, write_csv, CsvReaderConfig};
use std::env;
use std::error::Error;

/// This example demonstrates using CSV functionality for scientific data conversion
#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Scientific CSV Basic Example ===\n");

    // Use environment variable or temp directory for output
    let output_dir = env::var("SCIRS2_EXAMPLE_OUTPUT_DIR")
        .unwrap_or_else(|_| env::temp_dir().to_string_lossy().to_string());

    // Create sample scientific data
    let mut data = Array2::from_elem((10, 4), 0.0);

    // Fill with sample data: Time, Temperature, Pressure, Heat Index
    for i in 0..10 {
        let time = i as f64;
        let temp = 25.0 + 2.0 * i as f64;
        let pressure = 101.0 + 0.1 * i as f64;
        let heat_index = 0.5 * temp + 0.1 * pressure;

        data[[i, 0]] = time;
        data[[i, 1]] = temp;
        data[[i, 2]] = pressure;
        data[[i, 3]] = heat_index;
    }

    // Create headers for the CSV file
    let headers = vec![
        "Time (s)".to_string(),
        "Temperature (C)".to_string(),
        "Pressure (kPa)".to_string(),
        "Heat Index".to_string(),
    ];

    // Write data to CSV file
    let file_path = format!("{}/scirs2_scientific_basic.csv", output_dir);
    println!("Writing scientific data to CSV file...");
    write_csv(&file_path, &data, Some(&headers), None)?;
    println!("Data written to {}", file_path);

    // Read back the data
    println!("\nReading data back from CSV file...");
    let config = CsvReaderConfig {
        has_header: true,
        ..Default::default()
    };

    let (csv_headers, csv_data) = read_csv_numeric(&file_path, Some(config))?;

    println!("Headers: {:?}", csv_headers);
    println!("Data shape: {:?}", csv_data.shape());
    println!("First 3 rows:");
    for i in 0..3 {
        println!("  {:?}", csv_data.row(i));
    }

    // Convert temperature: Celsius to Fahrenheit
    println!("\nConverting temperature from Celsius to Fahrenheit...");
    let rows = csv_data.shape()[0];
    let cols = csv_data.shape()[1];
    let mut converted_data = Array2::from_elem((rows, cols), 0.0);

    for i in 0..csv_data.shape()[0] {
        converted_data[[i, 0]] = csv_data[[i, 0]]; // Time (unchanged)
        converted_data[[i, 1]] = csv_data[[i, 1]] * 9.0 / 5.0 + 32.0; // Temperature (C to F)
        converted_data[[i, 2]] = csv_data[[i, 2]]; // Pressure (unchanged)
        converted_data[[i, 3]] = csv_data[[i, 3]]; // Heat Index (unchanged)
    }

    // Create new headers with updated units
    let converted_headers = vec![
        "Time (s)".to_string(),
        "Temperature (F)".to_string(),
        "Pressure (kPa)".to_string(),
        "Heat Index".to_string(),
    ];

    // Write converted data to CSV file
    let converted_path = format!("{}/scirs2_scientific_basic_converted.csv", output_dir);
    println!("Writing converted data to CSV file...");
    write_csv(
        &converted_path,
        &converted_data,
        Some(&converted_headers),
        None,
    )?;
    println!("Converted data written to {}", converted_path);

    // Calculate statistics on the data
    println!("\nCalculating statistics on the data...");

    // Temperature statistics (from original data)
    let temp_column = csv_data.column(1).to_owned();
    let temp_stats = calculate_stats(&temp_column);
    println!("Temperature (C) statistics:");
    println!("  Min: {:.2}", temp_stats.min);
    println!("  Max: {:.2}", temp_stats.max);
    println!("  Mean: {:.2}", temp_stats.mean);
    println!("  Std Dev: {:.2}", temp_stats.std_dev);

    // Temperature statistics (from converted data)
    let temp_f_column = converted_data.column(1).to_owned();
    let temp_f_stats = calculate_stats(&temp_f_column);
    println!("Temperature (F) statistics:");
    println!("  Min: {:.2}", temp_f_stats.min);
    println!("  Max: {:.2}", temp_f_stats.max);
    println!("  Mean: {:.2}", temp_f_stats.mean);
    println!("  Std Dev: {:.2}", temp_f_stats.std_dev);

    println!("\nScientific CSV example completed successfully!");
    Ok(())
}

/// Simple statistics struct for data analysis
struct Stats {
    min: f64,
    max: f64,
    mean: f64,
    std_dev: f64,
}

/// Calculate basic statistics on a 1D array
#[allow(dead_code)]
fn calculate_stats(data: &Array1<f64>) -> Stats {
    let n = data.len() as f64;

    if n == 0.0 {
        return Stats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std_dev: 0.0,
        };
    }

    let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f64 = data.iter().sum();
    let mean = sum / n;

    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    Stats {
        min,
        max,
        mean,
        std_dev,
    }
}
