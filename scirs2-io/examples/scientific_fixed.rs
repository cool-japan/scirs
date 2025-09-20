use ndarray::Array2;
use scirs2_io::csv::{read_csv, write_csv, CsvReaderConfig, CsvWriterConfig};
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// This example demonstrates using CSV functionality for scientific data
/// with metadata in comments.
#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Scientific CSV Fixed Example ===\n");

    // Use environment variable or temp directory for output
    let output_dir = env::var("SCIRS2_EXAMPLE_OUTPUT_DIR")
        .unwrap_or_else(|_| env::temp_dir().to_string_lossy().to_string());

    // Create sample scientific data file with metadata
    create_sample_scientific_data(&output_dir)?;

    // Read and process the scientific data
    read_and_process_scientific_data(&output_dir)?;

    // Convert units and create a derived data file
    convert_units_and_create_derived_data(&output_dir)?;

    println!("\nScientific CSV fixed example completed successfully!");
    Ok(())
}

/// Create a sample scientific data file with metadata
#[allow(dead_code)]
fn create_sample_scientific_data(output_dir: &str) -> Result<(), Box<dyn Error>> {
    println!("Creating sample scientific data file...");

    // Create a file with metadata in comments at the top
    let file_path = format!("{}/scirs2_scientific_fixed.csv", output_dir);
    let file = File::create(&file_path)?;
    let mut writer = BufWriter::new(file);

    // Write metadata as comments
    writeln!(writer, "# Title: Sample Scientific Experiment Data")?;
    writeln!(writer, "# Author: SciRS2 Team")?;
    writeln!(writer, "# Date: 2025-04-10")?;
    writeln!(
        writer,
        "# Description: Temperature measurements across different materials"
    )?;
    writeln!(writer, "# Units:")?;
    writeln!(writer, "#   Time: seconds (s)")?;
    writeln!(writer, "#   Temperature: Celsius (°C)")?;
    writeln!(writer, "#   Pressure: kilopascals (kPa)")?;
    writeln!(writer, "# Experiment ID: EXP-20250410-001")?;

    // Write the header line
    writeln!(writer, "Time,Temperature,Pressure,Material")?;

    // Write data rows
    writeln!(writer, "0.0,25.0,101.3,\"Aluminum\"")?;
    writeln!(writer, "0.5,26.2,101.4,\"Aluminum\"")?;
    writeln!(writer, "1.0,27.5,101.6,\"Aluminum\"")?;
    writeln!(writer, "1.5,28.7,101.7,\"Aluminum\"")?;
    writeln!(writer, "2.0,30.0,101.9,\"Aluminum\"")?;
    writeln!(writer, "0.0,25.0,101.3,\"Copper\"")?;
    writeln!(writer, "0.5,28.1,101.4,\"Copper\"")?;
    writeln!(writer, "1.0,32.5,101.6,\"Copper\"")?;
    writeln!(writer, "1.5,36.7,101.7,\"Copper\"")?;
    writeln!(writer, "2.0,40.0,101.9,\"Copper\"")?;
    writeln!(writer, "0.0,25.0,101.3,\"Glass\"")?;
    writeln!(writer, "0.5,25.3,101.4,\"Glass\"")?;
    writeln!(writer, "1.0,25.8,101.6,\"Glass\"")?;
    writeln!(writer, "1.5,26.2,101.7,\"Glass\"")?;
    writeln!(writer, "2.0,26.5,101.9,\"Glass\"")?;

    writer.flush()?;
    println!("Sample scientific data created at {}", file_path);

    Ok(())
}

/// Extract metadata from comment lines
#[allow(dead_code)]
fn extract_metadata_from_file(_filepath: &str) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let file = File::open(_filepath)?;
    let reader = BufReader::new(file);
    let mut metadata = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') {
            if let Some(pos) = line.find(':') {
                let key = line[1..pos].trim().to_string();
                let value = line[pos + 1..].trim().to_string();
                metadata.insert(key, value);
            }
        } else {
            // Stop at the first non-comment line
            break;
        }
    }

    Ok(metadata)
}

/// Read and process the scientific data
#[allow(dead_code)]
fn read_and_process_scientific_data(output_dir: &str) -> Result<(), Box<dyn Error>> {
    println!("\nReading and processing scientific data...");

    // Extract metadata from comments
    let file_path = format!("{}/scirs2_scientific_fixed.csv", output_dir);
    let metadata = extract_metadata_from_file(&file_path)?;

    println!("Metadata:");
    for (key, value) in &metadata {
        println!("  {}: {}", key, value);
    }

    // Read the data with type detection
    let config = CsvReaderConfig {
        comment_char: Some('#'),
        has_header: true,
        ..Default::default()
    };

    let (headers, data) = read_csv(&file_path, Some(config))?;

    println!("\nData headers: {:?}", headers);
    println!("Number of data rows: {}", data.shape()[0]);
    println!("First 3 rows:");

    for i in 0..3.min(data.shape()[0]) {
        println!("  {:?}", data.row(i));
    }

    // Convert string data to numeric data for calculations
    let mut time_data = Vec::new();
    let mut temp_data = Vec::new();
    let mut pressure_data = Vec::new();
    let mut materials = HashMap::new();

    for i in 0..data.shape()[0] {
        let time = data[[i, 0]].parse::<f64>()?;
        let temp = data[[i, 1]].parse::<f64>()?;
        let pressure = data[[i, 2]].parse::<f64>()?;
        let material = data[[i, 3]].to_string();

        time_data.push(time);
        temp_data.push(temp);
        pressure_data.push(pressure);

        // Group data by material
        materials
            .entry(material.clone())
            .or_insert_with(Vec::new)
            .push((time, temp, pressure));
    }

    // Calculate statistics
    println!("\nBasic statistics:");

    // Temperature statistics
    let temp_min = temp_data.iter().cloned().fold(f64::INFINITY, f64::min);
    let temp_max = temp_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let temp_sum: f64 = temp_data.iter().sum();
    let temp_mean = temp_sum / temp_data.len() as f64;

    println!(
        "  Temperature (°C): min={:.2}, max={:.2}, mean={:.2}",
        temp_min, temp_max, temp_mean
    );

    // Pressure statistics
    let pressure_min = pressure_data.iter().cloned().fold(f64::INFINITY, f64::min);
    let pressure_max = pressure_data
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let pressure_sum: f64 = pressure_data.iter().sum();
    let pressure_mean = pressure_sum / pressure_data.len() as f64;

    println!(
        "  Pressure (kPa): min={:.2}, max={:.2}, mean={:.2}",
        pressure_min, pressure_max, pressure_mean
    );

    // Material-specific statistics
    println!("\nStatistics by material:");

    for (material, values) in &materials {
        let material_temps: Vec<f64> = values.iter().map(|(_, temp, _)| *temp).collect();
        let temp_sum: f64 = material_temps.iter().sum();
        let temp_mean = temp_sum / material_temps.len() as f64;

        println!(
            "  {}: {} measurements, mean temperature={:.2}°C",
            material,
            values.len(),
            temp_mean
        );

        // Calculate temperature change rate (if time series is long enough)
        if values.len() >= 2 {
            let mut rates = Vec::new();
            let mut sorted_values = values.clone();
            sorted_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for i in 1..sorted_values.len() {
                let time_diff = sorted_values[i].0 - sorted_values[i - 1].0;
                let temp_diff = sorted_values[i].1 - sorted_values[i - 1].1;

                if time_diff > 0.0 {
                    rates.push(temp_diff / time_diff);
                }
            }

            if !rates.is_empty() {
                let rate_sum: f64 = rates.iter().sum();
                let rate_mean = rate_sum / rates.len() as f64;
                println!("    Average temperature change rate: {:.2}°C/s", rate_mean);
            }
        }
    }

    Ok(())
}

/// Convert units and create a derived data file
#[allow(dead_code)]
fn convert_units_and_create_derived_data(output_dir: &str) -> Result<(), Box<dyn Error>> {
    println!("\nConverting units and creating derived data...");

    // Read the original data
    let file_path = format!("{}/scirs2_scientific_fixed.csv", output_dir);
    let config = CsvReaderConfig {
        comment_char: Some('#'),
        has_header: true,
        ..Default::default()
    };

    let (_headers, data) = read_csv(file_path, Some(config))?;

    // Create new array for converted data
    let rows = data.shape()[0];
    let cols = 7; // Will add 3 derived columns
    let mut converted_data = Array2::<String>::from_elem((rows, cols), String::new());

    // Create new headers with unit information
    let converted_headers = vec![
        "Time (min)".to_string(),          // s -> min
        "Temperature (K)".to_string(),     // °C -> K
        "Pressure (atm)".to_string(),      // kPa -> atm
        "Material".to_string(),            // unchanged
        "Temp Rate (K/min)".to_string(),   // derived
        "Pressure Change (%)".to_string(), // derived
        "Heat Index".to_string(),          // derived
    ];

    // Group by material to compute rates
    let mut material_groups = HashMap::new();

    for i in 0..rows {
        let material = data[[i, 3]].clone();
        material_groups
            .entry(material)
            .or_insert_with(Vec::new)
            .push(i);
    }

    // Process each material group to sort by time
    for (_material, indices) in &mut material_groups {
        indices.sort_by(|&a, &b| {
            let time_a = data[[a, 0]].parse::<f64>().unwrap_or(0.0);
            let time_b = data[[b, 0]].parse::<f64>().unwrap_or(0.0);
            time_a.partial_cmp(&time_b).unwrap()
        });
    }

    // Fill in the converted data
    for i in 0..rows {
        // Copy and convert existing data
        let time = data[[i, 0]].parse::<f64>().unwrap_or(0.0);
        let temp = data[[i, 1]].parse::<f64>().unwrap_or(0.0);
        let pressure = data[[i, 2]].parse::<f64>().unwrap_or(0.0);
        let material = data[[i, 3]].clone();

        // Unit conversions
        let time_min = time / 60.0; // seconds -> minutes
        let temp_k = temp + 273.15; // Celsius -> Kelvin
        let pressure_atm = pressure / 101.325; // kPa -> atm

        // Find this row's position in its material group
        let material_group = material_groups.get(&material).unwrap();
        let pos_in_group = material_group.iter().position(|&x| x == i).unwrap();

        let mut temp_rate = 0.0;
        let mut pressure_change = 0.0;

        // Calculate derived values if not first in group
        if pos_in_group > 0 {
            let prev_idx = material_group[pos_in_group - 1];
            let prev_time = data[[prev_idx, 0]].parse::<f64>().unwrap_or(0.0) / 60.0; // in min
            let prev_temp = data[[prev_idx, 1]].parse::<f64>().unwrap_or(0.0) + 273.15; // in K
            let prev_pressure = data[[prev_idx, 2]].parse::<f64>().unwrap_or(0.0);

            let time_diff = time_min - prev_time;
            if time_diff > 0.0 {
                temp_rate = (temp_k - prev_temp) / time_diff;
            }

            if prev_pressure > 0.0 {
                pressure_change = (pressure - prev_pressure) / prev_pressure * 100.0;
            }
        }

        // Calculate heat index (simplified model: just for demonstration)
        // In a real application, you would use a proper formula
        let heat_index = temp_k * 0.5 + pressure_atm * 10.0 + temp_rate.abs() * 5.0;

        // Store converted and derived values
        converted_data[[i, 0]] = format!("{:.4}", time_min);
        converted_data[[i, 1]] = format!("{:.2}", temp_k);
        converted_data[[i, 2]] = format!("{:.6}", pressure_atm);
        converted_data[[i, 3]] = material;
        converted_data[[i, 4]] = format!("{:.4}", temp_rate);
        converted_data[[i, 5]] = format!("{:.4}", pressure_change);
        converted_data[[i, 6]] = format!("{:.2}", heat_index);
    }

    // Write converted data to file
    let output_path = format!("{}/scirs2_scientific_derived_fixed.csv", output_dir);

    // Add metadata as comments
    let file = File::create(&output_path)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "# Title: Derived Scientific Data with Unit Conversions"
    )?;
    writeln!(writer, "# Based on: Sample Scientific Experiment Data")?;
    writeln!(writer, "# Converted Units:")?;
    writeln!(writer, "#   Time: minutes (min) [converted from seconds]")?;
    writeln!(
        writer,
        "#   Temperature: Kelvin (K) [converted from Celsius]"
    )?;
    writeln!(
        writer,
        "#   Pressure: atmospheres (atm) [converted from kilopascals]"
    )?;
    writeln!(writer, "# Derived Measurements:")?;
    writeln!(writer, "#   Temp Rate: temperature change rate (K/min)")?;
    writeln!(
        writer,
        "#   Pressure Change: percentage change from previous measurement (%)"
    )?;
    writeln!(
        writer,
        "#   Heat Index: simplified thermal index (arbitrary units)"
    )?;

    writer.flush()?;

    // Write the actual data using CSV writer
    let write_config = CsvWriterConfig {
        write_header: true,
        ..Default::default()
    };

    write_csv(
        &output_path,
        &converted_data,
        Some(&converted_headers),
        Some(write_config),
    )?;

    println!("Derived data written to {}", output_path);
    println!("Unit conversions performed:");
    println!("  - Time: seconds -> minutes");
    println!("  - Temperature: Celsius -> Kelvin");
    println!("  - Pressure: kilopascals -> atmospheres");
    println!("Derived values calculated:");
    println!("  - Temperature change rate (K/min)");
    println!("  - Pressure change percentage (%)");
    println!("  - Heat index (simplified thermal index)");

    Ok(())
}
