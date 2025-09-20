use scirs2_io::csv::{read_csv_typed, write_csv_typed, ColumnType, CsvReaderConfig, DataValue};
use statrs::statistics::Statistics;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};

/// This example demonstrates using CSV functionality for scientific data
/// with various units, metadata, and mixed types.
#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Scientific Data CSV Example ===\n");

    // Use environment variable or temp directory for output
    let output_dir = env::var("SCIRS2_EXAMPLE_OUTPUT_DIR")
        .unwrap_or_else(|_| env::temp_dir().to_string_lossy().to_string());

    // Create sample scientific data file with metadata header
    create_sample_scientific_data(&output_dir)?;

    // Read the file including metadata
    read_scientific_data_with_metadata(&output_dir)?;

    // Convert data and apply unit conversions
    convert_scientific_data_with_units(&output_dir)?;

    // Process time series data from CSV
    process_time_series_data(&output_dir)?;

    println!("\nScientific CSV example completed successfully!");
    Ok(())
}

/// Create a sample scientific data file with metadata
#[allow(dead_code)]
fn create_sample_scientific_data(output_dir: &str) -> Result<(), Box<dyn Error>> {
    println!("Creating sample scientific data file...");

    // Create a file with metadata in comments at the top
    let file_path = format!("{}/scirs2_scientific_data.csv", output_dir);
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
    writeln!(writer, "#   Concentration: moles per liter (mol/L)")?;
    writeln!(writer, "#   Energy: joules (J)")?;
    writeln!(writer, "# Experiment ID: EXP-20250410-001")?;
    writeln!(writer, "# Instrument: HighPrecThermometer-5000")?;

    // Write the header line
    writeln!(
        writer,
        "Time,Temperature,Pressure,Concentration,Energy,Material,Valid"
    )?;

    // Write data rows
    writeln!(writer, "0.0,25.0,101.3,0.5,0.0,\"Aluminum\",true")?;
    writeln!(writer, "0.5,26.2,101.4,0.48,125.5,\"Aluminum\",true")?;
    writeln!(writer, "1.0,27.5,101.6,0.45,250.0,\"Aluminum\",true")?;
    writeln!(writer, "1.5,28.7,101.7,0.43,374.5,\"Aluminum\",true")?;
    writeln!(writer, "2.0,30.0,101.9,0.40,500.0,\"Aluminum\",true")?;
    writeln!(writer, "0.0,25.0,101.3,0.5,0.0,\"Copper\",true")?;
    writeln!(writer, "0.5,28.1,101.4,0.47,225.5,\"Copper\",true")?;
    writeln!(writer, "1.0,32.5,101.6,0.42,450.0,\"Copper\",true")?;
    writeln!(writer, "1.5,36.7,101.7,0.38,674.5,\"Copper\",true")?;
    writeln!(writer, "2.0,40.0,101.9,0.35,900.0,\"Copper\",true")?;
    writeln!(writer, "0.0,25.0,101.3,0.5,0.0,\"Glass\",true")?;
    writeln!(writer, "0.5,25.3,101.4,0.49,75.5,\"Glass\",true")?;
    writeln!(writer, "1.0,25.8,101.6,0.48,150.0,\"Glass\",true")?;
    writeln!(writer, "1.5,26.2,101.7,0.47,224.5,\"Glass\",true")?;
    writeln!(writer, "2.0,26.5,101.9,0.46,300.0,\"Glass\",true")?;

    writer.flush()?;
    println!("Sample scientific data created at {}", file_path);

    Ok(())
}

/// Extract metadata from comment lines
#[allow(dead_code)]
fn extract_metadata_from_file(_filepath: &str) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let file = File::open(_filepath)?;
    let reader = std::io::BufReader::new(file);
    let mut metadata = HashMap::new();

    for line in std::io::BufRead::lines(reader) {
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

/// Read scientific data file with metadata
#[allow(dead_code)]
fn read_scientific_data_with_metadata(output_dir: &str) -> Result<(), Box<dyn Error>> {
    println!("\nReading scientific data file with metadata...");

    // Extract metadata from comments
    let file_path = format!("{}/scirs2_scientific_data.csv", output_dir);
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

    let col_types = vec![
        ColumnType::Float,   // Time
        ColumnType::Float,   // Temperature
        ColumnType::Float,   // Pressure
        ColumnType::Float,   // Concentration
        ColumnType::Float,   // Energy
        ColumnType::String,  // Material
        ColumnType::Boolean, // Valid
    ];

    let (_headers, data) = read_csv_typed(&file_path, Some(config), Some(&col_types), None)?;

    println!("\nData headers: Time, Temperature, Pressure, Concentration, Energy, Material, Valid");
    println!("Number of data rows: {}", data.len());

    // Calculate statistics for the numeric columns
    let mut temp_stats = ColumnStats::new("Temperature");
    let mut pressure_stats = ColumnStats::new("Pressure");
    let mut energy_stats = ColumnStats::new("Energy");

    for row in &data {
        if let DataValue::Float(temp) = row[1] {
            temp_stats.add_value(temp);
        }
        if let DataValue::Float(pressure) = row[2] {
            pressure_stats.add_value(pressure);
        }
        if let DataValue::Float(energy) = row[4] {
            energy_stats.add_value(energy);
        }
    }

    println!("\nStatistical summary:");
    println!("{}", temp_stats);
    println!("{}", pressure_stats);
    println!("{}", energy_stats);

    // Group data by material
    let mut materials = HashMap::new();

    for row in &data {
        if let DataValue::String(material) = &row[5] {
            materials
                .entry(material.clone())
                .or_insert_with(Vec::new)
                .push(row.clone());
        }
    }

    println!("\nData by material:");
    for (material, rows) in &materials {
        println!("  {}: {} measurements", material, rows.len());

        // Calculate average temperature change rate for each material
        let mut total_rate = 0.0;
        let mut count = 0;

        for i in 1..rows.len() {
            if let (
                DataValue::Float(t0),
                DataValue::Float(temp0),
                DataValue::Float(t1),
                DataValue::Float(temp1),
            ) = (&rows[i - 1][0], &rows[i - 1][1], &rows[i][0], &rows[i][1])
            {
                let time_diff = t1 - t0;
                if time_diff > 0.0 {
                    let temp_rate = (temp1 - temp0) / time_diff;
                    total_rate += temp_rate;
                    count += 1;
                }
            }
        }

        if count > 0 {
            let avg_rate = total_rate / count as f64;
            println!("    Average temperature change rate: {:.2} °C/s", avg_rate);
        }
    }

    Ok(())
}

/// Convert scientific data with unit conversions
#[allow(dead_code)]
fn convert_scientific_data_with_units(output_dir: &str) -> Result<(), Box<dyn Error>> {
    println!("\nConverting scientific data with unit conversions...");

    // Read the original data
    let file_path = format!("{}/scirs2_scientific_data.csv", output_dir);
    let config = CsvReaderConfig {
        comment_char: Some('#'),
        has_header: true,
        ..Default::default()
    };

    let col_types = vec![
        ColumnType::Float,   // Time
        ColumnType::Float,   // Temperature
        ColumnType::Float,   // Pressure
        ColumnType::Float,   // Concentration
        ColumnType::Float,   // Energy
        ColumnType::String,  // Material
        ColumnType::Boolean, // Valid
    ];

    let (_headers, data) = read_csv_typed(&file_path, Some(config), Some(&col_types), None)?;

    // Perform unit conversions
    let mut converted_data = Vec::new();

    // Define new headers with unit information
    let new_headers = vec![
        "Time (min)".to_string(),             // s -> min
        "Temperature (K)".to_string(),        // °C -> K
        "Pressure (atm)".to_string(),         // kPa -> atm
        "Concentration (mmol/L)".to_string(), // mol/L -> mmol/L
        "Energy (cal)".to_string(),           // J -> cal
        "Material".to_string(),               // unchanged
        "Valid".to_string(),                  // unchanged
    ];

    // Convert units for each row
    for row in data {
        let mut new_row = Vec::new();

        // Time: seconds -> minutes
        if let DataValue::Float(time) = row[0] {
            new_row.push(DataValue::Float(time / 60.0));
        } else {
            new_row.push(row[0].clone());
        }

        // Temperature: Celsius -> Kelvin
        if let DataValue::Float(temp) = row[1] {
            new_row.push(DataValue::Float(temp + 273.15));
        } else {
            new_row.push(row[1].clone());
        }

        // Pressure: kPa -> atm
        if let DataValue::Float(pressure) = row[2] {
            new_row.push(DataValue::Float(pressure / 101.325));
        } else {
            new_row.push(row[2].clone());
        }

        // Concentration: mol/L -> mmol/L
        if let DataValue::Float(conc) = row[3] {
            new_row.push(DataValue::Float(conc * 1000.0));
        } else {
            new_row.push(row[3].clone());
        }

        // Energy: J -> cal
        if let DataValue::Float(energy) = row[4] {
            new_row.push(DataValue::Float(energy / 4.184));
        } else {
            new_row.push(row[4].clone());
        }

        // Add remaining columns unchanged
        for i in 5..row.len() {
            new_row.push(row[i].clone());
        }

        converted_data.push(new_row);
    }

    // Write converted data to a new file
    let output_path = format!("{}/scirs2_scientific_data_converted.csv", output_dir);

    write_csv_typed(&output_path, &converted_data, Some(&new_headers), None)?;

    println!("Converted data written to {}", output_path);
    println!("Unit conversions performed:");
    println!("  - Time: seconds -> minutes");
    println!("  - Temperature: Celsius -> Kelvin");
    println!("  - Pressure: kilopascals -> atmospheres");
    println!("  - Concentration: mol/L -> mmol/L");
    println!("  - Energy: joules -> calories");

    Ok(())
}

/// Process time series data from CSV
#[allow(dead_code)]
fn process_time_series_data(output_dir: &str) -> Result<(), Box<dyn Error>> {
    println!("\nProcessing time series data...");

    // Read the original data, focusing on time and temperature
    let file_path = format!("{}/scirs2_scientific_data.csv", output_dir);
    let config = CsvReaderConfig {
        comment_char: Some('#'),
        has_header: true,
        ..Default::default()
    };

    let col_types = vec![
        ColumnType::Float,   // Time
        ColumnType::Float,   // Temperature
        ColumnType::Float,   // Pressure
        ColumnType::Float,   // Concentration
        ColumnType::Float,   // Energy
        ColumnType::String,  // Material
        ColumnType::Boolean, // Valid
    ];

    let (_headers, data) = read_csv_typed(&file_path, Some(config), Some(&col_types), None)?;

    // Group data by material to separate time series
    let mut time_series = HashMap::new();

    for row in &data {
        if let (DataValue::String(material), DataValue::Float(time), DataValue::Float(temp)) =
            (&row[5], &row[0], &row[1])
        {
            time_series
                .entry(material.clone())
                .or_insert_with(Vec::new)
                .push((*time, *temp));
        }
    }

    // Sort each time series by time
    for (_, series) in time_series.iter_mut() {
        series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    // Calculate derived values: rate of change and moving average
    let mut derived_data = Vec::new();

    for (material, series) in &time_series {
        // Calculate rate of change
        let mut times = Vec::new();
        let mut temps = Vec::new();
        let mut rates = Vec::new();
        let mut avg_temps = Vec::new();

        for (i, &(time, temp)) in series.iter().enumerate() {
            times.push(time);
            temps.push(temp);

            // Calculate rate of change (derivative)
            if i > 0 {
                let prev_time = series[i - 1].0;
                let prev_temp = series[i - 1].1;
                let time_diff = time - prev_time;

                if time_diff > 0.0 {
                    let rate = (temp - prev_temp) / time_diff;
                    rates.push(rate);
                } else {
                    rates.push(0.0);
                }
            } else {
                rates.push(0.0); // First point has no derivative
            }

            // Calculate centered moving average with window size 3
            if i >= 1 && i < series.len() - 1 {
                let avg = (series[i - 1].1 + temp + series[i + 1].1) / 3.0;
                avg_temps.push(avg);
            } else {
                avg_temps.push(temp); // End points use original value
            }
        }

        // Store derived values for this material
        for i in 0..times.len() {
            derived_data.push(vec![
                DataValue::Float(times[i]),
                DataValue::String(material.clone()),
                DataValue::Float(temps[i]),
                DataValue::Float(rates[i]),
                DataValue::Float(avg_temps[i]),
            ]);
        }
    }

    // Write derived data to a new file
    let derived_headers = vec![
        "Time (s)".to_string(),
        "Material".to_string(),
        "Temperature (°C)".to_string(),
        "Temperature Rate (°C/s)".to_string(),
        "Temperature Moving Avg (°C)".to_string(),
    ];

    let output_path = format!("{}/scirs2_scientific_data_derived.csv", output_dir);
    write_csv_typed(&output_path, &derived_data, Some(&derived_headers), None)?;

    println!("Derived time series data written to {}", output_path);
    println!("Derived measurements include:");
    println!("  - Original temperature values");
    println!("  - Rate of temperature change (°C/s)");
    println!("  - 3-point centered moving average of temperature");

    Ok(())
}

/// Helper struct to calculate column statistics
struct ColumnStats {
    name: String,
    count: usize,
    min: f64,
    max: f64,
    sum: f64,
    sum_squared: f64,
}

impl ColumnStats {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            sum_squared: 0.0,
        }
    }

    fn add_value(&mut self, value: f64) {
        self.count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value;
        self.sum_squared += value * value;
    }

    fn mean(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        }
    }

    fn variance(&self) -> f64 {
        if self.count > 1 {
            let mean = self.mean();
            self.sum_squared / self.count as f64 - mean * mean
        } else {
            0.0
        }
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl std::fmt::Display for ColumnStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "  {}: count={}, min={:.2}, max={:.2}, mean={:.2}, std_dev={:.2}",
            self.name,
            self.count,
            self.min,
            self.max,
            self.mean(),
            self.std_dev()
        )
    }
}
