use scirs2_io::csv::{read_csv_typed, write_csv_typed, ColumnType, CsvWriterConfig};
use std::env;
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    // Use environment variable or temp directory for output
    let output_dir = env::var("SCIRS2_EXAMPLE_OUTPUT_DIR")
        .unwrap_or_else(|_| env::temp_dir().to_string_lossy().to_string());

    let input_path = format!("{}/scirs2_simple_types.csv", output_dir);

    println!("Reading CSV file with advanced data types...");

    // Read the CSV file with automatic type detection
    let (headers, data) = read_csv_typed(&input_path, None, None, None)?;

    println!("Headers: {:?}", headers);
    println!("Detected types:");

    // Display the detected types and values for each row
    for (i, row) in data.iter().enumerate() {
        println!("Row {}: {:?}", i + 1, row);
    }

    // Read the CSV file with specified types
    let col_types = vec![
        ColumnType::String,
        ColumnType::Integer,
        ColumnType::Float,
        ColumnType::Boolean,
    ];

    println!("\nReading with explicit types...");

    let (headers, data) = read_csv_typed(&input_path, None, Some(&col_types), None)?;

    println!("Headers: {:?}", headers);

    // Display the values for each row
    for (i, row) in data.iter().enumerate() {
        println!("Row {}: {:?}", i + 1, row);
    }

    // Write the data back to a new CSV file
    println!("\nWriting data back to CSV...");

    let writer_config = CsvWriterConfig {
        delimiter: ',',
        quote_char: '"',
        always_quote: true,
        ..Default::default()
    };

    let output_path = format!("{}/scirs2_advanced_types_output.csv", output_dir);

    write_csv_typed(&output_path, &data, Some(&headers), Some(writer_config))?;

    println!("Data written to {}", output_path);

    println!("\nReading back the written file for verification...");

    let (output_headers, output_data) = read_csv_typed(&output_path, None, Some(&col_types), None)?;

    println!("Output headers: {:?}", output_headers);

    // Display the values for each row
    for (i, row) in output_data.iter().enumerate() {
        println!("Row {}: {:?}", i + 1, row);
    }

    println!("\nExample completed successfully!");

    Ok(())
}
