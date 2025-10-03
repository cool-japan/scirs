use scirs2_core::ndarray::{array, Array1};
use scirs2_io::csv::{
    read_csv, write_csv, write_csv_columns, CsvReaderConfig, CsvWriterConfig, LineEnding,
};
use std::env;
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    // Use environment variable or temp directory for output
    let output_dir = env::var("SCIRS2_EXAMPLE_OUTPUT_DIR")
        .unwrap_or_else(|_| env::temp_dir().to_string_lossy().to_string());

    // Path to example data file
    let data_path = format!("{}/scirs2_csv_example_data.csv", output_dir);

    println!("Reading CSV file as strings...");
    // Read CSV with default configuration
    let config = CsvReaderConfig {
        comment_char: Some('#'),
        ..Default::default()
    };
    let (headers, data) = read_csv(&data_path, Some(config))?;
    println!("Headers: {:?}", headers);
    println!("Data shape: {:?}", data.shape());
    println!("First row: {:?}", data.row(0));

    println!("\nReading CSV file with custom configuration...");
    // Read CSV with custom configuration
    let config = CsvReaderConfig {
        comment_char: Some('#'),
        has_header: true,
        ..Default::default()
    };
    let (_headers, data) = read_csv(&data_path, Some(config))?;
    println!("Number of rows read: {}", data.shape()[0]);

    println!("\nReading CSV file as numeric data...");
    // Read numeric data (skipping the first column which contains names)
    let config = CsvReaderConfig {
        has_header: true,
        comment_char: Some('#'),
        ..Default::default()
    };
    // Read the full file first to get headers
    let (all_headers, string_data) = read_csv(&data_path, Some(config.clone()))?;

    // Extract numeric columns
    let numeric_columns = vec![1, 2, 3]; // Age, Height, Weight columns
    let numeric_headers: Vec<String> = numeric_columns
        .iter()
        .map(|&i| all_headers[i].clone())
        .collect();

    // Create numeric data
    let mut numeric_data = Vec::new();
    for row_idx in 0..string_data.shape()[0] {
        let mut new_row = Vec::new();
        for &col_idx in &numeric_columns {
            let value = string_data[[row_idx, col_idx]].parse::<f64>()?;
            new_row.push(value);
        }
        numeric_data.push(new_row);
    }

    println!("Numeric headers: {:?}", numeric_headers);
    println!("First row of numeric data: {:?}", numeric_data[0]);

    println!("\nWriting data to CSV file...");
    // Create some data to write
    let data_to_write = array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ];
    let write_headers = vec![
        "Column A".to_string(),
        "Column B".to_string(),
        "Column C".to_string(),
        "Column D".to_string(),
    ];

    // Write with default configuration
    let output_path = format!("{}/scirs2_csv_example_output.csv", output_dir);
    write_csv(&output_path, &data_to_write, Some(&write_headers), None)?;
    println!("Wrote data to {}", output_path);

    // Write with custom configuration (semicolon separator, CRLF line endings)
    let write_config = CsvWriterConfig {
        delimiter: ';',
        line_ending: LineEnding::CRLF,
        always_quote: true,
        ..Default::default()
    };
    let semicolon_output_path = format!("{}/scirs2_csv_example_output_semicolon.csv", output_dir);
    write_csv(
        &semicolon_output_path,
        &data_to_write,
        Some(&write_headers),
        Some(write_config),
    )?;
    println!("Wrote data to {}", semicolon_output_path);

    println!("\nWriting columns to CSV file...");
    // Create some columns to write
    let col1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let col2 = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
    let col3 = Array1::from_vec(vec![9.0, 10.0, 11.0, 12.0]);

    let columns = vec![col1, col2, col3];
    let column_headers = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];

    // Write columns to CSV
    let columns_output_path = format!("{}/scirs2_csv_example_columns.csv", output_dir);
    write_csv_columns(&columns_output_path, &columns, Some(&column_headers), None)?;
    println!("Wrote columns to {}", columns_output_path);

    println!("\nCSV IO operations completed successfully!");
    Ok(())
}
