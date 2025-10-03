use scirs2_core::ndarray::Array2;
use scirs2_io::csv::{
    read_csv_chunked, read_csv_typed, write_csv_typed, ColumnType, DataValue, MissingValueOptions,
};
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Advanced CSV Functions Example ===\n");

    // Create sample data with mixed types and missing values
    let row1 = vec![
        DataValue::String("Alice".to_string()),
        DataValue::Integer(25),
        DataValue::Float(168.5),
        DataValue::Boolean(true),
        DataValue::Missing,
    ];
    let row2 = vec![
        DataValue::String("Bob".to_string()),
        DataValue::Integer(32),
        DataValue::Float(175.0),
        DataValue::Boolean(false),
        DataValue::Float(72.8),
    ];
    let row3 = vec![
        DataValue::String("Charlie".to_string()),
        DataValue::Integer(41),
        DataValue::Missing,
        DataValue::Boolean(true),
        DataValue::Float(90.5),
    ];

    let data = vec![row1, row2, row3];
    let headers = vec![
        "Name".to_string(),
        "Age".to_string(),
        "Height".to_string(),
        "Active".to_string(),
        "Weight".to_string(),
    ];

    // Write typed data to a file
    println!("Writing typed data to CSV file...");
    write_csv_typed(
        "scirs2-io/examples/typed_data.csv",
        &data,
        Some(&headers),
        None,
    )?;
    println!("Data written to scirs2-io/examples/typed_data.csv");

    // Read the typed data back with automatic type detection
    println!("\nReading typed data with automatic type detection...");
    let (read_headers, read_data) =
        read_csv_typed("scirs2-io/examples/typed_data.csv", None, None, None)?;

    println!("Headers: {:?}", read_headers);
    println!("First row: {:?}", read_data[0]);

    // Read data with custom missing value handling
    println!("\nReading with custom missing value handling...");
    let missing_opts = MissingValueOptions {
        values: vec!["NA".to_string(), "N/A".to_string(), "missing".to_string()],
        fill_value: Some(0.0),
    };

    let column_types = vec![
        ColumnType::String,
        ColumnType::Integer,
        ColumnType::Float,
        ColumnType::Boolean,
        ColumnType::Float,
    ];

    let (_, read_data_filled) = read_csv_typed(
        "scirs2-io/examples/typed_data.csv",
        None,
        Some(&column_types),
        Some(missing_opts),
    )?;

    println!(
        "First row with missing value handling: {:?}",
        read_data_filled[0]
    );
    println!(
        "Third row with missing value handling: {:?}",
        read_data_filled[2]
    );

    // Create a larger dataset for chunked reading demonstration
    println!("\nCreating a larger dataset for chunked reading...");
    let mut large_data = Array2::<String>::from_elem((1000, 3), String::new());

    for i in 0..1000 {
        large_data[[i, 0]] = format!("Row{}", i);
        large_data[[i, 1]] = i.to_string();
        large_data[[i, 2]] = format!("{:.1}", i as f64 / 10.0);
    }

    let large_headers = vec!["ID".to_string(), "Value".to_string(), "Float".to_string()];

    // Write the large dataset
    println!("Writing large dataset (1000 rows)...");
    scirs2_io::csv::write_csv(
        "scirs2-io/examples/large_data.csv",
        &large_data,
        Some(&large_headers),
        None,
    )?;

    // Read the large dataset in chunks
    println!("\nReading large dataset in chunks...");
    let mut total_rows = 0;
    let mut chunks_processed = 0;

    read_csv_chunked(
        "scirs2-io/examples/large_data.csv",
        None,
        200,
        |_headers, chunk| {
            chunks_processed += 1;
            total_rows += chunk.shape()[0];
            println!(
                "Processing chunk {}: {} rows (total processed: {})",
                chunks_processed,
                chunk.shape()[0],
                total_rows
            );
            true // continue processing
        },
    )?;

    println!("\nTotal rows processed: {}", total_rows);
    println!("Total chunks processed: {}", chunks_processed);

    println!("\nAdvanced CSV example completed successfully!");
    Ok(())
}
