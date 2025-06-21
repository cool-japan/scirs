//! Streaming and iterator interfaces example
//!
//! This example demonstrates the new streaming and iterator interfaces
//! for memory-efficient processing of large datasets.

use scirs2_io::streaming::{
    process_csv_chunked, process_file_chunked, ChunkedReader, LineChunkedReader, StreamingConfig,
    StreamingCsvReader,
};
use std::io::Write;
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Streaming and Iterator Interfaces Example");
    println!("============================================");

    // Demonstrate chunked file reading
    demonstrate_chunked_reading()?;

    // Demonstrate line-based reading
    demonstrate_line_reading()?;

    // Demonstrate streaming CSV processing
    demonstrate_csv_streaming()?;

    // Demonstrate large file processing with statistics
    demonstrate_large_file_processing()?;

    // Demonstrate parallel processing with iterators
    demonstrate_parallel_streaming()?;

    println!("\n✅ All streaming demonstrations completed successfully!");
    println!("💡 Streaming interfaces enable memory-efficient processing of large datasets");

    Ok(())
}

fn demonstrate_chunked_reading() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📁 Demonstrating Chunked File Reading...");

    let temp_dir = tempdir()?;
    let test_file = temp_dir.path().join("large_data.bin");

    // Create a large binary file (1MB)
    println!("  📝 Creating test file (1MB)...");
    let mut file = std::fs::File::create(&test_file)?;
    let chunk_data = vec![42u8; 1024]; // 1KB of data
    for i in 0..1024 {
        // Write 1KB chunks with different patterns
        let mut pattern_data = chunk_data.clone();
        pattern_data[0] = (i % 256) as u8;
        file.write_all(&pattern_data)?;
    }
    file.flush()?;

    println!("  🔄 Reading file in 64KB chunks...");
    let config = StreamingConfig::new()
        .chunk_size(64 * 1024) // 64KB chunks
        .buffer_size(8 * 1024); // 8KB buffer

    let reader = ChunkedReader::new(&test_file, config)?;

    let mut total_bytes = 0;
    let mut chunk_count = 0;
    let mut unique_patterns = std::collections::HashSet::new();

    for chunk_result in reader {
        let chunk = chunk_result?;
        total_bytes += chunk.len();
        chunk_count += 1;

        // Analyze first byte of each 1KB sub-chunk to find patterns
        for sub_chunk in chunk.chunks(1024) {
            if !sub_chunk.is_empty() {
                unique_patterns.insert(sub_chunk[0]);
            }
        }
    }

    println!("     Total bytes read: {}", total_bytes);
    println!("     Number of chunks: {}", chunk_count);
    println!("     Unique patterns found: {}", unique_patterns.len());
    println!(
        "     Average chunk size: {:.1} KB",
        total_bytes as f64 / chunk_count as f64 / 1024.0
    );

    Ok(())
}

fn demonstrate_line_reading() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📄 Demonstrating Line-based Reading...");

    let temp_dir = tempdir()?;
    let test_file = temp_dir.path().join("large_text.txt");

    // Create a large text file with numbered lines
    println!("  📝 Creating test file with 10,000 lines...");
    let mut file = std::fs::File::create(&test_file)?;
    for i in 0..10000 {
        writeln!(
            file,
            "This is line number {} with some additional text to make it longer",
            i
        )?;
    }
    file.flush()?;

    println!("  🔄 Reading file in chunks of 500 lines...");
    let config = StreamingConfig::new()
        .chunk_size(500)     // 500 lines per chunk
        .buffer_size(16384); // 16KB buffer

    let reader = LineChunkedReader::new(&test_file, config)?;

    let mut total_lines = 0;
    let mut chunk_count = 0;
    let mut longest_line = 0;
    let mut shortest_line = usize::MAX;

    for chunk_result in reader {
        let lines = chunk_result?;
        total_lines += lines.len();
        chunk_count += 1;

        // Analyze line lengths
        for line in &lines {
            longest_line = longest_line.max(line.len());
            shortest_line = shortest_line.min(line.len());
        }

        if chunk_count == 1 {
            println!("     First 3 lines of first chunk:");
            for (i, line) in lines.iter().take(3).enumerate() {
                println!("       {}: {}", i + 1, line);
            }
        }
    }

    println!("     Total lines read: {}", total_lines);
    println!("     Number of chunks: {}", chunk_count);
    println!("     Longest line: {} characters", longest_line);
    println!("     Shortest line: {} characters", shortest_line);

    Ok(())
}

fn demonstrate_csv_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Demonstrating Streaming CSV Processing...");

    let temp_dir = tempdir()?;
    let test_file = temp_dir.path().join("large_dataset.csv");

    // Create a large CSV file with scientific data
    println!("  📝 Creating test CSV with 5,000 measurement records...");
    let mut file = std::fs::File::create(&test_file)?;
    writeln!(file, "timestamp,temperature,humidity,pressure,location")?;

    for i in 0..5000 {
        let timestamp = format!(
            "2024-01-01T{:02}:{:02}:{:02}Z",
            (i / 3600) % 24,
            (i / 60) % 60,
            i % 60
        );
        let temperature = 20.0 + (i as f64 * 0.01).sin() * 10.0;
        let humidity = 50.0 + (i as f64 * 0.02).cos() * 20.0;
        let pressure = 1013.25 + (i as f64 * 0.005).sin() * 10.0;
        let location = format!("Station_{}", i % 10);

        writeln!(
            file,
            "{},{:.2},{:.1},{:.2},{}",
            timestamp, temperature, humidity, pressure, location
        )?;
    }
    file.flush()?;

    println!("  🔄 Processing CSV in chunks of 100 records...");
    let config = StreamingConfig::new()
        .chunk_size(100)     // 100 rows per chunk
        .buffer_size(8192); // 8KB buffer

    let reader = StreamingCsvReader::new(&test_file, config)?
        .with_header(true)   // Process header row
        .with_delimiter(','); // Comma delimiter

    let mut total_records = 0;
    let mut chunk_count = 0;
    let mut temperature_sum = 0.0;
    let mut station_counts = std::collections::HashMap::new();

    for chunk_result in reader.enumerate() {
        let (chunk_id, rows) = chunk_result;
        let rows = rows?;

        total_records += rows.len();
        chunk_count += 1;

        // Process each row in the chunk
        for row in &rows {
            if row.len() >= 5 {
                // Parse temperature (column 1)
                if let Ok(temp) = row[1].parse::<f64>() {
                    temperature_sum += temp;
                }

                // Count station occurrences (column 4)
                let station = &row[4];
                *station_counts.entry(station.clone()).or_insert(0) += 1;
            }
        }

        if chunk_id == 0 {
            println!("     First chunk sample:");
            for (i, row) in rows.iter().take(3).enumerate() {
                println!("       Record {}: {:?}", i + 1, row);
            }
        }
    }

    let avg_temperature = temperature_sum / total_records as f64;
    let most_common_station = station_counts
        .iter()
        .max_by_key(|(_, &count)| count)
        .map(|(station, count)| (station.clone(), *count));

    println!("     Total records processed: {}", total_records);
    println!("     Number of chunks: {}", chunk_count);
    println!("     Average temperature: {:.2}°C", avg_temperature);
    if let Some((station, count)) = most_common_station {
        println!("     Most common station: {} ({} records)", station, count);
    }

    Ok(())
}

fn demonstrate_large_file_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Demonstrating Large File Processing with Statistics...");

    let temp_dir = tempdir()?;
    let test_file = temp_dir.path().join("large_binary.dat");

    // Create a 5MB file with structured data
    println!("  📝 Creating large binary file (5MB)...");
    let mut file = std::fs::File::create(&test_file)?;
    for i in 0..5120 {
        // 5120 * 1KB = 5MB
        let mut data = vec![0u8; 1024];
        // Fill with structured data
        for (j, byte) in data.iter_mut().enumerate() {
            *byte = ((i + j) % 256) as u8;
        }
        file.write_all(&data)?;
    }
    file.flush()?;

    println!("  ⚡ Processing file with performance monitoring...");
    let config = StreamingConfig::new()
        .chunk_size(256 * 1024) // 256KB chunks
        .buffer_size(32 * 1024); // 32KB buffer

    // Process the file and calculate statistics
    let (byte_distribution, stats) = process_file_chunked(
        &test_file,
        config,
        |chunk, chunk_id| -> scirs2_io::error::Result<std::collections::HashMap<u8, usize>> {
            let mut distribution = std::collections::HashMap::new();

            // Count byte frequencies in this chunk
            for &byte in chunk {
                *distribution.entry(byte).or_insert(0) += 1;
            }

            if chunk_id % 5 == 0 {
                println!(
                    "     Processing chunk {} ({} bytes)...",
                    chunk_id,
                    chunk.len()
                );
            }

            Ok(distribution)
        },
    )?;

    // Display results
    println!("     📊 Processing Statistics:");
    println!("       {}", stats.summary());
    println!(
        "       Average chunk size: {:.1} KB",
        stats.avg_bytes_per_chunk / 1024.0
    );
    println!("       Processing speed: {:.2} MB/s", stats.avg_speed_mbps);

    println!("     📈 Data Analysis Results:");
    println!("       Unique byte values: {}", byte_distribution.len());

    // Find most common bytes
    let mut sorted_bytes: Vec<_> = byte_distribution.iter().collect();
    sorted_bytes.sort_by_key(|(_, &count)| std::cmp::Reverse(count));

    println!("       Most common bytes:");
    for (i, (&byte, &count)) in sorted_bytes.iter().take(5).enumerate() {
        println!("         {}. Byte {}: {} occurrences", i + 1, byte, count);
    }

    Ok(())
}

fn demonstrate_parallel_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Demonstrating Parallel Processing with Streaming...");

    let temp_dir = tempdir()?;
    let test_file = temp_dir.path().join("parallel_test.csv");

    // Create a CSV file suitable for parallel processing
    println!("  📝 Creating CSV file for parallel processing (2,000 records)...");
    let mut file = std::fs::File::create(&test_file)?;
    writeln!(file, "id,value1,value2,value3,result")?;

    for i in 0..2000 {
        let value1 = (i as f64 * 0.1).sin();
        let value2 = (i as f64 * 0.2).cos();
        let value3 = (i as f64 * 0.05).tan();
        writeln!(file, "{},{:.6},{:.6},{:.6},", i, value1, value2, value3)?;
    }
    file.flush()?;

    println!("  🚀 Processing with parallel computation...");
    let config = StreamingConfig::new()
        .chunk_size(50)      // 50 rows per chunk
        .buffer_size(4096); // 4KB buffer

    let (results, stats) = process_csv_chunked(
        &test_file,
        config,
        true, // Has header
        |rows, chunk_id, _header| -> scirs2_io::error::Result<Vec<f64>> {
            // Simulate parallel processing within chunk
            use rayon::prelude::*;

            let chunk_results: Vec<f64> = rows
                .par_iter()
                .filter_map(|row| {
                    if row.len() >= 4 {
                        // Parse numeric values and compute result
                        let val1: f64 = row[1].parse().ok()?;
                        let val2: f64 = row[2].parse().ok()?;
                        let val3: f64 = row[3].parse().ok()?;

                        // Complex computation that benefits from parallelization
                        let result = (val1.powi(2) + val2.powi(2) + val3.powi(2)).sqrt();
                        Some(result)
                    } else {
                        None
                    }
                })
                .collect();

            if chunk_id < 3 {
                println!(
                    "     Chunk {} processed {} rows in parallel",
                    chunk_id,
                    chunk_results.len()
                );
            }

            Ok(chunk_results)
        },
    )?;

    // Aggregate results
    let total_computed = results.len();
    let sum: f64 = results.iter().sum();
    let avg = sum / total_computed as f64;
    let max_val = results.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = results.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    println!("     📊 Parallel Processing Results:");
    println!("       {}", stats.summary());
    println!("       Values computed: {}", total_computed);
    println!("       Average result: {:.6}", avg);
    println!("       Min result: {:.6}", min_val);
    println!("       Max result: {:.6}", max_val);

    Ok(())
}
