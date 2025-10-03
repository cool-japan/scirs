//! Async I/O example
//!
//! This example demonstrates the new async I/O capabilities
//! for non-blocking processing of large datasets.

use futures::StreamExt;
use scirs2_io::async_io::{
    process_csv_async, process_file_async, AsyncChunkedReader, AsyncLineReader,
    AsyncStreamingConfig, CancellationToken,
};
use std::io::Write;
use tempfile::tempdir;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Async I/O Example");
    println!("==================");

    // Demonstrate basic async chunked reading
    demonstrate_async_chunked_reading().await?;

    // Demonstrate async line reading
    demonstrate_async_line_reading().await?;

    // Demonstrate concurrent processing
    demonstrate_concurrent_processing().await?;

    // Demonstrate cancellation
    demonstrate_cancellation().await?;

    // Demonstrate async CSV processing
    demonstrate_async_csv_processing().await?;

    println!("\n✅ All async I/O demonstrations completed successfully!");
    println!("💡 Async I/O enables non-blocking, concurrent processing of large datasets");

    Ok(())
}

async fn demonstrate_async_chunked_reading() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Demonstrating Async Chunked Reading...");

    let temp_dir = tempdir()?;
    let test_file = temp_dir.path().join("async_test.bin");

    // Create a test file with 500KB of data
    println!("  📝 Creating test file (500KB)...");
    let mut file = std::fs::File::create(&test_file)?;
    let chunk_data = vec![42u8; 1024]; // 1KB chunks
    for i in 0..500 {
        let mut pattern_data = chunk_data.clone();
        pattern_data[0] = (i % 256) as u8;
        file.write_all(&pattern_data)?;
    }
    file.flush()?;

    println!("  ⚡ Reading file asynchronously in 32KB chunks...");
    let config = AsyncStreamingConfig::new()
        .chunk_size(32 * 1024) // 32KB chunks
        .timeout(1000); // 1 second timeout

    let mut reader = AsyncChunkedReader::new(&test_file, config).await?;

    let mut total_bytes = 0;
    let mut chunk_count = 0;
    let start_time = std::time::Instant::now();

    // Process chunks asynchronously
    while let Some(chunk_result) = reader.read_next_chunk().await? {
        total_bytes += chunk_result.len();
        chunk_count += 1;

        // Simulate some async processing
        if chunk_count % 5 == 0 {
            tokio::task::yield_now().await; // Yield to allow other tasks
        }
    }

    let elapsed = start_time.elapsed();
    println!("     Total bytes read: {}", total_bytes);
    println!("     Number of chunks: {}", chunk_count);
    println!(
        "     Processing time: {:.2}ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "     Speed: {:.2} MB/s",
        (total_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64()
    );

    Ok(())
}

async fn demonstrate_async_line_reading() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📄 Demonstrating Async Line Reading...");

    let temp_dir = tempdir()?;
    let test_file = temp_dir.path().join("async_lines.txt");

    // Create a test file with many lines
    println!("  📝 Creating test file with 2,000 lines...");
    let mut file = std::fs::File::create(&test_file)?;
    for i in 0..2000 {
        writeln!(
            file,
            "This is async line number {} with timestamp {}",
            i,
            chrono::Utc::now().timestamp_millis()
        )?;
    }
    file.flush()?;

    println!("  ⚡ Reading lines asynchronously in batches of 100...");
    let config = AsyncStreamingConfig::new()
        .chunk_size(100)        // 100 lines per chunk
        .timeout(500); // 500ms timeout

    let mut reader = AsyncLineReader::new(&test_file, config).await?;

    let mut total_lines = 0;
    let mut batch_count = 0;
    let start_time = std::time::Instant::now();

    // Process line batches asynchronously using Stream trait
    while let Some(lines_result) = reader.next().await {
        let lines = lines_result?;
        total_lines += lines.len();
        batch_count += 1;

        if batch_count == 1 {
            println!("     First 3 lines of first batch:");
            for (i, line) in lines.iter().take(3).enumerate() {
                println!("       {}: {}", i + 1, line);
            }
        }

        // Simulate async processing
        tokio::task::yield_now().await;
    }

    let elapsed = start_time.elapsed();
    println!("     Total lines read: {}", total_lines);
    println!("     Number of batches: {}", batch_count);
    println!(
        "     Processing time: {:.2}ms",
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(())
}

async fn demonstrate_concurrent_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Demonstrating Concurrent Processing...");

    let temp_dir = tempdir()?;
    let test_file = temp_dir.path().join("concurrent_test.dat");

    // Create a larger test file for concurrent processing
    println!("  📝 Creating test file (2MB) for concurrent processing...");
    let mut file = std::fs::File::create(&test_file)?;
    for i in 0..2048 {
        // 2MB total
        let data: Vec<u8> = (0..1024).map(|j| ((i + j) % 256) as u8).collect();
        file.write_all(&data)?;
    }
    file.flush()?;

    println!("  🚀 Processing with 4 concurrent workers...");
    let config = AsyncStreamingConfig::new()
        .chunk_size(64 * 1024)  // 64KB chunks
        .concurrency(4)         // 4 concurrent workers
        .timeout(2000); // 2 second timeout

    let start_time = std::time::Instant::now();

    // Process file with concurrent chunk processing
    let (results, stats) = process_file_async(&test_file, config, |chunk, chunk_id| async move {
        // Simulate computation-intensive async work
        let mut checksum = 0u64;
        for &byte in &chunk {
            checksum = checksum.wrapping_add(byte as u64);

            // Occasionally yield to demonstrate async behavior
            if checksum.is_multiple_of(10000) {
                tokio::task::yield_now().await;
            }
        }

        // Simulate additional async work
        sleep(Duration::from_millis(10)).await;

        Ok((chunk_id, checksum, chunk.len()))
    })
    .await?;

    let elapsed = start_time.elapsed();

    // Analyze results
    let total_chunks = results.len();
    let total_checksum: u64 = results.iter().map(|(_, checksum, _)| *checksum).sum();
    let total_bytes: usize = results.iter().map(|(_, _, size)| *size).sum();

    println!("     📊 Concurrent Processing Results:");
    println!("       {}", stats.summary());
    println!("       Total chunks processed: {}", total_chunks);
    println!("       Total bytes processed: {}", total_bytes);
    println!("       Combined checksum: {}", total_checksum);
    println!(
        "       Wall clock time: {:.2}ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "       Effective speedup: {:.2}x",
        stats.processing_time_ms / (elapsed.as_secs_f64() * 1000.0)
    );

    Ok(())
}

async fn demonstrate_cancellation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⏹️  Demonstrating Cancellation Support...");

    let temp_dir = tempdir()?;
    let test_file = temp_dir.path().join("cancellation_test.dat");

    // Create a test file
    println!("  📝 Creating test file for cancellation demo...");
    let mut file = std::fs::File::create(&test_file)?;
    for i in 0..1000 {
        let data = vec![i as u8; 1024]; // 1MB total
        file.write_all(&data)?;
    }
    file.flush()?;

    println!("  ⏹️  Starting async processing with cancellation...");
    let config = AsyncStreamingConfig::new()
        .chunk_size(32 * 1024)  // 32KB chunks
        .concurrency(2);

    let token = CancellationToken::new();
    let token_clone = token.clone();

    // Start a task that will cancel the operation after a delay
    let cancel_task = tokio::spawn(async move {
        sleep(Duration::from_millis(100)).await;
        println!("     ⏹️  Cancelling operation...");
        token_clone.cancel();
    });

    // Process file with cancellation support
    let start_time = std::time::Instant::now();
    let mut reader = AsyncChunkedReader::new(&test_file, config).await?;
    let mut chunks_processed = 0;
    let mut bytes_processed = 0;

    while let Some(chunk_result) = reader.read_next_chunk().await? {
        // Check for cancellation
        if token.is_cancelled() {
            println!("     ⏹️  Operation cancelled!");
            break;
        }

        let chunk = chunk_result;
        bytes_processed += chunk.len();
        chunks_processed += 1;

        // Simulate some processing time
        sleep(Duration::from_millis(20)).await;
    }

    cancel_task.await?;
    let elapsed = start_time.elapsed();

    println!("     📊 Cancellation Results:");
    println!(
        "       Chunks processed before cancellation: {}",
        chunks_processed
    );
    println!("       Bytes processed: {}", bytes_processed);
    println!(
        "       Time before cancellation: {:.2}ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!("       Successfully demonstrated graceful cancellation!");

    Ok(())
}

async fn demonstrate_async_csv_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Demonstrating Async CSV Processing...");

    let temp_dir = tempdir()?;
    let test_file = temp_dir.path().join("async_data.csv");

    // Create a CSV file for async processing
    println!("  📝 Creating CSV file for async processing (1,000 records)...");
    let mut file = std::fs::File::create(&test_file)?;
    writeln!(file, "id,value,category,score")?;

    for i in 0..1000 {
        let value = (i as f64 * 0.1).sin() * 100.0;
        let category = match i % 4 {
            0 => "A",
            1 => "B",
            2 => "C",
            _ => "D",
        };
        let score = 50.0 + (i as f64 * 0.05).cos() * 30.0;
        writeln!(file, "{},{:.2},{},{:.1}", i, value, category, score)?;
    }
    file.flush()?;

    println!("  ⚡ Processing CSV asynchronously with concurrent workers...");
    let config = AsyncStreamingConfig::new()
        .chunk_size(1)          // Process 1 line at a time for fine granularity
        .concurrency(3)         // 3 concurrent workers
        .timeout(1000);

    let start_time = std::time::Instant::now();

    // Process CSV with async processing
    let (results, stats) = process_csv_async(&test_file, config, |lines, line_id| async move {
        if let Some(line) = lines.first() {
            // Parse CSV line (simple parsing)
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() >= 4 {
                // Simulate async computation
                sleep(Duration::from_millis(1)).await;

                let id: i32 = fields[0].parse().unwrap_or(0);
                let value: f64 = fields[1].parse().unwrap_or(0.0);
                let category = fields[2].to_string();
                let score: f64 = fields[3].parse().unwrap_or(0.0);

                // Compute derived metrics
                let adjusted_score = score + value.abs() * 0.1;

                return Ok((id, category, adjusted_score));
            }
        }

        Ok((line_id as i32, "unknown".to_string(), 0.0))
    })
    .await?;

    let elapsed = start_time.elapsed();

    // Analyze results
    let mut category_stats = std::collections::HashMap::new();
    let mut total_score = 0.0;

    for (_, category, score) in &results {
        *category_stats.entry(category.clone()).or_insert(0) += 1;
        total_score += score;
    }

    let avg_score = total_score / results.len() as f64;

    println!("     📊 Async CSV Processing Results:");
    println!("       {}", stats.summary());
    println!("       Records processed: {}", results.len());
    println!("       Average adjusted score: {:.2}", avg_score);
    println!("       Category distribution:");
    for (category, count) in category_stats {
        println!("         {}: {} records", category, count);
    }
    println!(
        "       Wall clock time: {:.2}ms",
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(())
}
