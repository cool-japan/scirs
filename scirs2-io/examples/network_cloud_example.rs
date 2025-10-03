//! Network I/O and cloud storage integration example
//!
//! This example demonstrates the network and cloud storage capabilities including:
//! - HTTP/HTTPS file download and upload with progress tracking
//! - Cloud storage integration with AWS S3, Google Cloud Storage, and Azure
//! - Streaming I/O for efficient handling of large files over network
//! - Authentication and credential management
//! - Local caching for offline access and performance optimization
//! - Retry logic and error recovery for network operations

use scirs2_io::network::cloud::{
    create_mock_metadata, validate_config, AzureConfig, CloudProvider, GcsConfig, S3Config,
};
use scirs2_io::network::http::{
    calculate_speed, format_file_size, format_speed, HttpClient, HttpMethod,
};
use scirs2_io::network::streaming::{
    copy_with_progress, ChunkedReader, ChunkedWriter, StreamConfig, StreamProgress,
};
#[cfg(feature = "reqwest")]
use scirs2_io::network::{batch_download, download_file, upload_file};
use scirs2_io::network::{
    batch_upload_to_cloud, create_cloud_client, NetworkClient, NetworkConfig,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tempfile::tempdir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 Network I/O and Cloud Storage Integration Example");
    println!("===================================================");

    // Demonstrate network configuration
    demonstrate_network_configuration().await?;

    // Demonstrate HTTP operations
    demonstrate_http_operations().await?;

    // Demonstrate cloud storage configuration
    demonstrate_cloud_storage_config().await?;

    // Demonstrate streaming operations
    demonstrate_streaming_operations().await?;

    // Demonstrate batch operations
    demonstrate_batch_operations().await?;

    // Demonstrate caching and offline access
    demonstrate_caching_operations().await?;

    println!("\n✅ All network and cloud storage demonstrations completed successfully!");
    println!("💡 Key benefits of the network I/O system:");
    println!("   - Unified interface for HTTP and cloud storage operations");
    println!("   - Streaming support for memory-efficient large file handling");
    println!("   - Robust error handling with automatic retry mechanisms");
    println!("   - Progress tracking and performance monitoring");
    println!("   - Local caching for improved performance and offline access");
    println!("   - Support for multiple cloud providers with consistent API");

    Ok(())
}

async fn demonstrate_network_configuration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚙️  Demonstrating Network Configuration...");

    // Create custom network configuration
    println!("  🔹 Creating custom network configuration:");
    let mut headers = HashMap::new();
    headers.insert("Authorization".to_string(), "Bearer demo-token".to_string());
    headers.insert(
        "X-Custom-Header".to_string(),
        "scirs2-io-example".to_string(),
    );

    let config = NetworkConfig {
        connect_timeout: Duration::from_secs(10),
        read_timeout: Duration::from_secs(120),
        max_retries: 5,
        user_agent: "scirs2-io-example/1.0".to_string(),
        headers,
        compression: true,
        cache_dir: Some("/tmp/scirs2_cache".to_string()),
        max_cache_size: 512, // 512MB
    };

    println!("    Connect timeout: {:?}", config.connect_timeout);
    println!("    Read timeout: {:?}", config.read_timeout);
    println!("    Max retries: {}", config.max_retries);
    println!("    User agent: {}", config.user_agent);
    println!("    Compression enabled: {}", config.compression);
    println!("    Cache directory: {:?}", config.cache_dir);
    println!("    Max cache size: {} MB", config.max_cache_size);
    println!("    Custom headers: {} items", config.headers.len());

    // Create network client with custom config
    let temp_dir = tempdir()?;
    let cache_path = temp_dir.path().join("cache");
    let client = NetworkClient::with_config(config).with_cache_dir(&cache_path);

    // Test cache operations
    println!("  🔹 Testing cache operations:");
    let (cache_size, cache_files) = client.get_cache_info()?;
    println!(
        "    Initial cache: {} bytes, {} files",
        cache_size, cache_files
    );

    client.clear_cache()?;
    let (cache_size_after, cache_files_after) = client.get_cache_info()?;
    println!(
        "    After clear: {} bytes, {} files",
        cache_size_after, cache_files_after
    );

    Ok(())
}

async fn demonstrate_http_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌍 Demonstrating HTTP Operations...");

    // Test HTTP client creation and configuration
    println!("  🔹 Creating HTTP client:");
    let config = NetworkConfig::default();
    let mut http_client = HttpClient::new(config);

    // Initialize client (only works with reqwest feature)
    println!("  🔹 Testing HTTP functionality:");
    #[cfg(feature = "reqwest")]
    {
        http_client.init()?;
        println!("    HTTP client initialized successfully");

        // Test URL accessibility check
        let test_urls = vec![
            "https://httpbin.org/status/200",
            "https://httpbin.org/status/404",
            "https://example.com",
        ];

        for url in test_urls {
            match http_client.check_url(url).await {
                Ok(accessible) => println!("    URL {} accessible: {}", url, accessible),
                Err(e) => println!("    URL {} check failed: {}", url, e),
            }
        }

        // Test getting remote file size
        match http_client
            .get_remote_file_size("https://httpbin.org/bytes/1024")
            .await
        {
            Ok(Some(size)) => println!("    Remote file size: {} bytes", size),
            Ok(None) => println!("    Remote file size: unknown"),
            Err(e) => println!("    Failed to get remote file size: {}", e),
        }

        // Test custom HTTP request
        match http_client
            .request(HttpMethod::GET, "https://httpbin.org/json", None)
            .await
        {
            Ok(response) => {
                println!("    HTTP GET response: status {}", response.status);
                println!("    Response headers: {} items", response.headers.len());
                println!("    Response body: {} bytes", response.body.len());
            }
            Err(e) => println!("    HTTP request failed: {}", e),
        }
    }

    #[cfg(not(feature = "reqwest"))]
    {
        println!("    HTTP functionality requires 'reqwest' feature");

        // Test that functions return appropriate errors
        match http_client.check_url("https://example.com").await {
            Ok(_) => println!("    Unexpected success"),
            Err(e) => println!("    Expected error: {}", e),
        }
    }

    // Test utility functions
    println!("  🔹 Testing utility functions:");
    let file_sizes = vec![512, 1024, 1536, 1024 * 1024, 1024 * 1024 * 1024];
    for size in file_sizes {
        println!("    {} bytes = {}", size, format_file_size(size));
    }

    let speeds = vec![1024.0, 1024.0 * 1024.0, 1024.0 * 1024.0 * 1024.0];
    for speed in speeds {
        println!("    {:.0} bytes/sec = {}", speed, format_speed(speed));
    }

    let duration = Duration::from_secs(2);
    let calculated_speed = calculate_speed(2048, duration);
    println!(
        "    Speed calculation: 2048 bytes in 2 seconds = {:.1} bytes/sec",
        calculated_speed
    );

    Ok(())
}

async fn demonstrate_cloud_storage_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n☁️  Demonstrating Cloud Storage Configuration...");

    // AWS S3 configuration
    println!("  🔹 AWS S3 configuration:");
    let s3_config = S3Config::new(
        "demo-bucket",
        "us-east-1",
        "demo-access-key",
        "demo-secret-key",
    )
    .with_endpoint("http://localhost:9000")
    .with_path_style(true);

    println!("    Bucket: {}", s3_config.bucket);
    println!("    Region: {}", s3_config.region);
    println!("    Endpoint: {:?}", s3_config.endpoint);
    println!("    Path style: {}", s3_config.path_style);

    let s3_provider = CloudProvider::S3(s3_config);
    match validate_config(&s3_provider) {
        Ok(_) => println!("    S3 configuration valid ✅"),
        Err(e) => println!("    S3 configuration error: {}", e),
    }

    // Google Cloud Storage configuration
    println!("  🔹 Google Cloud Storage configuration:");
    let gcs_config = GcsConfig::new("demo-bucket", "demo-project")
        .with_credentials_file("/path/to/service-account.json")
        .with_credentials_json(r#"{"type": "service_account"}"#);

    println!("    Bucket: {}", gcs_config.bucket);
    println!("    Project ID: {}", gcs_config.project_id);
    println!("    Credentials file: {:?}", gcs_config.credentials_path);
    println!(
        "    Has credentials JSON: {}",
        gcs_config.credentials_json.is_some()
    );

    let gcs_provider = CloudProvider::GCS(gcs_config);
    match validate_config(&gcs_provider) {
        Ok(_) => println!("    GCS configuration valid ✅"),
        Err(e) => println!("    GCS configuration error: {}", e),
    }

    // Azure Blob Storage configuration
    println!("  🔹 Azure Blob Storage configuration:");
    let azure_config = AzureConfig::new("demoaccount", "democontainer", "demo-access-key")
        .with_endpoint("http://localhost:10000");

    println!("    Account: {}", azure_config.account);
    println!("    Container: {}", azure_config.container);
    println!("    Endpoint: {:?}", azure_config.endpoint);

    let azure_provider = CloudProvider::Azure(azure_config);
    match validate_config(&azure_provider) {
        Ok(_) => println!("    Azure configuration valid ✅"),
        Err(e) => println!("    Azure configuration error: {}", e),
    }

    // Test cloud client creation
    println!("  🔹 Creating cloud clients:");
    let s3_client = create_cloud_client(s3_provider);
    println!("    S3 client created");

    let gcs_client = create_cloud_client(gcs_provider);
    println!("    GCS client created");

    let azure_client = create_cloud_client(azure_provider);
    println!("    Azure client created");

    // Test cloud operations (these will return feature errors without cloud SDKs)
    println!("  🔹 Testing cloud operations:");
    match s3_client.cloud_file_exists("test-file.txt").await {
        Ok(exists) => println!("    S3 file exists: {}", exists),
        Err(e) => println!("    S3 operation error (expected): {}", e),
    }

    match gcs_client.list_cloud_files("path/").await {
        Ok(files) => println!("    GCS files found: {}", files.len()),
        Err(e) => println!("    GCS operation error (expected): {}", e),
    }

    match azure_client.get_cloud_file_metadata("test.txt").await {
        Ok(metadata) => println!("    Azure metadata: {} bytes", metadata.size),
        Err(e) => println!("    Azure operation error (expected): {}", e),
    }

    // Test metadata utilities
    println!("  🔹 Testing metadata utilities:");
    let mock_metadata = create_mock_metadata("demo-file.dat", 1024 * 1024);
    println!("    Mock metadata: {}", mock_metadata.name);
    println!("    Size: {} bytes", mock_metadata.size);
    println!("    Content type: {:?}", mock_metadata.content_type);
    println!("    ETag: {:?}", mock_metadata.etag);

    Ok(())
}

async fn demonstrate_streaming_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📡 Demonstrating Streaming Operations...");

    let temp_dir = tempdir()?;

    // Test streaming configuration
    println!("  🔹 Streaming configuration:");
    let stream_config = StreamConfig {
        buffer_size: 32 * 1024,      // 32KB chunks
        max_memory: 8 * 1024 * 1024, // 8MB max buffer
        compression: false,
        progress_interval: 512 * 1024, // Report every 512KB
    };

    println!("    Buffer size: {} bytes", stream_config.buffer_size);
    println!("    Max memory: {} bytes", stream_config.max_memory);
    println!("    Compression: {}", stream_config.compression);
    println!(
        "    Progress interval: {} bytes",
        stream_config.progress_interval
    );

    // Create test data file
    println!("  🔹 Creating test data for streaming:");
    let test_file = temp_dir.path().join("streaming_test.dat");
    let test_data = vec![42u8; 1024 * 1024]; // 1MB of test data
    std::fs::write(&test_file, &test_data)?;
    println!("    Created {} byte test file", test_data.len());

    // Test chunked reader
    println!("  🔹 Testing chunked reader:");
    let mut chunked_reader = ChunkedReader::new(&test_file, 64 * 1024)?;
    println!("    File size: {} bytes", chunked_reader.size());

    let mut chunks_read = 0;
    let mut total_read = 0;
    while let Some(chunk) = chunked_reader.read_chunk()? {
        chunks_read += 1;
        total_read += chunk.len();
        if chunks_read <= 3 {
            println!("    Chunk {}: {} bytes", chunks_read, chunk.len());
        }
    }
    println!(
        "    Total chunks: {}, Total bytes: {}",
        chunks_read, total_read
    );
    println!("    Progress: {:.1}%", chunked_reader.progress_percentage());

    // Test chunked writer
    println!("  🔹 Testing chunked writer:");
    let output_file = temp_dir.path().join("chunked_output.dat");
    let mut chunked_writer = ChunkedWriter::new(&output_file, 32 * 1024)?;

    let chunks = [vec![1u8; 10000], vec![2u8; 15000], vec![3u8; 20000]];

    for (i, chunk) in chunks.iter().enumerate() {
        chunked_writer.write_chunk(chunk)?;
        println!("    Wrote chunk {}: {} bytes", i + 1, chunk.len());
    }

    let total_written = chunked_writer.finish()?;
    println!("    Total written: {} bytes", total_written);

    // Test progress tracking
    println!("  🔹 Testing progress tracking:");
    let input_file = std::fs::File::open(&test_file)?;
    let output_file_path = temp_dir.path().join("progress_output.dat");
    let output_file = std::fs::File::create(&output_file_path)?;

    let progress_callback = Box::new(|progress: StreamProgress| {
        if let Some(percentage) = progress.percentage() {
            println!(
                "    Progress: {:.1}% ({} bytes, {:.1} KB/s, ETA: {:.1}s)",
                percentage,
                progress.bytes_transferred,
                progress.rate / 1024.0,
                progress.eta_seconds.unwrap_or(0.0)
            );
        } else {
            println!(
                "    Progress: {} bytes, {:.1} KB/s",
                progress.bytes_transferred,
                progress.rate / 1024.0
            );
        }
    });

    let start_time = Instant::now();
    let copied_bytes = copy_with_progress(
        input_file,
        output_file,
        Some(test_data.len() as u64),
        Some(progress_callback),
    )?;
    let copy_time = start_time.elapsed();

    println!(
        "    Copied {} bytes in {:.2}ms",
        copied_bytes,
        copy_time.as_secs_f64() * 1000.0
    );

    // Verify copied file
    let copied_data = std::fs::read(&output_file_path)?;
    println!(
        "    Verification: {} bytes copied correctly",
        if copied_data == test_data {
            "✅"
        } else {
            "❌"
        }
    );

    Ok(())
}

async fn demonstrate_batch_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📦 Demonstrating Batch Operations...");

    // Test batch download preparation (URLs would be real in practice)
    println!("  🔹 Preparing batch download operations:");
    let download_tasks = vec![
        ("https://httpbin.org/bytes/1024", "file1.dat"),
        ("https://httpbin.org/bytes/2048", "file2.dat"),
        ("https://httpbin.org/bytes/4096", "file3.dat"),
    ];

    println!("    Prepared {} download tasks", download_tasks.len());
    for (url, local_file) in &download_tasks {
        println!("      {} -> {}", url, local_file);
    }

    // Test batch download (only if reqwest feature is available)
    #[cfg(feature = "reqwest")]
    {
        println!("  🔹 Testing batch download:");
        let download_results = batch_download(download_tasks.to_vec()).await?;

        let mut successful = 0;
        let mut failed = 0;
        for (i, result) in download_results.iter().enumerate() {
            match result {
                Ok(_) => {
                    successful += 1;
                    println!("    Download {}: ✅ Success", i + 1);
                }
                Err(e) => {
                    failed += 1;
                    println!("    Download {}: ❌ Failed: {}", i + 1, e);
                }
            }
        }
        println!(
            "    Batch download summary: {} successful, {} failed",
            successful, failed
        );
    }

    #[cfg(not(feature = "reqwest"))]
    {
        println!("  🔹 Batch download requires 'reqwest' feature");
    }

    // Test batch cloud upload preparation
    println!("  🔹 Preparing batch cloud upload operations:");
    let temp_dir = tempdir()?;

    // Create test files
    let upload_files = vec![
        ("test1.txt", "Hello, cloud!"),
        ("test2.txt", "Cloud storage test data"),
        ("test3.txt", "Batch upload demonstration"),
    ];

    for (filename, content) in &upload_files {
        let file_path = temp_dir.path().join(filename);
        std::fs::write(&file_path, content)?;
        println!("    Created: {} ({} bytes)", filename, content.len());
    }

    // Test batch cloud upload (using mock cloud client)
    println!("  🔹 Testing batch cloud upload:");
    let s3_config = S3Config::new("demo-bucket", "us-east-1", "demo-key", "demo-secret");
    let cloud_client = create_cloud_client(CloudProvider::S3(s3_config));

    let upload_tasks: Vec<(String, String)> = upload_files
        .iter()
        .map(|(filename, _content)| {
            let local_path = temp_dir.path().join(filename).to_string_lossy().to_string();
            (local_path, format!("uploads/{}", filename))
        })
        .collect();

    let upload_task_refs: Vec<(&str, &str)> = upload_tasks
        .iter()
        .map(|(local, remote)| (local.as_str(), remote.as_str()))
        .collect();

    // This will show feature errors since cloud SDKs are not enabled
    let upload_results = batch_upload_to_cloud(&cloud_client, upload_task_refs).await?;

    let mut upload_successful = 0;
    let mut upload_failed = 0;
    for (i, result) in upload_results.iter().enumerate() {
        match result {
            Ok(_) => {
                upload_successful += 1;
                println!("    Upload {}: ✅ Success", i + 1);
            }
            Err(e) => {
                upload_failed += 1;
                println!("    Upload {}: ❌ Failed (expected): {}", i + 1, e);
            }
        }
    }
    println!(
        "    Batch upload summary: {} successful, {} failed",
        upload_successful, upload_failed
    );

    Ok(())
}

async fn demonstrate_caching_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💾 Demonstrating Caching and Offline Access...");

    let temp_dir = tempdir()?;
    let cache_dir = temp_dir.path().join("cache");

    // Create network client with caching
    println!("  🔹 Setting up caching configuration:");
    let client = NetworkClient::new().with_cache_dir(&cache_dir);

    // Test cache operations
    println!("  🔹 Testing cache operations:");
    let (initial_size, initial_count) = client.get_cache_info()?;
    println!(
        "    Initial cache: {} bytes, {} files",
        initial_size, initial_count
    );

    // Simulate adding files to cache
    std::fs::create_dir_all(&cache_dir)?;
    let cached_files = vec![
        ("cached_file1.dat", vec![1u8; 1024]),
        ("cached_file2.dat", vec![2u8; 2048]),
        ("cached_file3.dat", vec![3u8; 4096]),
    ];

    for (filename, data) in &cached_files {
        let file_path = cache_dir.join(filename);
        std::fs::write(&file_path, data)?;
        println!("    Added to cache: {} ({} bytes)", filename, data.len());
    }

    let (cache_size, cache_count) = client.get_cache_info()?;
    println!(
        "    Cache after adding files: {} bytes, {} files",
        cache_size, cache_count
    );

    // Test cache download simulation
    println!("  🔹 Simulating cached downloads:");
    #[cfg(feature = "reqwest")]
    {
        // In a real scenario, this would check cache first, then download if not found
        let cache_file = cache_dir.join("download_cache.dat");

        // Simulate cache miss - would normally download
        if !cache_file.exists() {
            println!("    Cache miss for download_cache.dat - would download from network");
            // download_with_cache("https://example.com/file.dat", &cache_file, Some(&cache_dir)).await?;
        } else {
            println!("    Cache hit for download_cache.dat - using cached version");
        }
    }

    #[cfg(not(feature = "reqwest"))]
    {
        println!("    Download with cache requires 'reqwest' feature");
    }

    // Test cache cleanup
    println!("  🔹 Testing cache cleanup:");
    client.clear_cache()?;
    let (final_size, final_count) = client.get_cache_info()?;
    println!(
        "    Cache after cleanup: {} bytes, {} files",
        final_size, final_count
    );

    // Simulate cache size management
    println!("  🔹 Cache size management:");
    let max_cache_size = 1024 * 1024; // 1MB
    let total_cached_size = cached_files
        .iter()
        .map(|(_, data)| data.len())
        .sum::<usize>();

    if total_cached_size > max_cache_size {
        println!(
            "    Cache size {} bytes exceeds limit {} bytes - cleanup needed",
            total_cached_size, max_cache_size
        );
    } else {
        println!(
            "    Cache size {} bytes within limit {} bytes",
            total_cached_size, max_cache_size
        );
    }

    // Test offline access simulation
    println!("  🔹 Simulating offline access:");
    let offline_files = vec![
        "important_data.csv",
        "analysis_results.json",
        "model_weights.bin",
    ];

    for filename in offline_files {
        let cached_path = cache_dir.join(filename);
        if cached_path.exists() {
            println!("    Offline access available for: {}", filename);
        } else {
            println!(
                "    Offline access not available for: {} (not cached)",
                filename
            );
        }
    }

    Ok(())
}
