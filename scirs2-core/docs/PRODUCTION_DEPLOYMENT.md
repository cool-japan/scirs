# Production Deployment Guide for SciRS2

This guide provides comprehensive instructions for deploying SciRS2 in production environments, including monitoring, observability, and operational best practices.

## Table of Contents

1. [Environment Requirements](#environment-requirements)
2. [Installation and Configuration](#installation-and-configuration)
3. [Performance Tuning](#performance-tuning)
4. [Monitoring and Observability](#monitoring-and-observability)
5. [Security Configuration](#security-configuration)
6. [High Availability Setup](#high-availability-setup)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance and Updates](#maintenance-and-updates)

## Environment Requirements

### Minimum System Requirements

- **CPU**: x86_64 or ARM64 with SIMD support
- **Memory**: 4GB RAM minimum, 16GB+ recommended for large datasets
- **Storage**: 100MB for base installation, additional space for data caching
- **OS**: Linux (Ubuntu 20.04+, RHEL 8+), macOS 10.15+, Windows 10+

### Recommended Production Environment

- **CPU**: 8+ cores with AVX2/NEON support
- **Memory**: 32GB+ RAM with ECC
- **Storage**: NVMe SSD with high IOPS
- **Network**: High-bandwidth, low-latency for distributed workloads
- **GPU**: CUDA-compatible or Metal-compatible for acceleration (optional)

### Software Dependencies

```toml
# Cargo.toml dependencies for production
[dependencies]
scirs2-core = { version = "0.1.0-beta.2", features = [
    "production",
    "monitoring",
    "security",
    "parallel",
    "simd",
    "memory_efficient",
] }
```

## Installation and Configuration

### 1. Basic Installation

```bash
# Install via Cargo
cargo install scirs2-core --features production

# Or add to your project
cargo add scirs2-core --features production,monitoring,security
```

### 2. Environment Configuration

Create a production configuration file:

```toml
# scirs2.toml
[production]
log_level = "info"
enable_metrics = true
enable_profiling = false
cache_size = "1GB"

[performance]
simd_enabled = true
parallel_threads = 0  # Auto-detect
gpu_acceleration = true
memory_limit = "16GB"

[security]
input_validation = "strict"
audit_logging = true
secure_defaults = true

[monitoring]
metrics_endpoint = "127.0.0.1:9090"
health_check_interval = 30
log_rotation = true
max_log_size = "100MB"
```

### 3. Initialization

```rust
use scirs2_core::{Config, set_global_config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load production configuration
    let config = Config::from_file("scirs2.toml")?;
    set_global_config(config)?;
    
    // Initialize production features
    scirs2_core::production::initialize()?;
    
    // Your application code here
    Ok(())
}
```

## Performance Tuning

### 1. Memory Configuration

```rust
use scirs2_core::memory_efficient::{AdaptiveChunkingBuilder, MemoryMappedArray};

// Configure adaptive chunking for large datasets
let chunking = AdaptiveChunkingBuilder::new()
    .with_memory_limit(8 * 1024 * 1024 * 1024) // 8GB
    .with_parallel_threshold(100_000)
    .with_cache_size(512 * 1024 * 1024) // 512MB cache
    .build();

// Use memory-mapped arrays for very large data
let data = MemoryMappedArray::<f64>::create("large_dataset.dat", &[1_000_000, 1_000])?;
```

### 2. SIMD and Parallel Optimization

```rust
use scirs2_core::{simd_ops::SimdUnifiedOps, parallel_ops::*};

// Enable automatic SIMD detection
scirs2_core::simd_ops::PlatformCapabilities::detect();

// Configure parallel execution
set_num_threads(num_cpus::get());

// Use optimized operations
let result = f64::simd_add(&array_a.view(), &array_b.view());
```

### 3. GPU Acceleration

```rust
#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuContext, GpuBackend};

#[cfg(feature = "gpu")]
fn setup_gpu() -> Result<(), Box<dyn std::error::Error>> {
    let context = GpuContext::new(GpuBackend::Cuda)?;
    context.set_memory_limit(4 * 1024 * 1024 * 1024)?; // 4GB
    Ok(())
}
```

## Monitoring and Observability

### 1. Metrics Collection

```rust
use scirs2_core::observability::{Counter, Gauge, Histogram, Timer};

// Define application metrics
static OPERATIONS_TOTAL: Counter = Counter::new("scirs2_operations_total");
static MEMORY_USAGE: Gauge = Gauge::new("scirs2_memory_usage_bytes");
static COMPUTATION_TIME: Histogram = Histogram::new("scirs2_computation_seconds");

fn tracked_computation() -> Result<(), Box<dyn std::error::Error>> {
    let _timer = Timer::start("computation");
    OPERATIONS_TOTAL.increment();
    
    // Your computation here
    
    MEMORY_USAGE.set(get_memory_usage());
    Ok(())
}
```

### 2. Health Checks

```rust
use scirs2_core::observability::health::{HealthCheck, HealthStatus};

fn health_check() -> HealthStatus {
    let mut checks = Vec::new();
    
    // Check memory usage
    checks.push(HealthCheck {
        name: "memory_usage".to_string(),
        status: if get_memory_usage() < 0.9 * get_memory_limit() {
            "healthy".to_string()
        } else {
            "unhealthy".to_string()
        },
        details: Some(format!("Memory usage: {:.1}%", get_memory_usage_percent())),
    });
    
    // Check GPU availability
    #[cfg(feature = "gpu")]
    checks.push(HealthCheck {
        name: "gpu_availability".to_string(),
        status: if scirs2_core::gpu::is_available() {
            "healthy".to_string()
        } else {
            "degraded".to_string()
        },
        details: None,
    });
    
    HealthStatus { checks }
}
```

### 3. Logging Configuration

```rust
use scirs2_core::observability::audit::{AuditEvent, AuditLogger};

fn setup_logging() -> Result<(), Box<dyn std::error::Error>> {
    let logger = AuditLogger::new()
        .with_file_output("/var/log/scirs2/audit.log")
        .with_rotation(100_000_000) // 100MB
        .with_format("json");
    
    logger.log_event(AuditEvent {
        timestamp: std::time::SystemTime::now(),
        event_type: "application_start".to_string(),
        user: std::env::var("USER").unwrap_or("unknown".to_string()),
        details: "SciRS2 application started".to_string(),
    })?;
    
    Ok(())
}
```

### 4. Prometheus Integration

```toml
# Add to Cargo.toml
[dependencies]
prometheus = "0.13"
```

```rust
use prometheus::{register_counter, register_histogram, Counter, Histogram};

lazy_static::lazy_static! {
    static ref OPERATIONS_COUNTER: Counter = register_counter!(
        "scirs2_operations_total", "Total number of operations"
    ).unwrap();
    
    static ref COMPUTATION_HISTOGRAM: Histogram = register_histogram!(
        "scirs2_computation_duration_seconds", "Time spent in computations"
    ).unwrap();
}

// Export metrics endpoint
fn metrics_handler() -> String {
    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();
    encoder.encode_to_string(&metric_families).unwrap()
}
```

## Security Configuration

### 1. Input Validation

```rust
use scirs2_core::validation::{ValidationConfig, Validator};

fn setup_security() -> Result<(), Box<dyn std::error::Error>> {
    let config = ValidationConfig::production()
        .with_strict_bounds_checking()
        .with_input_sanitization()
        .with_memory_limits();
    
    Validator::set_global_config(config)?;
    Ok(())
}
```

### 2. Audit Logging

```rust
use scirs2_core::observability::audit::AuditLogger;

fn security_event(event: &str, details: &str) {
    AuditLogger::global().log_security_event(event, details);
}

// Usage in critical operations
fn process_sensitive_data(data: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    security_event("data_processing_start", &format!("Processing {} elements", data.len()));
    
    // Validate input
    scirs2_core::validation::check_array_finite(data, "input_data")?;
    
    // Process data
    let result = data.iter().map(|x| x * 2.0).collect();
    
    security_event("data_processing_complete", "Processing completed successfully");
    Ok(result)
}
```

## High Availability Setup

### 1. Load Balancing Configuration

```rust
use scirs2_core::distributed::{LoadBalancer, WorkerNode};

fn setup_ha_cluster() -> Result<(), Box<dyn std::error::Error>> {
    let nodes = vec![
        WorkerNode::new("worker-1", "192.168.1.10:8080"),
        WorkerNode::new("worker-2", "192.168.1.11:8080"),
        WorkerNode::new("worker-3", "192.168.1.12:8080"),
    ];
    
    let load_balancer = LoadBalancer::new()
        .with_strategy("round_robin")
        .with_health_checks(true)
        .with_failure_timeout(std::time::Duration::from_secs(30));
    
    for node in nodes {
        load_balancer.add_node(node);
    }
    
    Ok(())
}
```

### 2. State Management

```rust
use scirs2_core::cache::{Cache, CacheConfig};

fn setup_distributed_cache() -> Result<(), Box<dyn std::error::Error>> {
    let cache_config = CacheConfig::new()
        .with_size_limit(1024 * 1024 * 1024) // 1GB
        .with_ttl(std::time::Duration::from_secs(3600)) // 1 hour
        .with_persistence("/var/cache/scirs2");
    
    Cache::initialize_global(cache_config)?;
    Ok(())
}
```

## Troubleshooting

### Common Issues

#### 1. Memory Exhaustion

**Symptoms**: OutOfMemory errors, slow performance
**Solution**:
```rust
// Reduce memory usage
let chunking = AdaptiveChunkingBuilder::new()
    .with_memory_limit(available_memory() / 2)
    .with_aggressive_cleanup(true)
    .build();
```

#### 2. SIMD Not Available

**Symptoms**: Slower than expected performance
**Diagnosis**:
```rust
use scirs2_core::simd_ops::PlatformCapabilities;

fn diagnose_simd() {
    let caps = PlatformCapabilities::detect();
    println!("SIMD capabilities: {:?}", caps);
    
    if !caps.has_avx2() && !caps.has_neon() {
        println!("Warning: No advanced SIMD support detected");
    }
}
```

#### 3. GPU Initialization Failure

**Symptoms**: GPU features not working
**Diagnosis**:
```rust
#[cfg(feature = "gpu")]
fn diagnose_gpu() {
    match scirs2_core::gpu::GpuContext::new(scirs2_core::gpu::GpuBackend::Cuda) {
        Ok(_) => println!("CUDA GPU available"),
        Err(e) => println!("CUDA not available: {}", e),
    }
}
```

### Performance Analysis

```rust
use scirs2_core::profiling::{Profiler, ProfileScope};

fn performance_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let profiler = Profiler::new();
    profiler.start()?;
    
    {
        let _scope = ProfileScope::new("computation");
        // Your computation here
    }
    
    let report = profiler.generate_report()?;
    println!("Performance report: {}", report);
    
    Ok(())
}
```

### Log Analysis

```bash
# Analyze error patterns
grep "ERROR" /var/log/scirs2/app.log | tail -20

# Check memory usage trends
grep "memory_usage" /var/log/scirs2/metrics.log | tail -50

# Monitor GPU utilization
grep "gpu" /var/log/scirs2/performance.log | tail -30
```

## Maintenance and Updates

### 1. Version Compatibility Checking

```rust
use scirs2_core::api_freeze::{is_version_compatible, current_library_version};

fn check_compatibility() -> Result<(), Box<dyn std::error::Error>> {
    let current = current_library_version();
    let required = scirs2_core::Version::new(0, 1, 0);
    
    if !is_version_compatible(&required) {
        return Err("Version compatibility check failed".into());
    }
    
    println!("Version compatibility: OK (current: {})", current);
    Ok(())
}
```

### 2. Graceful Shutdown

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn setup_graceful_shutdown() -> Arc<AtomicBool> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    
    ctrlc::set_handler(move || {
        println!("Received shutdown signal, cleaning up...");
        shutdown_clone.store(true, Ordering::Relaxed);
    }).expect("Error setting shutdown handler");
    
    shutdown
}

fn main_loop(shutdown: Arc<AtomicBool>) {
    while !shutdown.load(Ordering::Relaxed) {
        // Main application logic
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    
    // Cleanup
    scirs2_core::cache::flush_all();
    scirs2_core::observability::audit::AuditLogger::global().flush();
    println!("Shutdown complete");
}
```

### 3. Backup and Recovery

```rust
use scirs2_core::io::{backup_state, restore_state};

fn backup_application_state() -> Result<(), Box<dyn std::error::Error>> {
    let backup_path = format!("/backup/scirs2-state-{}.bak", 
        chrono::Utc::now().format("%Y%m%d-%H%M%S"));
    
    backup_state(&backup_path)?;
    println!("Application state backed up to: {}", backup_path);
    
    Ok(())
}

fn restore_application_state(backup_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    restore_state(backup_path)?;
    println!("Application state restored from: {}", backup_path);
    
    Ok(())
}
```

## Configuration Examples

### Development Environment

```toml
# scirs2-dev.toml
[production]
log_level = "debug"
enable_metrics = true
enable_profiling = true

[performance]
simd_enabled = true
parallel_threads = 4
memory_limit = "4GB"

[security]
input_validation = "relaxed"
audit_logging = false
```

### Production Environment

```toml
# scirs2-prod.toml
[production]
log_level = "warn"
enable_metrics = true
enable_profiling = false

[performance]
simd_enabled = true
parallel_threads = 0  # Auto-detect
memory_limit = "32GB"

[security]
input_validation = "strict"
audit_logging = true
secure_defaults = true

[monitoring]
metrics_endpoint = "0.0.0.0:9090"
health_check_interval = 10
```

### High-Performance Computing

```toml
# scirs2-hpc.toml
[production]
log_level = "error"
enable_metrics = true
enable_profiling = true

[performance]
simd_enabled = true
parallel_threads = 0
gpu_acceleration = true
memory_limit = "128GB"

[memory_efficient]
aggressive_chunking = true
memory_mapping = true
compression = true
```

## Deployment Checklist

- [ ] System requirements verified
- [ ] Configuration files created and validated
- [ ] Security settings configured
- [ ] Monitoring and logging enabled
- [ ] Health checks implemented
- [ ] Performance tuning applied
- [ ] Backup procedures established
- [ ] Graceful shutdown implemented
- [ ] Documentation updated
- [ ] Team trained on operations

## Support and Resources

- **Documentation**: Full API documentation at [docs.rs](https://docs.rs/scirs2-core)
- **Examples**: Production examples in `/examples/production/`
- **Community**: GitHub Discussions for questions and support
- **Security**: Report security issues to security@scirs2.org
- **Performance**: Performance optimization guides in `/docs/performance/`

For enterprise support and consulting, contact: enterprise@scirs2.org