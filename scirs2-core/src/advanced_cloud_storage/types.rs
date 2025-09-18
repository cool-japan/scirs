//! Core types and configurations for advanced cloud storage
//!
//! This module contains all the shared types, configurations, request/response structures,
//! and basic enums used throughout the cloud storage system.

use crate::error::{CoreError, CoreResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for advanced cloud storage
#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct AdvancedCloudConfig {
    /// Enable multi-cloud optimization
    pub enable_multi_cloud: bool,
    /// Enable adaptive streaming
    pub enable_adaptive_streaming: bool,
    /// Enable intelligent caching
    pub enable_intelligent_caching: bool,
    /// Enable automatic compression
    pub enable_auto_compression: bool,
    /// Enable parallel transfers
    pub enable_parallel_transfers: bool,
    /// Maximum concurrent transfers
    pub max_concurrent_transfers: usize,
    /// Cache size limit (GB)
    pub cache_size_limit_gb: f64,
    /// Adaptive streaming buffer size (MB)
    pub streaming_buffer_size_mb: usize,
    /// Prefetch threshold
    pub prefetch_threshold: f64,
    /// Compression threshold (KB)
    pub compression_threshold_kb: usize,
    /// Transfer retry attempts
    pub transfer_retry_attempts: u32,
    /// Health check interval (seconds)
    pub health_check_interval_seconds: u64,
    /// Enable cost optimization
    pub enable_cost_optimization: bool,
}

impl Default for AdvancedCloudConfig {
    fn default() -> Self {
        Self {
            enable_multi_cloud: true,
            enable_adaptive_streaming: true,
            enable_intelligent_caching: true,
            enable_auto_compression: true,
            enable_parallel_transfers: true,
            max_concurrent_transfers: 16,
            cache_size_limit_gb: 10.0,
            streaming_buffer_size_mb: 64,
            prefetch_threshold: 0.7,
            compression_threshold_kb: 1024,
            transfer_retry_attempts: 3,
            health_check_interval_seconds: 60,
            enable_cost_optimization: true,
        }
    }
}

/// Cloud provider identifier
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[derive(Serialize, Deserialize)]
pub struct CloudProviderId(pub String);

/// Cloud provider types
#[derive(Debug, Clone, PartialEq)]
#[derive(Serialize, Deserialize)]
pub enum CloudProviderType {
    AmazonS3,
    GoogleCloudStorage,
    AzureBlobStorage,
    DigitalOceanSpaces,
    BackblazeB2,
    WasabiHotStorage,
    CloudflareR2,
    Custom(String),
}

/// Cloud provider configuration
#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct CloudProviderConfig {
    /// Provider type
    pub provider_type: CloudProviderType,
    /// Access credentials
    pub credentials: CloudCredentials,
    /// Region/endpoint configuration
    pub region_config: RegionConfig,
    /// Performance settings
    pub performance_settings: ProviderPerformanceSettings,
    /// Security settings
    pub security_settings: ProviderSecuritySettings,
}

/// Cloud credentials
#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct CloudCredentials {
    /// Access key or client ID
    pub access_key: String,
    /// Secret key or client secret
    pub secret_key: String,
    /// Session token (if applicable)
    pub session_token: Option<String>,
    /// Service account key (for GCS)
    pub service_account_key: Option<String>,
    /// Credential type
    pub credential_type: CredentialType,
}

/// Credential types
#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub enum CredentialType {
    AccessKey,
    ServiceAccount,
    OAuth2,
    IAMRole,
    ManagedIdentity,
}

/// Region configuration
#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct RegionConfig {
    /// Primary region
    pub primary_region: String,
    /// Secondary regions for replication
    pub secondary_regions: Vec<String>,
    /// Custom endpoint URL
    pub custom_endpoint: Option<String>,
    /// Enable dual stack (IPv4/IPv6)
    pub enable_dual_stack: bool,
}

/// Provider performance settings
#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct ProviderPerformanceSettings {
    /// Connection timeout (seconds)
    pub connection_timeout_seconds: u64,
    /// Read timeout (seconds)
    pub read_timeout_seconds: u64,
    /// Write timeout (seconds)
    pub write_timeout_seconds: u64,
    /// Maximum retry attempts
    pub max_retry_attempts: u32,
    /// Retry backoff strategy
    pub retry_strategy: RetryStrategy,
    /// Connection pool size
    pub connection_pool_size: usize,
    /// Enable transfer acceleration
    pub enable_transfer_acceleration: bool,
}

/// Retry strategies
#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub enum RetryStrategy {
    Exponential {
        base_delay_ms: u64,
        max_delay_ms: u64,
    },
    Linear {
        delay_ms: u64,
    },
    Fixed {
        delay_ms: u64,
    },
    Adaptive,
}

/// Provider security settings
#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct ProviderSecuritySettings {
    /// Enable encryption in transit
    pub enable_encryption_in_transit: bool,
    /// Enable encryption at rest
    pub enable_encryption_at_rest: bool,
    /// Encryption algorithm
    pub encryption_algorithm: EncryptionAlgorithm,
    /// Key management
    pub key_management: KeyManagement,
    /// Enable signature verification
    pub enable_signature_verification: bool,
    /// Certificate validation
    pub certificate_validation: CertificateValidation,
}

/// Encryption algorithms
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    AES256,
    AES128,
    ChaCha20Poly1305,
    ProviderManaged,
}

/// Key management
#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct KeyManagement {
    /// Key management service
    pub kms_provider: Option<String>,
    /// Key ID
    pub key_id: Option<String>,
    /// Client-side encryption
    pub client_side_encryption: bool,
    /// Key rotation interval (days)
    pub key_rotation_interval_days: Option<u32>,
}

/// Certificate validation
#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct CertificateValidation {
    /// Validate certificate chain
    pub validate_chain: bool,
    /// Validate hostname
    pub validate_hostname: bool,
    /// Custom CA certificates
    pub custom_ca_certs: Vec<String>,
    /// Certificate pinning
    pub certificate_pinning: bool,
}

/// Upload request
#[derive(Debug, Clone)]
pub struct UploadRequest {
    /// Object key/path
    pub key: String,
    /// Bucket/container name
    pub bucket: String,
    /// Data to upload
    pub data: Vec<u8>,
    /// Content type
    pub content_type: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Storage class
    pub storage_class: Option<StorageClass>,
    /// Encryption settings
    pub encryption: Option<EncryptionSettings>,
    /// Access control
    pub access_control: Option<AccessControl>,
    /// Upload options
    pub options: UploadOptions,
}

/// Upload options
#[derive(Debug, Clone)]
pub struct UploadOptions {
    /// Enable multipart upload
    pub enable_multipart: bool,
    /// Multipart chunk size (MB)
    pub chunk_size_mb: usize,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression algorithm
    pub compression_algorithm: Option<CompressionAlgorithm>,
    /// Enable checksums
    pub enable_checksums: bool,
    /// Progress callback interval
    pub progress_callback_interval: Option<Duration>,
}

/// Storage classes
#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub enum StorageClass {
    Standard,
    ReducedRedundancy,
    StandardIA,
    OneZoneIA,
    Glacier,
    GlacierDeepArchive,
    ColdStorage,
    Archive,
    Custom(String),
}

/// Encryption settings
#[derive(Debug, Clone)]
pub struct EncryptionSettings {
    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,
    /// Encryption key
    pub key: Option<Vec<u8>>,
    /// Key ID for KMS
    pub key_id: Option<String>,
    /// Encryption context
    pub context: HashMap<String, String>,
}

/// Access control
#[derive(Debug, Clone)]
pub struct AccessControl {
    /// Access control list
    pub acl: Option<String>,
    /// Bucket policy
    pub policy: Option<String>,
    /// CORS settings
    pub cors: Option<CorsSettings>,
}

/// CORS settings
#[derive(Debug, Clone)]
pub struct CorsSettings {
    /// Allowed origins
    pub allowed_origins: Vec<String>,
    /// Allowed methods
    pub allowed_methods: Vec<String>,
    /// Allowed headers
    pub allowed_headers: Vec<String>,
    /// Max age (seconds)
    pub max_age_seconds: u64,
}

/// Compression algorithms
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    Brotli,
    Snappy,
    Adaptive,
}

/// Upload response
#[derive(Debug, Clone)]
pub struct UploadResponse {
    /// Object key
    pub key: String,
    /// ETag or version ID
    pub etag: String,
    /// Upload timestamp
    pub timestamp: Instant,
    /// Final size after compression
    pub final_size_bytes: usize,
    /// Upload performance metrics
    pub performance: TransferPerformance,
}

/// Download request
#[derive(Debug, Clone)]
pub struct DownloadRequest {
    /// Object key/path
    pub key: String,
    /// Bucket/container name
    pub bucket: String,
    /// Byte range (optional)
    pub range: Option<ByteRange>,
    /// Version ID (optional)
    pub version_id: Option<String>,
    /// Download options
    pub options: DownloadOptions,
}

/// Byte range
#[derive(Debug, Clone)]
pub struct ByteRange {
    /// Start byte (inclusive)
    pub start: u64,
    /// End byte (inclusive, optional)
    pub end: Option<u64>,
}

/// Download options
#[derive(Debug, Clone)]
pub struct DownloadOptions {
    /// Enable streaming
    pub enable_streaming: bool,
    /// Buffer size for streaming (MB)
    pub buffer_size_mb: usize,
    /// Enable decompression
    pub enable_decompression: bool,
    /// Enable checksums verification
    pub verify_checksums: bool,
    /// Progress callback interval
    pub progress_callback_interval: Option<Duration>,
}

/// Download response
#[derive(Debug, Clone)]
pub struct DownloadResponse {
    /// Object key
    pub key: String,
    /// Downloaded data
    pub data: Vec<u8>,
    /// Content type
    pub content_type: Option<String>,
    /// Last modified timestamp
    pub last_modified: Option<Instant>,
    /// ETag
    pub etag: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Download performance metrics
    pub performance: TransferPerformance,
}

/// Stream request
#[derive(Debug, Clone)]
pub struct StreamRequest {
    /// Object key/path
    pub key: String,
    /// Bucket/container name
    pub bucket: String,
    /// Stream options
    pub options: StreamOptions,
}

/// Stream options
#[derive(Debug, Clone)]
pub struct StreamOptions {
    /// Buffer size (MB)
    pub buffer_size_mb: usize,
    /// Prefetch size (MB)
    pub prefetch_size_mb: usize,
    /// Enable adaptive buffering
    pub enable_adaptive_buffering: bool,
    /// Enable compression
    pub enable_compression: bool,
    /// Stream direction
    pub direction: StreamDirection,
}

/// Stream direction
#[derive(Debug, Clone)]
pub enum StreamDirection {
    Read,
    Write,
    Bidirectional,
}

/// Data stream trait
pub trait DataStream: std::fmt::Debug {
    /// Read data from stream
    fn read(&mut self, buffer: &mut [u8]) -> CoreResult<usize>;

    /// Write data to stream
    fn write(&mut self, data: &[u8]) -> CoreResult<usize>;

    /// Seek to position
    fn seek(&mut self, position: u64) -> CoreResult<u64>;

    /// Get current position
    fn position(&self) -> u64;

    /// Get stream size
    fn size(&self) -> Option<u64>;

    /// Close stream
    fn close(&mut self) -> CoreResult<()>;
}

/// List request
#[derive(Debug, Clone)]
pub struct ListRequest {
    /// Bucket/container name
    pub bucket: String,
    /// Prefix filter
    pub prefix: Option<String>,
    /// Delimiter
    pub delimiter: Option<String>,
    /// Maximum keys to return
    pub max_keys: Option<u32>,
    /// Continuation token
    pub continuation_token: Option<String>,
}

/// List response
#[derive(Debug, Clone)]
pub struct ListResponse {
    /// List of objects
    pub objects: Vec<ObjectInfo>,
    /// Common prefixes
    pub common_prefixes: Vec<String>,
    /// Truncated flag
    pub is_truncated: bool,
    /// Next continuation token
    pub next_continuation_token: Option<String>,
}

/// Object information
#[derive(Debug, Clone)]
pub struct ObjectInfo {
    /// Object key
    pub key: String,
    /// Size in bytes
    pub size: u64,
    /// Last modified timestamp
    pub last_modified: Instant,
    /// ETag
    pub etag: String,
    /// Storage class
    pub storage_class: Option<StorageClass>,
    /// Owner
    pub owner: Option<ObjectOwner>,
}

/// Object owner
#[derive(Debug, Clone)]
pub struct ObjectOwner {
    /// Owner ID
    pub id: String,
    /// Display name
    pub display_name: Option<String>,
}

/// Delete request
#[derive(Debug, Clone)]
pub struct DeleteRequest {
    /// Bucket/container name
    pub bucket: String,
    /// Objects to delete
    pub objects: Vec<DeleteObject>,
    /// Quiet mode
    pub quiet: bool,
}

/// Delete object
#[derive(Debug, Clone)]
pub struct DeleteObject {
    /// Object key
    pub key: String,
    /// Version ID (optional)
    pub version_id: Option<String>,
}

/// Delete response
#[derive(Debug, Clone)]
pub struct DeleteResponse {
    /// Successfully deleted objects
    pub deleted: Vec<DeletedObject>,
    /// Errors encountered
    pub errors: Vec<DeleteError>,
}

/// Deleted object
#[derive(Debug, Clone)]
pub struct DeletedObject {
    /// Object key
    pub key: String,
    /// Version ID
    pub version_id: Option<String>,
    /// Delete marker
    pub delete_marker: bool,
}

/// Delete error
#[derive(Debug, Clone)]
pub struct DeleteError {
    /// Object key
    pub key: String,
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
}

/// Metadata request
#[derive(Debug, Clone)]
pub struct MetadataRequest {
    /// Object key/path
    pub key: String,
    /// Bucket/container name
    pub bucket: String,
    /// Version ID (optional)
    pub version_id: Option<String>,
}

/// Object metadata
#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    /// Object key
    pub key: String,
    /// Size in bytes
    pub size: u64,
    /// Content type
    pub content_type: Option<String>,
    /// Last modified timestamp
    pub last_modified: Option<Instant>,
    /// ETag
    pub etag: Option<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Storage class
    pub storage_class: Option<StorageClass>,
    /// Encryption information
    pub encryption: Option<EncryptionInfo>,
}

/// Encryption information
#[derive(Debug, Clone)]
pub struct EncryptionInfo {
    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,
    /// Key ID
    pub key_id: Option<String>,
    /// Encryption context
    pub context: HashMap<String, String>,
}

/// Provider health
#[derive(Debug, Clone)]
pub struct ProviderHealth {
    /// Health status
    pub status: HealthStatus,
    /// Response time
    pub response_time: Duration,
    /// Error rate
    pub error_rate: f64,
    /// Available regions
    pub available_regions: Vec<String>,
    /// Service limits
    pub service_limits: ServiceLimits,
}

/// Health status
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Service limits
#[derive(Debug, Clone)]
pub struct ServiceLimits {
    /// Max object size (bytes)
    pub max_object_size: u64,
    /// Max request rate (requests/second)
    pub max_request_rate: u32,
    /// Max bandwidth (MB/s)
    pub max_bandwidth_mbps: f64,
    /// Request quotas
    pub request_quotas: HashMap<String, u64>,
}

/// Cost operation
#[derive(Debug, Clone)]
pub struct CostOperation {
    /// Operation type
    pub operation_type: OperationType,
    /// Data size (bytes)
    pub data_size_bytes: u64,
    /// Number of requests
    pub request_count: u32,
    /// Storage duration (hours)
    pub storage_duration_hours: Option<u64>,
    /// Transfer type
    pub transfer_type: Option<TransferType>,
}

/// Operation types
#[derive(Debug, Clone)]
pub enum OperationType {
    Upload,
    Download,
    Storage,
    Request,
    DataTransfer,
}

/// Transfer types
#[derive(Debug, Clone)]
pub enum TransferType {
    Inbound,
    Outbound,
    InterRegion,
    InterProvider,
}

/// Cost estimate
#[derive(Debug, Clone)]
pub struct CostEstimate {
    /// Total estimated cost
    pub total_cost: f64,
    /// Currency
    pub currency: String,
    /// Cost breakdown
    pub breakdown: HashMap<String, f64>,
    /// Optimization suggestions
    pub optimization_suggestions: Vec<String>,
}

/// Transfer performance metrics
#[derive(Debug, Clone)]
pub struct TransferPerformance {
    /// Transfer duration
    pub duration: Duration,
    /// Transfer rate (MB/s)
    pub transfer_rate_mbps: f64,
    /// Retry count
    pub retry_count: u32,
    /// Compression ratio (if applicable)
    pub compression_ratio: Option<f64>,
    /// Network efficiency
    pub network_efficiency: f64,
}

/// Multi-cloud optimization result
#[derive(Debug)]
pub struct MultiCloudOptimizationResult {
    /// Provider performance analysis
    pub provider_analysis: HashMap<CloudProviderId, ProviderPerformanceAnalysis>,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Optimization results
    pub optimization_results: Vec<OptimizationResult>,
    /// Timestamp
    pub timestamp: Instant,
}

/// Provider performance analysis
#[derive(Debug, Clone)]
pub struct ProviderPerformanceAnalysis {
    /// Average response time
    pub avg_response_time: Duration,
    /// Throughput (MB/s)
    pub throughput_mbps: f64,
    /// Error rate
    pub error_rate: f64,
    /// Cost per GB
    pub cost_per_gb: f64,
    /// Availability
    pub availability: f64,
    /// Geographic performance
    pub geographic_performance: HashMap<String, f64>,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: String,
    /// Description
    pub description: String,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Implementation complexity
    pub complexity: ComplexityLevel,
    /// Cost impact
    pub cost_impact: f64,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimization applied
    pub optimization_applied: String,
    /// Actual improvement
    pub actual_improvement: f64,
    /// Success flag
    pub success: bool,
    /// Error message (if failed)
    pub error_message: Option<String>,
}

/// Complexity levels
#[derive(Debug, Clone)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
}