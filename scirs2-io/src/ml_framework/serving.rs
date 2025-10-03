//! Model serving capabilities with REST API and gRPC support

use crate::error::{IoError, Result};
use crate::ml_framework::{DataType, MLModel, MLTensor, TensorMetadata};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::RwLock as StdRwLock;
use std::time::{Duration, Instant};

#[cfg(feature = "async")]
use tokio::sync::{Mutex, RwLock};
#[cfg(feature = "async")]
use tokio::time::{sleep, timeout};

/// Comprehensive model server with multiple API endpoints
#[cfg(feature = "async")]
pub struct ModelServer {
    model: Arc<RwLock<MLModel>>,
    config: ServerConfig,
    metrics: Arc<Mutex<ServerMetrics>>,
    request_queue: Arc<Mutex<VecDeque<InferenceRequest>>>,
    health_status: Arc<RwLock<HealthStatus>>,
}

#[cfg(not(feature = "async"))]
pub struct ModelServer {
    model: Arc<StdRwLock<MLModel>>,
    config: ServerConfig,
    metrics: Arc<StdRwLock<ServerMetrics>>,
    health_status: Arc<StdRwLock<HealthStatus>>,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub max_batch_size: usize,
    pub timeout_ms: u64,
    pub num_workers: usize,
    pub enable_batching: bool,
    pub batch_timeout_ms: u64,
    pub max_queue_size: usize,
    pub enable_streaming: bool,
    pub api_config: ApiConfig,
}

#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub rest_enabled: bool,
    pub grpc_enabled: bool,
    pub rest_port: u16,
    pub grpc_port: u16,
    pub enable_cors: bool,
    pub enable_auth: bool,
    pub auth_token: Option<String>,
    pub rate_limit: Option<RateLimit>,
}

#[derive(Debug, Clone)]
pub struct RateLimit {
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            timeout_ms: 5000,
            num_workers: 4,
            enable_batching: true,
            batch_timeout_ms: 100,
            max_queue_size: 1000,
            enable_streaming: false,
            api_config: ApiConfig::default(),
        }
    }
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            rest_enabled: true,
            grpc_enabled: false,
            rest_port: 8080,
            grpc_port: 9090,
            enable_cors: true,
            enable_auth: false,
            auth_token: None,
            rate_limit: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub id: String,
    pub inputs: HashMap<String, MLTensor>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub timestamp: Instant,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub request_id: String,
    pub outputs: HashMap<String, MLTensor>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub processing_time_ms: u64,
    pub status: ResponseStatus,
}

#[derive(Debug, Clone)]
pub enum ResponseStatus {
    Success,
    Error { code: u16, message: String },
    Timeout,
    QueueFull,
}

#[derive(Debug, Clone, Default)]
pub struct ServerMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_latency_ms: f64,
    pub requests_per_second: f64,
    pub current_queue_size: usize,
    pub max_queue_size_reached: usize,
    pub model_load_time_ms: u64,
    pub uptime_seconds: u64,
    pub batch_stats: BatchStats,
}

#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    pub total_batches: u64,
    pub average_batch_size: f64,
    pub batch_processing_time_ms: f64,
}

#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Degraded { reason: String },
    Unhealthy { reason: String },
    Starting,
    Stopping,
}

#[cfg(feature = "async")]
impl ModelServer {
    pub async fn new(model: MLModel, config: ServerConfig) -> Self {
        Self {
            model: Arc::new(RwLock::new(model)),
            config,
            metrics: Arc::new(Mutex::new(ServerMetrics::default())),
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
            health_status: Arc::new(RwLock::new(HealthStatus::Starting)),
        }
    }

    /// Start the model server with all enabled APIs
    pub async fn start(&self) -> Result<()> {
        // Update health status
        {
            let mut status = self.health_status.write().await;
            *status = HealthStatus::Healthy;
        }

        // Start metrics collection
        self.start_metrics_collection().await;

        // Start request processing workers
        self.start_workers().await?;

        // Start REST API if enabled
        if self.config.api_config.rest_enabled {
            self.start_rest_api().await?;
        }

        // Start gRPC API if enabled
        if self.config.api_config.grpc_enabled {
            self.start_grpc_api().await?;
        }

        Ok(())
    }

    /// Perform single inference
    pub async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start_time = Instant::now();

        // Check queue capacity
        {
            let queue = self.request_queue.lock().await;
            if queue.len() >= self.config.max_queue_size {
                return Ok(InferenceResponse {
                    request_id: request.id,
                    outputs: HashMap::new(),
                    metadata: HashMap::new(),
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    status: ResponseStatus::QueueFull,
                });
            }
        }

        // Add to queue if batching is enabled
        if self.config.enable_batching {
            {
                let mut queue = self.request_queue.lock().await;
                queue.push_back(request.clone());
            }

            // Wait for response (simplified - would use proper async coordination)
            sleep(Duration::from_millis(self.config.batch_timeout_ms)).await;
        }

        // Process inference
        let result = self.process_inference(&request.inputs).await;

        // Update metrics
        self.update_metrics(start_time, result.is_ok()).await;

        match result {
            Ok(outputs) => Ok(InferenceResponse {
                request_id: request.id,
                outputs,
                metadata: HashMap::new(),
                processing_time_ms: start_time.elapsed().as_millis() as u64,
                status: ResponseStatus::Success,
            }),
            Err(e) => Ok(InferenceResponse {
                request_id: request.id,
                outputs: HashMap::new(),
                metadata: HashMap::new(),
                processing_time_ms: start_time.elapsed().as_millis() as u64,
                status: ResponseStatus::Error {
                    code: 500,
                    message: e.to_string(),
                },
            }),
        }
    }

    /// Batch inference
    pub async fn batch_infer(
        &self,
        requests: Vec<InferenceRequest>,
    ) -> Result<Vec<InferenceResponse>> {
        let start_time = Instant::now();
        let mut responses = Vec::new();

        for batch in requests.chunks(self.config.max_batch_size) {
            let mut batch_inputs = HashMap::new();

            // Combine inputs from batch
            for (i, request) in batch.iter().enumerate() {
                for (name, tensor) in &request.inputs {
                    let batch_name = format!("{}_{}", name, i);
                    batch_inputs.insert(batch_name, tensor.clone());
                }
            }

            // Process batch
            let batch_outputs = self.process_inference(&batch_inputs).await?;

            // Split outputs back to individual responses
            for (i, request) in batch.iter().enumerate() {
                let mut outputs = HashMap::new();
                for name in request.inputs.keys() {
                    let batch_name = format!("{}_{}", name, i);
                    if let Some(output) = batch_outputs.get(&batch_name) {
                        outputs.insert(name.clone(), output.clone());
                    }
                }

                responses.push(InferenceResponse {
                    request_id: request.id.clone(),
                    outputs,
                    metadata: HashMap::new(),
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    status: ResponseStatus::Success,
                });
            }
        }

        // Update batch metrics
        self.update_batch_metrics(requests.len(), start_time).await;

        Ok(responses)
    }

    /// Process actual inference
    async fn process_inference(
        &self,
        inputs: &HashMap<String, MLTensor>,
    ) -> Result<HashMap<String, MLTensor>> {
        let model = self.model.read().await;

        // Simplified inference - in practice would use actual model inference
        let mut outputs = HashMap::new();
        for (name, tensor) in inputs {
            // Mock output - same as input for demonstration
            outputs.insert(format!("output_{}", name), tensor.clone());
        }

        Ok(outputs)
    }

    /// Start REST API server
    async fn start_rest_api(&self) -> Result<()> {
        // This would start an actual REST server (e.g., with warp, axum, or actix-web)
        // For demonstration, we'll just log that it's starting
        println!(
            "Starting REST API server on port {}",
            self.config.api_config.rest_port
        );

        // Simplified REST endpoints:
        // POST /predict - Single prediction
        // POST /batch_predict - Batch prediction
        // GET /health - Health check
        // GET /metrics - Server metrics
        // POST /model/update - Update model
        // GET /model/info - Model information

        Ok(())
    }

    /// Start gRPC API server
    async fn start_grpc_api(&self) -> Result<()> {
        // This would start an actual gRPC server (e.g., with tonic)
        println!(
            "Starting gRPC API server on port {}",
            self.config.api_config.grpc_port
        );

        // Simplified gRPC services:
        // ModelInference service with predict, batch_predict methods
        // ModelManagement service with update_model, get_info methods
        // HealthCheck service
        // Metrics service

        Ok(())
    }

    /// Start request processing workers
    async fn start_workers(&self) -> Result<()> {
        for _worker_id in 0..self.config.num_workers {
            let queue = self.request_queue.clone();
            let _config = self.config.clone();

            tokio::spawn(async move {
                loop {
                    // Process requests from queue
                    let request = {
                        let mut queue_guard = queue.lock().await;
                        queue_guard.pop_front()
                    };

                    if let Some(_request) = request {
                        // Process the request
                        sleep(Duration::from_millis(10)).await; // Simulate processing
                    } else {
                        // No requests, sleep briefly
                        sleep(Duration::from_millis(1)).await;
                    }
                }
            });
        }

        Ok(())
    }

    /// Start metrics collection
    async fn start_metrics_collection(&self) {
        let metrics = self.metrics.clone();
        let start_time = Instant::now();

        tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(1)).await;

                // Update uptime
                {
                    let mut m = metrics.lock().await;
                    m.uptime_seconds = start_time.elapsed().as_secs();
                }
            }
        });
    }

    /// Update server metrics
    async fn update_metrics(&self, start_time: Instant, success: bool) {
        let mut metrics = self.metrics.lock().await;
        metrics.total_requests += 1;

        if success {
            metrics.successful_requests += 1;
        } else {
            metrics.failed_requests += 1;
        }

        let latency = start_time.elapsed().as_millis() as f64;
        metrics.average_latency_ms =
            (metrics.average_latency_ms * (metrics.total_requests - 1) as f64 + latency)
                / metrics.total_requests as f64;
    }

    /// Update batch metrics
    async fn update_batch_metrics(&self, batch_size: usize, start_time: Instant) {
        let mut metrics = self.metrics.lock().await;
        metrics.batch_stats.total_batches += 1;

        let current_avg = metrics.batch_stats.average_batch_size;
        let total_batches = metrics.batch_stats.total_batches as f64;
        metrics.batch_stats.average_batch_size =
            (current_avg * (total_batches - 1.0) + batch_size as f64) / total_batches;

        let processing_time = start_time.elapsed().as_millis() as f64;
        let current_time_avg = metrics.batch_stats.batch_processing_time_ms;
        metrics.batch_stats.batch_processing_time_ms =
            (current_time_avg * (total_batches - 1.0) + processing_time) / total_batches;
    }

    /// Get server health status
    pub async fn get_health(&self) -> HealthStatus {
        self.health_status.read().await.clone()
    }

    /// Get server metrics
    pub async fn get_metrics(&self) -> ServerMetrics {
        self.metrics.lock().await.clone()
    }

    /// Update model
    pub async fn update_model(&self, newmodel: MLModel) -> Result<()> {
        let start_time = Instant::now();

        {
            let mut model = self.model.write().await;
            *model = newmodel;
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().await;
            metrics.model_load_time_ms = start_time.elapsed().as_millis() as u64;
        }

        Ok(())
    }

    /// Get model information
    pub async fn get_model_info(&self) -> ModelInfo {
        let model = self.model.read().await;
        ModelInfo {
            name: model
                .metadata
                .model_name
                .clone()
                .unwrap_or_else(|| "Unknown".to_string()),
            framework: model.metadata.framework.clone(),
            version: model.metadata.model_version.clone(),
            inputshapes: model.metadata.inputshapes.clone(),
            outputshapes: model.metadata.outputshapes.clone(),
            parameters: model.weights.len(),
            loaded_at: Instant::now(), // Simplified
        }
    }

    /// Graceful shutdown
    pub async fn shutdown(&self) -> Result<()> {
        {
            let mut status = self.health_status.write().await;
            *status = HealthStatus::Stopping;
        }

        // Wait for in-flight requests to complete
        sleep(Duration::from_millis(self.config.timeout_ms)).await;

        // Clear request queue
        {
            let mut queue = self.request_queue.lock().await;
            queue.clear();
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub framework: String,
    pub version: Option<String>,
    pub inputshapes: HashMap<String, Vec<usize>>,
    pub outputshapes: HashMap<String, Vec<usize>>,
    pub parameters: usize,
    pub loaded_at: Instant,
}

/// REST API utilities
pub mod rest {
    use super::*;

    #[derive(Debug, Serialize, Deserialize)]
    pub struct PredictRequest {
        pub inputs: HashMap<String, Vec<f32>>,
        pub metadata: Option<HashMap<String, serde_json::Value>>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct PredictResponse {
        pub outputs: HashMap<String, Vec<f32>>,
        pub metadata: HashMap<String, serde_json::Value>,
        pub processing_time_ms: u64,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct BatchPredictRequest {
        pub inputs: Vec<HashMap<String, Vec<f32>>>,
        pub metadata: Option<HashMap<String, serde_json::Value>>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct BatchPredictResponse {
        pub outputs: Vec<HashMap<String, Vec<f32>>>,
        pub metadata: HashMap<String, serde_json::Value>,
        pub processing_time_ms: u64,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct HealthResponse {
        pub status: String,
        pub uptime_seconds: u64,
        pub version: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct MetricsResponse {
        pub total_requests: u64,
        pub successful_requests: u64,
        pub failed_requests: u64,
        pub average_latency_ms: f64,
        pub requests_per_second: f64,
        pub queue_size: usize,
        pub uptime_seconds: u64,
    }

    /// Convert MLTensor to REST format
    pub fn tensor_to_rest(tensor: &MLTensor) -> Vec<f32> {
        tensor.data.as_slice().unwrap().to_vec()
    }

    /// Convert REST format to MLTensor
    pub fn rest_to_tensor(
        data: Vec<f32>,
        shape: Vec<usize>,
        name: Option<String>,
    ) -> Result<MLTensor> {
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| IoError::Other(e.to_string()))?;
        Ok(MLTensor::new(array, name))
    }
}

/// gRPC utilities
pub mod grpc {
    use super::*;

    // gRPC message definitions would go here
    // These would typically be generated from .proto files

    #[derive(Debug, Clone)]
    pub struct GrpcTensor {
        pub name: String,
        pub shape: Vec<i64>,
        pub dtype: String,
        pub data: Vec<u8>,
    }

    #[derive(Debug, Clone)]
    pub struct GrpcPredictRequest {
        pub model_name: String,
        pub inputs: Vec<GrpcTensor>,
        pub metadata: HashMap<String, String>,
    }

    #[derive(Debug, Clone)]
    pub struct GrpcPredictResponse {
        pub outputs: Vec<GrpcTensor>,
        pub metadata: HashMap<String, String>,
        pub status: GrpcStatus,
    }

    #[derive(Debug, Clone)]
    pub struct GrpcStatus {
        pub code: i32,
        pub message: String,
    }

    /// Convert MLTensor to gRPC format
    pub fn tensor_to_grpc(tensor: &MLTensor) -> GrpcTensor {
        GrpcTensor {
            name: tensor.metadata.name.clone().unwrap_or_default(),
            shape: tensor.metadata.shape.iter().map(|&s| s as i64).collect(),
            dtype: format!("{:?}", tensor.metadata.dtype),
            data: tensor
                .data
                .as_slice()
                .unwrap()
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
        }
    }

    /// Convert gRPC format to MLTensor
    pub fn grpc_to_tensor(grpctensor: &GrpcTensor) -> Result<MLTensor> {
        let shape: Vec<usize> = grpctensor.shape.iter().map(|&s| s as usize).collect();

        // Convert bytes back to f32
        let float_data: Vec<f32> = grpctensor
            .data
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().unwrap();
                f32::from_le_bytes(bytes)
            })
            .collect();

        let array = ArrayD::from_shape_vec(IxDyn(&shape), float_data)
            .map_err(|e| IoError::Other(e.to_string()))?;

        Ok(MLTensor::new(array, Some(grpctensor.name.clone())))
    }
}

/// Load balancer for multiple model servers
pub struct LoadBalancer {
    servers: Vec<ModelServer>,
    strategy: LoadBalancingStrategy,
    health_checker: HealthChecker,
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin { weights: Vec<f32> },
    Random,
    HealthBased,
}

pub struct HealthChecker {
    check_interval: Duration,
    timeout: Duration,
}

#[cfg(feature = "async")]
impl LoadBalancer {
    pub fn new(servers: Vec<ModelServer>, strategy: LoadBalancingStrategy) -> Self {
        Self {
            servers,
            strategy,
            health_checker: HealthChecker {
                check_interval: Duration::from_secs(30),
                timeout: Duration::from_secs(5),
            },
        }
    }

    /// Route request to appropriate server
    pub async fn route_request(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let server = self.select_server().await?;
        server.infer(request).await
    }

    /// Select server based on load balancing strategy
    async fn select_server(&self) -> Result<&ModelServer> {
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                // Simplified round-robin
                Ok(&self.servers[0])
            }
            LoadBalancingStrategy::HealthBased => {
                // Select first healthy server
                for server in &self.servers {
                    if matches!(server.get_health().await, HealthStatus::Healthy) {
                        return Ok(server);
                    }
                }
                Err(IoError::Other("No healthy servers available".to_string()))
            }
            _ => Ok(&self.servers[0]), // Simplified
        }
    }

    /// Start health checking
    pub async fn start_health_checking(&self) {
        let interval = self.health_checker.check_interval;

        tokio::spawn(async move {
            loop {
                sleep(interval).await;
                // Check health of all servers
                // This would be implemented with actual health checks
            }
        });
    }
}
