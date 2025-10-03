//! Specialized hardware support module
//!
//! This module provides support for specialized hardware accelerators including
//! FPGAs, custom ASICs, and other domain-specific processors.

pub mod accelerator;
pub mod custom_asic;
pub mod device_manager;
pub mod fpga;
pub mod kernel_compiler;
pub mod memory_mapping;
pub mod model_partitioning;
pub mod partial_reconfiguration;
pub use accelerator::{Accelerator, AcceleratorCapabilities, AcceleratorType};
pub use custom_asic::{ASICConfig, ASICOperation, CustomASIC, DataType, NativeOperation};
pub use device_manager::{DeviceInfo, DeviceManager, DeviceSelector};
pub use fpga::{FPGAConfig, FPGADevice, FPGAKernel};
pub use kernel_compiler::{CompilationTarget, KernelCompiler, OptimizationLevel};
pub use memory_mapping::{BufferAllocation, MemoryLayout, MemoryMapRequirements, MemoryMapper};
pub use model_partitioning::{
    LayerProfile, ModelPartition, ModelPartitioner, PartitioningStrategy,
};
pub use partial_reconfiguration::{
    DPRManager, PartialBitstream, PartialRegion, ReconfigurationState,
use crate::error::Result;
use scirs2_core::ndarray::prelude::*;
use std::sync::Arc;
/// Hardware acceleration configuration
#[derive(Debug, Clone)]
pub struct HardwareConfig {
    /// Device type to use
    pub device_type: AcceleratorType,
    /// Device ID (for multi-device systems)
    pub device_id: usize,
    /// Memory allocation strategy
    pub memory_strategy: MemoryStrategy,
    /// Kernel optimization level
    pub optimization_level: OptimizationLevel,
    /// Enable automatic kernel fusion
    pub enable_kernel_fusion: bool,
    /// Enable automatic memory layout optimization
    pub enable_layout_optimization: bool,
    /// Maximum batch size for kernels
    pub max_batch_size: usize,
    /// Precision mode
    pub precision_mode: PrecisionMode,
}
impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            device_type: AcceleratorType::CPU,
            device_id: 0,
            memory_strategy: MemoryStrategy::Automatic,
            optimization_level: OptimizationLevel::O2,
            enable_kernel_fusion: true,
            enable_layout_optimization: true,
            max_batch_size: 256,
            precision_mode: PrecisionMode::Mixed,
        }
    }
/// Memory allocation strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryStrategy {
    /// Automatic memory management
    Automatic,
    /// Preallocate all memory
    Preallocated,
    /// On-demand allocation
    OnDemand,
    /// Custom memory pool
    PoolBased(usize),
/// Precision mode for computations
pub enum PrecisionMode {
    /// Full precision (FP32)
    Full,
    /// Half precision (FP16)
    Half,
    /// Mixed precision (FP16 compute, FP32 accumulate)
    Mixed,
    /// Integer quantized (INT8)
    Quantized,
    /// Binary neural networks
    Binary,
/// Hardware-accelerated neural network layer
pub trait HardwareLayer: Send + Sync {
    /// Compile the layer for specific hardware
    fn compile(&mut self, device: &dyn Accelerator, config: &HardwareConfig) -> Result<()>;
    /// Execute forward pass on hardware
    fn forward_hardware(
        &self,
        input: &ArrayView2<f32>,
        device: &dyn Accelerator,
    ) -> Result<Array2<f32>>;
    /// Execute backward pass on hardware
    fn backward_hardware(
        grad_output: &ArrayView2<f32>,
    /// Get memory requirements
    fn memory_requirements(&self) -> MemoryRequirements;
    /// Check if layer is compiled for hardware
    fn is_compiled(&self) -> bool;
/// Memory requirements for a layer
pub struct MemoryRequirements {
    /// Input buffer size in bytes
    pub input_size: usize,
    /// Output buffer size in bytes
    pub output_size: usize,
    /// Weight buffer size in bytes
    pub weight_size: usize,
    /// Workspace size for computations
    pub workspace_size: usize,
    /// Preferred memory alignment
    pub alignment: usize,
/// Hardware execution context
pub struct HardwareContext {
    device_manager: DeviceManager,
    active_device: Arc<dyn Accelerator>,
    memory_mapper: MemoryMapper,
    kernel_compiler: KernelCompiler,
    config: HardwareConfig,
impl HardwareContext {
    /// Create a new hardware context
    pub fn new(config: HardwareConfig) -> Result<Self> {
        let device_manager = DeviceManager::new()?;
        let active_device = device_manager.get_device(_config.device_type, config.device_id)?;
        let memory_mapper = MemoryMapper::new(active_device.clone(), config.memory_strategy)?;
        let kernel_compiler = KernelCompiler::new(_config.optimization_level);
        Ok(Self {
            device_manager,
            active_device,
            memory_mapper,
            kernel_compiler,
            config,
        })
    /// List available devices
    pub fn list_devices(&self) -> Vec<DeviceInfo> {
        self.device_manager.list_devices()
    /// Switch to a different device
    pub fn switch_device(&mut self, device_type: AcceleratorType, deviceid: usize) -> Result<()> {
        self.active_device = self.device_manager.get_device(device_type, device_id)?;
        self.memory_mapper =
            MemoryMapper::new(self.active_device.clone(), self.config.memory_strategy)?;
        self.config.device_type = device_type;
        self.config.device_id = device_id;
        Ok(())
    /// Compile a model for hardware execution
    pub fn compile_model(&mut self, model: &mut dyn HardwareModel) -> Result<()> {
        model.compile(&*self.active_device, &self.config)?;
    /// Execute a model on hardware
    pub fn execute_model(
        model: &dyn HardwareModel,
    ) -> Result<Array2<f32>> {
        if !model.is_compiled() {
            return Err(crate::error::NeuralError::InvalidArgument(
                "Model must be compiled before execution".to_string(),
            ));
        model.forward_hardware(input, &*self.active_device)
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStatistics {
        self.memory_mapper.get_statistics()
    /// Optimize memory layout for a model
    pub fn optimize_memory_layout(&mut self, model: &dyn HardwareModel) -> Result<()> {
        if !self.config.enable_layout_optimization {
            return Ok(());
        let requirements = model.memory_requirements();
        self.memory_mapper.optimize_layout(&requirements)?;
/// Hardware-accelerated model trait
pub trait HardwareModel: Send + Sync {
    /// Compile the model for hardware
    /// Forward pass on hardware
    /// Get total memory requirements
    fn memory_requirements(&self) -> Vec<MemoryRequirements>;
    /// Check if model is compiled
    /// Get model statistics
    fn statistics(&self) -> ModelStatistics;
/// Model statistics for hardware execution
pub struct ModelStatistics {
    /// Total number of parameters
    pub total_params: usize,
    /// Total FLOPs for forward pass
    pub total_flops: usize,
    /// Memory bandwidth required (bytes/sec)
    pub memory_bandwidth: usize,
    /// Compute intensity (FLOPs/byte)
    pub compute_intensity: f32,
    /// Estimated latency in microseconds
    pub estimated_latency: f32,
/// Memory usage statistics
pub struct MemoryStatistics {
    /// Total allocated memory in bytes
    pub allocated: usize,
    /// Memory currently in use
    pub used: usize,
    /// Peak memory usage
    pub peak: usize,
    /// Number of allocations
    pub num_allocations: usize,
    /// Memory fragmentation ratio
    pub fragmentation: f32,
/// Kernel fusion optimizer
pub struct KernelFusion {
    enabled: bool,
    fusion_threshold: usize,
    max_fusion_depth: usize,
impl KernelFusion {
    /// Create a new kernel fusion optimizer
    pub fn new(enabled: bool) -> Self {
            enabled,
            fusion_threshold: 2,
            max_fusion_depth: 5,
    /// Analyze and fuse eligible kernels
    pub fn optimize_kernels(&self, kernels: Vec<KernelDescriptor>) -> Result<Vec<FusedKernel>> {
        if !self._enabled || kernels.len() < self.fusion_threshold {
            // Convert kernels to fused kernels without fusion
            return Ok(kernels
                .into_iter()
                .map(|k| FusedKernel {
                    kernels: vec![k],
                    fusion_type: FusionType::None,
                })
                .collect());
        // Simple fusion strategy: fuse consecutive element-wise operations
        let mut fused = Vec::new();
        let mut i = 0;
        while i < kernels.len() {
            if i + 1 < kernels.len() && self.can_fuse(&kernels[i], &kernels[i + 1]) {
                let mut fusion_group = vec![kernels[i].clone(), kernels[i + 1].clone()];
                i += 2;
                // Try to extend the fusion group
                while i < kernels.len()
                    && fusion_group.len() < self.max_fusion_depth
                    && self.can_fuse(fusion_group.last().unwrap(), &kernels[i])
                {
                    fusion_group.push(kernels[i].clone());
                    i += 1;
                }
                fused.push(FusedKernel {
                    kernels: fusion_group,
                    fusion_type: FusionType::ElementWise,
                });
            } else {
                    kernels: vec![kernels[i].clone()],
                i += 1;
            }
        Ok(fused)
    /// Check if two kernels can be fused
    fn can_fuse(&self, kernel1: &KernelDescriptor, kernel2: &KernelDescriptor) -> bool {
        // Simple heuristic: fuse element-wise operations with matching shapes
        kernel1.operation_type.is_element_wise()
            && kernel2.operation_type.is_element_wise()
            && kernel1.outputshape == kernel2.inputshape
/// Kernel descriptor
pub struct KernelDescriptor {
    pub name: String,
    pub operation_type: OperationType,
    pub inputshape: Vec<usize>,
    pub outputshape: Vec<usize>,
    pub memory_access_pattern: MemoryAccessPattern,
/// Operation type
#[derive(Debug, Clone, PartialEq)]
pub enum OperationType {
    MatMul,
    Conv2D,
    ElementWise(ElementWiseOp),
    Reduction(ReductionOp),
    Reshape,
    Transpose,
impl OperationType {
    fn is_element_wise(&self) -> bool {
        matches!(self, OperationType::ElementWise(_))
/// Element-wise operations
pub enum ElementWiseOp {
    Add,
    Multiply,
    ReLU,
    Sigmoid,
    Tanh,
/// Reduction operations
pub enum ReductionOp {
    Sum,
    Mean,
    Max,
    Min,
/// Memory access pattern
pub enum MemoryAccessPattern {
    Sequential,
    Strided(usize),
    Random,
    Tiled(usize, usize),
/// Fused kernel
pub struct FusedKernel {
    pub kernels: Vec<KernelDescriptor>,
    pub fusion_type: FusionType,
/// Fusion type
pub enum FusionType {
    None,
    ElementWise,
    ConvBiasReLU,
    MatMulBiasActivation,
    Custom(String),
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_hardware_config_default() {
        let config = HardwareConfig::default();
        assert_eq!(config.device_type, AcceleratorType::CPU);
        assert_eq!(config.optimization_level, OptimizationLevel::O2);
        assert!(config.enable_kernel_fusion);
    fn test_kernel_fusion() {
        let fusion = KernelFusion::new(true);
        let kernels = vec![
            KernelDescriptor {
                name: "add".to_string(),
                operation_type: OperationType::ElementWise(ElementWiseOp::Add),
                inputshape: vec![32, 64],
                outputshape: vec![32, 64],
                memory_access_pattern: MemoryAccessPattern::Sequential,
            },
                name: "relu".to_string(),
                operation_type: OperationType::ElementWise(ElementWiseOp::ReLU),
        ];
        let fused = fusion.optimize_kernels(kernels).unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].kernels.len(), 2);
        assert_eq!(fused[0].fusion_type, FusionType::ElementWise);
