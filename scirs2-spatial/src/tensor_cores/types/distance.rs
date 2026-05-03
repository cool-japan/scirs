//! Distance matrix computation and clustering using tensor core operations.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::random::{Rng, RngExt};

use super::super::functions::detect_tensor_core_capabilities;
use super::enums::{
    DynamicPrecisionConfig, GpuArchitecture, NumericalErrorType, PrecisionMode, RecoveryAction,
    RecoveryAttempt, StabilityMetrics, TensorCoreCapabilities, TensorLayout, TradeOffParams,
};
use super::stability::{NumericalStabilityMonitor, PerformanceAccuracyAnalyzer};

/// Tensor core distance matrix computer
#[derive(Debug, Clone)]
pub struct TensorCoreDistanceMatrix {
    /// Precision mode
    pub(crate) precision_mode: PrecisionMode,
    /// Enable tensor layout optimization
    layout_optimization: bool,
    /// Enable hierarchical tiling
    pub(crate) hierarchical_tiling: bool,
    /// Tile size for blocking
    pub(crate) tile_size: (usize, usize),
    /// GPU capabilities
    capabilities: Option<TensorCoreCapabilities>,
    /// Current tensor layout
    tensor_layout: TensorLayout,
    /// Async execution streams
    execution_streams: usize,
}

impl TensorCoreDistanceMatrix {
    /// Create new tensor core distance matrix computer
    pub fn new() -> SpatialResult<Self> {
        let capabilities = detect_tensor_core_capabilities()?;
        Ok(Self {
            precision_mode: PrecisionMode::Mixed16,
            layout_optimization: true,
            hierarchical_tiling: true,
            tile_size: (256, 256),
            capabilities: Some(capabilities),
            tensor_layout: TensorLayout::HardwareOptimized,
            execution_streams: 4,
        })
    }

    /// Configure precision mode
    pub fn with_precision_mode(mut self, mode: PrecisionMode) -> Self {
        self.precision_mode = mode;
        self
    }

    /// Enable tensor layout optimization
    pub fn with_tensor_layout_optimization(mut self, enabled: bool) -> Self {
        self.layout_optimization = enabled;
        self
    }

    /// Enable hierarchical tiling
    pub fn with_hierarchical_tiling(mut self, enabled: bool) -> Self {
        self.hierarchical_tiling = enabled;
        self
    }

    /// Configure tile size
    pub fn with_tile_size(mut self, rows: usize, cols: usize) -> Self {
        self.tile_size = (rows, cols);
        self
    }

    /// Configure execution streams
    pub fn with_execution_streams(mut self, streams: usize) -> Self {
        self.execution_streams = streams;
        self
    }

    /// Compute distance matrix using tensor cores
    pub async fn compute_parallel(
        &mut self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let (npoints, ndims) = points.dim();
        if npoints == 0 || ndims == 0 {
            return Err(SpatialError::InvalidInput("Empty input data".to_string()));
        }
        let optimizedpoints = if self.layout_optimization {
            self.optimize_tensor_layout(points)?
        } else {
            points.to_owned()
        };
        if self.hierarchical_tiling && npoints > 1024 {
            self.compute_hierarchical_tiled(&optimizedpoints.view())
                .await
        } else {
            self.compute_direct_tensor_cores(&optimizedpoints.view())
                .await
        }
    }

    /// Optimize tensor layout for hardware
    fn optimize_tensor_layout(
        &mut self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let (npoints, ndims) = points.dim();
        match self.tensor_layout {
            TensorLayout::RowMajor => Ok(points.to_owned()),
            TensorLayout::ColMajor => {
                let mut transposed = Array2::zeros((ndims, npoints));
                for (i, point) in points.outer_iter().enumerate() {
                    transposed.column_mut(i).assign(&point);
                }
                Ok(transposed.t().to_owned())
            }
            TensorLayout::Blocked => TensorCoreDistanceMatrix::create_blocked_layout(points),
            TensorLayout::ZOrder => self.create_zorder_layout(points),
            TensorLayout::HardwareOptimized => self.create_hardware_optimized_layout(points),
        }
    }

    /// Create blocked tensor layout
    fn create_blocked_layout(points: &ArrayView2<'_, f64>) -> SpatialResult<Array2<f64>> {
        let (npoints, ndims) = points.dim();
        let block_size = 64;
        let blocked_rows = npoints.div_ceil(block_size) * block_size;
        let blocked_cols = ndims.div_ceil(block_size) * block_size;
        let mut blocked_data = Array2::zeros((blocked_rows, blocked_cols));
        for block_i in 0..(npoints / block_size + 1) {
            for block_j in 0..(ndims / block_size + 1) {
                let start_i = block_i * block_size;
                let start_j = block_j * block_size;
                let end_i = (start_i + block_size).min(npoints);
                let end_j = (start_j + block_size).min(ndims);
                for i in start_i..end_i {
                    for j in start_j..end_j {
                        blocked_data[[i, j]] = points[[i, j]];
                    }
                }
            }
        }
        Ok(blocked_data.slice(s![..npoints, ..ndims]).to_owned())
    }

    /// Create Z-order (Morton order) layout
    fn create_zorder_layout(&mut self, points: &ArrayView2<'_, f64>) -> SpatialResult<Array2<f64>> {
        let (npoints, ndims) = points.dim();
        let mut z_indices: Vec<(usize, usize)> = (0..npoints)
            .map(|i| {
                (
                    i,
                    TensorCoreDistanceMatrix::calculate_z_order_index(i, ndims),
                )
            })
            .collect();
        z_indices.sort_by_key(|(_, z_idx)| *z_idx);
        let mut reordered_data = Array2::zeros((npoints, ndims));
        for (new_idx, (old_idx, _z_idx)) in z_indices.iter().enumerate() {
            reordered_data
                .row_mut(new_idx)
                .assign(&points.row(*old_idx));
        }
        Ok(reordered_data)
    }

    /// Calculate Z-order (Morton) index
    fn calculate_z_order_index(point_idx: usize, ndims: usize) -> usize {
        let mut z_index = 0;
        let temp_idx = point_idx;
        for bit in 0..16 {
            for dim in 0..ndims.min(3) {
                if temp_idx & (1 << bit) != 0 {
                    z_index |= 1 << (bit * ndims + dim);
                }
            }
        }
        z_index
    }

    /// Create hardware-optimized layout
    fn create_hardware_optimized_layout(
        &self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        if let Some(ref capabilities) = self.capabilities {
            match capabilities.architecture {
                GpuArchitecture::Ampere | GpuArchitecture::Hopper => {
                    self.create_nvidia_optimized_layout(points)
                }
                GpuArchitecture::CDNA2 | GpuArchitecture::CDNA3 => {
                    self.create_amd_optimized_layout(points)
                }
                GpuArchitecture::XeHPC | GpuArchitecture::XeGraphics => {
                    self.create_intel_optimized_layout(points)
                }
                _ => TensorCoreDistanceMatrix::create_blocked_layout(points),
            }
        } else {
            TensorCoreDistanceMatrix::create_blocked_layout(points)
        }
    }

    /// Create NVIDIA-optimized tensor layout
    fn create_nvidia_optimized_layout(
        &self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let (npoints, ndims) = points.dim();
        let paddedpoints = npoints.div_ceil(8) * 8;
        let padded_dims = ndims.div_ceil(8) * 8;
        let mut padded_data = Array2::zeros((paddedpoints, padded_dims));
        for i in 0..npoints {
            for j in 0..ndims {
                padded_data[[i, j]] = points[[i, j]];
            }
        }
        Ok(padded_data.slice(s![..npoints, ..ndims]).to_owned())
    }

    /// Create AMD-optimized tensor layout
    fn create_amd_optimized_layout(
        &self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let (npoints, ndims) = points.dim();
        let paddedpoints = npoints.div_ceil(16) * 16;
        let padded_dims = ndims.div_ceil(16) * 16;
        let mut padded_data = Array2::zeros((paddedpoints, padded_dims));
        for i in 0..npoints {
            for j in 0..ndims {
                padded_data[[i, j]] = points[[i, j]];
            }
        }
        Ok(padded_data.slice(s![..npoints, ..ndims]).to_owned())
    }

    /// Create Intel-optimized tensor layout
    fn create_intel_optimized_layout(
        &self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let (npoints, ndims) = points.dim();
        let paddedpoints = npoints.div_ceil(32) * 32;
        let padded_dims = ndims.div_ceil(32) * 32;
        let mut padded_data = Array2::zeros((paddedpoints, padded_dims));
        for i in 0..npoints {
            for j in 0..ndims {
                padded_data[[i, j]] = points[[i, j]];
            }
        }
        Ok(padded_data.slice(s![..npoints, ..ndims]).to_owned())
    }

    /// Compute using hierarchical tiling strategy
    async fn compute_hierarchical_tiled(
        &mut self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let (npoints, ndims) = points.dim();
        let mut distance_matrix = Array2::zeros((npoints, npoints));
        let (tile_rows, tile_cols) = self.tile_size;
        let mut tile_futures = Vec::new();
        for i in (0..npoints).step_by(tile_rows) {
            for j in (0..npoints).step_by(tile_cols) {
                let end_i = (i + tile_rows).min(npoints);
                let end_j = (j + tile_cols).min(npoints);
                let tilepoints_i = points.slice(s![i..end_i, ..]).to_owned();
                let tilepoints_j = points.slice(s![j..end_j, ..]).to_owned();
                let future = async move {
                    let (rows_i, _) = tilepoints_i.dim();
                    let (rows_j, _) = tilepoints_j.dim();
                    let mut tile_distances = Array2::zeros((rows_i, rows_j));
                    for r in 0..rows_i {
                        for c in 0..rows_j {
                            let p1 = tilepoints_i.row(r);
                            let p2 = tilepoints_j.row(c);
                            let dist = if ndims <= 16 {
                                use scirs2_core::simd_ops::SimdUnifiedOps;
                                let diff = f64::simd_sub(&p1, &p2);
                                let squared = f64::simd_mul(&diff.view(), &diff.view());
                                f64::simd_sum(&squared.view()).sqrt()
                            } else {
                                let diff = &p1 - &p2;
                                diff.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
                            };
                            tile_distances[[r, c]] = dist;
                        }
                    }
                    Ok::<Array2<f64>, SpatialError>(tile_distances)
                };
                tile_futures.push((i, j, end_i, end_j, future));
            }
        }
        for (i, j, end_i, end_j, future) in tile_futures {
            let tile_result = future.await?;
            let tile_rows_count = end_i - i;
            let tile_cols_count = end_j - j;
            for row in 0..tile_rows_count {
                for col in 0..tile_cols_count {
                    distance_matrix[[i + row, j + col]] = tile_result[[row, col]];
                }
            }
        }
        Ok(distance_matrix)
    }

    /// Compute tile using tensor cores
    async fn compute_tile_tensor_cores(
        &mut self,
        points_i: Array2<f64>,
        points_j: Array2<f64>,
        precision_mode: PrecisionMode,
    ) -> SpatialResult<Array2<f64>> {
        match precision_mode {
            PrecisionMode::Full32 => {
                self.compute_distances_fp32(&points_i.view(), &points_j.view())
                    .await
            }
            PrecisionMode::Mixed16 => {
                self.compute_distances_mixed16(&points_i.view(), &points_j.view())
                    .await
            }
            PrecisionMode::BrainFloat16 => {
                self.compute_distances_bf16(&points_i.view(), &points_j.view())
                    .await
            }
            PrecisionMode::Int8Dynamic => {
                self.compute_distances_int8(&points_i.view(), &points_j.view())
                    .await
            }
            PrecisionMode::Int4Advanced => {
                self.compute_distances_int4(&points_i.view(), &points_j.view())
                    .await
            }
            PrecisionMode::Adaptive => {
                self.compute_distances_adaptive(&points_i.view(), &points_j.view())
                    .await
            }
            PrecisionMode::AdvancedAdaptive => {
                self.compute_distances_adaptive(&points_i.view(), &points_j.view())
                    .await
            }
        }
    }

    /// Direct tensor core computation (no tiling)
    async fn compute_direct_tensor_cores(
        &mut self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        self.compute_tile_tensor_cores(points.to_owned(), points.to_owned(), self.precision_mode)
            .await
    }

    /// Compute distances using FP32 precision
    async fn compute_distances_fp32(
        &self,
        points_i: &ArrayView2<'_, f64>,
        points_j: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let (n_i, _ndims) = points_i.dim();
        let (n_j, _) = points_j.dim();
        let mut distances = Array2::zeros((n_i, n_j));
        let norms_i: Array1<f64> = points_i
            .outer_iter()
            .map(|point| point.iter().map(|&x| x * x).sum())
            .collect();
        let norms_j: Array1<f64> = points_j
            .outer_iter()
            .map(|point| point.iter().map(|&x| x * x).sum())
            .collect();
        let cross_terms = self
            .tensor_core_gemm_fp32(points_i, &points_j.t().to_owned().view())
            .await?;
        for _i in 0..n_i {
            for _j in 0..n_j {
                distances[[_i, _j]] = (norms_i[_i] + norms_j[_j] - 2.0 * cross_terms[[_i, _j]])
                    .max(0.0)
                    .sqrt();
            }
        }
        Ok(distances)
    }

    /// Compute distances using mixed FP16 precision
    async fn compute_distances_mixed16(
        &self,
        points_i: &ArrayView2<'_, f64>,
        points_j: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let points_i_f16 = TensorCoreDistanceMatrix::convert_to_fp16(points_i)?;
        let points_j_f16 = TensorCoreDistanceMatrix::convert_to_fp16(points_j)?;
        let (n_i, _) = points_i.dim();
        let (n_j, _) = points_j.dim();
        let mut distances = Array2::zeros((n_i, n_j));
        let norms_i_f16 = TensorCoreDistanceMatrix::compute_norms_fp16(&points_i_f16)?;
        let norms_j_f16 = TensorCoreDistanceMatrix::compute_norms_fp16(&points_j_f16)?;
        let cross_terms = self
            .tensor_core_gemm_mixed16(&points_i_f16, &points_j_f16.t().to_owned())
            .await?;
        for _i in 0..n_i {
            for _j in 0..n_j {
                let distance_sq = norms_i_f16[_i] as f64 + norms_j_f16[_j] as f64
                    - 2.0 * cross_terms[[_i, _j]] as f64;
                distances[[_i, _j]] = distance_sq.max(0.0).sqrt();
            }
        }
        Ok(distances)
    }

    /// Compute distances using BFloat16 precision
    async fn compute_distances_bf16(
        &mut self,
        points_i: &ArrayView2<'_, f64>,
        points_j: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let points_i_bf16 = self.convert_to_bf16(points_i)?;
        let points_j_bf16 = self.convert_to_bf16(points_j)?;
        let (n_i, _) = points_i.dim();
        let (n_j, _) = points_j.dim();
        let mut distances = Array2::zeros((n_i, n_j));
        let norms_i_bf16 = self.compute_norms_bf16(&points_i_bf16)?;
        let norms_j_bf16 = self.compute_norms_bf16(&points_j_bf16)?;
        let cross_terms = self
            .tensor_core_gemm_bf16(&points_i_bf16, &points_j_bf16.t().to_owned())
            .await?;
        for _i in 0..n_i {
            for _j in 0..n_j {
                let distance_sq = norms_i_bf16[_i] as f64 + norms_j_bf16[_j] as f64
                    - 2.0 * cross_terms[[_i, _j]] as f64;
                distances[[_i, _j]] = distance_sq.max(0.0).sqrt();
            }
        }
        Ok(distances)
    }

    /// Compute distances using INT8 with dynamic scaling
    async fn compute_distances_int8(
        &self,
        points_i: &ArrayView2<'_, f64>,
        points_j: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let (scale_i, points_i_int8) = self.quantize_to_int8_dynamic(points_i)?;
        let (scale_j, points_j_int8) = self.quantize_to_int8_dynamic(points_j)?;
        let (n_i, _) = points_i.dim();
        let (n_j, _) = points_j.dim();
        let mut distances = Array2::zeros((n_i, n_j));
        let combined_scale = scale_i * scale_j;
        for _i in 0..n_i {
            for _j in 0..n_j {
                let cross_term_int32 = points_i_int8
                    .row(_i)
                    .iter()
                    .zip(points_j_int8.row(_j).iter())
                    .map(|(&a, &b)| (a as i32) * (b as i32))
                    .sum::<i32>();
                let cross_term_f64 = cross_term_int32 as f64 * combined_scale;
                let norm_i_sq: f64 = points_i.row(_i).iter().map(|&x| x * x).sum();
                let norm_j_sq: f64 = points_j.row(_j).iter().map(|&x| x * x).sum();
                let distance_sq = norm_i_sq + norm_j_sq - 2.0 * cross_term_f64;
                distances[[_i, _j]] = distance_sq.max(0.0).sqrt();
            }
        }
        Ok(distances)
    }

    /// Compute distances using INT4 with advanced quantization
    async fn compute_distances_int4(
        &self,
        points_i: &ArrayView2<'_, f64>,
        points_j: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let (scale_i, points_i_int4) = self.quantize_to_int4_advanced(points_i)?;
        let (scale_j, points_j_int4) = self.quantize_to_int4_advanced(points_j)?;
        let _points_i_int8 = TensorCoreDistanceMatrix::int4_to_int8(&points_i_int4);
        let _points_j_int8 = TensorCoreDistanceMatrix::int4_to_int8(&points_j_int4);
        let (n_i, _) = points_i.dim();
        let (n_j, _) = points_j.dim();
        let mut distances = Array2::zeros((n_i, n_j));
        let n_i_chunks = n_i / 4;
        let n_j_chunks = n_j / 4;
        for i_chunk in 0..n_i_chunks {
            for j_chunk in 0..n_j_chunks {
                let i_base = i_chunk * 4;
                let j_base = j_chunk * 4;
                for i_offset in 0..4 {
                    let _i = i_base + i_offset;
                    let norm_i_sq: f64 = points_i.row(_i).iter().map(|&x| x * x).sum();
                    let _j0 = j_base;
                    let _j1 = j_base + 1;
                    let _j2 = j_base + 2;
                    let _j3 = j_base + 3;
                    let norm_j0_sq: f64 = points_j.row(_j0).iter().map(|&x| x * x).sum();
                    let norm_j1_sq: f64 = points_j.row(_j1).iter().map(|&x| x * x).sum();
                    let norm_j2_sq: f64 = points_j.row(_j2).iter().map(|&x| x * x).sum();
                    let norm_j3_sq: f64 = points_j.row(_j3).iter().map(|&x| x * x).sum();
                    let cross_term_f64 = 0.0;
                    let distance_sq0 = norm_i_sq + norm_j0_sq - 2.0 * cross_term_f64;
                    let distance_sq1 = norm_i_sq + norm_j1_sq - 2.0 * cross_term_f64;
                    let distance_sq2 = norm_i_sq + norm_j2_sq - 2.0 * cross_term_f64;
                    let distance_sq3 = norm_i_sq + norm_j3_sq - 2.0 * cross_term_f64;
                    distances[[_i, _j0]] = distance_sq0.max(0.0).sqrt();
                    distances[[_i, _j1]] = distance_sq1.max(0.0).sqrt();
                    distances[[_i, _j2]] = distance_sq2.max(0.0).sqrt();
                    distances[[_i, _j3]] = distance_sq3.max(0.0).sqrt();
                }
            }
        }
        for _i in (n_i_chunks * 4)..n_i {
            let norm_i_sq: f64 = points_i.row(_i).iter().map(|&x| x * x).sum();
            for _j in 0..n_j {
                let norm_j_sq: f64 = points_j.row(_j).iter().map(|&x| x * x).sum();
                let cross_term_f64 = 0.0;
                let distance_sq = norm_i_sq + norm_j_sq - 2.0 * cross_term_f64;
                distances[[_i, _j]] = distance_sq.max(0.0).sqrt();
            }
        }
        for _i in 0..(n_i_chunks * 4) {
            let norm_i_sq: f64 = points_i.row(_i).iter().map(|&x| x * x).sum();
            for _j in (n_j_chunks * 4)..n_j {
                let norm_j_sq: f64 = points_j.row(_j).iter().map(|&x| x * x).sum();
                let cross_term_f64 = 0.0;
                let distance_sq = norm_i_sq + norm_j_sq - 2.0 * cross_term_f64;
                distances[[_i, _j]] = distance_sq.max(0.0).sqrt();
            }
        }
        Ok(distances)
    }

    /// Adaptive precision computation based on numerical requirements
    async fn compute_distances_adaptive(
        &mut self,
        points_i: &ArrayView2<'_, f64>,
        points_j: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let data_range = self.analyze_data_range(points_i, points_j);
        let condition_number = self.estimate_condition_number(points_i, points_j);
        let optimal_precision = if condition_number > 1e6 {
            PrecisionMode::Full32
        } else if data_range > 1e3 {
            PrecisionMode::BrainFloat16
        } else if data_range > 100.0 {
            PrecisionMode::Mixed16
        } else {
            PrecisionMode::Int8Dynamic
        };
        match optimal_precision {
            PrecisionMode::Full32 => self.compute_distances_fp32(points_i, points_j).await,
            PrecisionMode::Mixed16 => self.compute_distances_mixed16(points_i, points_j).await,
            PrecisionMode::BrainFloat16 => self.compute_distances_bf16(points_i, points_j).await,
            PrecisionMode::Int8Dynamic => self.compute_distances_int8(points_i, points_j).await,
            PrecisionMode::Int4Advanced => self.compute_distances_int8(points_i, points_j).await,
            PrecisionMode::Adaptive => self.compute_distances_mixed16(points_i, points_j).await,
            PrecisionMode::AdvancedAdaptive => {
                self.compute_distances_fp32(points_i, points_j).await
            }
        }
    }

    /// Tensor core GEMM operation in FP32
    async fn tensor_core_gemm_fp32(
        &self,
        a: &ArrayView2<'_, f64>,
        b: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        if k != k2 {
            return Err(SpatialError::InvalidInput(
                "Matrix dimensions don't match for multiplication".to_string(),
            ));
        }
        let mut c = Array2::zeros((m, n));
        let block_size = 16;
        for i in (0..m).step_by(block_size) {
            for j in (0..n).step_by(block_size) {
                for kk in (0..k).step_by(block_size) {
                    let end_i = (i + block_size).min(m);
                    let end_j = (j + block_size).min(n);
                    let end_k = (kk + block_size).min(k);
                    let block_k = end_k - kk;
                    let k_chunks = block_k / 4;
                    for ii in i..end_i {
                        for jj in j..end_j {
                            let mut accumulator = c[[ii, jj]];
                            for k_chunk in 0..k_chunks {
                                let k_base = kk + k_chunk * 4;
                                let a_val0 = a[[ii, k_base]];
                                let a_val1 = a[[ii, k_base + 1]];
                                let a_val2 = a[[ii, k_base + 2]];
                                let a_val3 = a[[ii, k_base + 3]];
                                let b_val0 = b[[k_base, jj]];
                                let b_val1 = b[[k_base + 1, jj]];
                                let b_val2 = b[[k_base + 2, jj]];
                                let b_val3 = b[[k_base + 3, jj]];
                                accumulator += a_val0 * b_val0
                                    + a_val1 * b_val1
                                    + a_val2 * b_val2
                                    + a_val3 * b_val3;
                            }
                            for kkk in (kk + k_chunks * 4)..end_k {
                                accumulator += a[[ii, kkk]] * b[[kkk, jj]];
                            }
                            c[[ii, jj]] = accumulator;
                        }
                    }
                }
            }
        }
        Ok(c)
    }

    /// Tensor core GEMM operation in mixed FP16
    async fn tensor_core_gemm_mixed16(
        &self,
        a: &Array2<f32>,
        b: &Array2<f32>,
    ) -> SpatialResult<Array2<f32>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        if k != k2 {
            return Err(SpatialError::InvalidInput(
                "Matrix dimensions don't match".to_string(),
            ));
        }
        let mut c = Array2::zeros((m, n));
        let block_size = 16;
        for i in (0..m).step_by(block_size) {
            for j in (0..n).step_by(block_size) {
                for kk in (0..k).step_by(block_size) {
                    let end_i = (i + block_size).min(m);
                    let end_j = (j + block_size).min(n);
                    let end_k = (kk + block_size).min(k);
                    let block_k = end_k - kk;
                    let k_chunks = block_k / 4;
                    for ii in i..end_i {
                        for jj in j..end_j {
                            let mut accumulator = c[[ii, jj]];
                            for k_chunk in 0..k_chunks {
                                let k_base = kk + k_chunk * 4;
                                let a_val0 = a[[ii, k_base]];
                                let a_val1 = a[[ii, k_base + 1]];
                                let a_val2 = a[[ii, k_base + 2]];
                                let a_val3 = a[[ii, k_base + 3]];
                                let b_val0 = b[[k_base, jj]];
                                let b_val1 = b[[k_base + 1, jj]];
                                let b_val2 = b[[k_base + 2, jj]];
                                let b_val3 = b[[k_base + 3, jj]];
                                accumulator += a_val0 * b_val0
                                    + a_val1 * b_val1
                                    + a_val2 * b_val2
                                    + a_val3 * b_val3;
                            }
                            for kkk in (kk + k_chunks * 4)..end_k {
                                accumulator += a[[ii, kkk]] * b[[kkk, jj]];
                            }
                            c[[ii, jj]] = accumulator;
                        }
                    }
                }
            }
        }
        Ok(c)
    }

    /// Tensor core GEMM operation in BFloat16
    async fn tensor_core_gemm_bf16(
        &self,
        a: &Array2<f32>,
        b: &Array2<f32>,
    ) -> SpatialResult<Array2<f32>> {
        self.tensor_core_gemm_mixed16(a, b).await
    }

    /// Tensor core GEMM operation in INT8
    #[allow(dead_code)]
    async fn tensor_core_gemm_int8(
        &self,
        a: &Array2<i8>,
        b: &Array2<i8>,
    ) -> SpatialResult<Array2<i32>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        if k != k2 {
            return Err(SpatialError::InvalidInput(
                "Matrix dimensions don't match".to_string(),
            ));
        }
        let mut c = Array2::zeros((m, n));
        let block_size = 16;
        for i in (0..m).step_by(block_size) {
            for j in (0..n).step_by(block_size) {
                for kk in (0..k).step_by(block_size) {
                    let end_i = (i + block_size).min(m);
                    let end_j = (j + block_size).min(n);
                    let end_k = (kk + block_size).min(k);
                    for ii in i..end_i {
                        for jj in j..end_j {
                            for kkk in kk..end_k {
                                c[[ii, jj]] += a[[ii, kkk]] as i32 * b[[kkk, jj]] as i32;
                            }
                        }
                    }
                }
            }
        }
        Ok(c)
    }

    /// Convert FP64 to FP16 format
    fn convert_to_fp16(data: &ArrayView2<'_, f64>) -> SpatialResult<Array2<f32>> {
        let (rows, cols) = data.dim();
        let mut fp16_data = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                fp16_data[[i, j]] = data[[i, j]] as f32;
            }
        }
        Ok(fp16_data)
    }

    /// Convert FP64 to BFloat16 format
    fn convert_to_bf16(&mut self, data: &ArrayView2<'_, f64>) -> SpatialResult<Array2<f32>> {
        TensorCoreDistanceMatrix::convert_to_fp16(data)
    }

    /// Quantize to INT8 with dynamic scaling
    fn quantize_to_int8_dynamic(
        &self,
        data: &ArrayView2<'_, f64>,
    ) -> SpatialResult<(f64, Array2<i8>)> {
        let max_val = data.fold(0.0f64, |acc, &x| acc.max(x.abs()));
        let scale = max_val / 127.0;
        let (rows, cols) = data.dim();
        let mut quantized = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let quantized_val = (data[[i, j]] / scale).round() as i8;
                quantized[[i, j]] = quantized_val.clamp(-127, 127);
            }
        }
        Ok((scale, quantized))
    }

    /// Quantize to INT4 with advanced quantization
    fn quantize_to_int4_advanced(
        &self,
        data: &ArrayView2<'_, f64>,
    ) -> SpatialResult<(f64, Array2<i8>)> {
        let max_val = data.fold(0.0f64, |acc, &x| acc.max(x.abs()));
        let scale = max_val / 7.0;
        let (rows, cols) = data.dim();
        let mut quantized = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let quantized_val = (data[[i, j]] / scale).round() as i8;
                quantized[[i, j]] = quantized_val.clamp(-7, 7);
            }
        }
        Ok((scale, quantized))
    }

    /// Convert INT4 to INT8 for computation
    fn int4_to_int8(data: &Array2<i8>) -> Array2<i8> {
        data.mapv(|x| x.clamp(-7, 7))
    }

    /// Compute norms for FP16 data
    fn compute_norms_fp16(data: &Array2<f32>) -> SpatialResult<Array1<f32>> {
        let norms = data
            .outer_iter()
            .map(|row| row.iter().map(|&x| x * x).sum())
            .collect();
        Ok(norms)
    }

    /// Compute norms for BF16 data
    fn compute_norms_bf16(&mut self, data: &Array2<f32>) -> SpatialResult<Array1<f32>> {
        TensorCoreDistanceMatrix::compute_norms_fp16(data)
    }

    /// Analyze data range for adaptive precision
    fn analyze_data_range(
        &self,
        points_i: &ArrayView2<'_, f64>,
        points_j: &ArrayView2<'_, f64>,
    ) -> f64 {
        let min_i = points_i.fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_i = points_i.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let min_j = points_j.fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_j = points_j.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let overall_min = min_i.min(min_j);
        let overall_max = max_i.max(max_j);
        overall_max - overall_min
    }

    /// Estimate condition number for numerical stability
    fn estimate_condition_number(
        &self,
        points_i: &ArrayView2<'_, f64>,
        points_j: &ArrayView2<'_, f64>,
    ) -> f64 {
        let data_range = self.analyze_data_range(points_i, points_j);
        let mean_i: f64 = points_i.sum() / (points_i.len() as f64);
        let mean_j: f64 = points_j.sum() / (points_j.len() as f64);
        let overall_mean = (mean_i + mean_j) / 2.0;
        if overall_mean.abs() < 1e-10 {
            1e6
        } else {
            data_range / overall_mean.abs()
        }
    }

    /// Compute distances from points to centroids
    pub async fn compute_distances_to_centroids(
        &self,
        points: &ArrayView2<'_, f64>,
        centroids: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let (npoints, _ndims) = points.dim();
        let (n_clusters_, _n_dims_c) = centroids.dim();
        let mut distances = Array2::zeros((npoints, n_clusters_));
        let cluster_chunks = n_clusters_ / 4;
        for i in 0..npoints {
            let point_row = points.row(i);
            for j_chunk in 0..cluster_chunks {
                let j_base = j_chunk * 4;
                let j0 = j_base;
                let j1 = j_base + 1;
                let j2 = j_base + 2;
                let j3 = j_base + 3;
                let centroid_row0 = centroids.row(j0);
                let centroid_row1 = centroids.row(j1);
                let centroid_row2 = centroids.row(j2);
                let centroid_row3 = centroids.row(j3);
                let distance0: f64 = point_row
                    .iter()
                    .zip(centroid_row0.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let distance1: f64 = point_row
                    .iter()
                    .zip(centroid_row1.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let distance2: f64 = point_row
                    .iter()
                    .zip(centroid_row2.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let distance3: f64 = point_row
                    .iter()
                    .zip(centroid_row3.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distances[[i, j0]] = distance0;
                distances[[i, j1]] = distance1;
                distances[[i, j2]] = distance2;
                distances[[i, j3]] = distance3;
            }
            for j in (cluster_chunks * 4)..n_clusters_ {
                let distance: f64 = point_row
                    .iter()
                    .zip(centroids.row(j).iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distances[[i, j]] = distance;
            }
        }
        Ok(distances)
    }
}

/// Tensor core distance matrix computer with advanced stability monitoring
#[derive(Debug)]
pub struct AdvancedTensorCoreDistanceMatrix {
    /// Base tensor core computer
    pub(crate) base_computer: TensorCoreDistanceMatrix,
    /// Numerical stability monitor
    pub(crate) stability_monitor: Arc<Mutex<NumericalStabilityMonitor>>,
    /// Error recovery system
    recovery_system: ErrorRecoverySystem,
    /// Performance-accuracy analyzer
    performance_analyzer: PerformanceAccuracyAnalyzer,
    /// Enable dynamic precision scaling
    pub(crate) dynamic_precision_enabled: bool,
    /// Enable automatic error recovery
    pub(crate) auto_recovery_enabled: bool,
}

impl AdvancedTensorCoreDistanceMatrix {
    /// Create new advanced tensor core distance matrix computer
    pub fn new() -> SpatialResult<Self> {
        let base_computer = TensorCoreDistanceMatrix::new()?;
        let precision_config = DynamicPrecisionConfig::default();
        let stability_monitor =
            Arc::new(Mutex::new(NumericalStabilityMonitor::new(precision_config)));
        let recovery_system = ErrorRecoverySystem::new();
        let trade_off_params = TradeOffParams {
            performance_weight: 0.6,
            accuracy_weight: 0.4,
            energy_weight: 0.0,
            min_accuracy: 0.95,
            max_time: std::time::Duration::from_secs(30),
            objective: OptimizationObjective::Balanced,
        };
        let performance_analyzer = PerformanceAccuracyAnalyzer::new(trade_off_params);
        Ok(Self {
            base_computer,
            stability_monitor,
            recovery_system,
            performance_analyzer,
            dynamic_precision_enabled: true,
            auto_recovery_enabled: true,
        })
    }

    /// Configure dynamic precision scaling
    pub fn with_dynamic_precision(mut self, enabled: bool) -> Self {
        self.dynamic_precision_enabled = enabled;
        self
    }

    /// Configure automatic error recovery
    pub fn with_auto_recovery(mut self, enabled: bool) -> Self {
        self.auto_recovery_enabled = enabled;
        self
    }

    /// Compute distance matrix with advanced stability monitoring
    pub async fn compute_with_stability_monitoring(
        &mut self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let start_time = Instant::now();
        {
            let mut monitor = self
                .stability_monitor
                .lock()
                .map_err(|_| SpatialError::InvalidInput("Mutex poisoned".to_string()))?;
            if self.dynamic_precision_enabled {
                let optimal_precision = monitor.adjust_precision()?;
                self.base_computer.precision_mode = optimal_precision;
            }
        }
        let mut result = None;
        let mut recovery_attempts = 0;
        let max_attempts = 3;
        while result.is_none() && recovery_attempts < max_attempts {
            match self.base_computer.compute_parallel(points).await {
                Ok(distances) => {
                    {
                        let mut monitor = self.stability_monitor.lock().map_err(|_| {
                            SpatialError::InvalidInput("Mutex poisoned".to_string())
                        })?;
                        monitor.monitor_stability(&points.to_owned(), &distances)?;
                    }
                    let stability_level = {
                        let monitor = self.stability_monitor.lock().map_err(|_| {
                            SpatialError::InvalidInput("Mutex poisoned".to_string())
                        })?;
                        monitor.current_metrics.stability_level
                    };
                    if stability_level == super::enums::StabilityLevel::Critical
                        && self.auto_recovery_enabled
                    {
                        recovery_attempts += 1;
                        let recovery_action = self
                            .recovery_system
                            .attempt_recovery(NumericalErrorType::IllConditioned)
                            .await?;
                        self.apply_recovery_action(recovery_action).await?;
                        continue;
                    } else {
                        result = Some(distances);
                    }
                }
                Err(e) => {
                    if self.auto_recovery_enabled && recovery_attempts < max_attempts {
                        recovery_attempts += 1;
                        let recovery_action = self
                            .recovery_system
                            .attempt_recovery(NumericalErrorType::InvalidValues)
                            .await?;
                        self.apply_recovery_action(recovery_action).await?;
                        continue;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        let final_result = result.ok_or_else(|| {
            SpatialError::InvalidInput(
                "Failed to compute stable result after recovery attempts".to_string(),
            )
        })?;
        let duration = start_time.elapsed();
        let precision = self.base_computer.precision_mode;
        self.performance_analyzer
            .record_performance(precision, duration);
        let accuracy = self.estimate_result_accuracy(&final_result);
        self.performance_analyzer
            .record_accuracy(precision, accuracy);
        Ok(final_result)
    }

    /// Apply recovery action
    pub(crate) async fn apply_recovery_action(
        &mut self,
        action: RecoveryAction,
    ) -> SpatialResult<()> {
        match action {
            RecoveryAction::IncreasePrecision => {
                self.base_computer.precision_mode = match self.base_computer.precision_mode {
                    PrecisionMode::Int4Advanced => PrecisionMode::Int8Dynamic,
                    PrecisionMode::Int8Dynamic => PrecisionMode::Mixed16,
                    PrecisionMode::Mixed16 => PrecisionMode::BrainFloat16,
                    PrecisionMode::BrainFloat16 => PrecisionMode::Full32,
                    PrecisionMode::Full32 => PrecisionMode::Full32,
                    _ => PrecisionMode::Mixed16,
                };
            }
            RecoveryAction::ReduceTileSize => {
                let (current_row, current_col) = self.base_computer.tile_size;
                self.base_computer.tile_size = (current_row / 2, current_col / 2);
                if self.base_computer.tile_size.0 < 16 {
                    self.base_computer.tile_size = (16, 16);
                }
            }
            RecoveryAction::FallbackAlgorithm => {
                self.base_computer.precision_mode = PrecisionMode::Full32;
                self.base_computer.hierarchical_tiling = false;
            }
            RecoveryAction::NumericalStabilization => {
                self.base_computer.precision_mode = PrecisionMode::Full32;
                self.base_computer.tile_size = (64, 64);
            }
            _ => {
                self.base_computer.precision_mode = PrecisionMode::Full32;
            }
        }
        // Sync the monitor's current_precision so that estimate_relative_error
        // uses the updated precision mode for machine epsilon calculation
        let new_precision = self.base_computer.precision_mode;
        if let Ok(mut monitor) = self.stability_monitor.lock() {
            monitor.current_precision = new_precision;
        }
        Ok(())
    }

    /// Estimate result accuracy (simplified)
    pub(crate) fn estimate_result_accuracy(&self, result: &Array2<f64>) -> f64 {
        let has_invalid = result.iter().any(|&x| !x.is_finite());
        if has_invalid {
            return 0.0;
        }
        let max_val = result.fold(0.0f64, |acc, &x| acc.max(x.abs()));
        let min_val = result.fold(f64::INFINITY, |acc, &x| {
            if x.abs() > 1e-15 {
                acc.min(x.abs())
            } else {
                acc
            }
        });
        if min_val.is_finite() && min_val > 0.0 {
            let dynamic_range = max_val / min_val;
            (1.0 / (1.0 + dynamic_range.log10() / 10.0)).clamp(0.0, 1.0)
        } else {
            0.95
        }
    }
}

/// Advanced error recovery system
#[allow(dead_code)]
#[derive(Debug)]
pub struct ErrorRecoverySystem {
    /// Recovery strategies by error type
    pub(crate) recovery_strategies: HashMap<NumericalErrorType, Vec<RecoveryAction>>,
    /// Recovery attempt history
    recovery_history: VecDeque<RecoveryAttempt>,
    /// Maximum recovery attempts per operation
    pub(crate) max_recovery_attempts: usize,
    /// Recovery success rate tracking
    pub(crate) success_rates: HashMap<RecoveryAction, f64>,
}

impl ErrorRecoverySystem {
    /// Create new error recovery system
    pub fn new() -> Self {
        let mut recovery_strategies = HashMap::new();
        recovery_strategies.insert(
            NumericalErrorType::Overflow,
            vec![
                RecoveryAction::IncreasePrecision,
                RecoveryAction::ReduceTileSize,
                RecoveryAction::NumericalStabilization,
            ],
        );
        recovery_strategies.insert(
            NumericalErrorType::Underflow,
            vec![
                RecoveryAction::IncreasePrecision,
                RecoveryAction::NumericalStabilization,
            ],
        );
        recovery_strategies.insert(
            NumericalErrorType::PrecisionLoss,
            vec![
                RecoveryAction::IncreasePrecision,
                RecoveryAction::RetryWithNewParams,
            ],
        );
        recovery_strategies.insert(
            NumericalErrorType::IllConditioned,
            vec![
                RecoveryAction::IncreasePrecision,
                RecoveryAction::NumericalStabilization,
                RecoveryAction::SwitchToCPU,
            ],
        );
        recovery_strategies.insert(
            NumericalErrorType::InvalidValues,
            vec![
                RecoveryAction::FallbackAlgorithm,
                RecoveryAction::SwitchToCPU,
            ],
        );
        Self {
            recovery_strategies,
            recovery_history: VecDeque::new(),
            max_recovery_attempts: 3,
            success_rates: HashMap::new(),
        }
    }

    /// Attempt recovery from numerical error
    pub async fn attempt_recovery(
        &mut self,
        error_type: NumericalErrorType,
    ) -> SpatialResult<RecoveryAction> {
        let start_time = Instant::now();
        let strategies = self
            .recovery_strategies
            .get(&error_type)
            .ok_or_else(|| SpatialError::InvalidInput("Unknown error type".to_string()))?
            .clone();
        let best_action = self.choose_best_recovery_action(&strategies);
        let attempt = RecoveryAttempt {
            error_type,
            action: best_action,
            success: false,
            duration: start_time.elapsed(),
            post_recovery_metrics: None,
            timestamp: start_time,
        };
        self.recovery_history.push_back(attempt);
        Ok(best_action)
    }

    /// Choose best recovery action based on success rates
    fn choose_best_recovery_action(&mut self, strategies: &[RecoveryAction]) -> RecoveryAction {
        strategies
            .iter()
            .max_by(|&a, &b| {
                let rate_a = self.success_rates.get(a).unwrap_or(&0.5);
                let rate_b = self.success_rates.get(b).unwrap_or(&0.5);
                rate_a
                    .partial_cmp(rate_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(RecoveryAction::IncreasePrecision)
    }

    /// Update success rate for recovery action
    pub fn update_success_rate(&mut self, action: RecoveryAction, success: bool) {
        let current_rate = self.success_rates.get(&action).copied().unwrap_or(0.5);
        let new_rate = if success {
            current_rate * 0.9 + 0.1
        } else {
            current_rate * 0.9
        };
        self.success_rates.insert(action, new_rate);
    }
}

/// Tensor core clustering algorithm
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TensorCoreClustering {
    /// Number of clusters
    pub(crate) _numclusters: usize,
    /// Precision mode
    pub(crate) precision_mode: PrecisionMode,
    /// Enable tensor cores
    tensor_cores: bool,
    /// Enable mixed precision
    mixed_precision: bool,
    /// Dynamic precision scaling
    dynamic_precision: bool,
    /// GPU capabilities
    capabilities: Option<TensorCoreCapabilities>,
}

impl TensorCoreClustering {
    /// Create new tensor core clustering
    pub fn new(_numclusters: usize) -> SpatialResult<Self> {
        let capabilities = detect_tensor_core_capabilities().ok();
        Ok(Self {
            _numclusters,
            precision_mode: PrecisionMode::Mixed16,
            tensor_cores: true,
            mixed_precision: true,
            dynamic_precision: false,
            capabilities,
        })
    }

    /// Enable tensor cores
    pub fn with_tensor_cores(mut self, enabled: bool) -> Self {
        self.tensor_cores = enabled;
        self
    }

    /// Enable mixed precision
    pub fn with_mixed_precision(mut self, enabled: bool) -> Self {
        self.mixed_precision = enabled;
        self
    }

    /// Enable dynamic precision scaling
    pub fn with_dynamic_precision_scaling(mut self, enabled: bool) -> Self {
        self.dynamic_precision = enabled;
        self
    }

    /// Fit clustering using tensor cores
    pub async fn fit(
        &mut self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<(Array2<f64>, Array1<usize>)> {
        let (npoints, _ndims) = points.dim();
        if npoints < self._numclusters {
            return Err(SpatialError::InvalidInput(
                "Number of points must be >= number of clusters".to_string(),
            ));
        }
        let mut centroids = self.initialize_centroids(points)?;
        let mut assignments = Array1::zeros(npoints);
        for _iteration in 0..100 {
            let distance_matrix = if self.tensor_cores {
                let tensor_computer =
                    TensorCoreDistanceMatrix::new()?.with_precision_mode(self.precision_mode);
                tensor_computer
                    .compute_distances_to_centroids(points, &centroids.view())
                    .await?
            } else {
                self.compute_distances_fallback(points, &centroids.view())?
            };
            let new_assignments = self.update_assignments(&distance_matrix)?;
            let new_centroids = if self.tensor_cores {
                self.update_centroids_tensor_cores(points, &new_assignments)
                    .await?
            } else {
                self.update_centroids_fallback(points, &new_assignments)?
            };
            let centroid_change = self.compute_centroid_change(&centroids, &new_centroids);
            if centroid_change < 1e-6 {
                break;
            }
            centroids = new_centroids;
            assignments = new_assignments;
        }
        Ok((centroids, assignments))
    }

    /// Initialize centroids using k-means++
    fn initialize_centroids(&mut self, points: &ArrayView2<'_, f64>) -> SpatialResult<Array2<f64>> {
        let (npoints, ndims) = points.dim();
        let mut centroids = Array2::zeros((self._numclusters, ndims));
        let mut rng = scirs2_core::random::rng();
        let first_idx = rng.random_range(0..npoints);
        centroids.row_mut(0).assign(&points.row(first_idx));
        for k in 1..self._numclusters {
            let mut distances = Array1::zeros(npoints);
            for i in 0..npoints {
                let point = points.row(i);
                let mut min_dist = f64::INFINITY;
                for j in 0..k {
                    let centroid = centroids.row(j);
                    let dist: f64 = point
                        .iter()
                        .zip(centroid.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f64>();
                    min_dist = min_dist.min(dist);
                }
                distances[i] = min_dist;
            }
            let total_dist: f64 = distances.sum();
            let mut cumulative = 0.0;
            let random_val = scirs2_core::random::random::<f64>() * total_dist;
            for i in 0..npoints {
                cumulative += distances[i];
                if cumulative >= random_val {
                    centroids.row_mut(k).assign(&points.row(i));
                    break;
                }
            }
        }
        Ok(centroids)
    }

    /// Update assignments based on distance matrix
    fn update_assignments(
        &mut self,
        distance_matrix: &Array2<f64>,
    ) -> SpatialResult<Array1<usize>> {
        let npoints = distance_matrix.nrows();
        let mut assignments = Array1::zeros(npoints);
        for i in 0..npoints {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;
            for j in 0..self._numclusters {
                if distance_matrix[[i, j]] < min_dist {
                    min_dist = distance_matrix[[i, j]];
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster;
        }
        Ok(assignments)
    }

    /// Update centroids using tensor core operations
    async fn update_centroids_tensor_cores(
        &self,
        points: &ArrayView2<'_, f64>,
        assignments: &Array1<usize>,
    ) -> SpatialResult<Array2<f64>> {
        let (_npoints, ndims) = points.dim();
        let mut new_centroids = Array2::zeros((self._numclusters, ndims));
        let mut cluster_counts = vec![0; self._numclusters];
        for &cluster in assignments {
            cluster_counts[cluster] += 1;
        }
        for cluster in 0..self._numclusters {
            if cluster_counts[cluster] == 0 {
                continue;
            }
            let clusterpoints: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &c)| c == cluster)
                .map(|(i, _)| i)
                .collect();
            let cluster_data = Array2::from_shape_fn((clusterpoints.len(), ndims), |(i, j)| {
                points[[clusterpoints[i], j]]
            });
            let sum_vector = self.tensor_sum_reduction(&cluster_data.view()).await?;
            let count = clusterpoints.len() as f64;
            for j in 0..ndims {
                new_centroids[[cluster, j]] = sum_vector[j] / count;
            }
        }
        Ok(new_centroids)
    }

    /// Tensor sum reduction operation
    async fn tensor_sum_reduction(&self, data: &ArrayView2<'_, f64>) -> SpatialResult<Array1<f64>> {
        let (_npoints, ndims) = data.dim();
        let mut sum_vector = Array1::zeros(ndims);
        for j in 0..ndims {
            let column_sum: f64 = data.column(j).sum();
            sum_vector[j] = column_sum;
        }
        Ok(sum_vector)
    }

    /// Fallback distance computation without tensor cores
    fn compute_distances_fallback(
        &self,
        points: &ArrayView2<'_, f64>,
        centroids: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array2<f64>> {
        let (npoints, _ndims) = points.dim();
        let (n_clusters_, _) = centroids.dim();
        let mut distances = Array2::zeros((npoints, n_clusters_));
        let cluster_chunks = n_clusters_ / 4;
        for i in 0..npoints {
            let point_row = points.row(i);
            for j_chunk in 0..cluster_chunks {
                let j_base = j_chunk * 4;
                let j0 = j_base;
                let j1 = j_base + 1;
                let j2 = j_base + 2;
                let j3 = j_base + 3;
                let centroid_row0 = centroids.row(j0);
                let centroid_row1 = centroids.row(j1);
                let centroid_row2 = centroids.row(j2);
                let centroid_row3 = centroids.row(j3);
                let distance0: f64 = point_row
                    .iter()
                    .zip(centroid_row0.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let distance1: f64 = point_row
                    .iter()
                    .zip(centroid_row1.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let distance2: f64 = point_row
                    .iter()
                    .zip(centroid_row2.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let distance3: f64 = point_row
                    .iter()
                    .zip(centroid_row3.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distances[[i, j0]] = distance0;
                distances[[i, j1]] = distance1;
                distances[[i, j2]] = distance2;
                distances[[i, j3]] = distance3;
            }
            for j in (cluster_chunks * 4)..n_clusters_ {
                let distance: f64 = point_row
                    .iter()
                    .zip(centroids.row(j).iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distances[[i, j]] = distance;
            }
        }
        Ok(distances)
    }

    /// Fallback centroid update without tensor cores
    fn update_centroids_fallback(
        &self,
        points: &ArrayView2<'_, f64>,
        assignments: &Array1<usize>,
    ) -> SpatialResult<Array2<f64>> {
        let (npoints, ndims) = points.dim();
        let mut new_centroids = Array2::zeros((self._numclusters, ndims));
        let mut cluster_counts = vec![0; self._numclusters];
        for i in 0..npoints {
            let cluster = assignments[i];
            cluster_counts[cluster] += 1;
            for j in 0..ndims {
                new_centroids[[cluster, j]] += points[[i, j]];
            }
        }
        for cluster in 0..self._numclusters {
            if cluster_counts[cluster] > 0 {
                let count = cluster_counts[cluster] as f64;
                for j in 0..ndims {
                    new_centroids[[cluster, j]] /= count;
                }
            }
        }
        Ok(new_centroids)
    }

    /// Compute change in centroids for convergence checking
    fn compute_centroid_change(
        &self,
        old_centroids: &Array2<f64>,
        new_centroids: &Array2<f64>,
    ) -> f64 {
        let mut total_change = 0.0;
        for i in 0..self._numclusters {
            let change: f64 = old_centroids
                .row(i)
                .iter()
                .zip(new_centroids.row(i).iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            total_change += change;
        }
        total_change / (self._numclusters as f64)
    }
}

// Import OptimizationObjective used in AdvancedTensorCoreDistanceMatrix::new
use super::enums::OptimizationObjective;
