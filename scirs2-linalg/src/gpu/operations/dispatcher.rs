//! GPU operation dispatcher that automatically selects CPU or GPU

use super::super::{AutoGpuSelector, GpuBuffer, GpuContext, GpuDeviceInfo, GpuLinalgOps};
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, Zero};
use std::fmt::Debug;

/// Default GPU threshold for switching from CPU to GPU (number of elements)
pub const DEFAULT_GPU_THRESHOLD: usize = 50_000;

/// GPU operation dispatcher that automatically selects CPU or GPU
pub struct GpuOperationDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    gpu_threshold: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> GpuOperationDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    /// Create a new GPU operation dispatcher
    pub fn new() -> Self {
        Self {
            gpu_threshold: DEFAULT_GPU_THRESHOLD,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create dispatcher with custom GPU threshold
    pub fn with_threshold(threshold: usize) -> Self {
        Self {
            gpu_threshold: threshold,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the GPU threshold
    pub fn set_threshold(&mut self, threshold: usize) {
        self.gpu_threshold = threshold;
    }

    /// Get the current GPU threshold
    pub fn threshold(&self) -> usize {
        self.gpu_threshold
    }
}

impl<T> Default for GpuOperationDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GpuLinalgOps<T> for GpuOperationDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    fn gpu_matvec(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        x: &ArrayView1<T>,
    ) -> LinalgResult<Array1<T>> {
        let (m, n) = a.dim();

        if n != x.len() {
            return Err(LinalgError::ShapeError(format!(
                "Matrix columns ({}) must match vector length ({})",
                n,
                x.len()
            )));
        }

        // Check available memory
        let required_memory = (m * n + n + m) * std::mem::size_of::<T>();
        let available_memory = ctx.available_memory()?;

        if required_memory > available_memory {
            // Fall back to CPU if not enough GPU memory
            return self.cpu_matvec(a, x);
        }

        // Create GPU buffers
        let mut a_buffer = self.allocate_buffer_from_context::<T>(ctx, m * n)?;
        let mut x_buffer = self.allocate_buffer_from_context::<T>(ctx, n)?;
        let mut y_buffer = self.allocate_buffer_from_context::<T>(ctx, m)?;

        // Copy data to GPU
        let a_flat: Vec<T> = a.iter().cloned().collect();
        let x_flat: Vec<T> = x.iter().cloned().collect();

        a_buffer.copy_from_host(&a_flat)?;
        x_buffer.copy_from_host(&x_flat)?;

        // Run the kernel. No physical GPU runtime is linked, so this executes the
        // real computation on the CPU using the host-backed buffers (no fabrication).
        self.execute_matvec_kernel(
            ctx,
            a_buffer.as_ref(),
            x_buffer.as_ref(),
            y_buffer.as_mut(),
            m,
            n,
        )?;

        // Copy result back to host
        let mut result_data = vec![T::zero(); m];
        y_buffer.copy_to_host(&mut result_data)?;

        // Convert to ndarray
        Ok(Array1::from_vec(result_data))
    }

    fn gpu_matmul(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
    ) -> LinalgResult<Array2<T>> {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();

        if k1 != k2 {
            return Err(LinalgError::ShapeError(format!(
                "Matrix dimensions mismatch: {}x{} * {}x{}",
                m, k1, k2, n
            )));
        }

        let k = k1;

        // Check available memory
        let required_memory = (m * k + k * n + m * n) * std::mem::size_of::<T>();
        let available_memory = ctx.available_memory()?;

        if required_memory > available_memory {
            // Fall back to CPU if not enough GPU memory
            return self.cpu_matmul(a, b);
        }

        // Create GPU buffers
        let mut a_buffer = self.allocate_buffer_from_context::<T>(ctx, m * k)?;
        let mut b_buffer = self.allocate_buffer_from_context::<T>(ctx, k * n)?;
        let mut c_buffer = self.allocate_buffer_from_context::<T>(ctx, m * n)?;

        // Copy data to GPU
        let a_flat: Vec<T> = a.iter().cloned().collect();
        let b_flat: Vec<T> = b.iter().cloned().collect();

        a_buffer.copy_from_host(&a_flat)?;
        b_buffer.copy_from_host(&b_flat)?;

        // Execute GPU kernel
        self.execute_matmul_kernel(
            ctx,
            a_buffer.as_ref(),
            b_buffer.as_ref(),
            c_buffer.as_mut(),
            m,
            n,
            k,
        )?;

        // Copy result back to host
        let mut result_data = vec![T::zero(); m * n];
        c_buffer.copy_to_host(&mut result_data)?;

        // Convert to ndarray
        let result_array = Array2::from_shape_vec((m, n), result_data)
            .map_err(|e| LinalgError::ComputationError(format!("Shape error: {}", e)))?;
        Ok(result_array)
    }

    fn gpu_dot(
        &self,
        ctx: &dyn GpuContext,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
    ) -> LinalgResult<T> {
        if x.len() != y.len() {
            return Err(LinalgError::ShapeError(format!(
                "Vector lengths must match: {} != {}",
                x.len(),
                y.len()
            )));
        }

        // For now, fall back to CPU implementation
        Ok(Self::cpu_dot_static(x, y))
    }

    fn gpu_norm(&self, ctx: &dyn GpuContext, x: &ArrayView1<T>) -> LinalgResult<T> {
        // For now, fall back to CPU implementation
        Ok(Self::cpu_norm_static(x))
    }

    fn gpu_elementwise_add(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
    ) -> LinalgResult<Array2<T>> {
        if a.shape() != b.shape() {
            return Err(LinalgError::ShapeError(format!(
                "Matrix shapes must match: {:?} != {:?}",
                a.shape(),
                b.shape()
            )));
        }

        // For now, fall back to CPU implementation
        Self::cpu_elementwise_add_static(a, b)
    }

    fn gpu_elementwise_mul(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
    ) -> LinalgResult<Array2<T>> {
        if a.shape() != b.shape() {
            return Err(LinalgError::ShapeError(format!(
                "Matrix shapes must match: {:?} != {:?}",
                a.shape(),
                b.shape()
            )));
        }

        // For now, fall back to CPU implementation
        Self::cpu_elementwise_mul_static(a, b)
    }
}

impl<T> GpuOperationDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    /// Execute a matrix-vector multiplication on behalf of `gpu_matvec`.
    ///
    /// scirs2-linalg does not link a physical GPU runtime, so there is no real
    /// device kernel to launch. Rather than fabricate a successful kernel launch,
    /// we run the mathematically-equivalent computation on the CPU using the
    /// host-backed buffers. The result is therefore always real.
    fn execute_matvec_kernel(
        &self,
        _ctx: &dyn GpuContext,
        a_buffer: &dyn GpuBuffer<T>,
        x_buffer: &dyn GpuBuffer<T>,
        y_buffer: &mut dyn GpuBuffer<T>,
        m: usize,
        n: usize,
    ) -> LinalgResult<()> {
        self.cpu_fallback_matvec(a_buffer, x_buffer, y_buffer, m, n)
    }

    /// Execute a matrix-matrix multiplication on behalf of `gpu_matmul`.
    ///
    /// See [`Self::execute_matvec_kernel`] for the rationale: no physical GPU
    /// runtime is linked, so the real result is computed on the host instead of
    /// pretending a device kernel ran.
    fn execute_matmul_kernel(
        &self,
        _ctx: &dyn GpuContext,
        a_buffer: &dyn GpuBuffer<T>,
        b_buffer: &dyn GpuBuffer<T>,
        c_buffer: &mut dyn GpuBuffer<T>,
        m: usize,
        n: usize,
        k: usize,
    ) -> LinalgResult<()> {
        self.cpu_fallback_matmul(a_buffer, b_buffer, c_buffer, m, n, k)
    }

    /// CPU fallback matrix-vector multiply operating on host-backed buffers.
    ///
    /// Reads the operands back from the buffers, performs the multiply on the
    /// CPU, and writes the result into `y_buffer`. Used by the GPU dispatch path
    /// because no physical device kernel is linked into the crate.
    fn cpu_fallback_matvec(
        &self,
        a_buffer: &dyn GpuBuffer<T>,
        x_buffer: &dyn GpuBuffer<T>,
        y_buffer: &mut dyn GpuBuffer<T>,
        m: usize,
        n: usize,
    ) -> LinalgResult<()> {
        // Copy the operands back from the (host-backed) buffers and compute on CPU.
        let mut a_data = vec![T::zero(); m * n];
        let mut x_data = vec![T::zero(); n];
        let mut y_data = vec![T::zero(); m];

        a_buffer.copy_to_host(&mut a_data)?;
        x_buffer.copy_to_host(&mut x_data)?;

        // Simulate GPU computation
        for i in 0..m {
            let mut sum = T::zero();
            for j in 0..n {
                sum += a_data[i * n + j] * x_data[j];
            }
            y_data[i] = sum;
        }

        y_buffer.copy_from_host(&y_data)?;
        Ok(())
    }

    /// CPU fallback matrix-matrix multiply operating on host-backed buffers.
    fn cpu_fallback_matmul(
        &self,
        a_buffer: &dyn GpuBuffer<T>,
        b_buffer: &dyn GpuBuffer<T>,
        c_buffer: &mut dyn GpuBuffer<T>,
        m: usize,
        n: usize,
        k: usize,
    ) -> LinalgResult<()> {
        // Copy the operands back from the (host-backed) buffers and compute on CPU.
        let mut a_data = vec![T::zero(); m * k];
        let mut b_data = vec![T::zero(); k * n];
        let mut c_data = vec![T::zero(); m * n];

        a_buffer.copy_to_host(&mut a_data)?;
        b_buffer.copy_to_host(&mut b_data)?;

        // Simulate GPU GEMM
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum += a_data[i * k + l] * b_data[l * n + j];
                }
                c_data[i * n + j] = sum;
            }
        }

        c_buffer.copy_from_host(&c_data)?;
        Ok(())
    }

    /// CPU fallback for matrix-vector multiplication
    pub fn cpu_matvec(&self, a: &ArrayView2<T>, x: &ArrayView1<T>) -> LinalgResult<Array1<T>> {
        let (m, n) = a.dim();
        let mut result = Array1::zeros(m);

        for i in 0..m {
            let mut sum = T::zero();
            for j in 0..n {
                sum += a[[i, j]] * x[j];
            }
            result[i] = sum;
        }

        Ok(result)
    }

    /// CPU fallback for matrix-matrix multiplication
    pub fn cpu_matmul(&self, a: &ArrayView2<T>, b: &ArrayView2<T>) -> LinalgResult<Array2<T>> {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        let mut result = Array2::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum += a[[i, l]] * b[[l, j]];
                }
                result[[i, j]] = sum;
            }
        }

        Ok(result)
    }

    /// CPU fallback for dot product
    fn cpu_dot(&self, x: &ArrayView1<T>, y: &ArrayView1<T>) -> T {
        let mut result = T::zero();
        for (a, b) in x.iter().zip(y.iter()) {
            result += *a * *b;
        }
        result
    }

    /// Static CPU fallback for dot product
    fn cpu_dot_static(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T {
        let mut result = T::zero();
        for (a, b) in x.iter().zip(y.iter()) {
            result += *a * *b;
        }
        result
    }

    /// CPU fallback for vector norm
    fn cpu_norm(&self, x: &ArrayView1<T>) -> T {
        let mut sum_sq = T::zero();
        for &val in x.iter() {
            sum_sq += val * val;
        }
        sum_sq.sqrt()
    }

    /// Static CPU fallback for vector norm
    fn cpu_norm_static(x: &ArrayView1<T>) -> T {
        let mut sum_sq = T::zero();
        for &val in x.iter() {
            sum_sq += val * val;
        }
        sum_sq.sqrt()
    }

    /// CPU fallback for element-wise addition
    fn cpu_elementwise_add(&self, a: &ArrayView2<T>, b: &ArrayView2<T>) -> LinalgResult<Array2<T>> {
        let mut result = Array2::zeros(a.dim());
        for ((i, j), &val_a) in a.indexed_iter() {
            result[[i, j]] = val_a + b[[i, j]];
        }
        Ok(result)
    }

    /// Static CPU fallback for element-wise addition
    fn cpu_elementwise_add_static(a: &ArrayView2<T>, b: &ArrayView2<T>) -> LinalgResult<Array2<T>> {
        let mut result = Array2::zeros(a.dim());
        for ((i, j), &val_a) in a.indexed_iter() {
            result[[i, j]] = val_a + b[[i, j]];
        }
        Ok(result)
    }

    /// CPU fallback for element-wise multiplication
    fn cpu_elementwise_mul(&self, a: &ArrayView2<T>, b: &ArrayView2<T>) -> LinalgResult<Array2<T>> {
        let mut result = Array2::zeros(a.dim());
        for ((i, j), &val_a) in a.indexed_iter() {
            result[[i, j]] = val_a * b[[i, j]];
        }
        Ok(result)
    }

    /// Static CPU fallback for element-wise multiplication
    fn cpu_elementwise_mul_static(a: &ArrayView2<T>, b: &ArrayView2<T>) -> LinalgResult<Array2<T>> {
        let mut result = Array2::zeros(a.dim());
        for ((i, j), &val_a) in a.indexed_iter() {
            result[[i, j]] = val_a * b[[i, j]];
        }
        Ok(result)
    }

    /// Helper function to allocate buffer from a dyn GpuContext
    fn allocate_buffer_from_context<U: Clone + Send + Sync + Copy + std::fmt::Debug + 'static>(
        &self,
        ctx: &dyn GpuContext,
        size: usize,
    ) -> LinalgResult<Box<dyn GpuBuffer<U>>> {
        // A `&dyn GpuContext` does not expose `GpuContextAlloc`, so we cannot ask
        // the context for a device-native allocation here. Rather than fabricate an
        // opaque buffer, hand out a real CPU-backed buffer that actually stores the
        // data; the kernel-execution path below runs the equivalent computation on
        // the host so results are always real.
        use crate::gpu::acceleration::CpuFallbackBuffer;
        Ok(Box::new(CpuFallbackBuffer::new(size)))
    }
}

impl<T> AutoGpuSelector<T> for GpuOperationDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    fn auto_matvec(
        &self,
        a: &ArrayView2<T>,
        x: &ArrayView1<T>,
        gpu_context: Option<&dyn GpuContext>,
    ) -> LinalgResult<Array1<T>> {
        let elements = a.len();

        if let Some(ctx) = gpu_context {
            if elements > self.gpu_threshold {
                // Use GPU implementation
                return self.gpu_matvec(ctx, a, x);
            }
        }

        // Use CPU implementation
        self.cpu_matvec(a, x)
    }

    fn auto_matmul(
        &self,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
        gpu_context: Option<&dyn GpuContext>,
    ) -> LinalgResult<Array2<T>> {
        let elements = a.len() + b.len();

        if let Some(ctx) = gpu_context {
            if elements > self.gpu_threshold {
                // Use GPU implementation
                return self.gpu_matmul(ctx, a, b);
            }
        }

        // Use CPU implementation
        self.cpu_matmul(a, b)
    }
}
