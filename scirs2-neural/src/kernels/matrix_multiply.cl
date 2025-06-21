// OpenCL kernel for GPU-accelerated matrix multiplication
// Optimized for neural network workloads with block-wise computation

__kernel void matrix_multiply_f32(
    __global const float* A,    // Input matrix A (m x k)
    __global const float* B,    // Input matrix B (k x n) 
    __global float* C,          // Output matrix C (m x n)
    const unsigned int M,       // Number of rows in A and C
    const unsigned int N,       // Number of columns in B and C
    const unsigned int K        // Number of columns in A and rows in B
) {
    // Get work item indices
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    // Check bounds
    if (row >= M || col >= N) {
        return;
    }
    
    // Compute dot product for C[row][col]
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    // Store result
    C[row * N + col] = sum;
}

// Optimized matrix multiplication using local memory (for larger matrices)
__kernel void matrix_multiply_local_f32(
    __global const float* A,
    __global const float* B, 
    __global float* C,
    const unsigned int M,
    const unsigned int N,
    const unsigned int K
) {
    // Tile size (should match local work group size)
    const int TILE_SIZE = 16;
    
    // Local memory for tiles
    __local float localA[16][16];
    __local float localB[16][16];
    
    // Work group and local indices
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into local memory
        const int aCol = t * TILE_SIZE + localCol;
        if (row < M && aCol < K) {
            localA[localRow][localCol] = A[row * K + aCol];
        } else {
            localA[localRow][localCol] = 0.0f;
        }
        
        // Load tile of B into local memory
        const int bRow = t * TILE_SIZE + localRow;
        if (bRow < K && col < N) {
            localB[localRow][localCol] = B[bRow * N + col];
        } else {
            localB[localRow][localCol] = 0.0f;
        }
        
        // Synchronize to ensure all data is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial sum using local data
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += localA[localRow][k] * localB[k][localCol];
        }
        
        // Synchronize before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Batch matrix multiplication kernel
__kernel void batch_matrix_multiply_f32(
    __global const float* A,    // Batch of input matrices A
    __global const float* B,    // Batch of input matrices B
    __global float* C,          // Batch of output matrices C
    const unsigned int batch_size,
    const unsigned int M,
    const unsigned int N,
    const unsigned int K
) {
    // Get batch, row, and column indices
    const int batch = get_global_id(2);
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    // Check bounds
    if (batch >= batch_size || row >= M || col >= N) {
        return;
    }
    
    // Calculate matrix offsets for this batch
    const int matrix_size_A = M * K;
    const int matrix_size_B = K * N;
    const int matrix_size_C = M * N;
    
    const int offset_A = batch * matrix_size_A;
    const int offset_B = batch * matrix_size_B;
    const int offset_C = batch * matrix_size_C;
    
    // Compute dot product for C[batch][row][col]
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[offset_A + row * K + k] * B[offset_B + k * N + col];
    }
    
    // Store result
    C[offset_C + row * N + col] = sum;
}