//! Example demonstrating the use of quantization-aware linear algebra operations
//!
//! This example demonstrates how to use the quantization module for
//! various linear algebra operations with reduced precision.

use ndarray::{Array1, Array2};
use scirs2_linalg::quantization::{
    dequantize_matrix, dequantize_vector, fake_quantize, quantize_matrix, quantize_vector,
    quantized_dot, quantized_matmul, quantized_matvec, QuantizationMethod, QuantizedData2D,
};

fn main() {
    println!("Quantization-aware Linear Algebra Example");
    println!("=======================================\n");

    // Section for 4-bit quantization demonstration
    println!("4-bit Quantization Demonstration");
    println!("-------------------------------\n");

    // Create a sample matrix
    let int4_test =
        Array2::from_shape_vec((2, 4), vec![1.0_f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])
            .unwrap();

    println!("Original Matrix:");
    println!("{:?}\n", int4_test);

    // Demonstrate Int4 quantization
    let (int4_quantized, int4_params) =
        quantize_matrix(&int4_test.view(), 4, QuantizationMethod::Int4);

    println!("Storage comparison:");
    println!(
        "Original size: {} bytes",
        int4_test.len() * std::mem::size_of::<f32>()
    );
    println!(
        "Int4 size: {} bytes\n",
        int4_quantized.data.len() * std::mem::size_of::<i8>()
    );

    println!("Int4 Quantization Parameters:");
    println!("  Bits: {}", int4_params.bits);
    println!("  Scale: {}", int4_params.scale);
    println!("  Zero point: {}", int4_params.zero_point);
    println!("  Min value: {}", int4_params.min_val);
    println!("  Max value: {}\n", int4_params.max_val);

    println!("Int4 Data (packed, 2 values per byte):");
    if let QuantizedData2D::Int8(data) = &int4_quantized.data {
        for row in 0..data.nrows() {
            print!("  ");
            for col in 0..data.ncols() {
                print!("{:02x} ", data[[row, col]] as u8);
            }
            println!();
        }
    }
    println!();

    println!("Decoded Int4 Values:");
    for row in 0..int4_test.nrows() {
        print!("  ");
        for col in 0..int4_test.ncols() {
            print!("{:2} ", int4_quantized.get_i8(row, col));
        }
        println!();
    }
    println!();

    // Dequantize and show error
    let int4_dequantized = dequantize_matrix(&int4_quantized, &int4_params);

    println!("Dequantized Matrix:");
    println!("{:?}\n", int4_dequantized);

    println!("Quantization Error:");
    println!("{:?}\n", &int4_test - &int4_dequantized);

    // Do the same for UInt4
    let uint4_test =
        Array2::from_shape_vec((2, 4), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

    println!("\nUInt4 Quantization Example");
    println!("-------------------------\n");

    println!("Original Matrix (positive values):");
    println!("{:?}\n", uint4_test);

    // Demonstrate UInt4 quantization
    let (uint4_quantized, uint4_params) =
        quantize_matrix(&uint4_test.view(), 4, QuantizationMethod::UInt4);

    println!("UInt4 Quantization Parameters:");
    println!("  Bits: {}", uint4_params.bits);
    println!("  Scale: {}", uint4_params.scale);
    println!("  Zero point: {}", uint4_params.zero_point);
    println!("  Min value: {}", uint4_params.min_val);
    println!("  Max value: {}\n", uint4_params.max_val);

    println!("UInt4 Data (packed, 2 values per byte):");
    if let QuantizedData2D::Int8(data) = &uint4_quantized.data {
        for row in 0..data.nrows() {
            print!("  ");
            for col in 0..data.ncols() {
                print!("{:02x} ", data[[row, col]] as u8);
            }
            println!();
        }
    }
    println!();

    println!("Decoded UInt4 Values:");
    for row in 0..uint4_test.nrows() {
        print!("  ");
        for col in 0..uint4_test.ncols() {
            print!("{:2} ", uint4_quantized.get_i8(row, col));
        }
        println!();
    }
    println!();

    // Dequantize and show error
    let uint4_dequantized = dequantize_matrix(&uint4_quantized, &uint4_params);

    println!("Dequantized Matrix:");
    println!("{:?}\n", uint4_dequantized);

    println!("Quantization Error:");
    println!("{:?}\n", &uint4_test - &uint4_dequantized);

    // Original example continues below
    println!("Standard 8-bit Quantization Examples");
    println!("----------------------------------\n");

    // Create some sample matrices and vectors
    let a =
        Array2::from_shape_vec((3, 3), vec![1.2, 2.5, 3.7, 4.2, 5.0, 6.1, 7.3, 8.4, 9.5]).unwrap();

    let b = Array2::from_shape_vec((3, 2), vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5]).unwrap();

    let x = Array1::from_shape_vec(3, vec![0.1, 0.2, 0.3]).unwrap();

    println!("Original Matrix A:");
    println!("{:?}\n", a);

    println!("Original Matrix B:");
    println!("{:?}\n", b);

    println!("Original Vector x:");
    println!("{:?}\n", x);

    // Demonstrate basic quantization and dequantization
    println!("Basic Quantization-Dequantization");
    println!("--------------------------------");

    // Uniform quantization
    let (a_q_uniform, a_params_uniform) =
        quantize_matrix(&a.view(), 8, QuantizationMethod::Uniform);
    let a_dequant_uniform = dequantize_matrix(&a_q_uniform, &a_params_uniform);

    println!("Uniform Quantization Parameters:");
    println!("  Bits: {}", a_params_uniform.bits);
    println!("  Scale: {}", a_params_uniform.scale);
    println!("  Zero point: {}", a_params_uniform.zero_point);
    println!("  Min value: {}", a_params_uniform.min_val);
    println!("  Max value: {}", a_params_uniform.max_val);
    println!();

    println!("Quantized Matrix A (Uniform, 8-bit):");
    if let QuantizedData2D::Int8(data) = &a_q_uniform.data {
        println!("{:?}\n", data);
    }

    println!("Dequantized Matrix A (Uniform):");
    println!("{:?}\n", a_dequant_uniform);

    println!("Quantization Error (Uniform):");
    println!("{:?}\n", &a - &a_dequant_uniform);

    // Symmetric quantization
    let (a_q_symmetric, a_params_symmetric) =
        quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
    let a_dequant_symmetric = dequantize_matrix(&a_q_symmetric, &a_params_symmetric);

    println!("Symmetric Quantization Parameters:");
    println!("  Bits: {}", a_params_symmetric.bits);
    println!("  Scale: {}", a_params_symmetric.scale);
    println!("  Zero point: {}", a_params_symmetric.zero_point);
    println!("  Min value: {}", a_params_symmetric.min_val);
    println!("  Max value: {}", a_params_symmetric.max_val);
    println!();

    println!("Quantized Matrix A (Symmetric, 8-bit):");
    if let QuantizedData2D::Int8(data) = &a_q_symmetric.data {
        println!("{:?}\n", data);
    }

    println!("Dequantized Matrix A (Symmetric):");
    println!("{:?}\n", a_dequant_symmetric);

    println!("Quantization Error (Symmetric):");
    println!("{:?}\n", &a - &a_dequant_symmetric);

    // Affine quantization
    let (a_q_affine, a_params_affine) = quantize_matrix(&a.view(), 8, QuantizationMethod::Affine);
    let a_dequant_affine = dequantize_matrix(&a_q_affine, &a_params_affine);

    println!("Affine Quantization Parameters:");
    println!("  Bits: {}", a_params_affine.bits);
    println!("  Scale: {}", a_params_affine.scale);
    println!("  Zero point: {}", a_params_affine.zero_point);
    println!("  Min value: {}", a_params_affine.min_val);
    println!("  Max value: {}", a_params_affine.max_val);
    println!();

    println!("Quantized Matrix A (Affine, 8-bit):");
    if let QuantizedData2D::Int8(data) = &a_q_affine.data {
        println!("{:?}\n", data);
    }

    println!("Dequantized Matrix A (Affine):");
    println!("{:?}\n", a_dequant_affine);

    println!("Quantization Error (Affine):");
    println!("{:?}\n", &a - &a_dequant_affine);

    // Quantized Matrix Operations
    println!("Quantized Matrix Operations");
    println!("--------------------------");

    // Regular matrix multiplication
    println!("Regular Matrix Multiplication A * B:");
    let c = a.dot(&b);
    println!("{:?}\n", c);

    // Quantized matrix multiplication
    let (b_q, b_params) = quantize_matrix(&b.view(), 8, QuantizationMethod::Symmetric);
    println!("Quantized Matrix Multiplication (8-bit):");
    let c_q = quantized_matmul(&a_q_symmetric, &a_params_symmetric, &b_q, &b_params).unwrap();
    println!("{:?}\n", c_q);

    println!("Quantization Error for Matrix Multiplication:");
    println!("{:?}\n", &c - &c_q);

    // Calculate relative error
    let rel_error = (&c - &c_q).mapv(|x| x.abs()).sum() / c.sum();
    println!(
        "Relative Error for Matrix Multiplication: {:.6}\n",
        rel_error
    );

    // Regular matrix-vector multiplication
    println!("Regular Matrix-Vector Multiplication A * x:");
    let y = a.dot(&x);
    println!("{:?}\n", y);

    // Quantized matrix-vector multiplication
    let (x_q, x_params) = quantize_vector(&x.view(), 8, QuantizationMethod::Symmetric);
    println!("Quantized Matrix-Vector Multiplication (8-bit):");
    let y_q = quantized_matvec(&a_q_symmetric, &a_params_symmetric, &x_q, &x_params).unwrap();
    println!("{:?}\n", y_q);

    println!("Quantization Error for Matrix-Vector Multiplication:");
    println!("{:?}\n", &y - &y_q);

    // Calculate relative error
    let rel_error = (&y - &y_q).mapv(|x| x.abs()).sum() / y.sum();
    println!(
        "Relative Error for Matrix-Vector Multiplication: {:.6}\n",
        rel_error
    );

    // Regular dot product
    println!("Regular Dot Product x . x:");
    let dot = x.dot(&x);
    println!("{:?}\n", dot);

    // Quantized dot product
    println!("Quantized Dot Product (8-bit):");
    let dot_q = quantized_dot(&x_q, &x_params, &x_q, &x_params).unwrap();
    println!("{:?}\n", dot_q);

    println!("Quantization Error for Dot Product:");
    println!("{:?}\n", dot - dot_q);

    // Calculate relative error
    let rel_error = (dot - dot_q).abs() / dot;
    println!("Relative Error for Dot Product: {:.6}\n", rel_error);

    // Fake Quantization Example
    println!("Fake Quantization Example");
    println!("------------------------");

    // Apply fake quantization with different bit widths
    let bits_list = [4, 6, 8];
    for &bits in &bits_list {
        println!("Fake Quantization with {}-bit:", bits);
        let a_fake_q = fake_quantize(&a.view(), bits, QuantizationMethod::Symmetric);

        println!("Fake Quantized Matrix:");
        println!("{:?}\n", a_fake_q);

        println!("Quantization Error:");
        println!("{:?}\n", &a - &a_fake_q);

        // Calculate mean squared error
        let mse = (&a - &a_fake_q).mapv(|x| x * x).sum() / a.len() as f32;
        println!("Mean Squared Error: {:.6}\n", mse);
    }

    // Comparison across different quantization methods and bit widths
    println!("Quantization Comparison");
    println!("----------------------");

    let methods = [
        QuantizationMethod::Uniform,
        QuantizationMethod::Symmetric,
        QuantizationMethod::Affine,
        QuantizationMethod::PowerOfTwo,
        QuantizationMethod::Int4,
        QuantizationMethod::UInt4,
        QuantizationMethod::Float16,
        QuantizationMethod::BFloat16,
    ];

    let bits_list = [4, 6, 8, 10, 16];

    println!("Mean Squared Error (MSE) for different methods and bit widths:");
    println!();

    // Print header
    print!("{:<12}", "Bits");
    for method in &methods {
        let method_name = match method {
            QuantizationMethod::Uniform => "Uniform",
            QuantizationMethod::Symmetric => "Symmetric",
            QuantizationMethod::Affine => "Affine",
            QuantizationMethod::PowerOfTwo => "PowerOfTwo",
            QuantizationMethod::Int4 => "Int4",
            QuantizationMethod::UInt4 => "UInt4",
            QuantizationMethod::Float16 => "Float16",
            QuantizationMethod::BFloat16 => "BFloat16",
            QuantizationMethod::PerChannelSymmetric => "PC-Symmetric",
            QuantizationMethod::PerChannelAffine => "PC-Affine",
        };
        print!("{:<12}", method_name);
    }
    println!();

    // Print MSE for each combination
    for &bits in &bits_list {
        print!("{:<12}", bits);

        for &method in &methods {
            // Some methods have fixed bit widths
            let effective_bits = match method {
                QuantizationMethod::Int4 | QuantizationMethod::UInt4 => 4,
                QuantizationMethod::Float16 | QuantizationMethod::BFloat16 => 16,
                _ => bits,
            };

            let a_fake_q = fake_quantize(&a.view(), effective_bits, method);
            let mse = (&a - &a_fake_q).mapv(|x| x * x).sum() / a.len() as f32;
            print!("{:<12.6}", mse);
        }

        println!();
    }

    // Add a section specifically for float16 and bfloat16 demonstrations
    println!();
    println!("16-bit Floating-Point Quantization");
    println!("----------------------------------");

    // Create some sample data with a wide range of values
    let wide_range = Array1::from_shape_vec(
        8,
        vec![
            0.000001,             // Very small positive
            123456.0,             // Large positive
            -0.000002,            // Very small negative
            -98765.0,             // Large negative
            std::f32::consts::PI, // Pi
            std::f32::consts::E,  // e
            0.0,                  // Zero
            1.0,                  // One
        ],
    )
    .unwrap();

    println!("Original values with wide dynamic range:");
    println!("{:?}\n", wide_range);

    // Test float16 precision
    let (f16_quantized, f16_params) =
        quantize_vector(&wide_range.view(), 16, QuantizationMethod::Float16);
    let f16_dequantized = dequantize_vector(&f16_quantized, &f16_params);

    println!("After Float16 quantization and dequantization:");
    println!("{:?}\n", f16_dequantized);

    println!("Float16 absolute errors:");
    println!("{:?}\n", &wide_range - &f16_dequantized);

    println!("Float16 relative errors (%):");
    for (i, (&orig, &dequant)) in wide_range.iter().zip(f16_dequantized.iter()).enumerate() {
        if orig.abs() > 1e-10 {
            println!(
                "  Value {}: {:.6}%",
                i,
                100.0 * (orig - dequant).abs() / orig.abs()
            );
        } else {
            println!(
                "  Value {}: {:.6} (absolute error for near-zero value)",
                i,
                (orig - dequant).abs()
            );
        }
    }
    println!();

    // Test bfloat16 precision
    let (bf16_quantized, bf16_params) =
        quantize_vector(&wide_range.view(), 16, QuantizationMethod::BFloat16);
    let bf16_dequantized = dequantize_vector(&bf16_quantized, &bf16_params);

    println!("After BFloat16 quantization and dequantization:");
    println!("{:?}\n", bf16_dequantized);

    println!("BFloat16 absolute errors:");
    println!("{:?}\n", &wide_range - &bf16_dequantized);

    println!("BFloat16 relative errors (%):");
    for (i, (&orig, &dequant)) in wide_range.iter().zip(bf16_dequantized.iter()).enumerate() {
        if orig.abs() > 1e-10 {
            println!(
                "  Value {}: {:.6}%",
                i,
                100.0 * (orig - dequant).abs() / orig.abs()
            );
        } else {
            println!(
                "  Value {}: {:.6} (absolute error for near-zero value)",
                i,
                (orig - dequant).abs()
            );
        }
    }
    println!();

    // Demonstrate matrix operations with float16
    println!("Float16 Matrix Operations");
    println!("-----------------------");

    // Create matrices for multiplication
    let a_for_f16 = Array2::from_shape_vec((2, 3), vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6]).unwrap();
    let b_for_f16 = Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();

    // Quantize to float16
    let (a_f16, a_f16_params) = quantize_matrix(&a_for_f16.view(), 16, QuantizationMethod::Float16);
    let (b_f16, b_f16_params) = quantize_matrix(&b_for_f16.view(), 16, QuantizationMethod::Float16);

    // Regular matrix multiplication
    let c_full = a_for_f16.dot(&b_for_f16);
    println!("Regular matrix multiplication result (f32):");
    println!("{:?}\n", c_full);

    // Float16 matrix multiplication
    let c_f16 = quantized_matmul(&a_f16, &a_f16_params, &b_f16, &b_f16_params).unwrap();
    println!("Float16 matrix multiplication result:");
    println!("{:?}\n", c_f16);

    // Error analysis
    let rel_error_f16 = (&c_full - &c_f16).mapv(|x| x.abs()).sum() / c_full.sum();
    println!("Float16 matmul relative error: {:.6e}\n", rel_error_f16);

    // Demonstrate storage efficiency
    let matrix_size = 100;
    let large_matrix = Array2::from_elem((matrix_size, matrix_size), 1.0f32);

    let original_size = matrix_size * matrix_size * std::mem::size_of::<f32>();

    // Quantize with different methods
    let (int8_large, _) = quantize_matrix(&large_matrix.view(), 8, QuantizationMethod::Symmetric);
    let (int4_large, _) = quantize_matrix(&large_matrix.view(), 4, QuantizationMethod::Int4);
    let (f16_large, _) = quantize_matrix(&large_matrix.view(), 16, QuantizationMethod::Float16);
    let (bf16_large, _) = quantize_matrix(&large_matrix.view(), 16, QuantizationMethod::BFloat16);

    // Sizes in bytes
    let int8_size = int8_large.data.len() * std::mem::size_of::<i8>();
    let int4_size = int4_large.data.len() * std::mem::size_of::<i8>(); // Packed, 2 values per byte
    let f16_size = f16_large.data.len() * 2; // 2 bytes per f16 value
    let bf16_size = bf16_large.data.len() * 2; // 2 bytes per bf16 value

    println!(
        "Storage Efficiency Comparison ({}x{} matrix):",
        matrix_size, matrix_size
    );
    println!("  Original f32: {} bytes (100.0%)", original_size);
    println!(
        "  Int8:         {} bytes ({:.1}%)",
        int8_size,
        100.0 * int8_size as f32 / original_size as f32
    );
    println!(
        "  Int4:         {} bytes ({:.1}%)",
        int4_size,
        100.0 * int4_size as f32 / original_size as f32
    );
    println!(
        "  Float16:      {} bytes ({:.1}%)",
        f16_size,
        100.0 * f16_size as f32 / original_size as f32
    );
    println!(
        "  BFloat16:     {} bytes ({:.1}%)",
        bf16_size,
        100.0 * bf16_size as f32 / original_size as f32
    );
}
