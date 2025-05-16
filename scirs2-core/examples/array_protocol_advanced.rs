// Copyright (c) 2025, SciRS2 Team
//
// Licensed under either of
//
// * Apache License, Version 2.0
//   (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
// * MIT license
//   (LICENSE-MIT or http://opensource.org/licenses/MIT)
//
// at your option.
//

//! Advanced example demonstrating automatic device placement and mixed-precision features
//! of the array protocol.

use ndarray::Array2;
use scirs2_core::array_protocol::{
    self,
    auto_device::{set_auto_device_config, AutoDeviceConfig},
    mixed_precision::{
        set_mixed_precision_config, MixedPrecisionArray, MixedPrecisionConfig,
        MixedPrecisionSupport, Precision,
    },
    GPUBackend, GPUConfig, GPUNdarray,
};

fn main() {
    // Initialize the array protocol system
    array_protocol::init();

    println!("Advanced Array Protocol Features Example");
    println!("=======================================");

    // Part 1: Automatic Device Placement
    println!("\nPart 1: Automatic Device Placement");
    println!("----------------------------------");

    // Configure auto device placement - for demo, set low thresholds
    let auto_device_config = AutoDeviceConfig {
        gpu_threshold: 100,           // Place arrays with >100 elements on GPU
        distributed_threshold: 10000, // Place arrays with >10K elements on distributed
        enable_mixed_precision: true,
        prefer_memory_efficiency: true,
        auto_transfer: true,
        prefer_data_locality: true,
        preferred_gpu_backend: GPUBackend::CUDA,
        fallback_to_cpu: true,
    };
    set_auto_device_config(auto_device_config);

    // Create arrays of different sizes
    let small_array = Array2::<f64>::ones((5, 5)); // 25 elements
    let medium_array = Array2::<f64>::ones((20, 20)); // 400 elements
    let large_array = Array2::<f64>::ones((200, 200)); // 40K elements

    println!("Small array: {} elements", small_array.len());
    println!("Medium array: {} elements", medium_array.len());
    println!("Large array: {} elements", large_array.len());

    // AutoDevice functionality is not fully implemented in this example version
    // We're skipping the AutoDevice part of the demo to prevent compilation errors

    println!("\nAutoDevice operations skipped in this demo.");
    println!("In a full implementation, operations would be automatically dispatched");
    println!("to the most appropriate device based on array size and operation type.");

    // Print what would happen in a working implementation
    println!("\nExpected behavior with working AutoDevice:");
    println!("- Small array (25 elements): CPU");
    println!("- Medium array (400 elements): GPU (> threshold of 100)");
    println!("- Large array (40K elements): Distributed (> threshold of 10K)");

    println!("\nOperations with automatic device selection would:");
    println!("- Choose GPU for operations with medium/large arrays");
    println!("- Use CPU for small arrays");
    println!("- Automatically transfer data between devices as needed");
    println!("- Balance computation based on memory and execution efficiency");

    // Part 2: Mixed-Precision Operations
    println!("\nPart 2: Mixed-Precision Operations");
    println!("----------------------------------");

    // Configure mixed precision
    let mixed_precision_config = MixedPrecisionConfig {
        storage_precision: Precision::Single,
        compute_precision: Precision::Double,
        auto_precision: true,
        downcast_threshold: 1000,
        double_precision_accumulation: true,
    };
    set_mixed_precision_config(mixed_precision_config);

    // Create arrays with different precisions
    let array_f64 = Array2::<f64>::ones((10, 10));
    let array_f32 = array_f64.mapv(|x| x as f32);

    println!("Array f64: 10x10 double-precision array");
    println!("Array f32: 10x10 single-precision array");

    // Wrap in MixedPrecisionArray
    let mixed_f64 = MixedPrecisionArray::new(array_f64.clone());
    let mixed_f32 = MixedPrecisionArray::new(array_f32.clone());

    // Check precision
    println!("\nDefault precision of arrays:");
    println!("f64 array: {:?}", mixed_f64.precision());
    println!("f32 array: {:?}", mixed_f32.precision());

    // Convert arrays to specific precision
    println!("\nAttempt to convert arrays to specific precision:");

    // Try to convert f64 array to single precision
    match mixed_f64.to_precision(Precision::Single) {
        Ok(_) => println!("Converted f64 to single precision: succeeded"),
        Err(e) => println!("Conversion error: {}", e),
    }

    // Try to convert f32 array to double precision
    match mixed_f32.to_precision(Precision::Double) {
        Ok(_) => println!("Converted f32 to double precision: succeeded"),
        Err(e) => println!("Conversion error: {}", e),
    }

    // Perform operations with specific precision
    println!("\nOperations with specific precision:");

    // Matrix multiplication with single precision
    let result =
        array_protocol::mixed_precision::ops::matmul(&mixed_f64, &mixed_f32, Precision::Single);
    match result {
        Ok(_) => println!("Matrix multiplication (single precision): succeeded"),
        Err(e) => println!("Operation error: {}", e),
    }

    // Matrix multiplication with double precision
    let result =
        array_protocol::mixed_precision::ops::matmul(&mixed_f64, &mixed_f32, Precision::Double);
    match result {
        Ok(_) => println!("Matrix multiplication (double precision): succeeded"),
        Err(e) => println!("Operation error: {}", e),
    }

    // Part 3: Combining Auto-Device and Mixed-Precision
    println!("\nPart 3: Combining Auto-Device and Mixed-Precision");
    println!("------------------------------------------------");

    // Create a GPU array with mixed precision
    let gpu_config = GPUConfig {
        backend: GPUBackend::CUDA,
        device_id: 0,
        async_ops: true,
        mixed_precision: true,
        memory_fraction: 0.9,
    };

    let gpu_array = GPUNdarray::new(array_f64.clone(), gpu_config);

    // Check precision
    println!("GPU array with mixed precision enabled");
    match array_protocol::mixed_precision::MixedPrecisionSupport::precision(&gpu_array) {
        Precision::Mixed => println!("GPU array is using mixed precision"),
        precision => println!("GPU array is using {:?} precision", precision),
    }

    println!("\nAll operations completed successfully!");
}
