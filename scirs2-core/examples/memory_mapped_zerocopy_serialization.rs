//! Example demonstrating zero-copy serialization and deserialization with memory-mapped arrays.
//!
//! This example shows how to:
//! 1. Create and save arrays with zero-copy serialization
//! 2. Load arrays with zero-copy deserialization
//! 3. Work with metadata in zero-copy serialized files
//! 4. Modify arrays and save changes
//! 5. Compare performance with traditional serialization methods
//! 6. Implement and use custom types with zero-copy serialization

use ndarray::{Array, Array1, Array2, Array3, IxDyn};
use scirs2_core::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use scirs2_core::memory_efficient::{
    AccessMode, MemoryMappedArray, ZeroCopySerializable, ZeroCopySerialization,
};
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::mem;
use std::path::Path;
use std::slice;
use std::time::Instant;
use tempfile::tempdir;

// Custom complex number type that implements ZeroCopySerializable
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
struct Complex64 {
    real: f64,
    imag: f64,
}

impl Complex64 {
    fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }
}

// Implementation of ZeroCopySerializable for our custom type
impl ZeroCopySerializable for Complex64 {
    unsafe fn from_bytes(bytes: &[u8]) -> CoreResult<Self> {
        if !Self::validate_bytes(bytes) {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Invalid byte length for Complex64: expected {} got {}",
                    mem::size_of::<Self>(),
                    bytes.len()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let ptr = bytes.as_ptr() as *const Self;
        Ok(*ptr)
    }

    unsafe fn as_bytes(&self) -> &[u8] {
        let ptr = self as *const Self as *const u8;
        slice::from_raw_parts(ptr, mem::size_of::<Self>())
    }

    // Override the type identifier for more specific validation
    fn type_identifier() -> &'static str {
        "Complex64"
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory-Mapped Array Zero-Copy Serialization Example");
    println!("==================================================\n");

    // Create a temporary directory for our example files
    let dir = tempdir()?;
    println!("Using temporary directory: {:?}", dir.path());

    // Basic serialization example
    basic_serialization_example(dir.path())?;

    // Working with metadata example
    metadata_example(dir.path())?;

    // Multidimensional array example
    multidimensional_example(dir.path())?;

    // Performance comparison example
    performance_comparison(dir.path())?;

    // Updating data example
    updating_data_example(dir.path())?;

    // Custom type example
    custom_type_example(dir.path())?;

    println!("\nAll examples completed successfully!");
    Ok(())
}

/// Example demonstrating custom type serialization with zero-copy operations
fn custom_type_example(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n6. Custom Type Serialization Example");
    println!("-----------------------------------");

    // Create a 2D array of complex numbers in a spiral pattern
    let size = 10;
    println!("Creating a {}x{} array of complex numbers", size, size);

    let data = Array2::<Complex64>::from_shape_fn((size, size), |(i, j)| {
        // Create a spiral pattern for visually interesting values
        let distance = ((i as f64 - size as f64 / 2.0).powi(2)
            + (j as f64 - size as f64 / 2.0).powi(2))
        .sqrt();
        let angle =
            ((j as f64 - size as f64 / 2.0) / (i as f64 - size as f64 / 2.0 + 0.001)).atan();

        Complex64::new(distance * angle.cos(), distance * angle.sin())
    });

    // Display a small slice of the array
    println!("\nSample of the complex array (3x3 corner):");
    for i in 0..3 {
        let mut row = String::new();
        for j in 0..3 {
            let c = data[[i, j]];
            row.push_str(&format!("({:.2}+{:.2}i) ", c.real, c.imag));
        }
        println!("  {}", row);
    }

    // Save with metadata
    let file_path = temp_dir.join("complex_array.bin");

    let metadata = json!({
        "description": "Complex number array with spiral pattern",
        "type": "Complex64",
        "dimensions": [size, size],
        "created": "2023-05-20",
        "custom_properties": {
            "pattern": "spiral",
            "element_format": "real+imaginary"
        }
    });

    // Time the save operation
    let start = Instant::now();
    MemoryMappedArray::<Complex64>::save_array(&data, &file_path, Some(metadata.clone()))?;
    let save_time = start.elapsed();
    println!(
        "\nSaved complex array with zero-copy serialization in {:?}",
        save_time
    );
    println!("File size: {} bytes", file_path.metadata()?.len());

    // Now load the array back
    let start = Instant::now();
    let loaded = MemoryMappedArray::<Complex64>::open_zero_copy(&file_path, AccessMode::ReadOnly)?;
    let load_time = start.elapsed();
    println!(
        "Loaded complex array with zero-copy deserialization in {:?}",
        load_time
    );

    // Read the array
    let loaded_array = loaded.readonly_array::<ndarray::Ix2>()?;

    // Calculate magnitude for each element in a corner sample
    println!("\nCalculating magnitudes from loaded array (3x3 corner):");
    for i in 0..3 {
        let mut row = String::new();
        for j in 0..3 {
            let c = loaded_array[[i, j]];
            let mag = c.magnitude();
            row.push_str(&format!("{:.2} ", mag));
        }
        println!("  {}", row);
    }

    // Validate the data integrity
    let mut equal = true;
    for i in 0..size {
        for j in 0..size {
            if data[[i, j]] != loaded_array[[i, j]] {
                equal = false;
                println!("Data mismatch at [{}, {}]", i, j);
                break;
            }
        }
        if !equal {
            break;
        }
    }

    if equal {
        println!("\nVerification successful: All complex values loaded correctly");
    }

    // Read and display metadata
    let loaded_metadata = MemoryMappedArray::<Complex64>::read_metadata(&file_path)?;
    println!("\nMetadata from file:");
    println!("  Description: {}", loaded_metadata["description"]);
    println!("  Type: {}", loaded_metadata["type"]);
    println!(
        "  Pattern: {}",
        loaded_metadata["custom_properties"]["pattern"]
    );

    Ok(())
}

/// Basic example of zero-copy serialization and deserialization
fn basic_serialization_example(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n1. Basic Zero-Copy Serialization Example");
    println!("-----------------------------------------");

    // Create a 1D array with 1 million elements
    let size = 1_000_000;
    let data = Array1::<f64>::linspace(0.0, 999_999.0, size);
    println!("Created a 1D array with {} elements", size);

    // Set up file path for saving
    let file_path = temp_dir.join("basic_example.bin");

    // Create metadata for the array
    let metadata = json!({
        "description": "Basic example array",
        "created": "2023-05-20",
        "elements": size,
        "element_type": "f64"
    });

    // Save with zero-copy serialization
    let start = Instant::now();
    let mmap = MemoryMappedArray::<f64>::save_array(&data, &file_path, Some(metadata))?;
    let save_time = start.elapsed();
    println!(
        "Saved array with zero-copy serialization in {:?}",
        save_time
    );
    println!("File size: {} bytes", file_path.metadata()?.len());

    // Load with zero-copy deserialization
    let start = Instant::now();
    let loaded = MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly)?;
    let load_time = start.elapsed();
    println!(
        "Loaded array with zero-copy deserialization in {:?}",
        load_time
    );

    // Verify the loaded array
    let loaded_array = loaded.readonly_array::<ndarray::Ix1>()?;
    println!(
        "Loaded array shape: {:?}, elements: {}",
        loaded_array.shape(),
        loaded_array.len()
    );

    // Check a few values
    println!("Verifying values:");
    println!("  Element [0]: {}", loaded_array[0]);
    println!("  Element [500000]: {}", loaded_array[500000]);
    println!("  Element [999999]: {}", loaded_array[999999]);

    // Read metadata
    let loaded_metadata = MemoryMappedArray::<f64>::read_metadata(&file_path)?;
    println!("\nMetadata from file:");
    println!("  Description: {}", loaded_metadata["description"]);
    println!("  Created: {}", loaded_metadata["created"]);
    println!("  Elements: {}", loaded_metadata["elements"]);
    println!("  Element type: {}", loaded_metadata["element_type"]);

    Ok(())
}

/// Example demonstrating custom type serialization with zero-copy operations
fn custom_type_example(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n6. Custom Type Serialization Example");
    println!("-----------------------------------");

    // Create a 2D array of complex numbers in a spiral pattern
    let size = 10;
    println!("Creating a {}x{} array of complex numbers", size, size);

    let data = Array2::<Complex64>::from_shape_fn((size, size), |(i, j)| {
        // Create a spiral pattern for visually interesting values
        let distance = ((i as f64 - size as f64 / 2.0).powi(2)
            + (j as f64 - size as f64 / 2.0).powi(2))
        .sqrt();
        let angle =
            ((j as f64 - size as f64 / 2.0) / (i as f64 - size as f64 / 2.0 + 0.001)).atan();

        Complex64::new(distance * angle.cos(), distance * angle.sin())
    });

    // Display a small slice of the array
    println!("\nSample of the complex array (3x3 corner):");
    for i in 0..3 {
        let mut row = String::new();
        for j in 0..3 {
            let c = data[[i, j]];
            row.push_str(&format!("({:.2}+{:.2}i) ", c.real, c.imag));
        }
        println!("  {}", row);
    }

    // Save with metadata
    let file_path = temp_dir.join("complex_array.bin");

    let metadata = json!({
        "description": "Complex number array with spiral pattern",
        "type": "Complex64",
        "dimensions": [size, size],
        "created": "2023-05-20",
        "custom_properties": {
            "pattern": "spiral",
            "element_format": "real+imaginary"
        }
    });

    // Time the save operation
    let start = Instant::now();
    MemoryMappedArray::<Complex64>::save_array(&data, &file_path, Some(metadata.clone()))?;
    let save_time = start.elapsed();
    println!(
        "\nSaved complex array with zero-copy serialization in {:?}",
        save_time
    );
    println!("File size: {} bytes", file_path.metadata()?.len());

    // Now load the array back
    let start = Instant::now();
    let loaded = MemoryMappedArray::<Complex64>::open_zero_copy(&file_path, AccessMode::ReadOnly)?;
    let load_time = start.elapsed();
    println!(
        "Loaded complex array with zero-copy deserialization in {:?}",
        load_time
    );

    // Read the array
    let loaded_array = loaded.readonly_array::<ndarray::Ix2>()?;

    // Calculate magnitude for each element in a corner sample
    println!("\nCalculating magnitudes from loaded array (3x3 corner):");
    for i in 0..3 {
        let mut row = String::new();
        for j in 0..3 {
            let c = loaded_array[[i, j]];
            let mag = c.magnitude();
            row.push_str(&format!("{:.2} ", mag));
        }
        println!("  {}", row);
    }

    // Validate the data integrity
    let mut equal = true;
    for i in 0..size {
        for j in 0..size {
            if data[[i, j]] != loaded_array[[i, j]] {
                equal = false;
                println!("Data mismatch at [{}, {}]", i, j);
                break;
            }
        }
        if !equal {
            break;
        }
    }

    if equal {
        println!("\nVerification successful: All complex values loaded correctly");
    }

    // Read and display metadata
    let loaded_metadata = MemoryMappedArray::<Complex64>::read_metadata(&file_path)?;
    println!("\nMetadata from file:");
    println!("  Description: {}", loaded_metadata["description"]);
    println!("  Type: {}", loaded_metadata["type"]);
    println!(
        "  Pattern: {}",
        loaded_metadata["custom_properties"]["pattern"]
    );

    Ok(())
}

/// Example demonstrating working with metadata in zero-copy serialized files
fn metadata_example(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. Working with Metadata Example");
    println!("--------------------------------");

    // Create a small array for this example
    let data = Array1::<f32>::linspace(0.0, 99.0, 100);
    println!("Created a small array with 100 elements");

    // Set up file path for saving
    let file_path = temp_dir.join("metadata_example.bin");

    // Create rich metadata for the array
    let initial_metadata = json!({
        "description": "Scientific dataset example",
        "version": "1.0",
        "author": "SciRS2 Team",
        "created": "2023-05-20T10:30:00Z",
        "license": "MIT",
        "properties": {
            "sampling_rate": 1000,
            "units": "meters",
            "calibration_factor": 1.05,
            "valid_range": [0.0, 100.0]
        },
        "tags": ["example", "scientific", "numeric"]
    });

    // Save array with metadata
    MemoryMappedArray::<f32>::save_array(&data, &file_path, Some(initial_metadata))?;
    println!("Saved array with rich metadata");

    // Read metadata without loading the array
    let metadata = MemoryMappedArray::<f32>::read_metadata(&file_path)?;
    println!("\nMetadata before update:");
    println!("  Description: {}", metadata["description"]);
    println!("  Version: {}", metadata["version"]);
    println!(
        "  Sampling rate: {}",
        metadata["properties"]["sampling_rate"]
    );

    // Update metadata (without rewriting the entire array)
    println!("\nUpdating metadata...");
    let updated_metadata = json!({
        "description": "Scientific dataset example - Updated",
        "version": "1.1",
        "author": "SciRS2 Team",
        "created": "2023-05-20T10:30:00Z",
        "updated": "2023-05-20T11:45:00Z",
        "license": "MIT",
        "properties": {
            "sampling_rate": 1000,
            "units": "meters",
            "calibration_factor": 1.08,  // Updated calibration
            "valid_range": [0.0, 100.0],
            "processing": "filtered"      // Added new field
        },
        "tags": ["example", "scientific", "numeric", "processed"]
    });
    MemoryMappedArray::<f32>::update_metadata(&file_path, updated_metadata)?;

    // Read updated metadata
    let updated = MemoryMappedArray::<f32>::read_metadata(&file_path)?;
    println!("\nMetadata after update:");
    println!("  Description: {}", updated["description"]);
    println!("  Version: {}", updated["version"]);
    println!("  Updated: {}", updated["updated"]);
    println!(
        "  Calibration factor: {}",
        updated["properties"]["calibration_factor"]
    );
    println!("  Processing: {}", updated["properties"]["processing"]);
    println!("  Tags: {}", updated["tags"].to_string());

    // Load the array and verify data wasn't affected by metadata update
    let loaded = MemoryMappedArray::<f32>::open_zero_copy(&file_path, AccessMode::ReadOnly)?;
    let loaded_array = loaded.readonly_array::<ndarray::Ix1>()?;

    // Check a few values
    println!("\nVerifying data integrity after metadata update:");
    println!("  Element [0]: {}", loaded_array[0]);
    println!("  Element [50]: {}", loaded_array[50]);
    println!("  Element [99]: {}", loaded_array[99]);

    Ok(())
}

/// Example demonstrating custom type serialization with zero-copy operations
fn custom_type_example(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n6. Custom Type Serialization Example");
    println!("-----------------------------------");

    // Create a 2D array of complex numbers in a spiral pattern
    let size = 10;
    println!("Creating a {}x{} array of complex numbers", size, size);

    let data = Array2::<Complex64>::from_shape_fn((size, size), |(i, j)| {
        // Create a spiral pattern for visually interesting values
        let distance = ((i as f64 - size as f64 / 2.0).powi(2)
            + (j as f64 - size as f64 / 2.0).powi(2))
        .sqrt();
        let angle =
            ((j as f64 - size as f64 / 2.0) / (i as f64 - size as f64 / 2.0 + 0.001)).atan();

        Complex64::new(distance * angle.cos(), distance * angle.sin())
    });

    // Display a small slice of the array
    println!("\nSample of the complex array (3x3 corner):");
    for i in 0..3 {
        let mut row = String::new();
        for j in 0..3 {
            let c = data[[i, j]];
            row.push_str(&format!("({:.2}+{:.2}i) ", c.real, c.imag));
        }
        println!("  {}", row);
    }

    // Save with metadata
    let file_path = temp_dir.join("complex_array.bin");

    let metadata = json!({
        "description": "Complex number array with spiral pattern",
        "type": "Complex64",
        "dimensions": [size, size],
        "created": "2023-05-20",
        "custom_properties": {
            "pattern": "spiral",
            "element_format": "real+imaginary"
        }
    });

    // Time the save operation
    let start = Instant::now();
    MemoryMappedArray::<Complex64>::save_array(&data, &file_path, Some(metadata.clone()))?;
    let save_time = start.elapsed();
    println!(
        "\nSaved complex array with zero-copy serialization in {:?}",
        save_time
    );
    println!("File size: {} bytes", file_path.metadata()?.len());

    // Now load the array back
    let start = Instant::now();
    let loaded = MemoryMappedArray::<Complex64>::open_zero_copy(&file_path, AccessMode::ReadOnly)?;
    let load_time = start.elapsed();
    println!(
        "Loaded complex array with zero-copy deserialization in {:?}",
        load_time
    );

    // Read the array
    let loaded_array = loaded.readonly_array::<ndarray::Ix2>()?;

    // Calculate magnitude for each element in a corner sample
    println!("\nCalculating magnitudes from loaded array (3x3 corner):");
    for i in 0..3 {
        let mut row = String::new();
        for j in 0..3 {
            let c = loaded_array[[i, j]];
            let mag = c.magnitude();
            row.push_str(&format!("{:.2} ", mag));
        }
        println!("  {}", row);
    }

    // Validate the data integrity
    let mut equal = true;
    for i in 0..size {
        for j in 0..size {
            if data[[i, j]] != loaded_array[[i, j]] {
                equal = false;
                println!("Data mismatch at [{}, {}]", i, j);
                break;
            }
        }
        if !equal {
            break;
        }
    }

    if equal {
        println!("\nVerification successful: All complex values loaded correctly");
    }

    // Read and display metadata
    let loaded_metadata = MemoryMappedArray::<Complex64>::read_metadata(&file_path)?;
    println!("\nMetadata from file:");
    println!("  Description: {}", loaded_metadata["description"]);
    println!("  Type: {}", loaded_metadata["type"]);
    println!(
        "  Pattern: {}",
        loaded_metadata["custom_properties"]["pattern"]
    );

    Ok(())
}

/// Example demonstrating zero-copy serialization with multidimensional arrays
fn multidimensional_example(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3. Multidimensional Array Example");
    println!("---------------------------------");

    // Create a 3D array (5x5x5)
    let data = Array3::<i32>::from_shape_fn((5, 5, 5), |(i, j, k)| (i * 25 + j * 5 + k) as i32);
    println!("Created a 3D array with shape {:?}", data.shape());

    // Set up file path for saving
    let file_path = temp_dir.join("3d_array.bin");

    // Save with zero-copy serialization
    MemoryMappedArray::<i32>::save_array(&data, &file_path, None)?;
    println!("Saved 3D array to file");

    // Load with zero-copy deserialization
    let loaded = MemoryMappedArray::<i32>::open_zero_copy(&file_path, AccessMode::ReadOnly)?;
    println!("Loaded 3D array from file");

    // Verify the loaded array
    let loaded_array = loaded.readonly_array::<ndarray::Ix3>()?;
    println!(
        "Loaded array shape: {:?}, elements: {}",
        loaded_array.shape(),
        loaded_array.len()
    );

    // Print a slice of the 3D array
    println!("\nSlice of the 3D array (z=2):");
    for i in 0..5 {
        let mut row = String::new();
        for j in 0..5 {
            row.push_str(&format!("{:3} ", loaded_array[[i, j, 2]]));
        }
        println!("  {}", row);
    }

    // Create a dynamic-dimension array
    println!("\nCreating and saving a dynamic-dimension array...");
    let dyn_data = Array::from_shape_fn(IxDyn(&[3, 4, 2, 5]), |idx| {
        // Convert multidimensional index to a single value for this example
        let mut val = 0;
        let mut factor = 1;
        for &i in idx.slice().iter().rev() {
            val += i * factor;
            factor *= 10;
        }
        val as f64
    });

    let dyn_file_path = temp_dir.join("dyn_array.bin");
    MemoryMappedArray::<f64>::save_array(&dyn_data, &dyn_file_path, None)?;

    // Load dynamic array
    let loaded_dyn =
        MemoryMappedArray::<f64>::open_zero_copy(&dyn_file_path, AccessMode::ReadOnly)?;
    let loaded_dyn_array = loaded_dyn.readonly_array::<ndarray::IxDyn>()?;

    println!(
        "Loaded dynamic array shape: {:?}, elements: {}",
        loaded_dyn_array.shape(),
        loaded_dyn_array.len()
    );

    // Print a few values
    println!("Some values from the dynamic array:");
    println!(
        "  Value at [0,0,0,0]: {}",
        loaded_dyn_array[IxDyn(&[0, 0, 0, 0])]
    );
    println!(
        "  Value at [1,2,1,3]: {}",
        loaded_dyn_array[IxDyn(&[1, 2, 1, 3])]
    );
    println!(
        "  Value at [2,3,1,4]: {}",
        loaded_dyn_array[IxDyn(&[2, 3, 1, 4])]
    );

    Ok(())
}

/// Example demonstrating custom type serialization with zero-copy operations
fn custom_type_example(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n6. Custom Type Serialization Example");
    println!("-----------------------------------");

    // Create a 2D array of complex numbers in a spiral pattern
    let size = 10;
    println!("Creating a {}x{} array of complex numbers", size, size);

    let data = Array2::<Complex64>::from_shape_fn((size, size), |(i, j)| {
        // Create a spiral pattern for visually interesting values
        let distance = ((i as f64 - size as f64 / 2.0).powi(2)
            + (j as f64 - size as f64 / 2.0).powi(2))
        .sqrt();
        let angle =
            ((j as f64 - size as f64 / 2.0) / (i as f64 - size as f64 / 2.0 + 0.001)).atan();

        Complex64::new(distance * angle.cos(), distance * angle.sin())
    });

    // Display a small slice of the array
    println!("\nSample of the complex array (3x3 corner):");
    for i in 0..3 {
        let mut row = String::new();
        for j in 0..3 {
            let c = data[[i, j]];
            row.push_str(&format!("({:.2}+{:.2}i) ", c.real, c.imag));
        }
        println!("  {}", row);
    }

    // Save with metadata
    let file_path = temp_dir.join("complex_array.bin");

    let metadata = json!({
        "description": "Complex number array with spiral pattern",
        "type": "Complex64",
        "dimensions": [size, size],
        "created": "2023-05-20",
        "custom_properties": {
            "pattern": "spiral",
            "element_format": "real+imaginary"
        }
    });

    // Time the save operation
    let start = Instant::now();
    MemoryMappedArray::<Complex64>::save_array(&data, &file_path, Some(metadata.clone()))?;
    let save_time = start.elapsed();
    println!(
        "\nSaved complex array with zero-copy serialization in {:?}",
        save_time
    );
    println!("File size: {} bytes", file_path.metadata()?.len());

    // Now load the array back
    let start = Instant::now();
    let loaded = MemoryMappedArray::<Complex64>::open_zero_copy(&file_path, AccessMode::ReadOnly)?;
    let load_time = start.elapsed();
    println!(
        "Loaded complex array with zero-copy deserialization in {:?}",
        load_time
    );

    // Read the array
    let loaded_array = loaded.readonly_array::<ndarray::Ix2>()?;

    // Calculate magnitude for each element in a corner sample
    println!("\nCalculating magnitudes from loaded array (3x3 corner):");
    for i in 0..3 {
        let mut row = String::new();
        for j in 0..3 {
            let c = loaded_array[[i, j]];
            let mag = c.magnitude();
            row.push_str(&format!("{:.2} ", mag));
        }
        println!("  {}", row);
    }

    // Validate the data integrity
    let mut equal = true;
    for i in 0..size {
        for j in 0..size {
            if data[[i, j]] != loaded_array[[i, j]] {
                equal = false;
                println!("Data mismatch at [{}, {}]", i, j);
                break;
            }
        }
        if !equal {
            break;
        }
    }

    if equal {
        println!("\nVerification successful: All complex values loaded correctly");
    }

    // Read and display metadata
    let loaded_metadata = MemoryMappedArray::<Complex64>::read_metadata(&file_path)?;
    println!("\nMetadata from file:");
    println!("  Description: {}", loaded_metadata["description"]);
    println!("  Type: {}", loaded_metadata["type"]);
    println!(
        "  Pattern: {}",
        loaded_metadata["custom_properties"]["pattern"]
    );

    Ok(())
}

/// Example comparing performance of zero-copy serialization with traditional methods
fn performance_comparison(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n4. Performance Comparison Example");
    println!("--------------------------------");

    // Create a large 2D array (1000x1000) for performance testing
    let size = 1000;
    let data = Array2::<f64>::from_shape_fn((size, size), |(i, j)| (i * size + j) as f64);
    println!("Created a {}x{} array for performance testing", size, size);

    // Calculate memory size
    let memory_size = data.len() * std::mem::size_of::<f64>();
    println!(
        "Array size in memory: {:.2} MB",
        memory_size as f64 / (1024.0 * 1024.0)
    );

    // 1. Zero-copy serialization
    let zero_copy_path = temp_dir.join("zero_copy_perf.bin");
    let start = Instant::now();
    MemoryMappedArray::<f64>::save_array(&data, &zero_copy_path, None)?;
    let zero_copy_save_time = start.elapsed();

    // 2. Traditional serialization (using bincode)
    let traditional_path = temp_dir.join("traditional_perf.bin");
    let start = Instant::now();
    let serialized = bincode::serialize(&data)?;
    let mut file = File::create(&traditional_path)?;
    file.write_all(&serialized)?;
    let traditional_save_time = start.elapsed();

    // 3. Loading with zero-copy deserialization
    let start = Instant::now();
    let _loaded_zero_copy =
        MemoryMappedArray::<f64>::open_zero_copy(&zero_copy_path, AccessMode::ReadOnly)?;
    let zero_copy_load_time = start.elapsed();

    // 4. Loading with traditional deserialization
    let start = Instant::now();
    let mut file = File::open(&traditional_path)?;
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer)?;
    let _loaded_traditional: Array2<f64> = bincode::deserialize(&buffer)?;
    let traditional_load_time = start.elapsed();

    // 5. Array access time (zero-copy)
    let loaded = MemoryMappedArray::<f64>::open_zero_copy(&zero_copy_path, AccessMode::ReadOnly)?;
    let start = Instant::now();
    let array = loaded.readonly_array::<ndarray::Ix2>()?;
    let mut sum = 0.0;
    for i in 0..10 {
        for j in 0..10 {
            sum += array[[i, j]];
        }
    }
    let zero_copy_access_time = start.elapsed();

    // 6. Array access time (traditional)
    let loaded_traditional: Array2<f64> = bincode::deserialize(&buffer)?;
    let start = Instant::now();
    let mut sum = 0.0;
    for i in 0..10 {
        for j in 0..10 {
            sum += loaded_traditional[[i, j]];
        }
    }
    let traditional_access_time = start.elapsed();

    // Print performance results
    println!("\nPerformance Results:");
    println!("  File sizes:");
    println!(
        "    Zero-copy: {:.2} MB",
        zero_copy_path.metadata()?.len() as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    Traditional: {:.2} MB",
        traditional_path.metadata()?.len() as f64 / (1024.0 * 1024.0)
    );
    println!("\n  Serialization times:");
    println!("    Zero-copy: {:?}", zero_copy_save_time);
    println!("    Traditional: {:?}", traditional_save_time);
    println!("\n  Deserialization times:");
    println!("    Zero-copy: {:?}", zero_copy_load_time);
    println!("    Traditional: {:?}", traditional_load_time);
    println!("\n  Array access times (10x10 block):");
    println!("    Zero-copy: {:?}", zero_copy_access_time);
    println!("    Traditional: {:?}", traditional_access_time);

    println!("\nPerformance summary:");
    // Serialization
    let ser_ratio =
        zero_copy_save_time.as_micros() as f64 / traditional_save_time.as_micros() as f64;
    if ser_ratio < 1.0 {
        println!(
            "  Zero-copy serialization is {:.2}x faster than traditional",
            1.0 / ser_ratio
        );
    } else {
        println!(
            "  Traditional serialization is {:.2}x faster than zero-copy",
            ser_ratio
        );
    }

    // Deserialization
    let deser_ratio =
        zero_copy_load_time.as_micros() as f64 / traditional_load_time.as_micros() as f64;
    if deser_ratio < 1.0 {
        println!(
            "  Zero-copy deserialization is {:.2}x faster than traditional",
            1.0 / deser_ratio
        );
    } else {
        println!(
            "  Traditional deserialization is {:.2}x faster than zero-copy",
            deser_ratio
        );
    }

    // Access time
    let access_ratio =
        zero_copy_access_time.as_micros() as f64 / traditional_access_time.as_micros() as f64;
    if access_ratio < 1.0 {
        println!(
            "  Zero-copy access is {:.2}x faster than traditional",
            1.0 / access_ratio
        );
    } else {
        println!(
            "  Traditional access is {:.2}x faster than zero-copy",
            access_ratio
        );
    }

    println!("\nNote: Zero-copy serialization's main advantage is with extremely large arrays that don't fit in memory");

    Ok(())
}

/// Example demonstrating custom type serialization with zero-copy operations
fn custom_type_example(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n6. Custom Type Serialization Example");
    println!("-----------------------------------");

    // Create a 2D array of complex numbers in a spiral pattern
    let size = 10;
    println!("Creating a {}x{} array of complex numbers", size, size);

    let data = Array2::<Complex64>::from_shape_fn((size, size), |(i, j)| {
        // Create a spiral pattern for visually interesting values
        let distance = ((i as f64 - size as f64 / 2.0).powi(2)
            + (j as f64 - size as f64 / 2.0).powi(2))
        .sqrt();
        let angle =
            ((j as f64 - size as f64 / 2.0) / (i as f64 - size as f64 / 2.0 + 0.001)).atan();

        Complex64::new(distance * angle.cos(), distance * angle.sin())
    });

    // Display a small slice of the array
    println!("\nSample of the complex array (3x3 corner):");
    for i in 0..3 {
        let mut row = String::new();
        for j in 0..3 {
            let c = data[[i, j]];
            row.push_str(&format!("({:.2}+{:.2}i) ", c.real, c.imag));
        }
        println!("  {}", row);
    }

    // Save with metadata
    let file_path = temp_dir.join("complex_array.bin");

    let metadata = json!({
        "description": "Complex number array with spiral pattern",
        "type": "Complex64",
        "dimensions": [size, size],
        "created": "2023-05-20",
        "custom_properties": {
            "pattern": "spiral",
            "element_format": "real+imaginary"
        }
    });

    // Time the save operation
    let start = Instant::now();
    MemoryMappedArray::<Complex64>::save_array(&data, &file_path, Some(metadata.clone()))?;
    let save_time = start.elapsed();
    println!(
        "\nSaved complex array with zero-copy serialization in {:?}",
        save_time
    );
    println!("File size: {} bytes", file_path.metadata()?.len());

    // Now load the array back
    let start = Instant::now();
    let loaded = MemoryMappedArray::<Complex64>::open_zero_copy(&file_path, AccessMode::ReadOnly)?;
    let load_time = start.elapsed();
    println!(
        "Loaded complex array with zero-copy deserialization in {:?}",
        load_time
    );

    // Read the array
    let loaded_array = loaded.readonly_array::<ndarray::Ix2>()?;

    // Calculate magnitude for each element in a corner sample
    println!("\nCalculating magnitudes from loaded array (3x3 corner):");
    for i in 0..3 {
        let mut row = String::new();
        for j in 0..3 {
            let c = loaded_array[[i, j]];
            let mag = c.magnitude();
            row.push_str(&format!("{:.2} ", mag));
        }
        println!("  {}", row);
    }

    // Validate the data integrity
    let mut equal = true;
    for i in 0..size {
        for j in 0..size {
            if data[[i, j]] != loaded_array[[i, j]] {
                equal = false;
                println!("Data mismatch at [{}, {}]", i, j);
                break;
            }
        }
        if !equal {
            break;
        }
    }

    if equal {
        println!("\nVerification successful: All complex values loaded correctly");
    }

    // Read and display metadata
    let loaded_metadata = MemoryMappedArray::<Complex64>::read_metadata(&file_path)?;
    println!("\nMetadata from file:");
    println!("  Description: {}", loaded_metadata["description"]);
    println!("  Type: {}", loaded_metadata["type"]);
    println!(
        "  Pattern: {}",
        loaded_metadata["custom_properties"]["pattern"]
    );

    Ok(())
}

/// Example demonstrating updating data in a zero-copy serialized file
fn updating_data_example(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n5. Updating Data Example");
    println!("------------------------");

    // Create a 2D array (10x10)
    let data = Array2::<f32>::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f32);
    println!("Created a 10x10 array");

    // Set up file path for saving
    let file_path = temp_dir.join("updateable_array.bin");

    // Save with zero-copy serialization
    MemoryMappedArray::<f32>::save_array(&data, &file_path, None)?;
    println!("Saved initial array to file");

    // Display original data
    println!("\nOriginal array (first 5x5 corner):");
    for i in 0..5 {
        let mut row = String::new();
        for j in 0..5 {
            row.push_str(&format!("{:4.0} ", data[[i, j]]));
        }
        println!("  {}", row);
    }

    // Load with read-write access
    let mut mmap = MemoryMappedArray::<f32>::open_zero_copy(&file_path, AccessMode::ReadWrite)?;
    println!("\nLoaded array with read-write access for updating");

    // Modify the array
    {
        let mut array = mmap.as_array_mut::<ndarray::Ix2>()?;

        // Set diagonal elements to 1000
        for i in 0..10 {
            array[[i, i]] = 1000.0;
        }

        // Set top-right corner to -1
        for i in 0..5 {
            for j in 5..10 {
                array[[i, j]] = -1.0;
            }
        }
    }

    // Flush changes to disk
    mmap.flush()?;
    println!("Modified array and flushed changes to disk");

    // Load again to verify changes
    let loaded = MemoryMappedArray::<f32>::open_zero_copy(&file_path, AccessMode::ReadOnly)?;
    let loaded_array = loaded.readonly_array::<ndarray::Ix2>()?;

    // Display modified data
    println!("\nModified array (10x10):");
    for i in 0..10 {
        let mut row = String::new();
        for j in 0..10 {
            row.push_str(&format!("{:5.0} ", loaded_array[[i, j]]));
        }
        println!("  {}", row);
    }

    Ok(())
}

/// Example demonstrating custom type serialization with zero-copy operations
fn custom_type_example(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n6. Custom Type Serialization Example");
    println!("-----------------------------------");

    // Create a 2D array of complex numbers in a spiral pattern
    let size = 10;
    println!("Creating a {}x{} array of complex numbers", size, size);

    let data = Array2::<Complex64>::from_shape_fn((size, size), |(i, j)| {
        // Create a spiral pattern for visually interesting values
        let distance = ((i as f64 - size as f64 / 2.0).powi(2)
            + (j as f64 - size as f64 / 2.0).powi(2))
        .sqrt();
        let angle =
            ((j as f64 - size as f64 / 2.0) / (i as f64 - size as f64 / 2.0 + 0.001)).atan();

        Complex64::new(distance * angle.cos(), distance * angle.sin())
    });

    // Display a small slice of the array
    println!("\nSample of the complex array (3x3 corner):");
    for i in 0..3 {
        let mut row = String::new();
        for j in 0..3 {
            let c = data[[i, j]];
            row.push_str(&format!("({:.2}+{:.2}i) ", c.real, c.imag));
        }
        println!("  {}", row);
    }

    // Save with metadata
    let file_path = temp_dir.join("complex_array.bin");

    let metadata = json!({
        "description": "Complex number array with spiral pattern",
        "type": "Complex64",
        "dimensions": [size, size],
        "created": "2023-05-20",
        "custom_properties": {
            "pattern": "spiral",
            "element_format": "real+imaginary"
        }
    });

    // Time the save operation
    let start = Instant::now();
    MemoryMappedArray::<Complex64>::save_array(&data, &file_path, Some(metadata.clone()))?;
    let save_time = start.elapsed();
    println!(
        "\nSaved complex array with zero-copy serialization in {:?}",
        save_time
    );
    println!("File size: {} bytes", file_path.metadata()?.len());

    // Now load the array back
    let start = Instant::now();
    let loaded = MemoryMappedArray::<Complex64>::open_zero_copy(&file_path, AccessMode::ReadOnly)?;
    let load_time = start.elapsed();
    println!(
        "Loaded complex array with zero-copy deserialization in {:?}",
        load_time
    );

    // Read the array
    let loaded_array = loaded.readonly_array::<ndarray::Ix2>()?;

    // Calculate magnitude for each element in a corner sample
    println!("\nCalculating magnitudes from loaded array (3x3 corner):");
    for i in 0..3 {
        let mut row = String::new();
        for j in 0..3 {
            let c = loaded_array[[i, j]];
            let mag = c.magnitude();
            row.push_str(&format!("{:.2} ", mag));
        }
        println!("  {}", row);
    }

    // Validate the data integrity
    let mut equal = true;
    for i in 0..size {
        for j in 0..size {
            if data[[i, j]] != loaded_array[[i, j]] {
                equal = false;
                println!("Data mismatch at [{}, {}]", i, j);
                break;
            }
        }
        if !equal {
            break;
        }
    }

    if equal {
        println!("\nVerification successful: All complex values loaded correctly");
    }

    // Read and display metadata
    let loaded_metadata = MemoryMappedArray::<Complex64>::read_metadata(&file_path)?;
    println!("\nMetadata from file:");
    println!("  Description: {}", loaded_metadata["description"]);
    println!("  Type: {}", loaded_metadata["type"]);
    println!(
        "  Pattern: {}",
        loaded_metadata["custom_properties"]["pattern"]
    );

    Ok(())
}
