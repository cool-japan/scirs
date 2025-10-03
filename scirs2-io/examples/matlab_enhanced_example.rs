//! Enhanced MATLAB format support example
//!
//! This example demonstrates the enhanced MATLAB file format capabilities including:
//! - Complete MAT v5 format writing and reading
//! - Enhanced format support with automatic selection
//! - Cell array and structure creation
//! - Configuration options and format detection
//! - Error handling and performance monitoring

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_io::matlab::enhanced::{
    create_cell_array, create_struct, read_mat_enhanced, write_mat_enhanced, EnhancedMatFile,
    MatFileConfig,
};
use scirs2_io::matlab::{read_mat, write_mat, MatType};
use std::collections::HashMap;
use std::time::Instant;
use tempfile::tempdir;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Enhanced MATLAB Format Support Example");
    println!("=========================================");

    // Demonstrate basic MAT v5 format
    demonstrate_basic_mat_v5()?;

    // Demonstrate enhanced format features
    demonstrate_enhanced_features()?;

    // Demonstrate cell arrays
    demonstrate_cell_arrays()?;

    // Demonstrate structures
    demonstrate_structures()?;

    // Demonstrate format detection and automatic selection
    demonstrate_format_detection()?;

    // Demonstrate performance and error handling
    demonstrate_performance_and_errors()?;

    println!("\n✅ All MATLAB format demonstrations completed successfully!");
    println!("💡 Key benefits of the enhanced MATLAB system:");
    println!("   - Complete MAT v5 format writing support");
    println!("   - Enhanced format detection and automatic selection");
    println!("   - Cell array and structure support");
    println!("   - Robust error handling and validation");
    println!("   - Performance monitoring and optimization");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_basic_mat_v5() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Demonstrating Basic MAT v5 Format...");

    let temp_dir = tempdir()?;
    let mat_file = temp_dir.path().join("basic_test.mat");

    // Create various data types
    let mut vars = HashMap::new();

    // Numeric arrays
    println!("  🔹 Creating numeric arrays:");
    let double_array = Array1::from(vec![1.0, 2.0, std::f64::consts::PI, -5.5]).into_dyn();
    vars.insert("double_data".to_string(), MatType::Double(double_array));

    let single_array = Array1::from(vec![1.0f32, 2.5f32, std::f32::consts::PI]).into_dyn();
    vars.insert("single_data".to_string(), MatType::Single(single_array));

    let int32_array = Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j) as i32).into_dyn();
    vars.insert("int32_matrix".to_string(), MatType::Int32(int32_array));

    // Logical array
    let logical_array = Array1::from(vec![true, false, true, false]).into_dyn();
    vars.insert("logical_data".to_string(), MatType::Logical(logical_array));

    // Character data
    vars.insert(
        "text_data".to_string(),
        MatType::Char("Hello MATLAB!".to_string()),
    );

    println!("    Created {} variables", vars.len());

    // Write to MAT file
    println!("  🔹 Writing MAT file...");
    let write_start = Instant::now();
    write_mat(&mat_file, &vars)?;
    let write_time = write_start.elapsed();
    println!("    Write time: {:.2}ms", write_time.as_secs_f64() * 1000.0);

    // Read back from MAT file
    println!("  🔹 Reading MAT file...");
    let read_start = Instant::now();
    let loaded_vars = read_mat(&mat_file)?;
    let read_time = read_start.elapsed();
    println!("    Read time: {:.2}ms", read_time.as_secs_f64() * 1000.0);
    println!("    Loaded {} variables", loaded_vars.len());

    // Verify data integrity
    println!("  🔹 Verifying data integrity:");
    for (name, original) in &vars {
        if let Some(loaded) = loaded_vars.get(name) {
            let matches = match (original, loaded) {
                (MatType::Double(orig), MatType::Double(load)) => orig.shape() == load.shape(),
                (MatType::Single(orig), MatType::Single(load)) => orig.shape() == load.shape(),
                (MatType::Int32(orig), MatType::Int32(load)) => orig.shape() == load.shape(),
                (MatType::Logical(orig), MatType::Logical(load)) => orig.shape() == load.shape(),
                (MatType::Char(orig), MatType::Char(load)) => orig == load,
                _ => false,
            };
            println!(
                "    {}: {}",
                name,
                if matches { "✅ OK" } else { "❌ MISMATCH" }
            );
        } else {
            println!("    {}: ❌ MISSING", name);
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_enhanced_features() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Demonstrating Enhanced Format Features...");

    let temp_dir = tempdir()?;
    let enhanced_file = temp_dir.path().join("enhanced_test.mat");

    // Create configuration for enhanced features
    let config = MatFileConfig {
        use_v73: false, // Use v5 for now since we don't have full HDF5 integration
        compression: None,
        v73_threshold: 1024 * 1024, // 1MB threshold
    };

    println!("  🔹 Configuration:");
    println!("    Use v7.3 format: {}", config.use_v73);
    println!("    Size threshold: {} bytes", config.v73_threshold);

    // Create enhanced file handler
    let enhanced = EnhancedMatFile::new(config.clone());

    // Create test data
    let mut vars = HashMap::new();
    let large_array = Array2::from_shape_fn((100, 100), |(i, j)| (i + j) as f64).into_dyn();
    vars.insert("large_matrix".to_string(), MatType::Double(large_array));

    let small_array = Array1::from(vec![1.0, 2.0, 3.0]).into_dyn();
    vars.insert("small_vector".to_string(), MatType::Double(small_array));

    println!("  🔹 Writing with enhanced format:");
    let write_start = Instant::now();
    enhanced.write(&enhanced_file, &vars)?;
    let write_time = write_start.elapsed();
    println!(
        "    Enhanced write time: {:.2}ms",
        write_time.as_secs_f64() * 1000.0
    );

    // Read with enhanced format
    println!("  🔹 Reading with enhanced format:");
    let read_start = Instant::now();
    let loaded_vars = enhanced.read(&enhanced_file)?;
    let read_time = read_start.elapsed();
    println!(
        "    Enhanced read time: {:.2}ms",
        read_time.as_secs_f64() * 1000.0
    );
    println!("    Loaded {} variables", loaded_vars.len());

    // Demonstrate convenience functions
    println!("  🔹 Using convenience functions:");
    let convenience_file = temp_dir.path().join("convenience_test.mat");

    let conv_write_start = Instant::now();
    write_mat_enhanced(&convenience_file, &vars, Some(config))?;
    let conv_write_time = conv_write_start.elapsed();

    let conv_read_start = Instant::now();
    let _conv_loaded = read_mat_enhanced(&convenience_file, None)?;
    let conv_read_time = conv_read_start.elapsed();

    println!(
        "    Convenience write time: {:.2}ms",
        conv_write_time.as_secs_f64() * 1000.0
    );
    println!(
        "    Convenience read time: {:.2}ms",
        conv_read_time.as_secs_f64() * 1000.0
    );

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_cell_arrays() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📦 Demonstrating Cell Arrays...");

    // Create cell array with mixed data types
    println!("  🔹 Creating cell array with mixed data:");
    let cells = vec![
        MatType::Double(Array1::from(vec![1.0, 2.0, 3.0]).into_dyn()),
        MatType::Char("Cell string".to_string()),
        MatType::Int32(Array2::from_shape_fn((2, 2), |(i, j)| (i + j) as i32).into_dyn()),
        MatType::Logical(Array1::from(vec![true, false, true]).into_dyn()),
    ];

    let cell_array = create_cell_array(cells);

    match &cell_array {
        MatType::Cell(cells) => {
            println!("    Created cell array with {} elements:", cells.len());
            for (i, cell) in cells.iter().enumerate() {
                let type_name = match cell {
                    MatType::Double(_) => "Double array",
                    MatType::Char(_) => "Character string",
                    MatType::Int32(_) => "Int32 array",
                    MatType::Logical(_) => "Logical array",
                    _ => "Other type",
                };
                println!("      Cell {}: {}", i + 1, type_name);
            }
        }
        _ => unreachable!(),
    }

    // Nested cell arrays
    println!("  🔹 Creating nested cell arrays:");
    let inner_cells = vec![
        MatType::Double(Array1::from(vec![10.0, 20.0]).into_dyn()),
        MatType::Char("Nested".to_string()),
    ];
    let inner_cell_array = create_cell_array(inner_cells);

    let outer_cells = vec![
        cell_array,
        inner_cell_array,
        MatType::Char("Top level".to_string()),
    ];
    let nested_cell_array = create_cell_array(outer_cells);

    match &nested_cell_array {
        MatType::Cell(cells) => {
            println!(
                "    Created nested cell array with {} top-level elements",
                cells.len()
            );
        }
        _ => unreachable!(),
    }

    println!("  ✅ Cell array creation successful!");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_structures() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🏗️  Demonstrating Structures...");

    // Create simple structure
    println!("  🔹 Creating simple structure:");
    let mut simple_struct = HashMap::new();
    simple_struct.insert(
        "name".to_string(),
        MatType::Char("Test Structure".to_string()),
    );
    simple_struct.insert(
        "data".to_string(),
        MatType::Double(Array1::from(vec![1.0, 2.0, 3.0]).into_dyn()),
    );
    simple_struct.insert(
        "flag".to_string(),
        MatType::Logical(Array1::from(vec![true]).into_dyn()),
    );

    let structure = create_struct(simple_struct);

    match &structure {
        MatType::Struct(fields) => {
            println!("    Created structure with {} fields:", fields.len());
            for (field_name, field_value) in fields {
                let type_name = match field_value {
                    MatType::Double(_) => "Double array",
                    MatType::Char(_) => "Character string",
                    MatType::Logical(_) => "Logical array",
                    _ => "Other type",
                };
                println!("      {}: {}", field_name, type_name);
            }
        }
        _ => unreachable!(),
    }

    // Nested structures
    println!("  🔹 Creating nested structure:");
    let mut inner_struct = HashMap::new();
    inner_struct.insert(
        "x".to_string(),
        MatType::Double(Array1::from(vec![1.0, 2.0]).into_dyn()),
    );
    inner_struct.insert(
        "y".to_string(),
        MatType::Double(Array1::from(vec![3.0, 4.0]).into_dyn()),
    );
    let inner_structure = create_struct(inner_struct);

    let mut nested_struct = HashMap::new();
    nested_struct.insert("metadata".to_string(), structure);
    nested_struct.insert("coordinates".to_string(), inner_structure);
    nested_struct.insert(
        "timestamp".to_string(),
        MatType::Char("2024-01-01".to_string()),
    );

    let nested_structure = create_struct(nested_struct);

    match &nested_structure {
        MatType::Struct(fields) => {
            println!(
                "    Created nested structure with {} top-level fields:",
                fields.len()
            );
            for field_name in fields.keys() {
                println!("      {}", field_name);
            }
        }
        _ => unreachable!(),
    }

    println!("  ✅ Structure creation successful!");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_format_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Demonstrating Format Detection...");

    let temp_dir = tempdir()?;

    // Create files with different formats
    let v5_file = temp_dir.path().join("v5_format.mat");
    let auto_file = temp_dir.path().join("auto_format.mat");

    // Create test data
    let mut vars = HashMap::new();
    let test_array = Array2::from_shape_fn((50, 50), |(i, j)| (i + j) as f64).into_dyn();
    vars.insert("test_matrix".to_string(), MatType::Double(test_array));

    // Write with explicit v5 format
    println!("  🔹 Writing with explicit v5 format:");
    let v5_config = MatFileConfig {
        use_v73: false,
        compression: None,
        v73_threshold: 2 * 1024 * 1024 * 1024, // Very high threshold
    };

    let v5_enhanced = EnhancedMatFile::new(v5_config);
    v5_enhanced.write(&v5_file, &vars)?;
    println!("    v5 format file created");

    // Write with automatic format selection
    println!("  🔹 Writing with automatic format selection:");
    let auto_config = MatFileConfig::default();
    let auto_enhanced = EnhancedMatFile::new(auto_config);
    auto_enhanced.write(&auto_file, &vars)?;
    println!("    Auto-selected format file created");

    // Read and detect formats
    println!("  🔹 Reading and detecting formats:");

    let v5_detected = v5_enhanced.is_v73_file(&v5_file)?;
    println!("    v5 file detected as v7.3: {}", v5_detected);

    let auto_detected = auto_enhanced.is_v73_file(&auto_file)?;
    println!("    Auto file detected as v7.3: {}", auto_detected);

    // Verify reading works
    let _v5_loaded = auto_enhanced.read(&v5_file)?;
    let _auto_loaded = auto_enhanced.read(&auto_file)?;
    println!("    Both files read successfully");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_performance_and_errors() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Demonstrating Performance and Error Handling...");

    let temp_dir = tempdir()?;

    // Performance testing with different sizes
    println!("  🔹 Performance testing:");
    let sizes = vec![
        (10, 10, "Small (100 elements)"),
        (100, 100, "Medium (10K elements)"),
        (300, 300, "Large (90K elements)"),
    ];

    for (rows, cols, description) in sizes {
        let test_file = temp_dir.path().join(format!("perf_{}x{}.mat", rows, cols));

        // Create test data
        let mut vars = HashMap::new();
        let large_array = Array2::from_shape_fn((rows, cols), |(i, j)| (i + j) as f64).into_dyn();
        vars.insert(
            "performance_matrix".to_string(),
            MatType::Double(large_array),
        );

        // Write timing
        let write_start = Instant::now();
        write_mat(&test_file, &vars)?;
        let write_time = write_start.elapsed();

        // Read timing
        let read_start = Instant::now();
        let _loaded = read_mat(&test_file)?;
        let read_time = read_start.elapsed();

        // File size
        let file_size = std::fs::metadata(&test_file)?.len();

        println!(
            "    {}: Write {:.2}ms, Read {:.2}ms, Size {:.1}KB",
            description,
            write_time.as_secs_f64() * 1000.0,
            read_time.as_secs_f64() * 1000.0,
            file_size as f64 / 1024.0
        );
    }

    // Error handling demonstration
    println!("  🔹 Error handling:");

    // Try to read non-existent file
    let missing_file = temp_dir.path().join("missing.mat");
    match read_mat(&missing_file) {
        Ok(_) => println!("    Unexpected success reading missing file"),
        Err(e) => println!(
            "    Expected error reading missing file: {}",
            e.to_string().chars().take(50).collect::<String>()
        ),
    }

    // Try to write to invalid path
    let invalid_file = "/invalid/path/test.mat";
    let mut empty_vars = HashMap::new();
    empty_vars.insert("empty".to_string(), MatType::Char("test".to_string()));

    match write_mat(invalid_file, &empty_vars) {
        Ok(_) => println!("    Unexpected success writing to invalid path"),
        Err(e) => println!(
            "    Expected error writing to invalid path: {}",
            e.to_string().chars().take(50).collect::<String>()
        ),
    }

    println!("  ✅ Performance and error handling tests completed!");

    Ok(())
}
