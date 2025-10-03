//! Python Interoperability Demo
//!
//! This example demonstrates the Python interoperability infrastructure for scirs2-ndimage,
//! showing how to prepare data for Python bindings and generate API specifications.

use scirs2_core::ndarray::{array, Array2};
use scirs2_ndimage::{
    error::NdimageResult,
    python_interop::{
        api_spec::{generate_filter_api_specs, generate_python_docs},
        array_conversion::{array_to_py_info, validate_array_compatibility},
        binding_examples::{
            example_gaussian_filter_binding, example_median_filter_binding,
            generate_module_definition,
        },
        setup::{generate_init_py, generate_install_instructions, generate_setup_py},
        PyError,
    },
};

#[allow(dead_code)]
fn main() -> NdimageResult<()> {
    println!("🐍 Python Interoperability Infrastructure Demo");
    println!("============================================");

    println!("\n1. Array Conversion and Validation");
    println!("   📊 Testing array metadata conversion...");

    // Create sample arrays of different types and dimensions
    let f64_array = array![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let f32_array = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];

    // Convert to Python-compatible metadata
    let f64_info = array_to_py_info(&f64_array);
    let f32_info = array_to_py_info(&f32_array);

    println!("   ✅ F64 Array Info:");
    println!("      Shape: {:?}", f64_info.shape);
    println!("      Dtype: {}", f64_info.dtype);
    println!("      Strides: {:?}", f64_info.strides);
    println!("      Contiguous: {}", f64_info.contiguous);

    println!("   ✅ F32 Array Info:");
    println!("      Shape: {:?}", f32_info.shape);
    println!("      Dtype: {}", f32_info.dtype);
    println!("      Contiguous: {}", f32_info.contiguous);

    // Test validation
    println!("\n   🔍 Testing array compatibility validation...");
    match validate_array_compatibility::<f64>(&f64_info) {
        Ok(()) => println!("   ✅ F64 array validation: PASSED"),
        Err(e) => println!("   ❌ F64 array validation: FAILED - {}", e.message),
    }

    match validate_array_compatibility::<f64>(&f32_info) {
        Ok(()) => println!("   ✅ F32->F64 validation: PASSED (unexpected)"),
        Err(e) => println!(
            "   ✅ F32->F64 validation: FAILED as expected - {}",
            e.message
        ),
    }

    println!("\n2. API Specification Generation");
    println!("   📝 Generating Python API specifications...");

    let filter_specs = generate_filter_api_specs();
    println!(
        "   📊 Generated {} filter function specifications",
        filter_specs.len()
    );

    for spec in &filter_specs {
        println!(
            "      • {} - {} parameters",
            spec.name,
            spec.parameters.len()
        );
    }

    println!("\n3. Python Documentation Generation");
    println!("   📚 Generating comprehensive Python documentation...");

    let docs = generate_python_docs();
    let doc_lines = docs.lines().count();
    println!("   ✅ Generated {} lines of documentation", doc_lines);

    // Save documentation to file
    std::fs::create_dir_all("examples/outputs").ok();
    std::fs::write("examples/outputs/python_api_docs.md", &docs)
        .expect("Failed to write documentation");
    println!("   📁 Saved to: examples/outputs/python_api_docs.md");

    println!("\n4. Python Binding Examples");
    println!("   🔧 Generating PyO3 binding examples...");

    let gaussian_binding = example_gaussian_filter_binding();
    let median_binding = example_median_filter_binding();
    let module_def = generate_module_definition();

    println!(
        "   ✅ Generated Gaussian filter binding ({} chars)",
        gaussian_binding.len()
    );
    println!(
        "   ✅ Generated median filter binding ({} chars)",
        median_binding.len()
    );
    println!(
        "   ✅ Generated module definition ({} chars)",
        module_def.len()
    );

    // Save binding examples
    std::fs::write(
        "examples/outputs/gaussian_filter_binding.rs",
        &gaussian_binding,
    )
    .expect("Failed to write gaussian binding");
    std::fs::write("examples/outputs/median_filter_binding.rs", &median_binding)
        .expect("Failed to write median binding");
    std::fs::write("examples/outputs/module_definition.rs", &module_def)
        .expect("Failed to write module definition");

    println!("   📁 Binding examples saved to examples/outputs/");

    println!("\n5. Python Package Setup");
    println!("   📦 Generating Python package setup files...");

    let setup_py = generate_setup_py();
    let init_py = generate_init_py();
    let install_instructions = generate_install_instructions();

    println!("   ✅ Generated setup.py ({} chars)", setup_py.len());
    println!("   ✅ Generated _init__.py ({} chars)", init_py.len());
    println!(
        "   ✅ Generated installation instructions ({} chars)",
        install_instructions.len()
    );

    // Save setup files
    std::fs::write("examples/outputs/setup.py", &setup_py).expect("Failed to write setup.py");
    std::fs::write("examples/outputs/__init__.py", &init_py).expect("Failed to write _init__.py");
    std::fs::write("examples/outputs/INSTALL.md", &install_instructions)
        .expect("Failed to write install instructions");

    println!("   📁 Setup files saved to examples/outputs/");

    println!("\n6. Error Handling Integration");
    println!("   ⚠️  Testing error conversion for Python compatibility...");

    use scirs2_ndimage::error::NdimageError;

    let errors = vec![
        NdimageError::InvalidInput("Invalid input data".to_string()),
        NdimageError::DimensionError("Dimension mismatch".to_string()),
        NdimageError::ComputationError("Computation failed".to_string()),
        NdimageError::MemoryError("Out of memory".to_string()),
    ];

    for error in errors {
        let py_error: PyError = error.into();
        println!(
            "   • {} -> {}: {}",
            py_error.error_type, py_error.error_type, py_error.message
        );
    }

    println!("\n7. Performance Considerations");
    println!("   ⚡ Python interop performance notes...");

    let large_array = Array2::<f64>::zeros((1000, 1000));
    let large_info = array_to_py_info(&large_array);

    println!("   📊 Large array processing:");
    println!(
        "      Elements: {} million",
        large_info.shape.iter().product::<usize>() as f64 / 1_000_000.0
    );
    println!(
        "      Memory: ~{:.1} MB",
        large_info.shape.iter().product::<usize>() * 8 / 1_048_576
    );

    match validate_array_compatibility::<f64>(&large_info) {
        Ok(()) => println!("   ✅ Large array validation: PASSED"),
        Err(e) => println!("   ❌ Large array validation: FAILED - {}", e.message),
    }

    println!("\n8. Integration Summary");
    println!("   📋 Python interoperability infrastructure summary:");
    println!("      ✅ Array metadata conversion and validation");
    println!("      ✅ API specification generation for all major functions");
    println!("      ✅ Comprehensive Python documentation generation");
    println!("      ✅ PyO3 binding examples and templates");
    println!("      ✅ Python package setup and installation files");
    println!("      ✅ Error handling conversion for Python exceptions");
    println!("      ✅ Performance validation for large arrays");

    println!("\n🐍 Python Interop Demo Complete!");
    println!("\n📁 All generated files saved to examples/outputs/");
    println!("   • python_api_docs.md - Complete API documentation");
    println!("   • *.rs files - PyO3 binding examples");
    println!("   • setup.py - Python package setup");
    println!("   • _init__.py - Python package initialization");
    println!("   • INSTALL.md - Installation instructions");

    println!("\n🚀 Next Steps for Full Python Bindings:");
    println!("   1. Add PyO3 dependency to Cargo.toml");
    println!("   2. Implement the binding functions using the generated templates");
    println!("   3. Set up the Python package structure");
    println!("   4. Configure build system for cross-platform distribution");
    println!("   5. Add comprehensive Python tests");

    Ok(())
}
