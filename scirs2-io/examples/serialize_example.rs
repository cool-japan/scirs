use scirs2_core::ndarray::array;
use scirs2_io::serialize::{
    deserialize_array, deserialize_array_with_metadata, deserialize_sparse_matrix,
    deserialize_struct, serialize_array, serialize_array_with_metadata, serialize_sparse_matrix,
    serialize_struct, SerializationFormat, SparseMatrixCOO,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;

// Define a sample struct for serialization
#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct ExperimentResult {
    name: String,
    parameters: HashMap<String, f64>,
    measurements: Vec<f64>,
    success: bool,
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Data Serialization Example ===\n");

    // 1. Array serialization example
    println!("1. Array serialization example");

    // Create a simple 2D array
    let arr = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

    println!("Original array:\n{:?}", arr);

    // Convert to dynamic dimension for serialization
    let arr_dyn = arr.into_dyn();

    // Serialize to different formats
    println!("Serializing array to binary format...");
    serialize_array(
        "scirs2-io/examples/array.bin",
        &arr_dyn,
        SerializationFormat::Binary,
    )?;

    println!("Serializing array to JSON format...");
    serialize_array(
        "scirs2-io/examples/array.json",
        &arr_dyn,
        SerializationFormat::JSON,
    )?;

    println!("Serializing array to MessagePack format...");
    serialize_array(
        "scirs2-io/examples/array.msgpack",
        &arr_dyn,
        SerializationFormat::MessagePack,
    )?;

    // Deserialize from different formats
    println!("\nDeserializing array from binary format...");
    let arr_bin =
        deserialize_array::<_, f64>("scirs2-io/examples/array.bin", SerializationFormat::Binary)?;
    println!("Binary deserialized array:\n{:?}", arr_bin);

    println!("Deserializing array from JSON format...");
    let arr_json =
        deserialize_array::<_, f64>("scirs2-io/examples/array.json", SerializationFormat::JSON)?;
    println!("JSON deserialized array:\n{:?}", arr_json);

    // Check if deserialized arrays are the same as the original
    assert_eq!(arr_dyn, arr_bin);
    assert_eq!(arr_dyn, arr_json);

    // 2. Array with metadata example
    println!("\n2. Array with metadata example");

    // Create array metadata
    let mut metadata = HashMap::new();
    metadata.insert("description".to_string(), "Test array".to_string());
    metadata.insert("units".to_string(), "meters".to_string());
    metadata.insert("source".to_string(), "experiment_1".to_string());

    println!("Serializing array with metadata...");
    serialize_array_with_metadata(
        "scirs2-io/examples/array_metadata.json",
        &arr_dyn,
        metadata.clone(),
        SerializationFormat::JSON,
    )?;

    // Deserialize array with metadata
    println!("Deserializing array with metadata...");
    let (arr_meta, meta) = deserialize_array_with_metadata::<_, f64>(
        "scirs2-io/examples/array_metadata.json",
        SerializationFormat::JSON,
    )?;

    println!("Deserialized array with metadata:");
    println!("Array shape: {:?}", arr_meta.shape());
    println!("Metadata: {:?}", meta);

    // 3. Struct serialization example
    println!("\n3. Struct serialization example");

    // Create a sample struct
    let mut parameters = HashMap::new();
    parameters.insert("temperature".to_string(), 25.5);
    parameters.insert("pressure".to_string(), 101.3);
    parameters.insert("humidity".to_string(), 45.0);

    let result = ExperimentResult {
        name: "Experiment 1".to_string(),
        parameters,
        measurements: vec![10.5, 11.2, 9.8, 10.1, 10.3],
        success: true,
    };

    println!("Original struct:\n{:?}", result);

    // Serialize struct
    println!("Serializing struct to JSON format...");
    serialize_struct(
        "scirs2-io/examples/struct.json",
        &result,
        SerializationFormat::JSON,
    )?;

    // Deserialize struct
    println!("Deserializing struct from JSON format...");
    let result_deserialized: ExperimentResult =
        deserialize_struct("scirs2-io/examples/struct.json", SerializationFormat::JSON)?;

    println!("Deserialized struct:\n{:?}", result_deserialized);

    // Check if deserialized struct is the same as the original
    assert_eq!(result, result_deserialized);

    // 4. Sparse matrix example
    println!("\n4. Sparse matrix example");

    // Create a sparse matrix
    let mut sparse = SparseMatrixCOO::<f64>::new(1000, 1000);

    // Add some non-zero elements
    sparse.push(0, 0, 1.5);
    sparse.push(10, 10, 2.5);
    sparse.push(100, 100, 3.5);
    sparse.push(500, 500, 4.5);
    sparse.push(999, 999, 5.5);

    println!(
        "Created sparse matrix: {}x{} with {} non-zero elements",
        sparse.rows,
        sparse.cols,
        sparse.nnz()
    );

    // Add metadata
    sparse
        .metadata
        .insert("description".to_string(), "Sparse test matrix".to_string());
    sparse.metadata.insert(
        "density".to_string(),
        format!(
            "{:.6}",
            sparse.nnz() as f64 / (sparse.rows * sparse.cols) as f64
        ),
    );

    // Serialize sparse matrix
    println!("Serializing sparse matrix...");
    serialize_sparse_matrix(
        "scirs2-io/examples/sparse.json",
        &sparse,
        SerializationFormat::JSON,
    )?;

    // Deserialize sparse matrix
    println!("Deserializing sparse matrix...");
    let sparse_deserialized = deserialize_sparse_matrix::<_, f64>(
        "scirs2-io/examples/sparse.json",
        SerializationFormat::JSON,
    )?;

    println!(
        "Deserialized sparse matrix: {}x{} with {} non-zero elements",
        sparse_deserialized.rows,
        sparse_deserialized.cols,
        sparse_deserialized.nnz()
    );
    println!("Metadata: {:?}", sparse_deserialized.metadata);

    println!("\nSerialization example completed successfully!");
    Ok(())
}
