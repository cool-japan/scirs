use ndarray::{Array1, Array2};
use scirs2_io::matlab::{read_mat, write_mat, MatType};
use std::collections::HashMap;
use tempfile::tempdir;

#[test]
fn test_matlab_write_read_roundtrip() {
    let temp_dir = tempdir().unwrap();
    let mat_file = temp_dir.path().join("test.mat");

    // Create test data
    let mut vars = HashMap::new();

    // Numeric arrays
    let double_array = Array1::from(vec![1.0, 2.0, 3.14159, -5.5]).into_dyn();
    vars.insert("double_data".to_string(), MatType::Double(double_array));

    let single_array = Array1::from(vec![1.0f32, 2.5f32, 3.14f32]).into_dyn();
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

    // Write to MAT file
    write_mat(&mat_file, &vars).expect("Failed to write MAT file");

    // Read back from MAT file
    let loaded_vars = read_mat(&mat_file).expect("Failed to read MAT file");

    // Verify we got the same number of variables
    assert_eq!(loaded_vars.len(), vars.len());

    // Verify data integrity
    for (name, original) in &vars {
        let loaded = loaded_vars
            .get(name)
            .expect(&format!("Variable {} not found", name));

        match (original, loaded) {
            (MatType::Double(orig), MatType::Double(load)) => {
                assert_eq!(orig.shape(), load.shape());
                for (o, l) in orig.iter().zip(load.iter()) {
                    assert!(
                        (o - l).abs() < 1e-10,
                        "Double values don't match: {} vs {}",
                        o,
                        l
                    );
                }
            }
            (MatType::Single(orig), MatType::Single(load)) => {
                assert_eq!(orig.shape(), load.shape());
                for (o, l) in orig.iter().zip(load.iter()) {
                    assert!(
                        (o - l).abs() < 1e-6,
                        "Single values don't match: {} vs {}",
                        o,
                        l
                    );
                }
            }
            (MatType::Int32(orig), MatType::Int32(load)) => {
                assert_eq!(orig.shape(), load.shape());
                for (o, l) in orig.iter().zip(load.iter()) {
                    assert_eq!(o, l, "Int32 values don't match: {} vs {}", o, l);
                }
            }
            (MatType::Logical(orig), MatType::Logical(load)) => {
                assert_eq!(orig.shape(), load.shape());
                for (o, l) in orig.iter().zip(load.iter()) {
                    assert_eq!(o, l, "Logical values don't match: {} vs {}", o, l);
                }
            }
            (MatType::Char(orig), MatType::Char(load)) => {
                assert_eq!(
                    orig, load,
                    "Character values don't match: {} vs {}",
                    orig, load
                );
            }
            _ => panic!("Type mismatch for variable {}", name),
        }
    }
}
