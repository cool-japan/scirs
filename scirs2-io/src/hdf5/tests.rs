//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

use super::*;

#[cfg(test)]
mod legacy_tests {
    use super::*;
    #[test]
    fn test_group_creation() {
        let mut root = Group::new("/".to_string());
        let subgroup = root.create_group("data");
        assert_eq!(subgroup.name, "data");
        assert!(root.get_group("data").is_some());
    }
    #[test]
    fn test_attribute_setting() {
        let mut group = Group::new("test".to_string());
        group.set_attribute("version", AttributeValue::Integer(1));
        group.set_attribute(
            "description",
            AttributeValue::String("Test group".to_string()),
        );
        assert_eq!(group.attributes.len(), 2);
    }
    #[test]
    fn test_dataset_creation() {
        let dataset = Dataset {
            name: "test_data".to_string(),
            dtype: HDF5DataType::Float { size: 8 },
            shape: vec![2, 3],
            data: DataArray::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            attributes: HashMap::new(),
            options: DatasetOptions::default(),
        };
        assert_eq!(dataset.shape, vec![2, 3]);
        if let DataArray::Float(data) = &dataset.data {
            assert_eq!(data.len(), 6);
        }
    }
    #[test]
    fn test_compression_options() {
        let mut options = CompressionOptions::default();
        options.gzip = Some(6);
        options.shuffle = true;
        assert_eq!(options.gzip, Some(6));
        assert!(options.shuffle);
    }
    #[test]
    fn test_hdf5_file_creation() {
        let tmp = std::env::temp_dir();
        let path = tmp.join("scirs2_test_hdf5.h5");
        let file = HDF5File::create(path.to_str().unwrap()).expect("Operation failed");
        assert_eq!(file.mode, FileMode::Create);
        assert_eq!(file.root.name, "/");
        let _ = std::fs::remove_file(&path);
    }
    #[test]
    fn test_f64_dataset_slice_roundtrip() {
        let tmp = std::env::temp_dir();
        let path = tmp.join("scirs2_test_hdf5_slice.h5");
        let mut file = HDF5File::create(path.to_str().unwrap()).expect("create failed");
        let base: Vec<f64> = (0..16).map(|v| v as f64).collect();
        let array = ArrayD::from_shape_vec(IxDyn(&[4, 4]), base).expect("array build failed");
        file.create_dataset_from_array("grid", &array, None)
            .expect("create dataset failed");
        let slice = file
            .read_f64_dataset_slice("grid", &[2, 2], &[1, 1])
            .expect("slice read failed");
        assert_eq!(slice, vec![5.0, 6.0, 9.0, 10.0]);
        file.write_f64_dataset_slice("grid", &[100.0, 101.0, 102.0, 103.0], &[2, 2], &[1, 1])
            .expect("slice write failed");
        let full = file.read_dataset("grid").expect("read back failed");
        let full = full.as_slice().expect("contiguous");
        assert_eq!(full[5], 100.0);
        assert_eq!(full[6], 101.0);
        assert_eq!(full[9], 102.0);
        assert_eq!(full[10], 103.0);
        assert_eq!(full[0], 0.0);
        assert_eq!(full[15], 15.0);
        assert!(file
            .read_f64_dataset_slice("grid", &[2, 2], &[3, 3])
            .is_err());
        let _ = std::fs::remove_file(&path);
    }
    /// Directly test that the HDF5DataType::Array variant round-trips correctly.
    #[test]
    fn test_hdf5_datatype_array_f32_roundtrip() {
        let base = HDF5DataType::Float { size: 4 };
        let array_type = HDF5DataType::Array {
            base_type: Box::new(base),
            shape: vec![8],
        };
        if let HDF5DataType::Array { base_type, shape } = &array_type {
            assert!(matches!(**base_type, HDF5DataType::Float { size: 4 }));
            assert_eq!(shape, &[8]);
        } else {
            panic!("Expected HDF5DataType::Array");
        }
    }
    /// Test Array variant with f64 element type.
    #[test]
    fn test_hdf5_datatype_array_f64_roundtrip() {
        let base = HDF5DataType::Float { size: 8 };
        let array_type = HDF5DataType::Array {
            base_type: Box::new(base),
            shape: vec![16],
        };
        if let HDF5DataType::Array { base_type, shape } = &array_type {
            assert!(matches!(**base_type, HDF5DataType::Float { size: 8 }));
            assert_eq!(shape, &[16]);
        } else {
            panic!("Expected HDF5DataType::Array");
        }
    }
    /// Test nested Array (array of arrays).
    #[test]
    fn test_hdf5_datatype_nested_array() {
        let inner = HDF5DataType::Array {
            base_type: Box::new(HDF5DataType::Integer {
                size: 4,
                signed: true,
            }),
            shape: vec![4],
        };
        let outer = HDF5DataType::Array {
            base_type: Box::new(inner),
            shape: vec![2],
        };
        if let HDF5DataType::Array {
            base_type: outer_base,
            shape: outer_shape,
        } = &outer
        {
            assert_eq!(outer_shape, &[2]);
            if let HDF5DataType::Array {
                base_type: inner_base,
                shape: inner_shape,
            } = outer_base.as_ref()
            {
                assert_eq!(inner_shape, &[4]);
                assert!(matches!(
                    **inner_base,
                    HDF5DataType::Integer {
                        size: 4,
                        signed: true
                    }
                ));
            } else {
                panic!("Expected inner HDF5DataType::Array");
            }
        } else {
            panic!("Expected outer HDF5DataType::Array");
        }
    }
    /// Test that scalar types (Integer, Float) still produce the expected HDF5DataType.
    #[test]
    fn test_hdf5_scalar_types_still_correct() {
        let int_type = HDF5DataType::Integer {
            size: 8,
            signed: true,
        };
        let float_type = HDF5DataType::Float { size: 8 };
        let str_type = HDF5DataType::String {
            encoding: StringEncoding::UTF8,
        };
        assert!(matches!(
            int_type,
            HDF5DataType::Integer {
                size: 8,
                signed: true
            }
        ));
        assert!(matches!(float_type, HDF5DataType::Float { size: 8 }));
        assert!(matches!(
            str_type,
            HDF5DataType::String {
                encoding: StringEncoding::UTF8
            }
        ));
    }
    /// Test that Array type with VarLen semantics (shape=[0]) represents variable-length.
    #[test]
    fn test_hdf5_varlen_array_marker() {
        let base = HDF5DataType::Integer {
            size: 4,
            signed: false,
        };
        let varlen = HDF5DataType::Array {
            base_type: Box::new(base),
            shape: vec![0],
        };
        if let HDF5DataType::Array { shape, .. } = &varlen {
            assert_eq!(shape[0], 0, "VarLen marker must be shape=[0]");
        } else {
            panic!("Expected HDF5DataType::Array");
        }
    }
}
