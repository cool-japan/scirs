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

/// Round-trip tests against the real HDF5 byte format via the pure-Rust
/// `oxih5` backend.
///
/// The pre-existing `legacy_tests` above only ever touch the in-memory model,
/// so none of them would notice if `write()` produced nothing, produced a JSON
/// sidecar, or silently dropped every non-f64 dataset. These do.
#[cfg(test)]
mod oxih5_backend_tests {
    use super::*;
    use crate::error::IoError;

    /// HDF5 signature — the first 8 bytes of every conforming file.
    const HDF5_MAGIC: &[u8] = b"\x89HDF\r\n\x1a\n";

    /// A scratch path under the system temp directory, cleared before use.
    fn temp_path(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        let _ = std::fs::remove_file(&path);
        path
    }

    /// A dataset written through `HDF5File` must come back with the same shape,
    /// dtype and values — and the file on disk must actually be HDF5.
    #[test]
    fn test_root_dataset_round_trip_through_real_file() {
        let path = temp_path("scirs2_hdf5_root_round_trip.h5");
        let mut file = HDF5File::create(&path).expect("create");
        let array = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("array");
        file.create_dataset_from_array("values", &array, None)
            .expect("create dataset");
        file.write().expect("write");

        let bytes = std::fs::read(&path).expect("read back");
        assert_eq!(
            bytes.get(..8),
            Some(HDF5_MAGIC),
            "write() must emit a real HDF5 file"
        );

        let reopened = HDF5File::open(&path, FileMode::ReadOnly).expect("open");
        let dataset = reopened.get_dataset("values").expect("dataset");
        assert_eq!(dataset.shape, vec![2, 3]);
        assert_eq!(dataset.dtype, HDF5DataType::Float { size: 8 });
        assert_eq!(
            dataset.as_float_vec().expect("floats"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Groups nest to arbitrary depth and carry their own attributes.
    #[test]
    fn test_nested_groups_and_attributes_round_trip() {
        let path = temp_path("scirs2_hdf5_nested_round_trip.h5");
        let mut file = HDF5File::create(&path).expect("create");
        file.root_mut()
            .set_attribute("file_version", AttributeValue::String("1.0".to_string()));
        {
            let experiment = file.root_mut().create_group("experiment");
            experiment.set_attribute("experiment_id", AttributeValue::Integer(12345));
            experiment.set_attribute("temperature", AttributeValue::Float(25.5));
            let measurements = experiment.create_group("measurements");
            measurements
                .set_attribute("sensor_type", AttributeValue::String("thermal".to_string()));
        }
        let temps = ArrayD::from_shape_vec(IxDyn(&[3]), vec![25.1, 25.3, 25.2]).expect("array");
        file.create_dataset_from_array("experiment/measurements/temperature", &temps, None)
            .expect("create dataset");
        file.write().expect("write");

        let reopened = HDF5File::open(&path, FileMode::ReadOnly).expect("open");
        let root = reopened.root();
        assert!(
            matches!(root.get_attribute("file_version"), Some(AttributeValue::String(v)) if v == "1.0"),
            "root attribute lost: {:?}",
            root.get_attribute("file_version")
        );

        let experiment = root.get_group("experiment").expect("experiment group");
        assert!(
            matches!(
                experiment.get_attribute("experiment_id"),
                Some(AttributeValue::Integer(12345))
            ),
            "group attribute lost: {:?}",
            experiment.get_attribute("experiment_id")
        );
        assert!(
            matches!(experiment.get_attribute("temperature"), Some(AttributeValue::Float(v)) if (*v - 25.5).abs() < 1e-12)
        );

        let measurements = experiment
            .get_group("measurements")
            .expect("second-level group");
        assert!(
            matches!(measurements.get_attribute("sensor_type"), Some(AttributeValue::String(v)) if v == "thermal")
        );
        let dataset = measurements.get_dataset("temperature").expect("dataset");
        assert_eq!(
            dataset.as_float_vec().expect("floats"),
            vec![25.1, 25.3, 25.2]
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The migration's central regression: oxih5's accessors each match one
    /// exact datatype, so a literal port would have narrowed SciRS2 to reading
    /// f64 datasets only. Every existing fixture is f64, so nothing else here
    /// would catch it.
    #[test]
    fn test_non_f64_dataset_is_widened_on_read() {
        let path = temp_path("scirs2_hdf5_widening.h5");
        oxih5::FileWriter::new()
            .write_dataset_i32("counts", &[10, 20, 30], &[3])
            .expect("seed i32 dataset")
            .build(&path)
            .expect("build");

        let file = HDF5File::open(&path, FileMode::ReadOnly).expect("open");
        let dataset = file.get_dataset("counts").expect("dataset");
        assert_eq!(
            dataset.dtype,
            HDF5DataType::Integer {
                size: 4,
                signed: true
            },
            "dtype must report the on-disk width"
        );
        assert_eq!(dataset.as_integer_vec().expect("ints"), vec![10, 20, 30]);

        let widened: Vec<f64> = file
            .read_dataset("counts")
            .expect("an i32 dataset must still be readable as f64")
            .iter()
            .copied()
            .collect();
        assert_eq!(widened, vec![10.0, 20.0, 30.0]);
        let _ = std::fs::remove_file(&path);
    }

    /// `FileWriter::build` rewrites its target from scratch, so a write through
    /// a read-only handle would destroy the source. It must be refused, and
    /// `close()` must not perform one.
    #[test]
    fn test_write_through_readonly_handle_is_refused() {
        let path = temp_path("scirs2_hdf5_readonly_guard.h5");
        oxih5::FileWriter::new()
            .write_dataset_f64("d", &[1.0, 2.0], &[2])
            .expect("seed dataset")
            .build(&path)
            .expect("build");
        let before = std::fs::read(&path).expect("read seed");

        let file = HDF5File::open(&path, FileMode::ReadOnly).expect("open");
        assert!(
            file.write().is_err(),
            "writing through a read-only handle must fail"
        );
        file.close()
            .expect("closing a read-only handle must succeed");

        let after = std::fs::read(&path).expect("read after close");
        assert_eq!(
            before, after,
            "a read-only open/close cycle must leave the file byte-identical"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Variable-length string datasets survive a round trip.
    #[test]
    fn test_string_dataset_round_trip() {
        let path = temp_path("scirs2_hdf5_strings.h5");
        let mut file = HDF5File::create(&path).expect("create");
        file.root_mut().datasets.insert(
            "labels".to_string(),
            Dataset::new(
                "labels".to_string(),
                HDF5DataType::String {
                    encoding: StringEncoding::UTF8,
                },
                vec![3],
                DataArray::String(vec![
                    "alpha".to_string(),
                    "beta".to_string(),
                    "gamma".to_string(),
                ]),
                DatasetOptions::default(),
            ),
        );
        file.write().expect("write");

        let reopened = HDF5File::open(&path, FileMode::ReadOnly).expect("open");
        let dataset = reopened.get_dataset("labels").expect("dataset");
        assert_eq!(
            dataset.as_string_vec().expect("strings"),
            vec!["alpha", "beta", "gamma"]
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Scalar and array attributes both round-trip, and an attribute's
    /// object-header padding is not decoded as extra elements.
    #[test]
    fn test_dataset_attributes_round_trip() {
        let path = temp_path("scirs2_hdf5_attrs.h5");
        let mut file = HDF5File::create(&path).expect("create");
        let array = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).expect("array");
        file.create_dataset_from_array("series", &array, None)
            .expect("create dataset");
        if let Some(dataset) = file.root_mut().get_dataset_mut("series") {
            dataset.set_attribute("units", AttributeValue::String("celsius".to_string()));
            dataset.set_attribute("count", AttributeValue::Integer(2));
            dataset.set_attribute("scale", AttributeValue::Float(0.5));
            dataset.set_attribute("bounds", AttributeValue::FloatArray(vec![-1.5, 2.5]));
            dataset.set_attribute("ids", AttributeValue::IntegerArray(vec![7, 8, 9]));
        }
        file.write().expect("write");

        let reopened = HDF5File::open(&path, FileMode::ReadOnly).expect("open");
        let dataset = reopened.get_dataset("series").expect("dataset");
        assert!(
            matches!(dataset.get_attribute("units"), Some(AttributeValue::String(v)) if v == "celsius")
        );
        assert!(matches!(
            dataset.get_attribute("count"),
            Some(AttributeValue::Integer(2))
        ));
        assert!(
            matches!(dataset.get_attribute("scale"), Some(AttributeValue::Float(v)) if (*v - 0.5).abs() < 1e-12)
        );
        match dataset.get_attribute("bounds") {
            Some(AttributeValue::FloatArray(v)) => assert_eq!(v, &vec![-1.5, 2.5]),
            other => panic!("expected FloatArray for 'bounds', got {other:?}"),
        }
        match dataset.get_attribute("ids") {
            Some(AttributeValue::IntegerArray(v)) => assert_eq!(v, &vec![7, 8, 9]),
            other => panic!("expected IntegerArray for 'ids', got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    /// The one construct the writer still cannot express must be named in the
    /// error, never dropped.
    #[test]
    fn test_multidimensional_string_dataset_is_reported() {
        let path = temp_path("scirs2_hdf5_2d_strings.h5");
        let mut file = HDF5File::create(&path).expect("create");
        file.root_mut().datasets.insert(
            "grid".to_string(),
            Dataset::new(
                "grid".to_string(),
                HDF5DataType::String {
                    encoding: StringEncoding::UTF8,
                },
                vec![2, 2],
                DataArray::String(vec![
                    "a".to_string(),
                    "b".to_string(),
                    "c".to_string(),
                    "d".to_string(),
                ]),
                DatasetOptions::default(),
            ),
        );
        let err = file
            .write()
            .expect_err("2-D string datasets are not writable yet");
        assert!(
            matches!(err, IoError::UnsupportedFormat(_)),
            "expected UnsupportedFormat, got {err:?}"
        );
        assert!(
            format!("{err}").contains("2-D string dataset"),
            "the error must name the construct: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// `close()` must flush. The previous implementation discarded the write
    /// result, so a failed flush was indistinguishable from a clean close, and
    /// the no-feature build never wrote HDF5 at all.
    #[test]
    fn test_close_flushes_to_disk() {
        let path = temp_path("scirs2_hdf5_close_flush.h5");
        let mut file = HDF5File::create(&path).expect("create");
        let array = ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.5, 4.5]).expect("array");
        file.create_dataset_from_array("d", &array, None)
            .expect("create dataset");
        file.close().expect("close must flush");

        assert!(path.exists(), "close() must have written the file");
        let reopened = HDF5File::open(&path, FileMode::ReadOnly).expect("open");
        assert_eq!(
            reopened
                .get_dataset("d")
                .expect("dataset")
                .as_float_vec()
                .expect("floats"),
            vec![3.5, 4.5]
        );
        // No JSON sidecar: the old no-feature fallback wrote `{path}.json`
        // instead of HDF5, which silently produced unreadable output.
        let sidecar = std::path::PathBuf::from(format!("{}.json", path.display()));
        assert!(!sidecar.exists(), "no JSON sidecar may be produced");
        let _ = std::fs::remove_file(&path);
    }

    /// The free functions `scirs2-datasets::formats` calls must round-trip,
    /// including the group-per-path form `write_hdf5` accepts.
    #[test]
    fn test_write_hdf5_read_hdf5_round_trip() {
        let path = temp_path("scirs2_hdf5_free_functions.h5");
        let mut datasets = HashMap::new();
        datasets.insert(
            "mydata".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).expect("array"),
        );
        datasets.insert(
            "data/temperature".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![9.0, 8.0, 7.0]).expect("array"),
        );
        write_hdf5(&path, datasets).expect("write_hdf5");

        let root = read_hdf5(&path).expect("read_hdf5");
        let flat = root.get_dataset("mydata").expect("root dataset");
        assert_eq!(flat.shape, vec![2, 2]);
        assert_eq!(
            flat.as_float_vec().expect("floats"),
            vec![1.0, 2.0, 3.0, 4.0]
        );

        let nested = root
            .get_group("data")
            .and_then(|g| g.get_dataset("temperature"))
            .expect("grouped dataset");
        assert_eq!(nested.as_float_vec().expect("floats"), vec![9.0, 8.0, 7.0]);
        let _ = std::fs::remove_file(&path);
    }

    /// `create_dataset_from_array` must convert exactly. The previous
    /// implementation round-tripped each element through `format!("{:?}")` and
    /// `parse::<f64>()`, yielding 0.0 for anything that did not parse.
    #[test]
    fn test_create_dataset_from_array_converts_exactly() {
        let mut file = HDF5File::create(temp_path("scirs2_hdf5_exact_convert.h5")).expect("create");
        let array = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-7i32, 0, 42]).expect("array");
        file.create_dataset_from_array("ints", &array, None)
            .expect("create dataset");
        let values = file
            .root()
            .get_dataset("ints")
            .and_then(Dataset::as_float_vec)
            .expect("floats");
        assert_eq!(values, vec![-7.0, 0.0, 42.0]);
    }
}
