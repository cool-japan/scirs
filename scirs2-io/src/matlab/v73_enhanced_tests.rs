//! Tests for [`super::v73_enhanced`].
//!
//! Split into its own file to keep `v73_enhanced.rs` inside the 2000-line
//! limit; included via `#[path]` so `use super::*` still reaches the module
//! under test.

use super::*;

#[test]
fn test_v73_features_default() {
    let features = V73Features::default();
    assert!(features.enable_partial_io);
    assert!(features.support_objects);
    assert!(features.support_tables);
}

#[test]
fn test_matlab_table_creation() {
    let mut table = MatlabTable {
        variable_names: vec!["x".to_string(), "y".to_string()],
        row_names: Some(vec!["row1".to_string(), "row2".to_string()]),
        data: HashMap::new(),
        properties: HashMap::new(),
    };

    table.data.insert(
        "x".to_string(),
        MatType::Double(ArrayD::zeros(IxDyn(&[2, 1]))),
    );
    table.data.insert(
        "y".to_string(),
        MatType::Double(ArrayD::ones(IxDyn(&[2, 1]))),
    );

    assert_eq!(table.variable_names.len(), 2);
    assert_eq!(table.data.len(), 2);
}

/// Round-trip a table with 2 columns and 3 rows.
#[test]
fn test_round_trip_table() {
    let path = std::env::temp_dir().join("test_v73_table.h5");
    let handler = V73MatFile::new(V73Features::default());

    let mut data = HashMap::new();
    data.insert(
        "col_a".to_string(),
        MatType::Double(ArrayD::from_elem(IxDyn(&[3, 1]), 1.0_f64)),
    );
    data.insert(
        "col_b".to_string(),
        MatType::Double(ArrayD::from_elem(IxDyn(&[3, 1]), 2.0_f64)),
    );

    let table = MatlabTable {
        variable_names: vec!["col_a".to_string(), "col_b".to_string()],
        row_names: Some(vec!["r1".to_string(), "r2".to_string(), "r3".to_string()]),
        data,
        properties: HashMap::new(),
    };

    let mut vars = HashMap::new();
    vars.insert("tbl".to_string(), ExtendedMatType::Table(table));

    handler
        .write_extended(&path, &vars)
        .expect("write_extended table");

    let read_back = handler.read_extended(&path).expect("read_extended table");
    let ext = read_back.get("tbl").expect("key 'tbl' missing");

    if let ExtendedMatType::Table(t) = ext {
        assert_eq!(t.variable_names.len(), 2);
        assert!(t.variable_names.contains(&"col_a".to_string()));
        assert!(t.variable_names.contains(&"col_b".to_string()));
        assert_eq!(t.row_names.as_ref().map(|r| r.len()), Some(3));
    } else {
        panic!("Expected Table, got something else");
    }

    let _ = std::fs::remove_file(&path);
}

/// Round-trip a 5-element categorical array with 3 categories.
#[test]
fn test_round_trip_categorical() {
    let path = std::env::temp_dir().join("test_v73_categorical.h5");
    let handler = V73MatFile::new(V73Features::default());

    let cat = CategoricalArray {
        categories: vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ],
        data: ArrayD::from_shape_vec(IxDyn(&[5]), vec![0u32, 1, 2, 0, 1]).expect("shape vec"),
        ordered: true,
    };

    let mut vars = HashMap::new();
    vars.insert("cat".to_string(), ExtendedMatType::Categorical(cat));
    handler
        .write_extended(&path, &vars)
        .expect("write categorical");

    let read_back = handler.read_extended(&path).expect("read categorical");
    let ext = read_back.get("cat").expect("key 'cat' missing");

    if let ExtendedMatType::Categorical(c) = ext {
        assert_eq!(c.categories, vec!["apple", "banana", "cherry"]);
        assert_eq!(c.data.len(), 5);
        assert!(c.ordered);
    } else {
        panic!("Expected Categorical");
    }

    let _ = std::fs::remove_file(&path);
}

/// Round-trip a 3-element datetime array.
#[test]
fn test_round_trip_datetime() {
    let path = std::env::temp_dir().join("test_v73_datetime.h5");
    let handler = V73MatFile::new(V73Features::default());

    let dt = DateTimeArray {
        data: ArrayD::from_shape_vec(IxDyn(&[3]), vec![738200.0_f64, 738201.0, 738202.0])
            .expect("shape vec"),
        timezone: Some("UTC".to_string()),
        format: "yyyy-MM-dd".to_string(),
    };

    let mut vars = HashMap::new();
    vars.insert("dt".to_string(), ExtendedMatType::DateTime(dt));
    handler
        .write_extended(&path, &vars)
        .expect("write datetime");

    let read_back = handler.read_extended(&path).expect("read datetime");
    let ext = read_back.get("dt").expect("key 'dt' missing");

    if let ExtendedMatType::DateTime(d) = ext {
        assert_eq!(d.data.len(), 3);
        assert_eq!(d.timezone, Some("UTC".to_string()));
        assert_eq!(d.format, "yyyy-MM-dd");
        assert!((d.data[[0]] - 738200.0).abs() < 1e-6);
    } else {
        panic!("Expected DateTime");
    }

    let _ = std::fs::remove_file(&path);
}

/// Round-trip an array of 4 strings.
#[test]
fn test_round_trip_string_array() {
    let path = std::env::temp_dir().join("test_v73_string_array.h5");
    let handler = V73MatFile::new(V73Features::default());

    let strings = vec![
        "hello".to_string(),
        "world".to_string(),
        "foo".to_string(),
        "bar".to_string(),
    ];

    let mut vars = HashMap::new();
    vars.insert(
        "sa".to_string(),
        ExtendedMatType::StringArray(strings.clone()),
    );
    handler
        .write_extended(&path, &vars)
        .expect("write string array");

    let read_back = handler.read_extended(&path).expect("read string array");
    let ext = read_back.get("sa").expect("key 'sa' missing");

    if let ExtendedMatType::StringArray(sa) = ext {
        assert_eq!(sa.len(), 4);
        assert_eq!(sa[0], "hello");
        assert_eq!(sa[3], "bar");
    } else {
        panic!("Expected StringArray");
    }

    let _ = std::fs::remove_file(&path);
}

/// Round-trip a function handle with a simple workspace variable.
#[test]
fn test_round_trip_function_handle() {
    let path = std::env::temp_dir().join("test_v73_funchandle.h5");
    let handler = V73MatFile::new(V73Features::default());

    let mut ws = HashMap::new();
    ws.insert(
        "x".to_string(),
        MatType::Double(ArrayD::from_elem(IxDyn(&[1]), 42.0_f64)),
    );

    let fh = FunctionHandle {
        function: "@(x) x^2".to_string(),
        function_type: "anonymous".to_string(),
        workspace: Some(ws),
    };

    let mut vars = HashMap::new();
    vars.insert("fh".to_string(), ExtendedMatType::FunctionHandle(fh));
    handler
        .write_extended(&path, &vars)
        .expect("write function handle");

    let read_back = handler.read_extended(&path).expect("read function handle");
    let ext = read_back.get("fh").expect("key 'fh' missing");

    if let ExtendedMatType::FunctionHandle(f) = ext {
        assert_eq!(f.function, "@(x) x^2");
        assert_eq!(f.function_type, "anonymous");
        assert!(f.workspace.is_some());
    } else {
        panic!("Expected FunctionHandle");
    }

    let _ = std::fs::remove_file(&path);
}

/// Round-trip an object with 2 properties.
#[test]
fn test_round_trip_object() {
    let path = std::env::temp_dir().join("test_v73_object.h5");
    let handler = V73MatFile::new(V73Features::default());

    let mut props = HashMap::new();
    props.insert(
        "alpha".to_string(),
        MatType::Double(ArrayD::from_elem(IxDyn(&[1]), std::f64::consts::PI)),
    );
    props.insert(
        "beta".to_string(),
        MatType::Double(ArrayD::from_elem(IxDyn(&[1]), 2.71_f64)),
    );

    let obj = MatlabObject {
        class_name: "MyClass".to_string(),
        properties: props,
        superclass_data: None,
    };

    let mut vars = HashMap::new();
    vars.insert("obj".to_string(), ExtendedMatType::Object(obj));
    handler.write_extended(&path, &vars).expect("write object");

    let read_back = handler.read_extended(&path).expect("read object");
    let ext = read_back.get("obj").expect("key 'obj' missing");

    if let ExtendedMatType::Object(o) = ext {
        assert_eq!(o.class_name, "MyClass");
        assert_eq!(o.properties.len(), 2);
        assert!(o.properties.contains_key("alpha"));
        assert!(o.properties.contains_key("beta"));
    } else {
        panic!("Expected Object");
    }

    let _ = std::fs::remove_file(&path);
}

/// Write a simple double array via `write_standard_type`.
#[test]
fn test_write_standard_type() {
    let path = std::env::temp_dir().join("test_v73_standard.h5");
    let handler = V73MatFile::new(V73Features::default());

    let arr = MatType::Double(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0_f64, 2.0, 3.0]).expect("shape"),
    );
    let mut vars = HashMap::new();
    vars.insert("arr".to_string(), ExtendedMatType::Standard(Box::new(arr)));
    let result = handler.write_extended(&path, &vars);
    assert!(
        result.is_ok(),
        "write_standard_type failed: {:?}",
        result.err()
    );

    let _ = std::fs::remove_file(&path);
}

/// Write a 20×10 f64 array, read a 5×3 slice at offset [2,1], verify values.
#[test]
fn test_partial_io_round_trip() {
    let path = std::env::temp_dir().join("test_v73_partial_io.h5");

    // Build 20×10 array where element [i,j] = (i*10 + j) as f64
    let data: Vec<f64> = (0..200).map(|n| n as f64).collect();
    let full = ArrayD::from_shape_vec(IxDyn(&[20, 10]), data).expect("shape");

    // Write the full array
    {
        use crate::hdf5::HDF5File;
        let mut file = HDF5File::create(&path).expect("create");
        file.create_dataset_from_array("myvar", &full, None)
            .expect("create dataset");
        file.close().expect("close");
    }

    // Read slice at offset [2, 1], count [5, 3]
    let slice = PartialIoSupport::read_array_slice(&path, "myvar", &[2, 1], &[5, 3])
        .expect("read_array_slice");

    assert_eq!(slice.shape(), &[5, 3]);
    // Element [r, c] in slice = (2+r)*10 + (1+c)
    for r in 0..5 {
        for c in 0..3 {
            let expected = ((2 + r) * 10 + (1 + c)) as f64;
            assert!(
                (slice[[r, c]] - expected).abs() < 1e-9,
                "slice[{},{}]: expected {}, got {}",
                r,
                c,
                expected,
                slice[[r, c]]
            );
        }
    }

    // Write a patch at offset [5, 2], values all 999.0
    let patch = ArrayD::from_elem(IxDyn(&[2, 2]), 999.0_f64);
    PartialIoSupport::write_array_slice(&path, "myvar", &patch, &[5, 2])
        .expect("write_array_slice");

    // Read back to verify patch
    let after = PartialIoSupport::read_array_slice(&path, "myvar", &[5, 2], &[2, 2])
        .expect("read after write");
    for r in 0..2 {
        for c in 0..2 {
            assert!(
                (after[[r, c]] - 999.0).abs() < 1e-9,
                "patched element [{},{}] should be 999.0, got {}",
                r,
                c,
                after[[r, c]]
            );
        }
    }

    let _ = std::fs::remove_file(&path);
}

/// The point of the in-place write: everything the caller did not name
/// survives.
///
/// `write_array_slice` used to reach through a raw `libhdf5` handle for this,
/// and the only alternative on offer was `HDF5File::write`, which rebuilds
/// the file from SciRS2's in-memory model. That model does not represent the
/// MATLAB-specific constructs a real `.mat` file carries, so a rebuild would
/// quietly drop them. This test stands in for that: a sibling dataset, a
/// nested group, and attributes at both levels must all come back unchanged,
/// and the file must not have changed length — an in-place overwrite that
/// resized anything would invalidate every address recorded in the file.
#[test]
fn test_write_array_slice_preserves_the_rest_of_the_file() {
    let path = std::env::temp_dir().join(format!(
        "test_v73_inplace_preserve_{}.h5",
        std::process::id()
    ));
    let target = ArrayD::from_shape_vec(IxDyn(&[4, 3]), (0..12).map(f64::from).collect())
        .expect("4x3 target");
    let sibling =
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![10.0, 20.0, 30.0, 40.0, 50.0]).expect("sibling");

    {
        let mut file = HDF5File::create(&path).expect("create");
        file.create_dataset_from_array("target", &target, None)
            .expect("target dataset");
        file.create_group("meta").expect("meta group");
        file.create_dataset_from_array("meta/sibling", &sibling, None)
            .expect("sibling dataset");
        file.set_attribute(
            "meta",
            "MATLAB_class",
            AttributeValue::String("struct".to_string()),
        )
        .expect("group attribute");
        file.set_attribute(
            "/",
            "MATLAB_version",
            AttributeValue::String("7.3".to_string()),
        )
        .expect("root attribute");
        file.close().expect("flush");
    }

    let bytes_before = std::fs::read(&path).expect("read before");
    // Where `target`'s raw data lives. Everything outside this window must come
    // back bit-identical — that is the whole claim the in-place writer makes.
    let extent = oxih5::dataset_data_extent(&path, "target").expect("target is overwritable");

    let patch = ArrayD::from_elem(IxDyn(&[2, 2]), -1.0_f64);
    PartialIoSupport::write_array_slice(&path, "target", &patch, &[1, 1])
        .expect("in-place slice write");

    let bytes_after = std::fs::read(&path).expect("read after");
    assert_eq!(
        bytes_before.len(),
        bytes_after.len(),
        "an in-place overwrite must not change the file's length"
    );

    let start = extent.address as usize;
    let end = start + extent.size as usize;
    assert_eq!(
        bytes_before[..start],
        bytes_after[..start],
        "every byte before the dataset's data area must be untouched"
    );
    assert_eq!(
        bytes_before[end..],
        bytes_after[end..],
        "every byte after the dataset's data area must be untouched"
    );
    assert_ne!(
        bytes_before[start..end],
        bytes_after[start..end],
        "the dataset's own data area must actually have changed"
    );

    let reopened = HDF5File::open(&path, FileMode::ReadOnly).expect("reopen");

    // The sibling dataset is untouched.
    assert_eq!(
        reopened
            .read_dataset("meta/sibling")
            .expect("sibling survives")
            .iter()
            .copied()
            .collect::<Vec<f64>>(),
        vec![10.0, 20.0, 30.0, 40.0, 50.0]
    );

    // Attributes at both levels survive.
    assert!(
        matches!(
            reopened.get_attribute("meta", "MATLAB_class"),
            Ok(Some(AttributeValue::String(class))) if class == "struct"
        ),
        "the group attribute must survive an in-place data overwrite"
    );
    assert!(
        matches!(
            reopened.get_attribute("/", "MATLAB_version"),
            Ok(Some(AttributeValue::String(version))) if version == "7.3"
        ),
        "the root attribute must survive an in-place data overwrite"
    );

    // The patch landed, and only where it was aimed.
    let after = reopened.read_dataset("target").expect("target survives");
    let expected: Vec<f64> = vec![
        0.0, 1.0, 2.0, //
        3.0, -1.0, -1.0, //
        6.0, -1.0, -1.0, //
        9.0, 10.0, 11.0,
    ];
    assert_eq!(after.iter().copied().collect::<Vec<f64>>(), expected);

    let _ = std::fs::remove_file(&path);
}

/// Datasets that cannot be overwritten at a fixed size are refused by name,
/// before anything is read, and the file is left exactly as it was.
#[test]
fn test_write_array_slice_refuses_non_overwritable_datasets() {
    let path =
        std::env::temp_dir().join(format!("test_v73_inplace_refuse_{}.h5", std::process::id()));
    let numbers = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).expect("numbers");

    {
        let mut file = HDF5File::create(&path).expect("create");
        file.create_dataset_from_array("numbers", &numbers, None)
            .expect("numbers dataset");
        file.close().expect("flush");
    }
    let bytes_before = std::fs::read(&path).expect("read before");

    let patch = ArrayD::from_elem(IxDyn(&[1]), 9.0_f64);

    // A dataset that is not in the file at all.
    let missing = PartialIoSupport::write_array_slice(&path, "absent", &patch, &[0]);
    assert!(
        matches!(missing, Err(IoError::UnsupportedFormat(_))),
        "a missing dataset must be reported, got {missing:?}"
    );

    // A rank disagreement between `start` and the patch.
    let bad_rank = PartialIoSupport::write_array_slice(&path, "numbers", &patch, &[0, 0]);
    assert!(
        matches!(bad_rank, Err(IoError::Other(_))),
        "a rank mismatch must be reported, got {bad_rank:?}"
    );

    assert_eq!(
        bytes_before,
        std::fs::read(&path).expect("read after"),
        "a refused write must leave every byte of the file alone"
    );

    let _ = std::fs::remove_file(&path);
}
