//! HDF5 binary-format conformance fixtures.
//!
//! Each test hand-assembles a minimal HDF5 byte stream and feeds it to
//! [`oxih5::File::open_from_bytes`], which parses from memory and so needs no
//! temporary files. The fixtures cover the structures a real file is built from
//! — superblock v0 and v2, version-1 and version-2 object headers, link
//! messages, dataspace/datatype/layout messages and a contiguous data block —
//! at a granularity that reading a checked-in `.h5` file cannot: a wrong byte
//! here names the exact field that is wrong.
//!
//! These fixtures were inherited from the `hdf5_lite` reader that `oxih5`
//! replaced, and are kept because nothing else in the workspace exercises the
//! format at this level.
//!
//! # A note on `open_from_bytes`
//!
//! `open_from_bytes` is lazy: it stores the buffer and returns `Ok` without
//! looking at it. Malformed input is therefore reported by the first accessor
//! that parses — `info()`, `root()` or `dataset()` — not by the open itself, so
//! the rejection tests below assert on those.

use oxih5::File;

const MSG_LINK: u16 = 0x0006;
const MSG_DATASPACE: u16 = 0x0001;
const MSG_DATATYPE: u16 = 0x0003;
const MSG_DATA_LAYOUT: u16 = 0x0008;

/// A minimal, valid HDF5 file: version-0 superblock plus an empty version-1
/// object header for the root group.
fn build_minimal_hdf5_v0() -> Vec<u8> {
    let mut buf = Vec::new();

    // HDF5 signature (8 bytes)
    buf.extend_from_slice(&[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]);

    buf.push(0); // superblock version
    buf.push(0); // free-space storage version
    buf.push(0); // root group symbol table entry version
    buf.push(0); // reserved
    buf.push(0); // shared header message format version
    buf.push(8); // size of offsets
    buf.push(8); // size of lengths
    buf.push(0); // reserved

    buf.extend_from_slice(&4u16.to_le_bytes()); // group leaf node K
    buf.extend_from_slice(&16u16.to_le_bytes()); // group internal node K
    buf.extend_from_slice(&0u32.to_le_bytes()); // file consistency flags

    buf.extend_from_slice(&0u64.to_le_bytes()); // base address
    buf.extend_from_slice(&u64::MAX.to_le_bytes()); // free-space info (undefined)
    buf.extend_from_slice(&512u64.to_le_bytes()); // end-of-file address
    buf.extend_from_slice(&u64::MAX.to_le_bytes()); // driver info block (undefined)

    // Root group symbol table entry.
    buf.extend_from_slice(&0u64.to_le_bytes()); // link name offset
    let oh_address = 128u64;
    buf.extend_from_slice(&oh_address.to_le_bytes()); // object header address
    buf.extend_from_slice(&0u32.to_le_bytes()); // cache type
    buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
    buf.extend_from_slice(&[0u8; 16]); // scratch pad

    while buf.len() < oh_address as usize {
        buf.push(0);
    }

    // Version-1 object header at `oh_address`, carrying no messages.
    buf.push(1); // version
    buf.push(0); // reserved
    buf.extend_from_slice(&0u16.to_le_bytes()); // number of messages
    buf.extend_from_slice(&1u32.to_le_bytes()); // object reference count
    buf.extend_from_slice(&0u32.to_le_bytes()); // header data size

    while buf.len() < 512 {
        buf.push(0);
    }

    buf
}

/// A minimal, valid HDF5 file: version-2 superblock plus an empty version-2
/// ("OHDR") object header for the root group.
fn build_minimal_hdf5_v2() -> Vec<u8> {
    let mut buf = Vec::new();

    buf.extend_from_slice(&[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]);

    buf.push(2); // superblock version
    buf.push(8); // size of offsets
    buf.push(8); // size of lengths
    buf.push(0); // file consistency flags

    buf.extend_from_slice(&0u64.to_le_bytes()); // base address
    buf.extend_from_slice(&u64::MAX.to_le_bytes()); // superblock extension (undefined)
    buf.extend_from_slice(&512u64.to_le_bytes()); // end-of-file address
    let oh_address = 64u64;
    buf.extend_from_slice(&oh_address.to_le_bytes()); // root group object header
    buf.extend_from_slice(&0u32.to_le_bytes()); // superblock checksum

    while buf.len() < oh_address as usize {
        buf.push(0);
    }

    buf.extend_from_slice(b"OHDR");
    buf.push(2); // version
    buf.push(0); // flags: chunk#0 size is 1 byte
    buf.push(0); // chunk#0 size = 0, i.e. no messages

    while buf.len() < 512 {
        buf.push(0);
    }

    buf
}

/// A version-2 superblock file whose root group holds one dataset, `/data`,
/// reached through a Link message and stored as three contiguous `f64` values.
///
/// # Dataspace header width
///
/// The version-2 Dataspace message header is **four** bytes — version,
/// dimensionality, flags, type — before the dimension sizes begin. The `type`
/// byte (1 = simple) is mandatory; see the HDF5 File Format Specification,
/// "The Dataspace Message".
///
/// The `hdf5_lite` reader this fixture came from parsed that header as three
/// bytes, having mistaken the `flags` byte for `type`, and the fixture was
/// originally built to match: it emitted an 11-byte body where the spec requires
/// 12. oxih5 rejects that with
/// `dataspace body too short for 1 dims: have 11, need 12`, correctly — the
/// fixture was encoding the old reader's bug, not a property of HDF5. The
/// missing byte is restored below.
fn build_hdf5_v2_with_dataset() -> Vec<u8> {
    build_v2_dataset_file(true)
}

/// Datatype message for a little-endian IEEE-754 `f64` (class 1, version 1).
fn f64_datatype_msg() -> Vec<u8> {
    let mut dt = vec![
        0x11, // (version 1 << 4) | class 1 (floating point)
        0x20, // class bit field byte 0: little-endian, no padding
        0x00, // class bit field byte 1
        0x00, // class bit field byte 2
    ];
    dt.extend_from_slice(&8u32.to_le_bytes()); // element size
    dt.extend_from_slice(&0u16.to_le_bytes()); // bit offset
    dt.extend_from_slice(&64u16.to_le_bytes()); // bit precision
    dt.extend_from_slice(&[
        52, // exponent location
        11, // exponent size
        0,  // mantissa location
        52, // mantissa size
    ]);
    dt.extend_from_slice(&1023u32.to_le_bytes()); // exponent bias
    dt
}

/// Datatype message for a little-endian signed `i32` (class 0, version 1).
///
/// Bit 3 of the class bit field marks the value signed; the version-1 properties
/// section is bit offset then bit precision, both `u16`.
fn i32_datatype_msg() -> Vec<u8> {
    let mut dt = vec![
        0x10, // (version 1 << 4) | class 0 (fixed point)
        0x08, // bit field: little-endian, signed
        0, 0, // bit field, remaining two bytes
    ];
    dt.extend_from_slice(&4u32.to_le_bytes()); // element size
    dt.extend_from_slice(&0u16.to_le_bytes()); // bit offset
    dt.extend_from_slice(&32u16.to_le_bytes()); // bit precision
    dt
}

/// Body of [`build_hdf5_v2_with_dataset`], carrying three `f64` values.
///
/// `spec_conformant` selects the four-byte version-2 dataspace header the HDF5
/// specification requires; `false` reproduces the three-byte form the
/// `hdf5_lite` reader accepted, used by
/// [`test_v2_dataspace_missing_type_byte_is_rejected`].
fn build_v2_dataset_file(spec_conformant: bool) -> Vec<u8> {
    let payload: Vec<u8> = [1.0f64, 2.0, 3.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    build_v2_dataset_file_with(&f64_datatype_msg(), &payload, 3, spec_conformant)
}

/// The same file shape as [`build_v2_dataset_file`] with a caller-chosen element
/// type and payload, so the reader can be exercised on more than one datatype.
///
/// `n_elems` is the single dataspace dimension; `payload` must hold exactly that
/// many elements of the type `dt_msg` describes.
fn build_v2_dataset_file_with(
    dt_msg: &[u8],
    payload: &[u8],
    n_elems: u64,
    spec_conformant: bool,
) -> Vec<u8> {
    let mut buf = Vec::new();

    buf.extend_from_slice(&[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]);

    buf.push(2); // superblock version
    buf.push(8); // size of offsets
    buf.push(8); // size of lengths
    buf.push(0); // file consistency flags

    buf.extend_from_slice(&0u64.to_le_bytes()); // base address
    buf.extend_from_slice(&u64::MAX.to_le_bytes()); // superblock extension (undefined)
    buf.extend_from_slice(&1024u64.to_le_bytes()); // end-of-file address
    buf.extend_from_slice(&64u64.to_le_bytes()); // root group object header
    buf.extend_from_slice(&0u32.to_le_bytes()); // superblock checksum

    while buf.len() < 64 {
        buf.push(0);
    }

    // Root group object header: one Link message naming "data".
    let dataset_oh_addr = 256u64;
    let mut link_data = Vec::new();
    link_data.push(1); // link message version
    link_data.push(0); // flags: 1-byte name length, hard link
    link_data.push(4); // name length
    link_data.extend_from_slice(b"data");
    link_data.extend_from_slice(&dataset_oh_addr.to_le_bytes());

    buf.extend_from_slice(b"OHDR");
    buf.push(2); // version
    buf.push(0); // flags: chunk#0 size is 1 byte

    let msg_size = link_data.len() as u16;
    let chunk_size = 1 + 2 + 1 + link_data.len(); // type + size + flags + body
    buf.push(chunk_size as u8);
    buf.push(MSG_LINK as u8);
    buf.extend_from_slice(&msg_size.to_le_bytes());
    buf.push(0); // message flags
    buf.extend_from_slice(&link_data);

    while buf.len() < dataset_oh_addr as usize {
        buf.push(0);
    }

    let data_addr = 512u64;
    let data_size = payload.len() as u64;

    // Dataspace message: version 2, rank 1, simple, dim[0] = n_elems.
    let mut ds_msg = Vec::new();
    ds_msg.push(2); // version
    ds_msg.push(1); // dimensionality
    ds_msg.push(0); // flags: no max dims
    if spec_conformant {
        ds_msg.push(1); // type: simple
    }
    ds_msg.extend_from_slice(&n_elems.to_le_bytes());

    // Data layout message: version 3, class 1 (contiguous).
    let mut layout_msg = Vec::new();
    layout_msg.push(3); // version
    layout_msg.push(1); // class: contiguous
    layout_msg.extend_from_slice(&data_addr.to_le_bytes());
    layout_msg.extend_from_slice(&data_size.to_le_bytes());

    buf.extend_from_slice(b"OHDR");
    buf.push(2); // version
    buf.push(0); // flags

    let total_msg_size = (1 + 2 + 1) * 3 + ds_msg.len() + dt_msg.len() + layout_msg.len();
    buf.push(total_msg_size as u8);

    buf.push(MSG_DATASPACE as u8);
    buf.extend_from_slice(&(ds_msg.len() as u16).to_le_bytes());
    buf.push(0);
    buf.extend_from_slice(&ds_msg);

    buf.push(MSG_DATATYPE as u8);
    buf.extend_from_slice(&(dt_msg.len() as u16).to_le_bytes());
    buf.push(0);
    buf.extend_from_slice(dt_msg);

    buf.push(MSG_DATA_LAYOUT as u8);
    buf.extend_from_slice(&(layout_msg.len() as u16).to_le_bytes());
    buf.push(0);
    buf.extend_from_slice(&layout_msg);

    while buf.len() < data_addr as usize {
        buf.push(0);
    }

    buf.extend_from_slice(payload);

    while buf.len() < 1024 {
        buf.push(0);
    }

    buf
}

// =====================================================================
// Rejection of malformed input
// =====================================================================

#[test]
fn test_invalid_signature() {
    let data = vec![0u8; 64];
    let file = File::open_from_bytes(&data).expect("open_from_bytes is lazy");
    // `expect_err` is unavailable here: it needs `FileInfo: Debug`, which oxih5
    // does not implement. The same applies to the other rejection tests below.
    let Err(err) = file.info() else {
        panic!("all-zero bytes are not an HDF5 file");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("signature"),
        "expected a signature error, got: {msg}"
    );
}

#[test]
fn test_too_short_file() {
    let data = vec![0x89, 0x48, 0x44, 0x46]; // truncated signature
    let file = File::open_from_bytes(&data).expect("open_from_bytes is lazy");
    assert!(
        file.info().is_err(),
        "a 4-byte file cannot carry a superblock"
    );
}

#[test]
fn test_unsupported_superblock_version() {
    let mut data = vec![0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
    data.push(99); // no such superblock version
    data.extend_from_slice(&[0u8; 128]);
    let file = File::open_from_bytes(&data).expect("open_from_bytes is lazy");
    let Err(err) = file.info() else {
        panic!("superblock version 99 is not defined");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("99"),
        "the error should name the offending version, got: {msg}"
    );
}

#[test]
fn test_file_not_found() {
    let missing = std::env::temp_dir().join("scirs2_io_no_such_file_hdf5_conformance.h5");
    assert!(
        File::open(&missing).is_err(),
        "opening a nonexistent path must fail"
    );
}

// =====================================================================
// Superblock version 0
// =====================================================================

#[test]
fn test_parse_v0_superblock() {
    let data = build_minimal_hdf5_v0();
    let file = File::open_from_bytes(&data).expect("valid file");
    let info = file.info().expect("v0 superblock parses");
    assert_eq!(info.superblock_version, 0);
    assert_eq!(info.offset_size, 8);
    assert_eq!(info.length_size, 8);
}

#[test]
fn test_v0_root_group_empty() {
    let data = build_minimal_hdf5_v0();
    let file = File::open_from_bytes(&data).expect("valid file");
    let root = file.root().expect("root group opens");
    assert!(root.datasets().expect("dataset listing").is_empty());
    assert!(root.groups().expect("group listing").is_empty());
}

#[test]
fn test_v0_root_group_has_no_attributes() {
    let data = build_minimal_hdf5_v0();
    let file = File::open_from_bytes(&data).expect("valid file");
    let root = file.root().expect("root group opens");
    assert!(root.attr_views().expect("attribute listing").is_empty());
}

#[test]
fn test_walk_on_empty_file_yields_nothing() {
    let data = build_minimal_hdf5_v0();
    let file = File::open_from_bytes(&data).expect("valid file");
    let mut seen = Vec::new();
    file.walk(&mut |path, is_group| seen.push((path.to_string(), is_group)))
        .expect("walk succeeds");
    assert!(
        seen.is_empty(),
        "empty file should list nothing, got {seen:?}"
    );
}

#[test]
fn test_read_dataset_at_root_path_error() {
    let data = build_minimal_hdf5_v0();
    let file = File::open_from_bytes(&data).expect("valid file");
    assert!(
        file.dataset("/").is_err(),
        "the root group is not a dataset"
    );
}

#[test]
fn test_read_nonexistent_dataset() {
    let data = build_minimal_hdf5_v0();
    let file = File::open_from_bytes(&data).expect("valid file");
    assert!(file.dataset("/nonexistent").is_err());
}

// =====================================================================
// Superblock version 2
// =====================================================================

#[test]
fn test_parse_v2_superblock() {
    let data = build_minimal_hdf5_v2();
    let file = File::open_from_bytes(&data).expect("valid file");
    let info = file.info().expect("v2 superblock parses");
    assert_eq!(info.superblock_version, 2);
    assert_eq!(info.offset_size, 8);
    assert_eq!(info.length_size, 8);
}

#[test]
fn test_v2_root_group_empty() {
    let data = build_minimal_hdf5_v2();
    let file = File::open_from_bytes(&data).expect("valid file");
    let root = file.root().expect("root group opens");
    assert!(root.datasets().expect("dataset listing").is_empty());
    assert!(root.groups().expect("group listing").is_empty());
}

// =====================================================================
// Superblock version 2 with a contiguous dataset
// =====================================================================

#[test]
fn test_v2_with_dataset_root_group() {
    let data = build_hdf5_v2_with_dataset();
    let file = File::open_from_bytes(&data).expect("valid file");
    let root = file.root().expect("root group opens");
    assert_eq!(root.datasets().expect("dataset listing"), vec!["data"]);
}

#[test]
fn test_v2_with_dataset_walk() {
    let data = build_hdf5_v2_with_dataset();
    let file = File::open_from_bytes(&data).expect("valid file");
    let mut seen = Vec::new();
    file.walk(&mut |path, is_group| seen.push((path.to_string(), is_group)))
        .expect("walk succeeds");
    assert_eq!(seen, vec![("/data".to_string(), false)]);
}

#[test]
fn test_v2_read_dataset() {
    let data = build_hdf5_v2_with_dataset();
    let file = File::open_from_bytes(&data).expect("valid file");
    let dataset = file.dataset("/data").expect("dataset reads");
    assert_eq!(dataset.shape, vec![3]);
    assert_eq!(
        dataset.dtype,
        oxih5::Dtype::Float {
            size: 8,
            order: oxih5::ByteOrder::Little,
        }
    );
    let values = dataset.as_f64().expect("f64 payload decodes");
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

/// The version-2 Dataspace message header is four bytes wide. Dropping the
/// `type` byte — the shape the `hdf5_lite` reader accepted — must be reported,
/// not silently misparsed into a plausible-looking dimension list.
#[test]
fn test_v2_dataspace_missing_type_byte_is_rejected() {
    let data = build_v2_dataset_file(false);
    let file = File::open_from_bytes(&data).expect("open_from_bytes is lazy");
    let Err(err) = file.dataset("/data") else {
        panic!("an 11-byte version-2 dataspace body is malformed");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("dataspace"),
        "the error should name the dataspace message, got: {msg}"
    );
    // The rest of the file is well-formed, so the group listing still works —
    // the defect is confined to the one message.
    let root = file.root().expect("root group still opens");
    assert_eq!(root.datasets().expect("dataset listing"), vec!["data"]);
}

// =====================================================================
// End-to-end: universal_reader's HDF5 path
// =====================================================================

/// Write `bytes` to a uniquely named file under the system temp directory.
fn write_temp_h5(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "scirs2_io_hdf5_conformance_{tag}_{}.h5",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write temp HDF5 file");
    path
}

#[test]
fn test_universal_reader_reads_f64_dataset() {
    let path = write_temp_h5("f64", &build_hdf5_v2_with_dataset());
    let table = scirs2_io::universal_reader::read_data(&path, None).expect("universal read");
    let _ = std::fs::remove_file(&path);

    assert_eq!(table.metadata("superblock_version"), Some("2"));
    match table.column("/data").expect("column '/data' present") {
        scirs2_io::universal_reader::DataColumn::Float64(v) => {
            assert_eq!(v, &vec![1.0, 2.0, 3.0]);
        }
        other => panic!("expected Float64, got {}", other.type_name()),
    }
}

/// The regression the widening layer exists to prevent: oxih5's `as_f64` matches
/// only `Float { size: 8 }`, so a reader that called it directly would drop this
/// integer dataset entirely. Every pre-existing fixture is f64, so nothing else
/// would have caught it.
#[test]
fn test_universal_reader_widens_i32_dataset() {
    let payload: Vec<u8> = [-7i32, 0, 42]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let bytes = build_v2_dataset_file_with(&i32_datatype_msg(), &payload, 3, true);
    let path = write_temp_h5("i32", &bytes);
    let table = scirs2_io::universal_reader::read_data(&path, None).expect("universal read");
    let _ = std::fs::remove_file(&path);

    // A 4-byte signed integer keeps the `Int32` column width the `hdf5_lite`
    // reader produced for it.
    match table.column("/data").expect("column '/data' present") {
        scirs2_io::universal_reader::DataColumn::Int32(v) => {
            assert_eq!(v, &vec![-7, 0, 42]);
        }
        other => panic!("expected Int32, got {}", other.type_name()),
    }
}

/// `ReadOptions::hdf5_dataset` selects a single dataset and keys the column by
/// its leaf name rather than its full path.
#[test]
fn test_universal_reader_selects_single_dataset() {
    let path = write_temp_h5("single", &build_hdf5_v2_with_dataset());
    let opts = scirs2_io::universal_reader::ReadOptions {
        hdf5_dataset: Some("/data".to_string()),
        ..Default::default()
    };
    let table = scirs2_io::universal_reader::read_data(&path, Some(opts)).expect("universal read");
    let _ = std::fs::remove_file(&path);

    assert_eq!(table.metadata("dataset_path"), Some("/data"));
    assert_eq!(table.metadata("shape"), Some("[3]"));
    match table.column("data").expect("column 'data' present") {
        scirs2_io::universal_reader::DataColumn::Float64(v) => {
            assert_eq!(v, &vec![1.0, 2.0, 3.0]);
        }
        other => panic!("expected Float64, got {}", other.type_name()),
    }
}
