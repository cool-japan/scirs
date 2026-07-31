//! Real NetCDF-3 "classic" format (CDF-1) binary read/write backend.
//!
//! This module adapts [`NetCDFFile`](super::NetCDFFile) onto the pure-Rust
//! `netcdf3` crate, which implements the on-disk NetCDF-3 classic/64-bit-offset
//! format (magic `CDF\x01`/`CDF\x02`, the dimension/attribute/variable lists,
//! and the record vs. non-record data sections) per Unidata's file format
//! specification. All byte-level encoding/decoding, `begin` offset
//! computation, and 4-byte record padding is delegated to that crate; this
//! module only translates between it and [`NetCDFFile`]'s own
//! dimension/variable/attribute bookkeeping.
//!
//! # Working around a `netcdf3` 0.6.1 `FileWriter::close()` defect
//!
//! Empirically verified with a standalone reproduction against the `netcdf3`
//! crate directly (write a dataset with one fixed-size variable left
//! unwritten alongside a fully-written record variable, then re-read both):
//! `netcdf3` 0.6.1's `FileWriter::close()` fills every variable that was never
//! explicitly written using the *dataset-wide* record count and record
//! stride, rather than that variable's own chunk count. For a record
//! variable this happens to be correct (its own chunk count/stride equal the
//! dataset-wide ones). But for a *fixed-size* variable in a dataset that also
//! contains record variables with `numrecs > 1`, this incorrectly emits
//! `numrecs` fill copies at record-stride offsets instead of a single copy at
//! the variable's own `begin` offset -- silently corrupting whatever data
//! happens to follow (verified to corrupt an already fully-written sibling
//! record variable's data). To stay safe regardless of the exact conditions
//! that trigger it, this module *never* leaves a fixed-size variable for
//! `close()` to fill: every fixed-size variable is always written explicitly
//! (with its real buffered data, or with synthesized fill values if it was
//! declared but never written). Only record variables ever rely on
//! `close()`'s auto-fill, and only for individual missing records, which was
//! verified safe (including with multiple record variables sharing the
//! record dimension).
use std::collections::HashMap;

use super::{AttributeValue, NetCDFDataType, NetCDFFile, NetCDFFormat, VariableInfo};
use crate::error::{IoError, Result};

/// Buffered NetCDF-3 classic variable data, canonicalized to row-major `f64`
/// regardless of the variable's declared on-disk type. Either populated by
/// `NetCDFFile::write_variable` (pending a flush to disk) or by
/// [`open_classic`] when reading an existing file.
#[derive(Debug, Clone)]
pub(super) struct ClassicVarData {
    /// Row-major values; for record variables the outermost axis is the record axis.
    pub(super) values: Vec<f64>,
    /// Shape matching `values`, with the record axis (if any) as axis 0.
    pub(super) shape: Vec<usize>,
}

/// Maps our [`NetCDFDataType`] to the `netcdf3` crate's `DataType`.
///
/// A plain function (rather than `impl From`) because both `NetCDFDataType`
/// wouldn't be the issue, but the reverse direction's target type
/// (`netcdf3::DataType`) is foreign, and implementing a foreign trait
/// (`From`) for a foreign type is not allowed by Rust's orphan rules.
fn to_nc3_type(data_type: NetCDFDataType) -> netcdf3::DataType {
    match data_type {
        NetCDFDataType::Byte => netcdf3::DataType::I8,
        NetCDFDataType::Char => netcdf3::DataType::U8,
        NetCDFDataType::Short => netcdf3::DataType::I16,
        NetCDFDataType::Int => netcdf3::DataType::I32,
        NetCDFDataType::Float => netcdf3::DataType::F32,
        NetCDFDataType::Double => netcdf3::DataType::F64,
    }
}

/// Maps the `netcdf3` crate's `DataType` back to our [`NetCDFDataType`].
fn from_nc3_type(data_type: &netcdf3::DataType) -> NetCDFDataType {
    match data_type {
        netcdf3::DataType::I8 => NetCDFDataType::Byte,
        netcdf3::DataType::U8 => NetCDFDataType::Char,
        netcdf3::DataType::I16 => NetCDFDataType::Short,
        netcdf3::DataType::I32 => NetCDFDataType::Int,
        netcdf3::DataType::F32 => NetCDFDataType::Float,
        netcdf3::DataType::F64 => NetCDFDataType::Double,
    }
}

/// Returns the standard NetCDF default fill value for `data_type`, as an `f64`.
///
/// These are the same `NC_FILL_*` constants used by the reference NetCDF C
/// library: reading a declared-but-never-written variable returns these
/// values, not zero.
pub(super) fn fill_value_f64(data_type: NetCDFDataType) -> f64 {
    match data_type {
        NetCDFDataType::Byte => netcdf3::NC_FILL_I8 as f64,
        NetCDFDataType::Char => netcdf3::NC_FILL_U8 as f64,
        NetCDFDataType::Short => netcdf3::NC_FILL_I16 as f64,
        NetCDFDataType::Int => netcdf3::NC_FILL_I32 as f64,
        NetCDFDataType::Float => netcdf3::NC_FILL_F32 as f64,
        NetCDFDataType::Double => netcdf3::NC_FILL_F64,
    }
}

// `netcdf3::WriteError` implements `Debug` but not `Display` (unlike `ReadError`/
// `InvalidDataSet`, whose `Display` impls just forward to `{:?}` anyway), so format
// uniformly via `Debug` here to support all three error types this is called with.
fn classic_err(context: &str, err: impl std::fmt::Debug) -> IoError {
    IoError::FormatError(format!("{context}: {err:?}"))
}

/// Reconstructs one of our [`AttributeValue`]s from a parsed `netcdf3` attribute.
fn nc3_attribute_to_value(attr: &netcdf3::Attribute) -> AttributeValue {
    use netcdf3::DataType as T;
    match attr.data_type() {
        T::I8 => {
            let v = attr.get_i8().unwrap_or(&[]).to_vec();
            match v.len() {
                1 => AttributeValue::Byte(v[0]),
                _ => AttributeValue::ByteArray(v),
            }
        }
        T::U8 => {
            // NC_CHAR attributes are conventionally text; reconstruct as a `String`
            // (lossily, in the rare case the bytes aren't valid UTF-8) rather than
            // inventing a separate "raw char array" variant.
            let bytes = attr.get_u8().unwrap_or(&[]);
            AttributeValue::String(String::from_utf8_lossy(bytes).into_owned())
        }
        T::I16 => {
            let v = attr.get_i16().unwrap_or(&[]).to_vec();
            match v.len() {
                1 => AttributeValue::Short(v[0]),
                _ => AttributeValue::ShortArray(v),
            }
        }
        T::I32 => {
            let v = attr.get_i32().unwrap_or(&[]).to_vec();
            match v.len() {
                1 => AttributeValue::Int(v[0]),
                _ => AttributeValue::IntArray(v),
            }
        }
        T::F32 => {
            let v = attr.get_f32().unwrap_or(&[]).to_vec();
            match v.len() {
                1 => AttributeValue::Float(v[0]),
                _ => AttributeValue::FloatArray(v),
            }
        }
        T::F64 => {
            let v = attr.get_f64().unwrap_or(&[]).to_vec();
            match v.len() {
                1 => AttributeValue::Double(v[0]),
                _ => AttributeValue::DoubleArray(v),
            }
        }
    }
}

fn datavector_to_f64(vector: &netcdf3::DataVector) -> Vec<f64> {
    match vector {
        netcdf3::DataVector::I8(v) => v.iter().map(|&x| x as f64).collect(),
        netcdf3::DataVector::U8(v) => v.iter().map(|&x| x as f64).collect(),
        netcdf3::DataVector::I16(v) => v.iter().map(|&x| x as f64).collect(),
        netcdf3::DataVector::I32(v) => v.iter().map(|&x| x as f64).collect(),
        netcdf3::DataVector::F32(v) => v.iter().map(|&x| x as f64).collect(),
        netcdf3::DataVector::F64(v) => v.clone(),
    }
}

/// Opens and fully parses an existing NetCDF-3 classic (or 64-bit offset)
/// file, returning a read-mode [`NetCDFFile`] populated with its real
/// dimensions, variables, attributes and data.
pub(super) fn open_classic(path_str: &str) -> Result<NetCDFFile> {
    let mut reader = netcdf3::FileReader::open(path_str)
        .map_err(|e| classic_err(&format!("Failed to read NetCDF3 file '{path_str}'"), e))?;

    // Snapshot everything the parsed header/data-set tells us into our own owned
    // structures before calling `read_all_vars`, so that this immutable borrow of
    // `reader` (via `reader.data_set()`) is released before the mutable borrow below.
    let (dimensions, dim_order, attributes, attr_order, variables, var_order, mut shapes) = {
        let data_set = reader.data_set();

        let mut dimensions = HashMap::new();
        let mut dim_order = Vec::new();
        for dim in data_set.get_dims() {
            let name = dim.name();
            dim_order.push(name.clone());
            dimensions.insert(
                name,
                if dim.is_unlimited() {
                    None
                } else {
                    Some(dim.size())
                },
            );
        }

        let mut attributes = HashMap::new();
        let mut attr_order = Vec::new();
        for attr in data_set.get_global_attrs() {
            attr_order.push(attr.name().to_string());
            attributes.insert(attr.name().to_string(), nc3_attribute_to_value(attr));
        }

        let mut variables = HashMap::new();
        let mut var_order = Vec::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        for var in data_set.get_vars() {
            let name = var.name().to_string();
            var_order.push(name.clone());
            shapes.insert(
                name.clone(),
                var.get_dims().iter().map(|d| d.size()).collect(),
            );

            let mut var_attrs = HashMap::new();
            let mut var_attr_order = Vec::new();
            for attr in var.get_attrs() {
                var_attr_order.push(attr.name().to_string());
                var_attrs.insert(attr.name().to_string(), nc3_attribute_to_value(attr));
            }

            variables.insert(
                name.clone(),
                VariableInfo {
                    name,
                    data_type: from_nc3_type(&var.data_type()),
                    dimensions: var.dim_names(),
                    attributes: var_attrs,
                    attr_order: var_attr_order,
                },
            );
        }

        (
            dimensions, dim_order, attributes, attr_order, variables, var_order, shapes,
        )
    };

    let all_data = reader.read_all_vars().map_err(|e| {
        classic_err(
            &format!("Failed to read NetCDF3 variable data in '{path_str}'"),
            e,
        )
    })?;
    let mut classic_data = HashMap::new();
    for (name, vector) in all_data {
        let values = datavector_to_f64(&vector);
        let shape = shapes.remove(&name).unwrap_or_default();
        classic_data.insert(name, ClassicVarData { values, shape });
    }

    Ok(NetCDFFile {
        path: path_str.to_string(),
        mode: "r".to_string(),
        format: NetCDFFormat::Classic,
        dimensions,
        dim_order,
        variables,
        var_order,
        attributes,
        attr_order,
        classic_data,
        hdf5_backend: None,
    })
}

fn f64_to_i8(v: &[f64]) -> Vec<i8> {
    v.iter().map(|&x| x as i8).collect()
}
fn f64_to_u8(v: &[f64]) -> Vec<u8> {
    v.iter().map(|&x| x as u8).collect()
}
fn f64_to_i16(v: &[f64]) -> Vec<i16> {
    v.iter().map(|&x| x as i16).collect()
}
fn f64_to_i32(v: &[f64]) -> Vec<i32> {
    v.iter().map(|&x| x as i32).collect()
}
fn f64_to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn write_whole_var(
    writer: &mut netcdf3::FileWriter,
    name: &str,
    data_type: NetCDFDataType,
    values: &[f64],
) -> Result<()> {
    let res = match data_type {
        NetCDFDataType::Byte => writer.write_var_i8(name, &f64_to_i8(values)),
        NetCDFDataType::Char => writer.write_var_u8(name, &f64_to_u8(values)),
        NetCDFDataType::Short => writer.write_var_i16(name, &f64_to_i16(values)),
        NetCDFDataType::Int => writer.write_var_i32(name, &f64_to_i32(values)),
        NetCDFDataType::Float => writer.write_var_f32(name, &f64_to_f32(values)),
        NetCDFDataType::Double => writer.write_var_f64(name, values),
    };
    res.map_err(|e| classic_err(&format!("Failed to write variable '{name}'"), e))
}

fn write_one_record(
    writer: &mut netcdf3::FileWriter,
    name: &str,
    record_index: usize,
    data_type: NetCDFDataType,
    values: &[f64],
) -> Result<()> {
    let res = match data_type {
        NetCDFDataType::Byte => writer.write_record_i8(name, record_index, &f64_to_i8(values)),
        NetCDFDataType::Char => writer.write_record_u8(name, record_index, &f64_to_u8(values)),
        NetCDFDataType::Short => writer.write_record_i16(name, record_index, &f64_to_i16(values)),
        NetCDFDataType::Int => writer.write_record_i32(name, record_index, &f64_to_i32(values)),
        NetCDFDataType::Float => writer.write_record_f32(name, record_index, &f64_to_f32(values)),
        NetCDFDataType::Double => writer.write_record_f64(name, record_index, values),
    };
    res.map_err(|e| {
        classic_err(
            &format!("Failed to write record {record_index} of variable '{name}'"),
            e,
        )
    })
}

fn add_global_attr(
    data_set: &mut netcdf3::DataSet,
    name: &str,
    value: &AttributeValue,
) -> Result<()> {
    let res = match value {
        AttributeValue::String(s) => data_set.add_global_attr_string(name, s),
        AttributeValue::Byte(b) => data_set.add_global_attr_i8(name, vec![*b]),
        AttributeValue::Short(s) => data_set.add_global_attr_i16(name, vec![*s]),
        AttributeValue::Int(i) => data_set.add_global_attr_i32(name, vec![*i]),
        AttributeValue::Float(f) => data_set.add_global_attr_f32(name, vec![*f]),
        AttributeValue::Double(d) => data_set.add_global_attr_f64(name, vec![*d]),
        AttributeValue::ByteArray(v) => data_set.add_global_attr_i8(name, v.clone()),
        AttributeValue::ShortArray(v) => data_set.add_global_attr_i16(name, v.clone()),
        AttributeValue::IntArray(v) => data_set.add_global_attr_i32(name, v.clone()),
        AttributeValue::FloatArray(v) => data_set.add_global_attr_f32(name, v.clone()),
        AttributeValue::DoubleArray(v) => data_set.add_global_attr_f64(name, v.clone()),
    };
    res.map_err(|e| classic_err(&format!("Failed to add global attribute '{name}'"), e))
}

fn add_var_attr(
    data_set: &mut netcdf3::DataSet,
    var_name: &str,
    attr_name: &str,
    value: &AttributeValue,
) -> Result<()> {
    let res = match value {
        AttributeValue::String(s) => data_set.add_var_attr_string(var_name, attr_name, s),
        AttributeValue::Byte(b) => data_set.add_var_attr_i8(var_name, attr_name, vec![*b]),
        AttributeValue::Short(s) => data_set.add_var_attr_i16(var_name, attr_name, vec![*s]),
        AttributeValue::Int(i) => data_set.add_var_attr_i32(var_name, attr_name, vec![*i]),
        AttributeValue::Float(f) => data_set.add_var_attr_f32(var_name, attr_name, vec![*f]),
        AttributeValue::Double(d) => data_set.add_var_attr_f64(var_name, attr_name, vec![*d]),
        AttributeValue::ByteArray(v) => data_set.add_var_attr_i8(var_name, attr_name, v.clone()),
        AttributeValue::ShortArray(v) => data_set.add_var_attr_i16(var_name, attr_name, v.clone()),
        AttributeValue::IntArray(v) => data_set.add_var_attr_i32(var_name, attr_name, v.clone()),
        AttributeValue::FloatArray(v) => data_set.add_var_attr_f32(var_name, attr_name, v.clone()),
        AttributeValue::DoubleArray(v) => data_set.add_var_attr_f64(var_name, attr_name, v.clone()),
    };
    res.map_err(|e| {
        classic_err(
            &format!("Failed to add attribute '{attr_name}' to variable '{var_name}'"),
            e,
        )
    })
}

/// Computes the number of records to declare for the unlimited dimension
/// named `record_dim_name`: the maximum leading-axis extent among all
/// buffered record variables that use it (0 if none have been written yet).
fn compute_numrecs(file: &NetCDFFile, record_dim_name: &str) -> usize {
    file.var_order
        .iter()
        .filter_map(|var_name| {
            let var_info = file.variables.get(var_name)?;
            if var_info.dimensions.first().map(String::as_str) != Some(record_dim_name) {
                return None;
            }
            file.classic_data.get(var_name)?.shape.first().copied()
        })
        .max()
        .unwrap_or(0)
}

/// Serializes `file`'s current dimensions/variables/attributes/buffered data
/// to disk as a NetCDF-3 classic file, truncating and overwriting any
/// existing file at `file.path`.
pub(super) fn flush_classic(file: &NetCDFFile) -> Result<()> {
    let mut data_set = netcdf3::DataSet::new();

    for dim_name in &file.dim_order {
        match file.dimensions.get(dim_name).copied().flatten() {
            Some(fixed_size) => data_set
                .add_fixed_dim(dim_name, fixed_size)
                .map_err(|e| classic_err(&format!("Failed to define dimension '{dim_name}'"), e))?,
            None => {
                let numrecs = compute_numrecs(file, dim_name);
                data_set.set_unlimited_dim(dim_name, numrecs).map_err(|e| {
                    classic_err(&format!("Failed to define dimension '{dim_name}'"), e)
                })?;
            }
        }
    }

    for attr_name in &file.attr_order {
        if let Some(value) = file.attributes.get(attr_name) {
            add_global_attr(&mut data_set, attr_name, value)?;
        }
    }

    for var_name in &file.var_order {
        let var_info = &file.variables[var_name];
        let nc_type = to_nc3_type(var_info.data_type);
        data_set
            .add_var(var_name, &var_info.dimensions, nc_type)
            .map_err(|e| classic_err(&format!("Failed to define variable '{var_name}'"), e))?;
        for attr_name in &var_info.attr_order {
            if let Some(value) = var_info.attributes.get(attr_name) {
                add_var_attr(&mut data_set, var_name, attr_name, value)?;
            }
        }
    }

    let mut writer = netcdf3::FileWriter::open(&file.path)
        .map_err(|e| classic_err(&format!("Failed to open '{}' for writing", file.path), e))?;
    writer
        .set_def(&data_set, netcdf3::Version::Classic, 0)
        .map_err(|e| {
            classic_err(
                &format!("Failed to write NetCDF3 header to '{}'", file.path),
                e,
            )
        })?;

    for var_name in &file.var_order {
        let var_info = &file.variables[var_name];
        // Use `netcdf3`'s own authoritative `is_record_var` (dataset var, not a
        // name comparison): a variable with zero dimensions has no "first
        // dimension" to compare, and a naive `record_dim_name == first_dim_name`
        // comparison would wrongly treat it as a record variable whenever the
        // dataset has no unlimited dimension at all (`None == None`).
        let is_record_var = data_set
            .get_var(var_name)
            .map(|v| v.is_record_var())
            .unwrap_or(false);
        match (is_record_var, file.classic_data.get(var_name)) {
            (true, Some(buf)) => {
                // Write only the records we actually have; `writer.close()` fills any
                // remaining records (up to numrecs) with the standard NC fill value --
                // verified safe for record variables (see module docs).
                let num_records = buf.shape.first().copied().unwrap_or(0);
                let chunk_len: usize = buf.shape.iter().skip(1).product();
                for r in 0..num_records {
                    let start = r * chunk_len;
                    let end = start + chunk_len;
                    write_one_record(
                        &mut writer,
                        var_name,
                        r,
                        var_info.data_type,
                        &buf.values[start..end],
                    )?;
                }
            }
            (true, None) => {
                // Declared record variable, never written: leave entirely to
                // `close()`'s auto-fill.
            }
            (false, Some(buf)) => {
                write_whole_var(&mut writer, var_name, var_info.data_type, &buf.values)?;
            }
            (false, None) => {
                // Never leave a fixed-size variable for `close()` to fill (see module
                // docs for the corruption this otherwise triggers in mixed datasets):
                // write it explicitly with synthesized fill values instead.
                let chunk_len = data_set
                    .get_var(var_name)
                    .map(|v| v.chunk_len())
                    .unwrap_or(0);
                let fill = vec![fill_value_f64(var_info.data_type); chunk_len];
                write_whole_var(&mut writer, var_name, var_info.data_type, &fill)?;
            }
        }
    }

    writer.close().map_err(|e| {
        classic_err(
            &format!("Failed to finalize NetCDF3 file '{}'", file.path),
            e,
        )
    })?;
    Ok(())
}
