//! NetCDF3 dataset reader for climate and geospatial data
//!
//! Provides `NetCdfDataset` — reads NetCDF-3 Classic or 64-bit-offset files
//! using the pure-Rust `netcdf3` crate. No feature gate is required; this
//! module is available in all build configurations.
//!
//! NetCDF (Network Common Data Form) is widely used to store gridded
//! scientific data such as climate model output, reanalysis data, and
//! satellite observations.
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_datasets::netcdf_dataset::NetCdfDataset;
//!
//! # fn example() -> Result<(), scirs2_datasets::error::DatasetsError> {
//! let ds = NetCdfDataset::from_file("temperature.nc")?;
//! println!("Variables: {:?}", ds.variable_names());
//!
//! if let Ok(arr) = ds.to_float_array("temperature") {
//!     println!("First value: {}", arr[0]);
//! }
//! # Ok(())
//! # }
//! ```

use crate::error::{DatasetsError, Result};
use netcdf3::{DataSet, DataType, DataVector, FileReader, FileWriter, Version};
use scirs2_core::ndarray::Array1;
use std::path::Path;

// ============================================================================
// Public types
// ============================================================================

/// A named dimension with its size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetCdfDimension {
    /// Dimension name
    pub name: String,
    /// Dimension size (`None` for unlimited / record dimensions with zero
    /// records written)
    pub size: Option<usize>,
}

/// A scalar or vector attribute value.
#[derive(Debug, Clone)]
pub enum AttrValue {
    /// Signed byte data (NC_BYTE / i8)
    Byte(Vec<i8>),
    /// Unsigned byte data — also used for NC_CHAR strings
    UByte(Vec<u8>),
    /// Short integer data (NC_SHORT / i16)
    Short(Vec<i16>),
    /// Integer data (NC_INT / i32)
    Int(Vec<i32>),
    /// Single-precision float data (NC_FLOAT / f32)
    Float(Vec<f32>),
    /// Double-precision float data (NC_DOUBLE / f64)
    Double(Vec<f64>),
    /// Character (string) attribute — decoded from NC_CHAR (U8) bytes
    Char(String),
}

impl AttrValue {
    fn from_attribute(attr: &netcdf3::Attribute) -> Self {
        // Use the typed accessors on Attribute — `data` field is private
        let dt = attr.data_type();
        match dt {
            DataType::I8 => attr
                .get_i8()
                .map(|s| AttrValue::Byte(s.to_vec()))
                .unwrap_or(AttrValue::Byte(vec![])),
            DataType::U8 => {
                // U8 is also used for NC_CHAR strings; try as_string first
                if let Some(s) = attr.get_as_string() {
                    AttrValue::Char(s)
                } else {
                    attr.get_u8()
                        .map(|s| AttrValue::UByte(s.to_vec()))
                        .unwrap_or(AttrValue::UByte(vec![]))
                }
            }
            DataType::I16 => attr
                .get_i16()
                .map(|s| AttrValue::Short(s.to_vec()))
                .unwrap_or(AttrValue::Short(vec![])),
            DataType::I32 => attr
                .get_i32()
                .map(|s| AttrValue::Int(s.to_vec()))
                .unwrap_or(AttrValue::Int(vec![])),
            DataType::F32 => attr
                .get_f32()
                .map(|s| AttrValue::Float(s.to_vec()))
                .unwrap_or(AttrValue::Float(vec![])),
            DataType::F64 => attr
                .get_f64()
                .map(|s| AttrValue::Double(s.to_vec()))
                .unwrap_or(AttrValue::Double(vec![])),
        }
    }
}

/// A single attribute (name + value).
#[derive(Debug, Clone)]
pub struct NetCdfAttribute {
    /// Attribute name
    pub name: String,
    /// Attribute value
    pub value: AttrValue,
}

/// Column-data of a NetCDF variable.
#[derive(Debug, Clone)]
pub enum NcData {
    /// Single-precision floats (NC_FLOAT)
    Float(Array1<f32>),
    /// Double-precision floats (NC_DOUBLE)
    Double(Array1<f64>),
    /// 32-bit integers (NC_INT)
    Int(Array1<i32>),
    /// 16-bit integers (NC_SHORT)
    Short(Array1<i16>),
    /// Signed bytes (NC_BYTE)
    Byte(Vec<i8>),
    /// Unsigned bytes (NC_CHAR — raw bytes, not decoded as string)
    UByte(Vec<u8>),
}

impl NcData {
    fn from_data_vector(dv: DataVector) -> Self {
        match dv {
            DataVector::F32(v) => NcData::Float(Array1::from_vec(v)),
            DataVector::F64(v) => NcData::Double(Array1::from_vec(v)),
            DataVector::I32(v) => NcData::Int(Array1::from_vec(v)),
            DataVector::I16(v) => NcData::Short(Array1::from_vec(v)),
            DataVector::I8(v) => NcData::Byte(v),
            DataVector::U8(v) => NcData::UByte(v),
        }
    }

    /// Cast contents to a flat `Array1<f32>` (for Float data only).
    ///
    /// Returns `None` for non-Float variants.
    pub fn as_float_array(&self) -> Option<&Array1<f32>> {
        if let NcData::Float(arr) = self {
            Some(arr)
        } else {
            None
        }
    }

    /// Cast contents to a flat `Array1<f64>` (for Double data only).
    pub fn as_double_array(&self) -> Option<&Array1<f64>> {
        if let NcData::Double(arr) = self {
            Some(arr)
        } else {
            None
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            NcData::Float(a) => a.len(),
            NcData::Double(a) => a.len(),
            NcData::Int(a) => a.len(),
            NcData::Short(a) => a.len(),
            NcData::Byte(v) => v.len(),
            NcData::UByte(v) => v.len(),
        }
    }

    /// Returns `true` if there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A NetCDF variable including its dimensions, attributes, and data.
#[derive(Debug, Clone)]
pub struct NetCdfVariable {
    /// Variable name
    pub name: String,
    /// Dimension names (in order)
    pub dimensions: Vec<String>,
    /// Variable attributes
    pub attributes: Vec<NetCdfAttribute>,
    /// NetCDF data type
    pub dtype: DataType,
    /// Actual data, read on load
    pub data: NcData,
}

/// A complete NetCDF3 dataset loaded into memory.
#[derive(Debug, Clone)]
pub struct NetCdfDataset {
    /// Ordered list of dimensions
    pub dimensions: Vec<NetCdfDimension>,
    /// Global attributes
    pub global_attributes: Vec<NetCdfAttribute>,
    /// Variables (including coordinate variables)
    pub variables: Vec<NetCdfVariable>,
}

// ============================================================================
// Implementation
// ============================================================================

impl NetCdfDataset {
    /// Read a NetCDF3 file from disk.
    ///
    /// # Errors
    ///
    /// Returns `DatasetsError` if the file cannot be opened, is not a valid
    /// NetCDF3 file, or a variable's data cannot be read.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(DatasetsError::NotFound(format!(
                "NetCDF file not found: {}",
                path.display()
            )));
        }

        let mut reader = FileReader::open(path)
            .map_err(|e| DatasetsError::InvalidFormat(format!("NetCDF3 open error: {e:?}")))?;

        Self::from_reader(&mut reader)
    }

    /// Parse a NetCDF3 dataset from an in-memory byte slice.
    ///
    /// The bytes must start with the NetCDF3 magic (`CDF\x01` or `CDF\x02`).
    ///
    /// Internally the bytes are written to a temporary file and read back via
    /// the `netcdf3` crate, which requires a seekable `Read` backed by a real
    /// file.
    ///
    /// # Errors
    ///
    /// Returns `DatasetsError::InvalidFormat` if the bytes are not valid
    /// NetCDF3 or if reading fails.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        use std::io::Write;

        let dir = tempfile::tempdir().map_err(DatasetsError::IoError)?;
        let path = dir.path().join("from_bytes.nc");

        std::fs::File::create(&path)
            .map_err(DatasetsError::IoError)?
            .write_all(bytes)
            .map_err(DatasetsError::IoError)?;

        let mut reader = FileReader::open(&path)
            .map_err(|e| DatasetsError::InvalidFormat(format!("NetCDF3 parse error: {e:?}")))?;

        Self::from_reader(&mut reader)
    }

    /// Internal: extract dataset from an already-opened `FileReader`.
    fn from_reader(reader: &mut FileReader) -> Result<Self> {
        let ds: &DataSet = reader.data_set();

        // ── Dimensions ────────────────────────────────────────────────────
        let dimensions: Vec<NetCdfDimension> = ds
            .get_dims()
            .iter()
            .map(|dim| NetCdfDimension {
                name: dim.name(),
                size: ds.dim_size(&dim.name()),
            })
            .collect();

        // ── Global attributes ─────────────────────────────────────────────
        let global_attributes: Vec<NetCdfAttribute> = ds
            .get_global_attrs()
            .iter()
            .map(|attr| NetCdfAttribute {
                name: attr.name().to_owned(),
                value: AttrValue::from_attribute(attr),
            })
            .collect();

        // ── Variable metadata ─────────────────────────────────────────────
        let var_names: Vec<String> = ds.get_var_names();
        // Collect metadata while we have &DataSet
        let var_meta: Vec<(String, Vec<String>, Vec<NetCdfAttribute>, DataType)> = var_names
            .iter()
            .filter_map(|var_name| {
                let var_def = ds.get_var(var_name)?;
                let dim_names: Vec<String> = var_def.dim_names();
                let attrs: Vec<NetCdfAttribute> = ds
                    .get_var_attrs(var_name)
                    .unwrap_or_default()
                    .iter()
                    .map(|attr| NetCdfAttribute {
                        name: attr.name().to_owned(),
                        value: AttrValue::from_attribute(attr),
                    })
                    .collect();
                let dtype = var_def.data_type();
                Some((var_name.clone(), dim_names, attrs, dtype))
            })
            .collect();

        // ── Read all variable data ────────────────────────────────────────
        let var_data_map = reader
            .read_all_vars()
            .map_err(|e| DatasetsError::InvalidFormat(format!("NetCDF3 read vars error: {e:?}")))?;

        // ── Assemble variables ────────────────────────────────────────────
        let mut variables: Vec<NetCdfVariable> = Vec::with_capacity(var_meta.len());
        for (var_name, dim_names, attrs, dtype) in var_meta {
            let data_vec = var_data_map
                .get(&var_name)
                .ok_or_else(|| DatasetsError::NotFound(format!("Data for '{var_name}' missing")))?
                .clone();

            let data = NcData::from_data_vector(data_vec);

            variables.push(NetCdfVariable {
                name: var_name,
                dimensions: dim_names,
                attributes: attrs,
                dtype,
                data,
            });
        }

        Ok(Self {
            dimensions,
            global_attributes,
            variables,
        })
    }

    // ── Lookup helpers ────────────────────────────────────────────────────

    /// Look up a variable by name.
    pub fn variable(&self, name: &str) -> Option<&NetCdfVariable> {
        self.variables.iter().find(|v| v.name == name)
    }

    /// Look up a dimension by name.
    pub fn dimension(&self, name: &str) -> Option<&NetCdfDimension> {
        self.dimensions.iter().find(|d| d.name == name)
    }

    /// Return all variable names.
    pub fn variable_names(&self) -> Vec<&str> {
        self.variables.iter().map(|v| v.name.as_str()).collect()
    }

    /// Return all dimension names.
    pub fn dimension_names(&self) -> Vec<&str> {
        self.dimensions.iter().map(|d| d.name.as_str()).collect()
    }

    /// Get a float (f32) array for the named variable.
    ///
    /// # Errors
    ///
    /// Returns `NotFound` if no variable with that name exists, or
    /// `InvalidFormat` if the variable's data is not a Float32 type.
    pub fn to_float_array(&self, var_name: &str) -> Result<Array1<f32>> {
        let var = self
            .variable(var_name)
            .ok_or_else(|| DatasetsError::NotFound(format!("Variable '{var_name}' not found")))?;
        match &var.data {
            NcData::Float(arr) => Ok(arr.clone()),
            _ => Err(DatasetsError::InvalidFormat(format!(
                "Variable '{var_name}' is not Float32 (actual dtype: {:?})",
                var.dtype
            ))),
        }
    }

    /// Get a double (f64) array for the named variable.
    ///
    /// # Errors
    ///
    /// Returns `NotFound` if no variable with that name exists, or
    /// `InvalidFormat` if the variable's data is not a Float64 type.
    pub fn to_double_array(&self, var_name: &str) -> Result<Array1<f64>> {
        let var = self
            .variable(var_name)
            .ok_or_else(|| DatasetsError::NotFound(format!("Variable '{var_name}' not found")))?;
        match &var.data {
            NcData::Double(arr) => Ok(arr.clone()),
            _ => Err(DatasetsError::InvalidFormat(format!(
                "Variable '{var_name}' is not Float64 (actual dtype: {:?})",
                var.dtype
            ))),
        }
    }

    /// Promote any numeric variable to f64, regardless of underlying type.
    ///
    /// Characters and raw bytes are excluded.
    ///
    /// # Errors
    ///
    /// Returns `NotFound` or `InvalidFormat` as appropriate.
    pub fn to_f64_array(&self, var_name: &str) -> Result<Array1<f64>> {
        let var = self
            .variable(var_name)
            .ok_or_else(|| DatasetsError::NotFound(format!("Variable '{var_name}' not found")))?;
        match &var.data {
            NcData::Float(a) => Ok(a.mapv(|v| v as f64)),
            NcData::Double(a) => Ok(a.clone()),
            NcData::Int(a) => Ok(a.mapv(|v| v as f64)),
            NcData::Short(a) => Ok(a.mapv(|v| v as f64)),
            _ => Err(DatasetsError::InvalidFormat(format!(
                "Variable '{var_name}' cannot be cast to f64 (dtype: {:?})",
                var.dtype
            ))),
        }
    }
}

// ============================================================================
// Helper: build a minimal in-memory NetCDF3 file for tests
// ============================================================================

/// Write a minimal NetCDF3 Classic file with one fixed dimension and one
/// Float32 variable, returning the raw bytes. Used in unit tests.
///
/// # Errors
///
/// Returns `DatasetsError` if the temporary file cannot be created or written.
#[doc(hidden)]
pub fn write_test_nc3_bytes(
    dim_name: &str,
    dim_size: usize,
    var_name: &str,
    data: &[f32],
) -> Result<Vec<u8>> {
    use std::io::Read;

    let dir = tempfile::tempdir().map_err(DatasetsError::IoError)?;
    let path = dir.path().join("test.nc");

    let mut dataset = DataSet::new();
    dataset
        .add_fixed_dim(dim_name, dim_size)
        .map_err(|e| DatasetsError::InvalidFormat(format!("NC3 dim error: {e:?}")))?;
    dataset
        .add_var_f32(var_name, &[dim_name])
        .map_err(|e| DatasetsError::InvalidFormat(format!("NC3 var error: {e:?}")))?;

    let mut writer = FileWriter::open(&path)
        .map_err(|e| DatasetsError::InvalidFormat(format!("NC3 writer error: {e:?}")))?;
    writer
        .set_def(&dataset, Version::Classic, 0)
        .map_err(|e| DatasetsError::InvalidFormat(format!("NC3 set_def error: {e:?}")))?;
    writer
        .write_var_f32(var_name, data)
        .map_err(|e| DatasetsError::InvalidFormat(format!("NC3 write error: {e:?}")))?;
    writer
        .close()
        .map_err(|e| DatasetsError::InvalidFormat(format!("NC3 close error: {e:?}")))?;

    let mut bytes = Vec::new();
    std::fs::File::open(&path)
        .map_err(DatasetsError::IoError)?
        .read_to_end(&mut bytes)
        .map_err(DatasetsError::IoError)?;

    Ok(bytes)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a temp NC3 file with one dimension and one Float32 variable.
    fn make_nc3_file_f32(
        dim_name: &str,
        dim_size: usize,
        var_name: &str,
        data: &[f32],
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.nc");

        let mut dataset = DataSet::new();
        dataset.add_fixed_dim(dim_name, dim_size).expect("add_dim");
        dataset.add_var_f32(var_name, &[dim_name]).expect("add_var");

        let mut writer = FileWriter::open(&path).expect("writer open");
        writer
            .set_def(&dataset, Version::Classic, 0)
            .expect("set_def");
        writer.write_var_f32(var_name, data).expect("write_var");
        writer.close().expect("close");

        (dir, path)
    }

    /// Create a temp NC3 file with Float64 variable.
    fn make_nc3_file_f64(
        dim_name: &str,
        dim_size: usize,
        var_name: &str,
        data: &[f64],
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.nc");

        let mut dataset = DataSet::new();
        dataset.add_fixed_dim(dim_name, dim_size).expect("add_dim");
        dataset.add_var_f64(var_name, &[dim_name]).expect("add_var");

        let mut writer = FileWriter::open(&path).expect("writer open");
        writer
            .set_def(&dataset, Version::Classic, 0)
            .expect("set_def");
        writer.write_var_f64(var_name, data).expect("write_var");
        writer.close().expect("close");

        (dir, path)
    }

    #[test]
    fn test_from_file_f32_roundtrip() {
        let data = vec![1.0_f32, 2.5, std::f32::consts::PI, -1.0];
        let (_dir, path) = make_nc3_file_f32("time", 4, "temperature", &data);

        let ds = NetCdfDataset::from_file(&path).expect("from_file");

        assert_eq!(ds.variable_names(), vec!["temperature"]);
        assert_eq!(ds.dimension_names(), vec!["time"]);

        let arr = ds.to_float_array("temperature").expect("to_float_array");
        assert_eq!(arr.len(), 4);
        assert!((arr[0] - 1.0).abs() < 1e-6);
        assert!((arr[2] - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn test_from_file_f64_roundtrip() {
        let data = vec![100.0_f64, 200.0, 300.5];
        let (_dir, path) = make_nc3_file_f64("x", 3, "altitude", &data);

        let ds = NetCdfDataset::from_file(&path).expect("from_file");
        let arr = ds.to_double_array("altitude").expect("to_double_array");

        assert_eq!(arr.len(), 3);
        assert!((arr[1] - 200.0).abs() < 1e-12);
    }

    #[test]
    fn test_dimension_lookup() {
        let data = vec![0.0_f32; 5];
        let (_dir, path) = make_nc3_file_f32("lat", 5, "temp", &data);

        let ds = NetCdfDataset::from_file(&path).expect("from_file");
        let dim = ds.dimension("lat").expect("dimension lat");
        assert_eq!(dim.name, "lat");
        assert_eq!(dim.size, Some(5));
    }

    #[test]
    fn test_variable_not_found() {
        let data = vec![1.0_f32];
        let (_dir, path) = make_nc3_file_f32("d", 1, "v", &data);

        let ds = NetCdfDataset::from_file(&path).expect("from_file");
        let result = ds.to_float_array("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_file_not_found() {
        let result =
            NetCdfDataset::from_file(std::env::temp_dir().join("__scirs2_nonexistent_9999.nc"));
        assert!(matches!(result, Err(DatasetsError::NotFound(_))));
    }

    #[test]
    fn test_from_bytes_roundtrip() {
        let data = vec![10.0_f32, 20.0, 30.0];
        let bytes = write_test_nc3_bytes("x", 3, "signal", &data).expect("write bytes");

        // Check magic
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..3], b"CDF");

        let ds = NetCdfDataset::from_bytes(&bytes).expect("from_bytes");
        let arr = ds.to_float_array("signal").expect("to_float_array");

        assert_eq!(arr.len(), 3);
        assert!((arr[0] - 10.0).abs() < 1e-6);
        assert!((arr[2] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_to_f64_array_from_f32_variable() {
        let data = vec![1.5_f32, 2.5, 3.5];
        let (_dir, path) = make_nc3_file_f32("n", 3, "values", &data);

        let ds = NetCdfDataset::from_file(&path).expect("from_file");
        let arr = ds.to_f64_array("values").expect("to_f64_array");

        assert_eq!(arr.len(), 3);
        assert!((arr[0] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_variable_dim_references() {
        let data = vec![0.0_f32; 4];
        let (_dir, path) = make_nc3_file_f32("time", 4, "u_wind", &data);

        let ds = NetCdfDataset::from_file(&path).expect("from_file");
        let var = ds.variable("u_wind").expect("variable u_wind");
        assert_eq!(var.dimensions, vec!["time"]);
        assert_eq!(var.dtype, DataType::F32);
    }

    #[test]
    fn test_nc3_magic_bytes() {
        let data = vec![0.0_f32; 1];
        let bytes = write_test_nc3_bytes("d", 1, "v", &data).expect("write");
        // Classic format starts with CDF\x01
        assert_eq!(bytes[0], b'C');
        assert_eq!(bytes[1], b'D');
        assert_eq!(bytes[2], b'F');
        assert_eq!(bytes[3], 0x01);
    }

    #[test]
    fn test_global_attribute_reading() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("with_attr.nc");

        let mut dataset = DataSet::new();
        dataset.add_fixed_dim("t", 2).expect("add_dim");
        dataset.add_var_f32("temp", &["t"]).expect("add_var");
        dataset
            .add_global_attr_string("institution", "Test Institute")
            .expect("add_attr");

        let mut writer = FileWriter::open(&path).expect("open");
        writer
            .set_def(&dataset, Version::Classic, 0)
            .expect("set_def");
        writer.write_var_f32("temp", &[1.0, 2.0]).expect("write");
        writer.close().expect("close");

        let ds = NetCdfDataset::from_file(&path).expect("from_file");
        assert!(!ds.global_attributes.is_empty());

        let inst = ds
            .global_attributes
            .iter()
            .find(|a| a.name == "institution")
            .expect("institution attr");
        if let AttrValue::Char(s) = &inst.value {
            assert_eq!(s, "Test Institute");
        } else {
            // Accept UByte too — NC_CHAR may come back as raw bytes depending on crate version
            if let AttrValue::UByte(bytes) = &inst.value {
                let decoded = String::from_utf8_lossy(bytes);
                assert!(decoded.contains("Test Institute"));
            } else {
                panic!(
                    "Expected Char or UByte attribute for institution, got: {:?}",
                    inst.value
                );
            }
        }
    }

    #[test]
    fn test_variable_attribute_reading() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("var_attr.nc");

        let mut dataset = DataSet::new();
        dataset.add_fixed_dim("z", 3).expect("add_dim");
        dataset.add_var_f32("pressure", &["z"]).expect("add_var");
        dataset
            .add_var_attr_string("pressure", "units", "hPa")
            .expect("add_attr");

        let mut writer = FileWriter::open(&path).expect("open");
        writer
            .set_def(&dataset, Version::Classic, 0)
            .expect("set_def");
        writer
            .write_var_f32("pressure", &[1013.0, 850.0, 500.0])
            .expect("write");
        writer.close().expect("close");

        let ds = NetCdfDataset::from_file(&path).expect("from_file");
        let var = ds.variable("pressure").expect("pressure variable");

        let units_attr = var
            .attributes
            .iter()
            .find(|a| a.name == "units")
            .expect("units attr");
        // Accept Char or UByte (NC_CHAR) for the "units" attribute
        match &units_attr.value {
            AttrValue::Char(s) => assert_eq!(s, "hPa"),
            AttrValue::UByte(b) => {
                let decoded = String::from_utf8_lossy(b);
                assert!(decoded.contains("hPa"));
            }
            other => panic!("Unexpected attribute variant: {:?}", other),
        }
    }

    #[test]
    fn test_i32_variable() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("int_var.nc");

        let mut dataset = DataSet::new();
        dataset.add_fixed_dim("n", 3).expect("add_dim");
        dataset.add_var_i32("counts", &["n"]).expect("add_var");

        let mut writer = FileWriter::open(&path).expect("open");
        writer
            .set_def(&dataset, Version::Classic, 0)
            .expect("set_def");
        writer
            .write_var_i32("counts", &[10, 20, 30])
            .expect("write");
        writer.close().expect("close");

        let ds = NetCdfDataset::from_file(&path).expect("from_file");
        let arr = ds.to_f64_array("counts").expect("to_f64_array");

        assert_eq!(arr.len(), 3);
        assert!((arr[0] - 10.0).abs() < 1e-12);
        assert!((arr[2] - 30.0).abs() < 1e-12);
    }
}
