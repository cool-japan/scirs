//! Enhanced MATLAB v7.3+ format support
//!
//! This module provides comprehensive support for MATLAB v7.3+ files,
//! which are based on HDF5 format with MATLAB-specific conventions.

use crate::error::{IoError, Result};
use crate::matlab::MatType;
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::path::Path;

use crate::hdf5::{AttributeValue, CompressionOptions, DatasetOptions, FileMode, HDF5File};

/// MATLAB v7.3+ specific features
#[derive(Debug, Clone)]
pub struct V73Features {
    /// Enable subsref subsasgn support for partial I/O
    pub enable_partial_io: bool,
    /// Support for MATLAB objects
    pub support_objects: bool,
    /// Support for function handles
    pub support_function_handles: bool,
    /// Support for tables
    pub support_tables: bool,
    /// Support for tall arrays
    pub support_tall_arrays: bool,
    /// Support for categorical arrays
    pub support_categorical: bool,
    /// Support for datetime arrays
    pub support_datetime: bool,
    /// Support for string arrays (different from char arrays)
    pub support_string_arrays: bool,
}

impl Default for V73Features {
    fn default() -> Self {
        Self {
            enable_partial_io: true,
            support_objects: true,
            support_function_handles: true,
            support_tables: true,
            support_tall_arrays: false, // Requires special handling
            support_categorical: true,
            support_datetime: true,
            support_string_arrays: true,
        }
    }
}

/// Extended MATLAB data types for v7.3+
#[derive(Debug, Clone)]
pub enum ExtendedMatType {
    /// Standard MatType
    Standard(Box<MatType>),
    /// MATLAB table
    Table(MatlabTable),
    /// MATLAB categorical array
    Categorical(CategoricalArray),
    /// MATLAB datetime array
    DateTime(DateTimeArray),
    /// MATLAB string array (not char array)
    StringArray(Vec<String>),
    /// Function handle
    FunctionHandle(FunctionHandle),
    /// MATLAB object
    Object(MatlabObject),
    /// Complex double array
    ComplexDouble(ArrayD<scirs2_core::numeric::Complex<f64>>),
    /// Complex single array
    ComplexSingle(ArrayD<scirs2_core::numeric::Complex<f32>>),
}

/// MATLAB table representation
#[derive(Debug, Clone)]
pub struct MatlabTable {
    /// Variable names
    pub variable_names: Vec<String>,
    /// Row names (optional)
    pub row_names: Option<Vec<String>>,
    /// Table data (column-oriented)
    pub data: HashMap<String, MatType>,
    /// Table properties
    pub properties: HashMap<String, String>,
}

/// MATLAB categorical array
#[derive(Debug, Clone)]
pub struct CategoricalArray {
    /// Category names
    pub categories: Vec<String>,
    /// Data indices (0-based)
    pub data: ArrayD<u32>,
    /// Whether the categories are ordered
    pub ordered: bool,
}

/// MATLAB datetime array
#[derive(Debug, Clone)]
pub struct DateTimeArray {
    /// Serial date numbers (days since January 0, 0000)
    pub data: ArrayD<f64>,
    /// Time zone information
    pub timezone: Option<String>,
    /// Date format
    pub format: String,
}

/// MATLAB function handle
#[derive(Debug, Clone)]
pub struct FunctionHandle {
    /// Function name or anonymous function string
    pub function: String,
    /// Function type (simple, nested, anonymous, etc.)
    pub function_type: String,
    /// Workspace variables (for nested/anonymous functions)
    pub workspace: Option<HashMap<String, MatType>>,
}

/// MATLAB object
#[derive(Debug, Clone)]
pub struct MatlabObject {
    /// Class name
    pub class_name: String,
    /// Object properties
    pub properties: HashMap<String, MatType>,
    /// Superclass data
    pub superclass_data: Option<Box<MatlabObject>>,
}

/// Enhanced v7.3 MAT file handler
pub struct V73MatFile {
    #[allow(dead_code)]
    features: V73Features,
    compression: Option<CompressionOptions>,
}

impl V73MatFile {
    /// Create a new v7.3 MAT file handler
    pub fn new(features: V73Features) -> Self {
        Self {
            features,
            compression: None,
        }
    }

    /// Set compression options
    pub fn with_compression(mut self, compression: CompressionOptions) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Write extended MATLAB types to v7.3 file
    pub fn write_extended<P: AsRef<Path>>(
        &self,
        path: P,
        vars: &HashMap<String, ExtendedMatType>,
    ) -> Result<()> {
        let mut hdf5_file = HDF5File::create(path)?;

        // Add MATLAB v7.3 file signature
        hdf5_file.set_attribute(
            "/",
            "MATLAB_version",
            AttributeValue::String("7.3".to_string()),
        )?;

        for (name, ext_type) in vars {
            self.write_extended_type(&mut hdf5_file, name, ext_type)?;
        }

        hdf5_file.close()?;
        Ok(())
    }

    /// Read extended MATLAB types from v7.3 file
    pub fn read_extended<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<HashMap<String, ExtendedMatType>> {
        let hdf5_file = HDF5File::open(path, FileMode::ReadOnly)?;
        let mut vars = HashMap::new();

        // Get all top-level datasets and groups
        let items = hdf5_file.list_all_items();

        for item in items {
            if let Ok(ext_type) = self.read_extended_type(&hdf5_file, &item) {
                vars.insert(item.trim_start_matches('/').to_string(), ext_type);
            }
        }

        Ok(vars)
    }

    /// Write an extended type to HDF5
    fn write_extended_type(
        &self,
        file: &mut HDF5File,
        name: &str,
        ext_type: &ExtendedMatType,
    ) -> Result<()> {
        match ext_type {
            ExtendedMatType::Standard(mat_type) => self.write_standard_type(file, name, mat_type),
            ExtendedMatType::Table(table) => self.write_table(file, name, table),
            ExtendedMatType::Categorical(cat_array) => {
                self.write_categorical(file, name, cat_array)
            }
            ExtendedMatType::DateTime(dt_array) => self.write_datetime(file, name, dt_array),
            ExtendedMatType::StringArray(strings) => self.write_string_array(file, name, strings),
            ExtendedMatType::FunctionHandle(func_handle) => {
                self.write_function_handle(file, name, func_handle)
            }
            ExtendedMatType::Object(object) => self.write_object(file, name, object),
            ExtendedMatType::ComplexDouble(array) => self.write_complex_double(file, name, array),
            ExtendedMatType::ComplexSingle(array) => self.write_complex_single(file, name, array),
        }
    }

    /// Write a MATLAB table.
    ///
    /// Layout:
    /// - group `{name}/`  with attr `MATLAB_class = "table"`
    /// - attr `VariableNames` → StringArray of column names
    /// - attr `RowNames`      → StringArray (optional)
    /// - attr `property_{k}` → String for each table property
    /// - dataset `{name}/{var_name}` for each column
    fn write_table(&self, file: &mut HDF5File, name: &str, table: &MatlabTable) -> Result<()> {
        Self::create_nested_group(file, name)?;
        Self::set_group_attribute(
            file,
            name,
            "MATLAB_class",
            AttributeValue::String("table".to_string()),
        )?;

        // Write variable names as a string array attribute (round-trippable)
        Self::set_group_attribute(
            file,
            name,
            "VariableNames",
            AttributeValue::StringArray(table.variable_names.clone()),
        )?;

        // Write table data columns
        for (var_name, var_data) in &table.data {
            let var_path = format!("{}/{}", name, var_name);
            self.write_standard_type(file, &var_path, var_data)?;
        }

        // Write row names if present
        if let Some(ref row_names) = table.row_names {
            Self::set_group_attribute(
                file,
                name,
                "RowNames",
                AttributeValue::StringArray(row_names.clone()),
            )?;
        }

        // Write properties
        for (prop_name, prop_value) in &table.properties {
            Self::set_group_attribute(
                file,
                name,
                &format!("property_{}", prop_name),
                AttributeValue::String(prop_value.clone()),
            )?;
        }

        Ok(())
    }

    /// Write a categorical array.
    ///
    /// Layout:
    /// - group `{name}/` with attr `MATLAB_class = "categorical"`
    /// - attr `Categories` → StringArray
    /// - attr `ordered`    → Boolean
    /// - dataset `{name}/data` → u32 indices
    fn write_categorical(
        &self,
        file: &mut HDF5File,
        name: &str,
        cat_array: &CategoricalArray,
    ) -> Result<()> {
        Self::create_nested_group(file, name)?;
        Self::set_group_attribute(
            file,
            name,
            "MATLAB_class",
            AttributeValue::String("categorical".to_string()),
        )?;

        // Write categories as a string array attribute (round-trippable)
        Self::set_group_attribute(
            file,
            name,
            "Categories",
            AttributeValue::StringArray(cat_array.categories.clone()),
        )?;

        // Write data indices
        file.create_dataset_from_array(
            &format!("{}/data", name),
            &cat_array.data,
            Some(DatasetOptions::default()),
        )?;

        // Write ordered flag
        Self::set_group_attribute(
            file,
            name,
            "ordered",
            AttributeValue::Boolean(cat_array.ordered),
        )?;

        Ok(())
    }

    /// Write a datetime array.
    ///
    /// Layout:
    /// - group `{name}/` with attr `MATLAB_class = "datetime"`
    /// - attr `timezone`   → String (optional)
    /// - attr `format`     → String
    /// - dataset `{name}/data` of f64 serial-date values
    ///
    /// Using a group (rather than a top-level dataset) ensures attributes are
    /// flushed to native HDF5 — `write_dataset_to_hdf5` does not write dataset
    /// attributes, but `write_group_to_hdf5` does write group attributes.
    fn write_datetime(
        &self,
        file: &mut HDF5File,
        name: &str,
        dt_array: &DateTimeArray,
    ) -> Result<()> {
        Self::create_nested_group(file, name)?;

        Self::set_group_attribute(
            file,
            name,
            "MATLAB_class",
            AttributeValue::String("datetime".to_string()),
        )?;

        if let Some(ref tz) = dt_array.timezone {
            Self::set_group_attribute(file, name, "timezone", AttributeValue::String(tz.clone()))?;
        }

        Self::set_group_attribute(
            file,
            name,
            "format",
            AttributeValue::String(dt_array.format.clone()),
        )?;

        // Store the actual data as a sub-dataset
        file.create_dataset_from_array(
            &format!("{}/data", name),
            &dt_array.data,
            Some(DatasetOptions {
                compression: self.compression.clone().unwrap_or_default(),
                ..Default::default()
            }),
        )?;

        Ok(())
    }

    /// Write a string array.
    ///
    /// Layout:
    /// - group `{name}/` with attr `MATLAB_class = "string"`
    /// - attr `size` → Integer(n)  (scalar i64; IntegerArray silently fails to write due to ndarray 0.15/0.17 mismatch in hdf5-0.8.1)
    /// - dataset `{name}/string_{i}` for each element (stored as u16 UTF-16 values cast to f64)
    fn write_string_array(
        &self,
        file: &mut HDF5File,
        name: &str,
        strings: &[String],
    ) -> Result<()> {
        Self::create_nested_group(file, name)?;
        Self::set_group_attribute(
            file,
            name,
            "MATLAB_class",
            AttributeValue::String("string".to_string()),
        )?;

        for (i, string) in strings.iter().enumerate() {
            let string_data: Vec<u16> = string.encode_utf16().collect();
            let string_array = scirs2_core::ndarray::Array1::from_vec(string_data).into_dyn();
            file.create_dataset_from_array(
                &format!("{}/string_{}", name, i),
                &string_array,
                Some(DatasetOptions::default()),
            )?;
        }

        // Store count as Integer scalar — Integer writes reliably as a scalar i64 attr.
        // IntegerArray with shape [1] causes the HDF5 attr.write() to fail silently
        // when the ndarray 0.15 / 0.17 mismatch is in play.
        Self::set_group_attribute(
            file,
            name,
            "size",
            AttributeValue::Integer(strings.len() as i64),
        )?;

        Ok(())
    }

    /// Write a function handle.
    ///
    /// Layout:
    /// - group `{name}/` with attr `MATLAB_class = "function_handle"`
    /// - attr `type`                → String
    /// - dataset `{name}/function`  → u16 UTF-16 values
    /// - group `{name}/workspace/`  (optional) with one child per workspace variable
    fn write_function_handle(
        &self,
        file: &mut HDF5File,
        name: &str,
        func_handle: &FunctionHandle,
    ) -> Result<()> {
        Self::create_nested_group(file, name)?;
        Self::set_group_attribute(
            file,
            name,
            "MATLAB_class",
            AttributeValue::String("function_handle".to_string()),
        )?;

        let func_data: Vec<u16> = func_handle.function.encode_utf16().collect();
        let func_array = scirs2_core::ndarray::Array1::from_vec(func_data).into_dyn();
        file.create_dataset_from_array(
            &format!("{}/function", name),
            &func_array,
            Some(DatasetOptions::default()),
        )?;

        Self::set_group_attribute(
            file,
            name,
            "type",
            AttributeValue::String(func_handle.function_type.clone()),
        )?;

        if let Some(ref workspace) = func_handle.workspace {
            let ws_group = format!("{}/workspace", name);
            Self::create_nested_group(file, &ws_group)?;

            for (var_name, var_data) in workspace {
                let var_path = format!("{}/{}", ws_group, var_name);
                self.write_standard_type(file, &var_path, var_data)?;
            }
        }

        Ok(())
    }

    /// Write a MATLAB object.
    ///
    /// Layout:
    /// - group `{name}/` with attr `MATLAB_class = <class_name>`, `MATLAB_object = true`
    /// - group `{name}/properties/` with one child dataset per property
    /// - group `{name}/superclass/` (optional) recursively written
    fn write_object(&self, file: &mut HDF5File, name: &str, object: &MatlabObject) -> Result<()> {
        Self::create_nested_group(file, name)?;
        Self::set_group_attribute(
            file,
            name,
            "MATLAB_class",
            AttributeValue::String(object.class_name.clone()),
        )?;
        Self::set_group_attribute(file, name, "MATLAB_object", AttributeValue::Boolean(true))?;

        let props_group = format!("{}/properties", name);
        Self::create_nested_group(file, &props_group)?;

        // Store property names for round-trip reconstruction
        let prop_names: Vec<String> = object.properties.keys().cloned().collect();
        Self::set_group_attribute(
            file,
            name,
            "PropertyNames",
            AttributeValue::StringArray(prop_names),
        )?;

        for (prop_name, prop_data) in &object.properties {
            let prop_path = format!("{}/{}", props_group, prop_name);
            self.write_standard_type(file, &prop_path, prop_data)?;
        }

        if let Some(ref superclass) = object.superclass_data {
            let super_path = format!("{}/superclass", name);
            self.write_object(file, &super_path, superclass)?;
        }

        Ok(())
    }

    /// Write complex double array
    fn write_complex_double(
        &self,
        file: &mut HDF5File,
        name: &str,
        array: &ArrayD<scirs2_core::numeric::Complex<f64>>,
    ) -> Result<()> {
        let real_part = array.mapv(|x| x.re);
        let imag_part = array.mapv(|x| x.im);

        file.create_group(name)?;
        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String("double".to_string()),
        )?;
        file.set_attribute(name, "MATLAB_complex", AttributeValue::Boolean(true))?;

        file.create_dataset_from_array(
            &format!("{}/real", name),
            &real_part,
            Some(DatasetOptions {
                compression: self.compression.clone().unwrap_or_default(),
                ..Default::default()
            }),
        )?;
        file.create_dataset_from_array(
            &format!("{}/imag", name),
            &imag_part,
            Some(DatasetOptions {
                compression: self.compression.clone().unwrap_or_default(),
                ..Default::default()
            }),
        )?;

        Ok(())
    }

    /// Write complex single array
    fn write_complex_single(
        &self,
        file: &mut HDF5File,
        name: &str,
        array: &ArrayD<scirs2_core::numeric::Complex<f32>>,
    ) -> Result<()> {
        let real_part = array.mapv(|x| x.re);
        let imag_part = array.mapv(|x| x.im);

        file.create_group(name)?;
        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String("single".to_string()),
        )?;
        file.set_attribute(name, "MATLAB_complex", AttributeValue::Boolean(true))?;

        file.create_dataset_from_array(
            &format!("{}/real", name),
            &real_part,
            Some(DatasetOptions {
                compression: self.compression.clone().unwrap_or_default(),
                ..Default::default()
            }),
        )?;
        file.create_dataset_from_array(
            &format!("{}/imag", name),
            &imag_part,
            Some(DatasetOptions {
                compression: self.compression.clone().unwrap_or_default(),
                ..Default::default()
            }),
        )?;

        Ok(())
    }

    /// Create a group at a nested path, properly navigating the in-memory tree.
    ///
    /// `HDF5File::create_group` passes the full name directly to `Group::create_group`,
    /// which treats the entire string as a single key.  This helper splits the path
    /// and navigates (and creates) each level in the tree.
    fn create_nested_group(file: &mut HDF5File, path: &str) -> Result<()> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.is_empty() {
            return Ok(());
        }
        let mut g = file.root_mut();
        for part in parts {
            g = g.create_group(part);
        }
        Ok(())
    }

    /// Set an attribute on a group at a nested path.
    ///
    /// `HDF5File::set_attribute` navigates through the groups map using the path.
    /// This is fine for groups but fails when any path component is a dataset.
    /// For group-only paths this method is equivalent but explicit.
    fn set_group_attribute(
        file: &mut HDF5File,
        path: &str,
        key: &str,
        value: AttributeValue,
    ) -> Result<()> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.is_empty() {
            file.root_mut().set_attribute(key, value);
            return Ok(());
        }
        let mut g = file.root_mut();
        for part in &parts {
            g = g
                .get_group_mut(part)
                .ok_or_else(|| IoError::FormatError(format!("Group '{}' not found", part)))?;
        }
        g.set_attribute(key, value);
        Ok(())
    }

    /// Get an attribute from a group at a nested path.
    fn get_group_attribute<'a>(
        file: &'a HDF5File,
        path: &str,
        key: &str,
    ) -> Option<&'a AttributeValue> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.is_empty() {
            return file.root().get_attribute(key);
        }
        let mut g = file.root();
        for part in &parts {
            g = g.get_group(part)?;
        }
        g.get_attribute(key)
    }

    /// Get an attribute from a dataset identified by its full path.
    ///
    /// `HDF5File::get_attribute` only navigates the groups map; it cannot reach
    /// attributes stored on a leaf dataset.  This helper navigates to the parent
    /// group and fetches the attribute from the `Dataset` object directly.
    ///
    /// Returns `None` if the dataset or attribute does not exist.
    fn get_dataset_attribute<'a>(
        file: &'a HDF5File,
        path: &str,
        key: &str,
    ) -> Option<&'a AttributeValue> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.is_empty() {
            return None;
        }
        let dataset_name = *parts.last()?;
        let parent_group = if parts.len() == 1 {
            file.root()
        } else {
            let mut g = file.root();
            for &group_part in &parts[..parts.len() - 1] {
                g = g.get_group(group_part)?;
            }
            g
        };
        parent_group.get_dataset(dataset_name)?.get_attribute(key)
    }

    /// Set an attribute on a dataset identified by its full path.
    ///
    /// The in-house `HDF5File::set_attribute` only knows about the groups map,
    /// not the datasets map.  This helper navigates to the parent group and then
    /// sets the attribute directly on the `Dataset` object stored in that group's
    /// `datasets` map.  This is required for any dataset that lives under a parent
    /// group (e.g. path `"tbl/col_a"`).
    fn set_dataset_attribute(
        file: &mut HDF5File,
        path: &str,
        key: &str,
        value: AttributeValue,
    ) -> Result<()> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        let Some(dataset_name) = parts.last().copied() else {
            return Err(IoError::FormatError("Invalid dataset path".to_string()));
        };
        let parent_group = if parts.len() == 1 {
            // Top-level dataset — attribute lives on the root group's dataset entry.
            file.root_mut()
        } else {
            // Navigate to the parent group, creating groups as needed.
            let mut g = file.root_mut();
            for &group_part in &parts[..parts.len() - 1] {
                g = g.create_group(group_part);
            }
            g
        };
        let ds = parent_group
            .get_dataset_mut(dataset_name)
            .ok_or_else(|| IoError::FormatError(format!("Dataset '{}' not found", path)))?;
        ds.set_attribute(key, value);
        Ok(())
    }

    /// Write a standard MatType to HDF5.
    ///
    /// Inlines the same dispatch logic as `EnhancedMatFile::write_mat_type_to_hdf5`,
    /// using this handler's own compression settings.
    ///
    /// Dataset-level attributes (like `MATLAB_class`) are set via
    /// `set_dataset_attribute` rather than `HDF5File::set_attribute`, because
    /// the latter only navigates through the groups map and fails on leaf datasets
    /// that live under a parent group.
    fn write_standard_type(
        &self,
        file: &mut HDF5File,
        name: &str,
        mat_type: &MatType,
    ) -> Result<()> {
        let options = DatasetOptions {
            compression: self.compression.clone().unwrap_or_default(),
            ..Default::default()
        };

        match mat_type {
            MatType::Double(array) => {
                file.create_dataset_from_array(name, array, Some(options))?;
                Self::set_dataset_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("double".to_string()),
                )?;
            }
            MatType::Single(array) => {
                file.create_dataset_from_array(name, array, Some(options))?;
                Self::set_dataset_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("single".to_string()),
                )?;
            }
            MatType::Int8(array) => {
                file.create_dataset_from_array(name, array, Some(options))?;
                Self::set_dataset_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("int8".to_string()),
                )?;
            }
            MatType::Int16(array) => {
                file.create_dataset_from_array(name, array, Some(options))?;
                Self::set_dataset_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("int16".to_string()),
                )?;
            }
            MatType::Int32(array) => {
                file.create_dataset_from_array(name, array, Some(options))?;
                Self::set_dataset_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("int32".to_string()),
                )?;
            }
            MatType::Int64(array) => {
                // The HDF5 backing store is f64, so this cast is lossy above
                // 2^53. It is spelled out rather than hidden behind a blanket
                // conversion so the loss is visible where it happens.
                file.create_dataset_from_array(name, &array.mapv(|v| v as f64), Some(options))?;
                Self::set_dataset_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("int64".to_string()),
                )?;
            }
            MatType::UInt8(array) => {
                file.create_dataset_from_array(name, array, Some(options))?;
                Self::set_dataset_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("uint8".to_string()),
                )?;
            }
            MatType::UInt16(array) => {
                file.create_dataset_from_array(name, array, Some(options))?;
                Self::set_dataset_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("uint16".to_string()),
                )?;
            }
            MatType::UInt32(array) => {
                file.create_dataset_from_array(name, array, Some(options))?;
                Self::set_dataset_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("uint32".to_string()),
                )?;
            }
            MatType::UInt64(array) => {
                // Lossy above 2^53, as for int64 above.
                file.create_dataset_from_array(name, &array.mapv(|v| v as f64), Some(options))?;
                Self::set_dataset_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("uint64".to_string()),
                )?;
            }
            MatType::Logical(array) => {
                let u8_array = array.mapv(|x| if x { 1u8 } else { 0u8 });
                file.create_dataset_from_array(name, &u8_array, Some(options))?;
                Self::set_dataset_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("logical".to_string()),
                )?;
            }
            MatType::Char(string) => {
                let utf16_data: Vec<u16> = string.encode_utf16().collect();
                let array = scirs2_core::ndarray::Array1::from_vec(utf16_data).into_dyn();
                file.create_dataset_from_array(name, &array, Some(options))?;
                Self::set_dataset_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("char".to_string()),
                )?;
            }
            MatType::Cell(cells) => {
                Self::create_nested_group(file, name)?;
                Self::set_group_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("cell".to_string()),
                )?;
                let dims = vec![cells.len() as i64];
                Self::set_group_attribute(
                    file,
                    name,
                    "MATLAB_dims",
                    AttributeValue::IntegerArray(dims),
                )?;
                for (i, cell_value) in cells.iter().enumerate() {
                    let cell_name = format!("{}/cell_{}", name, i);
                    self.write_standard_type(file, &cell_name, cell_value)?;
                }
                return Ok(());
            }
            MatType::Struct(fields) => {
                Self::create_nested_group(file, name)?;
                Self::set_group_attribute(
                    file,
                    name,
                    "MATLAB_class",
                    AttributeValue::String("struct".to_string()),
                )?;
                let field_names: Vec<String> = fields.keys().cloned().collect();
                Self::set_group_attribute(
                    file,
                    name,
                    "MATLAB_fields",
                    AttributeValue::StringArray(field_names),
                )?;
                for (field_name, field_value) in fields {
                    let field_path = format!("{}/{}", name, field_name);
                    self.write_standard_type(file, &field_path, field_value)?;
                }
                return Ok(());
            }
            MatType::SparseDouble(_) | MatType::SparseSingle(_) | MatType::SparseLogical(_) => {
                return Err(IoError::Other(
                    "Sparse matrix write via write_standard_type not supported in V73MatFile; \
                     use EnhancedMatFile for sparse data"
                        .to_string(),
                ));
            }
        }

        // Set MATLAB_int_decode on the dataset.
        Self::set_dataset_attribute(file, name, "MATLAB_int_decode", AttributeValue::Integer(2))?;
        Ok(())
    }

    /// Read an extended type from HDF5.
    ///
    /// Checks both group-level attributes (via `get_group_attribute`) and
    /// dataset-level attributes (via `get_dataset_attribute`) to determine
    /// the MATLAB type.
    fn read_extended_type(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        // Try group attribute first (groups store their attrs in the Group struct)
        let class_opt = Self::get_group_attribute(file, name, "MATLAB_class")
            .or_else(|| Self::get_dataset_attribute(file, name, "MATLAB_class"));

        if let Some(class_attr) = class_opt {
            match class_attr {
                AttributeValue::String(class_name) => {
                    let class_name = class_name.clone();
                    match class_name.as_str() {
                        "table" => self.read_table(file, name),
                        "categorical" => self.read_categorical(file, name),
                        "datetime" => self.read_datetime(file, name),
                        "string" => self.read_string_array(file, name),
                        "function_handle" => self.read_function_handle(file, name),
                        _ => {
                            // MATLAB_object is written as boolean but read back as integer
                            // because write_attribute_to_hdf5 converts Boolean to i64.
                            let is_object = Self::get_group_attribute(file, name, "MATLAB_object")
                                .map(|v| match v {
                                    AttributeValue::Boolean(b) => *b,
                                    AttributeValue::Integer(i) => *i != 0,
                                    _ => false,
                                })
                                .unwrap_or(false);
                            if is_object {
                                self.read_object(file, name)
                            } else {
                                let mat = self.read_standard_type(file, name)?;
                                Ok(ExtendedMatType::Standard(Box::new(mat)))
                            }
                        }
                    }
                }
                _ => Err(IoError::Other("Invalid MATLAB_class attribute".to_string())),
            }
        } else {
            Err(IoError::Other("Missing MATLAB_class attribute".to_string()))
        }
    }

    /// Read a standard `MatType` from an HDF5 path.
    ///
    /// Mirrors `EnhancedMatFile::read_mat_type_from_hdf5`. Supports the common
    /// scalar/array types. Nested cell/struct/sparse are forwarded to error with a
    /// clear message — they can be added when needed.
    fn read_standard_type(&self, file: &HDF5File, name: &str) -> Result<MatType> {
        if file.is_group(name) {
            let class = Self::get_group_attribute(file, name, "MATLAB_class")
                .and_then(|v| {
                    if let AttributeValue::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| IoError::Other(format!("No MATLAB_class on group '{}'", name)))?;

            match class.as_str() {
                "cell" => {
                    let mut cells = Vec::new();
                    if let Some(AttributeValue::IntegerArray(dims)) =
                        Self::get_group_attribute(file, name, "MATLAB_dims")
                    {
                        let num_cells = dims
                            .first()
                            .copied()
                            .ok_or_else(|| IoError::Other("Empty MATLAB_dims".to_string()))?
                            as usize;
                        for i in 0..num_cells {
                            let cell_name = format!("{}/cell_{}", name, i);
                            let cell_value = self.read_standard_type(file, &cell_name)?;
                            cells.push(cell_value);
                        }
                    }
                    Ok(MatType::Cell(cells))
                }
                "struct" => {
                    let mut fields = HashMap::new();
                    if let Some(AttributeValue::StringArray(field_names)) =
                        Self::get_group_attribute(file, name, "MATLAB_fields")
                    {
                        for field_name in field_names {
                            let field_path = format!("{}/{}", name, field_name);
                            let field_value = self.read_standard_type(file, &field_path)?;
                            fields.insert(field_name.clone(), field_value);
                        }
                    }
                    Ok(MatType::Struct(fields))
                }
                other => Err(IoError::Other(format!(
                    "Group MATLAB class '{}' not supported in read_standard_type",
                    other
                ))),
            }
        } else {
            // Dataset path — attributes live on the Dataset object, not on groups.
            // Use get_dataset_attribute to navigate correctly.
            let class = Self::get_dataset_attribute(file, name, "MATLAB_class")
                .and_then(|v| {
                    if let AttributeValue::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                // Also try the group-level path (works for top-level datasets
                // where set_attribute and get_attribute both navigate groups only).
                .or_else(|| {
                    file.get_attribute(name, "MATLAB_class")
                        .ok()
                        .flatten()
                        .and_then(|v| {
                            if let AttributeValue::String(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                })
                .unwrap_or_else(|| "double".to_string());

            match class.as_str() {
                "double" => {
                    let array = file.read_dataset_typed::<f64>(name)?;
                    Ok(MatType::Double(array))
                }
                "single" => {
                    let array = file.read_dataset_typed::<f32>(name)?;
                    Ok(MatType::Single(array))
                }
                "int8" => {
                    let array = file.read_dataset_typed::<i8>(name)?;
                    Ok(MatType::Int8(array))
                }
                "int16" => {
                    let array = file.read_dataset_typed::<i16>(name)?;
                    Ok(MatType::Int16(array))
                }
                "int32" => {
                    let array = file.read_dataset_typed::<i32>(name)?;
                    Ok(MatType::Int32(array))
                }
                "int64" => {
                    let array = file.read_dataset_typed::<i64>(name)?;
                    Ok(MatType::Int64(array))
                }
                "uint8" => {
                    let array = file.read_dataset_typed::<u8>(name)?;
                    Ok(MatType::UInt8(array))
                }
                "uint16" => {
                    let array = file.read_dataset_typed::<u16>(name)?;
                    Ok(MatType::UInt16(array))
                }
                "uint32" => {
                    let array = file.read_dataset_typed::<u32>(name)?;
                    Ok(MatType::UInt32(array))
                }
                "uint64" => {
                    let array = file.read_dataset_typed::<u64>(name)?;
                    Ok(MatType::UInt64(array))
                }
                "logical" => {
                    let array = file.read_dataset_typed::<u8>(name)?;
                    let bool_array = array.mapv(|x| x != 0);
                    Ok(MatType::Logical(bool_array))
                }
                "char" => {
                    // Stored as u16 UTF-16 values (via create_dataset_from_array → f64 internally)
                    let array = file.read_dataset_typed::<u16>(name)?;
                    let utf16_data: Vec<u16> = array.iter().copied().collect();
                    let string = String::from_utf16(&utf16_data)
                        .map_err(|_| IoError::Other("Invalid UTF-16 char data".to_string()))?;
                    Ok(MatType::Char(string))
                }
                other => Err(IoError::Other(format!(
                    "MATLAB class '{}' not supported in read_standard_type",
                    other
                ))),
            }
        }
    }

    /// Read a MATLAB table from HDF5.
    ///
    /// Expects the layout written by `write_table`.
    fn read_table(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        // Read variable names from group attribute
        let variable_names = match Self::get_group_attribute(file, name, "VariableNames") {
            Some(AttributeValue::StringArray(names)) => names.clone(),
            _ => {
                return Err(IoError::Other(format!(
                    "Table '{}' missing VariableNames attribute",
                    name
                )))
            }
        };

        // Read optional row names
        let row_names = match Self::get_group_attribute(file, name, "RowNames") {
            Some(AttributeValue::StringArray(rn)) => Some(rn.clone()),
            _ => None,
        };

        // Read each column
        let mut data = HashMap::new();
        for var_name in &variable_names {
            let var_path = format!("{}/{}", name, var_name);
            match self.read_standard_type(file, &var_path) {
                Ok(mat) => {
                    data.insert(var_name.clone(), mat);
                }
                Err(_) => {
                    // Column missing or unreadable — skip rather than fail the whole table
                }
            }
        }

        // Read table properties (attr keys with `property_` prefix)
        let mut properties = HashMap::new();
        // Navigate to the group and iterate its attributes directly.
        if let Ok(group) = file.get_group(name) {
            for (key, val) in &group.attributes {
                if let Some(prop_key) = key.strip_prefix("property_") {
                    if let AttributeValue::String(prop_val) = val {
                        properties.insert(prop_key.to_string(), prop_val.clone());
                    }
                }
            }
        }

        Ok(ExtendedMatType::Table(MatlabTable {
            variable_names,
            row_names,
            data,
            properties,
        }))
    }

    /// Read a MATLAB categorical array from HDF5.
    ///
    /// Expects the layout written by `write_categorical`.
    fn read_categorical(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        let categories = match Self::get_group_attribute(file, name, "Categories") {
            Some(AttributeValue::StringArray(cats)) => cats.clone(),
            _ => {
                return Err(IoError::Other(format!(
                    "Categorical '{}' missing Categories attribute",
                    name
                )))
            }
        };

        let ordered = match Self::get_group_attribute(file, name, "ordered") {
            Some(AttributeValue::Boolean(b)) => *b,
            Some(AttributeValue::Integer(i)) => *i != 0,
            _ => false,
        };

        // Read u32 data indices (stored as f64 internally by create_dataset_from_array, cast back)
        let data_path = format!("{}/data", name);
        let raw = file.read_dataset(&data_path)?;
        let data = raw.mapv(|v| v as u32);

        Ok(ExtendedMatType::Categorical(CategoricalArray {
            categories,
            data,
            ordered,
        }))
    }

    /// Read a MATLAB datetime array from HDF5.
    ///
    /// Expects the layout written by `write_datetime`.  Attributes are stored on
    /// the parent GROUP (`{name}/`), and the numeric data lives in the sub-dataset
    /// `{name}/data`.  Use `get_group_attribute` for attrs and
    /// `file.read_dataset("{name}/data")` for the data.
    fn read_datetime(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        // `write_datetime` stores numeric values in a sub-dataset `{name}/data`
        // because `write_dataset_to_hdf5` does not flush dataset-level attributes;
        // only group attributes are written to native HDF5.
        let data_path = format!("{}/data", name.trim_start_matches('/'));
        let data = file.read_dataset(&data_path)?;

        let timezone = Self::get_group_attribute(file, name, "timezone").and_then(|v| {
            if let AttributeValue::String(tz) = v {
                Some(tz.clone())
            } else {
                None
            }
        });

        let format = Self::get_group_attribute(file, name, "format")
            .and_then(|v| {
                if let AttributeValue::String(f) = v {
                    Some(f.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "yyyy-MM-dd HH:mm:ss".to_string());

        Ok(ExtendedMatType::DateTime(DateTimeArray {
            data,
            timezone,
            format,
        }))
    }

    /// Read a MATLAB string array from HDF5.
    ///
    /// Expects the layout written by `write_string_array` (one dataset per element,
    /// stored as u16 UTF-16 values cast through f64).
    fn read_string_array(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        let count = match Self::get_group_attribute(file, name, "size") {
            Some(AttributeValue::Array(arr)) => arr
                .first()
                .copied()
                .ok_or_else(|| IoError::Other("Empty 'size' attribute".to_string()))?
                as usize,
            Some(AttributeValue::IntegerArray(arr)) => arr
                .first()
                .copied()
                .ok_or_else(|| IoError::Other("Empty 'size' attribute".to_string()))?
                as usize,
            Some(AttributeValue::Integer(n)) => *n as usize,
            _ => {
                return Err(IoError::Other(format!(
                    "StringArray '{}' missing 'size' attribute",
                    name
                )))
            }
        };

        let mut strings = Vec::with_capacity(count);
        for i in 0..count {
            let ds_path = format!("{}/string_{}", name, i);
            // Each element is u16 values stored as f64 by the in-house helper.
            let raw = file.read_dataset(&ds_path)?;
            let utf16: Vec<u16> = raw.iter().map(|&v| v as u16).collect();
            let s = String::from_utf16(&utf16)
                .map_err(|_| IoError::Other(format!("Invalid UTF-16 in '{}'", ds_path)))?;
            strings.push(s);
        }

        Ok(ExtendedMatType::StringArray(strings))
    }

    /// Read a MATLAB function handle from HDF5.
    ///
    /// Expects the layout written by `write_function_handle`.
    fn read_function_handle(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        // Read function name (u16 UTF-16 values stored as f64)
        let func_path = format!("{}/function", name);
        let raw = file.read_dataset(&func_path)?;
        let utf16: Vec<u16> = raw.iter().map(|&v| v as u16).collect();
        let function = String::from_utf16(&utf16)
            .map_err(|_| IoError::Other("Invalid UTF-16 in function handle".to_string()))?;

        let function_type = match Self::get_group_attribute(file, name, "type") {
            Some(AttributeValue::String(t)) => t.clone(),
            _ => "simple".to_string(),
        };

        // Read workspace if present
        let ws_group_path = format!("{}/workspace", name);
        let workspace = if file.is_group(&ws_group_path) {
            let mut ws_map = HashMap::new();
            if let Ok(ws_group) = file.get_group(&ws_group_path) {
                // Collect names first to avoid borrow issues
                let ds_names: Vec<String> = ws_group
                    .dataset_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                let grp_names: Vec<String> = ws_group
                    .group_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect();

                for var_name in ds_names.iter().chain(grp_names.iter()) {
                    let var_path = format!("{}/{}", ws_group_path, var_name);
                    if let Ok(mat) = self.read_standard_type(file, &var_path) {
                        ws_map.insert(var_name.clone(), mat);
                    }
                }
            }
            if ws_map.is_empty() {
                None
            } else {
                Some(ws_map)
            }
        } else {
            None
        };

        Ok(ExtendedMatType::FunctionHandle(FunctionHandle {
            function,
            function_type,
            workspace,
        }))
    }

    /// Read a MATLAB object from HDF5.
    ///
    /// Expects the layout written by `write_object`.
    fn read_object(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        let class_name = match Self::get_group_attribute(file, name, "MATLAB_class") {
            Some(AttributeValue::String(c)) => c.clone(),
            _ => {
                return Err(IoError::Other(format!(
                    "Object '{}' missing MATLAB_class attribute",
                    name
                )))
            }
        };

        // Read property names from attribute (set by write_object)
        let prop_names: Vec<String> = match Self::get_group_attribute(file, name, "PropertyNames") {
            Some(AttributeValue::StringArray(names)) => names.clone(),
            _ => Vec::new(),
        };

        let props_group_path = format!("{}/properties", name);
        let mut properties = HashMap::new();
        for prop_name in &prop_names {
            let prop_path = format!("{}/{}", props_group_path, prop_name);
            if let Ok(mat) = self.read_standard_type(file, &prop_path) {
                properties.insert(prop_name.clone(), mat);
            }
        }

        // Read superclass if present
        let super_path = format!("{}/superclass", name);
        let superclass_data = if file.is_group(&super_path) {
            match self.read_object(file, &super_path)? {
                ExtendedMatType::Object(obj) => Some(Box::new(obj)),
                _ => None,
            }
        } else {
            None
        };

        Ok(ExtendedMatType::Object(MatlabObject {
            class_name,
            properties,
            superclass_data,
        }))
    }
}

/// Partial I/O support for large variables.
///
/// [`PartialIoSupport::read_array_slice`] and
/// [`PartialIoSupport::write_array_slice`] give hyperslab-style access to one
/// hyper-rectangular region of an `f64` dataset without disturbing the rest of
/// the file.
///
/// # Why the write does not rebuild the file
///
/// [`HDF5File::write`] serialises SciRS2's in-memory model from scratch, and
/// that model knows about groups, datasets and simple attributes — not about
/// `MATLAB_class` markers on object references, the `#refs#` group, cell arrays
/// or compound structs. Rebuilding a `.mat` file authored by MATLAB or h5py
/// would therefore silently drop all of it.
///
/// The write instead goes through [`oxih5::write_dataset_in_place_f64`], which
/// replaces a contiguous dataset's data bytes and touches nothing else. It
/// insists the supplied buffer match the dataset's allocated size exactly, and
/// that is what makes it safe: no byte count changes, so no address recorded
/// anywhere in the file can shift.
///
/// The previous implementation reached past this module into a raw `libhdf5`
/// handle to call `write_raw`, because the ordinary write path fails once the
/// dataset already exists. That handle no longer exists, and it is no longer
/// needed.
pub struct PartialIoSupport;

impl PartialIoSupport {
    /// Read a contiguous hyper-rectangular slice from a large f64 dataset.
    ///
    /// `start` and `count` must have the same length as the dataset's rank, and
    /// the region must lie inside it.
    ///
    /// Delegates to [`HDF5File::read_f64_dataset_slice`], which performs the
    /// same row-major coordinate walk this method used to spell out inline —
    /// with per-axis bounds checks the inline copy lacked, and without the
    /// off-by-one that made an empty region (any `count` of zero) push one
    /// element before failing to reshape.
    pub fn read_array_slice<P: AsRef<Path>>(
        path: P,
        var_name: &str,
        start: &[usize],
        count: &[usize],
    ) -> Result<ArrayD<f64>> {
        if start.len() != count.len() {
            return Err(IoError::Other(
                "read_array_slice: start and count must have the same length".to_string(),
            ));
        }

        let file = HDF5File::open(path, FileMode::ReadOnly)?;
        let values = file.read_f64_dataset_slice(var_name, count, start)?;

        ArrayD::from_shape_vec(IxDyn(count), values)
            .map_err(|e| IoError::FormatError(format!("Failed to reshape slice: {}", e)))
    }

    /// Write a contiguous hyper-rectangular slice into an existing f64 dataset.
    ///
    /// `start` must have the same length as the dataset's rank, and the dataset
    /// must already exist and be large enough to hold `data` at that offset.
    ///
    /// The dataset is read, the sub-region is patched in memory, and the whole
    /// dataset is written back over its own bytes. Every other byte of the file
    /// survives untouched — `MATLAB_class` attributes, object references, the
    /// `#refs#` group, cell arrays and compound structs included.
    ///
    /// # Errors
    ///
    /// [`IoError::UnsupportedFormat`] when the dataset cannot be overwritten at
    /// a fixed size: chunked, compact or virtual layouts, a filter pipeline
    /// (compressed data), and variable-length datatypes are all rejected before
    /// anything is read, so the file is never left half-written. Also
    /// [`IoError::Other`] for a rank mismatch, and whatever
    /// [`HDF5File::write_f64_dataset_slice`] reports for an out-of-range region.
    pub fn write_array_slice<P: AsRef<Path>>(
        path: P,
        var_name: &str,
        data: &ArrayD<f64>,
        start: &[usize],
    ) -> Result<()> {
        let path = path.as_ref();
        let count: Vec<usize> = data.shape().to_vec();

        if start.len() != count.len() {
            return Err(IoError::Other(
                "write_array_slice: start rank must match data rank".to_string(),
            ));
        }

        // oxih5 addresses datasets without a leading slash.
        let dataset_path = var_name.trim_start_matches('/');

        // Ask first, write later. `dataset_data_extent` applies every
        // precondition the overwrite applies except the length check, so an Ok
        // here means the write will be accepted — and an Err names the reason
        // before a single byte has been read.
        let extent = oxih5::dataset_data_extent(path, dataset_path).map_err(|e| {
            IoError::UnsupportedFormat(format!(
                "write_array_slice: dataset '{dataset_path}' cannot be overwritten in place: {e}. \
                 A fixed-size overwrite needs a contiguous, unfiltered, fixed-length dataset; \
                 chunked, compressed and variable-length datasets do not qualify."
            ))
        })?;

        // Read the dataset, patch the sub-region, and take the full result back.
        // `write_f64_dataset_slice` owns the coordinate walk and the bounds
        // checks; this method used to duplicate both.
        let mut file = HDF5File::open(path, FileMode::ReadOnly)?;

        // The bytes written back are f64-encoded, so anything else on disk would
        // be corrupted by them. A same-width integer dataset in particular passes
        // the byte-count invariant below, and would otherwise be caught only by
        // oxih5's type check with a far less specific message.
        let stored_dtype = file.get_dataset(var_name)?.dtype.clone();
        if !matches!(stored_dtype, crate::hdf5::HDF5DataType::Float { size: 8 }) {
            return Err(IoError::UnsupportedFormat(format!(
                "write_array_slice: dataset '{dataset_path}' stores {stored_dtype:?}, but an \
                 in-place slice write encodes f64; rewrite the whole dataset instead"
            )));
        }

        let patch = data.as_slice().ok_or_else(|| {
            IoError::Other("write_array_slice: patch array is not contiguous".to_string())
        })?;
        file.write_f64_dataset_slice(var_name, patch, &count, start)?;
        let patched_array = file.read_dataset(var_name)?;
        let patched = patched_array.as_slice().ok_or_else(|| {
            IoError::Other("write_array_slice: patched dataset is not contiguous".to_string())
        })?;

        // The size invariant is the whole basis of the operation, so it is
        // restated here rather than left to surface as a byte-count error from
        // deeper down.
        let required = extent.size as usize;
        let supplied = std::mem::size_of_val(patched);
        if supplied != required {
            return Err(IoError::UnsupportedFormat(format!(
                "write_array_slice: dataset '{dataset_path}' holds {required} bytes on disk but \
                 the patched data occupies {supplied}; an in-place overwrite must not change the \
                 size of the data area"
            )));
        }

        oxih5::write_dataset_in_place_f64(path, dataset_path, patched).map_err(|e| {
            IoError::FormatError(format!(
                "write_array_slice: failed to overwrite dataset '{dataset_path}': {e}"
            ))
        })
    }
}

#[cfg(test)]
#[path = "v73_enhanced_tests.rs"]
mod tests;
