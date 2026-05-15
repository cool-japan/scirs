//! Enhanced MATLAB v7.3+ format support
//!
//! This module provides comprehensive support for MATLAB v7.3+ files,
//! which are based on HDF5 format with MATLAB-specific conventions.

use crate::error::{IoError, Result};
use crate::matlab::MatType;
#[allow(unused_imports)]
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
    compression: Option<CompressionOptions>,
}

impl V73MatFile {
    /// Create a new v7.3 MAT file handler
    pub fn new(features: V73Features) -> Self {
        Self {
            features,
            #[cfg(feature = "hdf5")]
            compression: None,
        }
    }

    /// Set compression options
    #[cfg(feature = "hdf5")]
    pub fn with_compression(mut self, compression: CompressionOptions) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Write extended MATLAB types to v7.3 file
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
                file.create_dataset_from_array(name, array, Some(options))?;
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
                file.create_dataset_from_array(name, array, Some(options))?;
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
    #[cfg(feature = "hdf5")]
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
/// `read_array_slice` and `write_array_slice` provide hyperslab-style access
/// using the in-house `HDF5File::read_dataset` / `create_dataset_from_array`
/// API.  True low-level HDF5 hyperslab selection is avoided here because the
/// hdf5-rust crate uses ndarray 0.15 while scirs2_core uses ndarray 0.17 —
/// mixing the two versions in the same call chain produces type-incompatibility
/// errors.  Instead we use the in-house module's `read_dataset` (which already
/// handles native-file delegation) and perform the slice extraction in plain
/// Rust.  For writes we use a read-modify-write pattern via `create_dataset_from_array`.
///
/// A comment documents this limitation so full HDF5 hyperslab support can be
/// wired in when both crates align on the same ndarray version.
pub struct PartialIoSupport;

impl PartialIoSupport {
    /// Read a contiguous hyper-rectangular slice from a large f64 dataset.
    ///
    /// `start` and `count` must have the same length as the dataset's rank.
    ///
    /// Implementation note: reads the entire dataset via the in-house wrapper
    /// and extracts the requested region in Rust.  True HDF5 hyperslab selection
    /// (reading only the needed bytes from disk) will be unlocked once
    /// hdf5-rust and scirs2_core share the same ndarray version.
    #[cfg(feature = "hdf5")]
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
        // `read_dataset` uses the native HDF5 handle when available,
        // falling back to the in-memory representation otherwise.
        let full = file.read_dataset(var_name)?;
        let full_shape = full.shape().to_vec();
        let ndim = full_shape.len();

        if start.len() != ndim {
            return Err(IoError::Other(format!(
                "read_array_slice: start/count rank {} does not match dataset rank {}",
                start.len(),
                ndim
            )));
        }

        // Extract the hyper-rectangular region element by element using flat indices.
        let total: usize = count.iter().product();
        let mut result = Vec::with_capacity(total);

        // Stride vector for the full array in C order.
        let mut strides = vec![1usize; ndim];
        for ax in (0..ndim.saturating_sub(1)).rev() {
            strides[ax] = strides[ax + 1] * full_shape[ax + 1];
        }

        // Iterate over all multi-indices in [0..count[0]) x ... x [0..count[ndim-1])
        let mut coords = vec![0usize; ndim];
        let full_flat = full
            .as_slice()
            .ok_or_else(|| IoError::Other("Dataset not contiguous".to_string()))?;

        loop {
            // Compute flat index in the full array.
            let flat_idx: usize = coords
                .iter()
                .enumerate()
                .map(|(ax, &c)| (start[ax] + c) * strides[ax])
                .sum();
            result.push(
                *full_flat
                    .get(flat_idx)
                    .ok_or_else(|| IoError::Other("Slice out of bounds".to_string()))?,
            );

            // Increment coords in C order (last axis fastest).
            let mut carry = true;
            for ax in (0..ndim).rev() {
                if carry {
                    coords[ax] += 1;
                    if coords[ax] < count[ax] {
                        carry = false;
                    } else {
                        coords[ax] = 0;
                    }
                }
            }
            if carry {
                break; // All indices exhausted.
            }
        }

        ArrayD::from_shape_vec(IxDyn(count), result)
            .map_err(|e| IoError::FormatError(format!("Failed to reshape slice: {}", e)))
    }

    /// Write a contiguous hyper-rectangular slice into an existing f64 dataset.
    ///
    /// `start` must have the same length as the dataset's rank.
    /// The dataset must already exist and be large enough to contain
    /// `data` at the given offset.
    ///
    /// Implementation note: uses read-modify-write via the in-house HDF5 wrapper.
    /// The full dataset is read, the slice region is patched in Rust, and the
    /// modified array is written back.  This is correct but not bandwidth-optimal;
    /// see `read_array_slice` for the reason true HDF5 hyperslab is deferred.
    #[cfg(feature = "hdf5")]
    pub fn write_array_slice<P: AsRef<Path>>(
        path: P,
        var_name: &str,
        data: &ArrayD<f64>,
        start: &[usize],
    ) -> Result<()> {
        let count: Vec<usize> = data.shape().to_vec();
        let ndim = start.len();

        if ndim != count.len() {
            return Err(IoError::Other(
                "write_array_slice: start rank must match data rank".to_string(),
            ));
        }

        // Step 1: read the full dataset.
        let file_ro = HDF5File::open(path.as_ref(), FileMode::ReadOnly)?;
        let full = file_ro.read_dataset(var_name)?;
        let full_shape = full.shape().to_vec();
        drop(file_ro);

        if full_shape.len() != ndim {
            return Err(IoError::Other(format!(
                "write_array_slice: start rank {} does not match dataset rank {}",
                ndim,
                full_shape.len()
            )));
        }

        // Step 2: patch in memory using flat-index arithmetic (C order).
        let mut full_flat: Vec<f64> = full.into_raw_vec_and_offset().0;
        let mut strides = vec![1usize; ndim];
        for ax in (0..ndim.saturating_sub(1)).rev() {
            strides[ax] = strides[ax + 1] * full_shape[ax + 1];
        }

        let patch_flat: Vec<f64> = data.iter().copied().collect();
        let mut coords = vec![0usize; ndim];
        for &val in &patch_flat {
            let flat_idx: usize = coords
                .iter()
                .enumerate()
                .map(|(ax, &c)| (start[ax] + c) * strides[ax])
                .sum();
            if let Some(elem) = full_flat.get_mut(flat_idx) {
                *elem = val;
            } else {
                return Err(IoError::Other(
                    "write_array_slice: patch out of bounds".to_string(),
                ));
            }
            // Increment coords.
            let mut carry = true;
            for ax in (0..ndim).rev() {
                if carry {
                    coords[ax] += 1;
                    if coords[ax] < count[ax] {
                        carry = false;
                    } else {
                        coords[ax] = 0;
                    }
                }
            }
        }

        // Step 3: write the patched array back via native HDF5 in-place overwrite.
        //
        // HDF5File::write() → write_group_to_hdf5 → new_dataset().create(path) fails
        // when the dataset already exists.  Instead, open the existing dataset with
        // the native handle and write_raw to overwrite its data without recreating it.
        let file_rw = HDF5File::open(path.as_ref(), FileMode::ReadWrite)?;

        if let Some(nf) = file_rw.native_file() {
            // var_name should be a bare name (no leading slash) for top-level datasets.
            let clean_name = var_name.trim_start_matches('/');
            let h5_ds = nf.dataset(clean_name).map_err(|e| {
                IoError::FormatError(format!(
                    "write_array_slice: cannot open dataset '{}': {}",
                    clean_name, e
                ))
            })?;
            h5_ds.write_raw(&full_flat).map_err(|e| {
                IoError::FormatError(format!(
                    "write_array_slice: failed to write dataset '{}': {}",
                    clean_name, e
                ))
            })?;
        } else {
            return Err(IoError::Other(
                "write_array_slice: native HDF5 handle not available".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
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
    #[cfg(all(test, feature = "hdf5"))]
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
    #[cfg(all(test, feature = "hdf5"))]
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
    #[cfg(all(test, feature = "hdf5"))]
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
    #[cfg(all(test, feature = "hdf5"))]
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
    #[cfg(all(test, feature = "hdf5"))]
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
    #[cfg(all(test, feature = "hdf5"))]
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
    #[cfg(all(test, feature = "hdf5"))]
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
    #[cfg(all(test, feature = "hdf5"))]
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
}
