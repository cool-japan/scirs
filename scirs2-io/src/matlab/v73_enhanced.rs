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
            ExtendedMatType::Standard(mat_type) => self.write_standard_type(file, name, &mat_type),
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

    /// Write a MATLAB table
    #[cfg(feature = "hdf5")]
    fn write_table(&self, file: &mut HDF5File, name: &str, table: &MatlabTable) -> Result<()> {
        // Create a group for the table
        file.create_group(name)?;
        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String("table".to_string()),
        )?;

        // Write variable names
        let var_names_data: Vec<u16> = table
            .variable_names
            .iter()
            .flat_map(|s| s.encode_utf16())
            .collect();
        // Convert Vec to ArrayD and use create_dataset_from_array
        let var_names_array = scirs2_core::ndarray::Array1::from_vec(var_names_data).into_dyn();
        file.create_dataset_from_array(
            &format!("{}/varnames", name),
            &var_names_array,
            Some(DatasetOptions::default()),
        )?;

        // Write table data
        for (var_name, var_data) in &table.data {
            let var_path = format!("{}/{}", name, var_name);
            self.write_standard_type(file, &var_path, var_data)?;
        }

        // Write row names if present
        if let Some(ref row_names) = table.row_names {
            let row_names_data: Vec<u16> =
                row_names.iter().flat_map(|s| s.encode_utf16()).collect();
            let row_names_array = scirs2_core::ndarray::Array1::from_vec(row_names_data).into_dyn();
            file.create_dataset_from_array(
                &format!("{}/rownames", name),
                &row_names_array,
                Some(DatasetOptions::default()),
            )?;
        }

        // Write properties
        for (prop_name, prop_value) in &table.properties {
            file.set_attribute(
                name,
                &format!("property_{}", prop_name),
                AttributeValue::String(prop_value.clone()),
            )?;
        }

        Ok(())
    }

    /// Write a categorical array
    #[cfg(feature = "hdf5")]
    fn write_categorical(
        &self,
        file: &mut HDF5File,
        name: &str,
        cat_array: &CategoricalArray,
    ) -> Result<()> {
        // Create a group for the categorical _array
        file.create_group(name)?;
        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String("categorical".to_string()),
        )?;

        // Write categories
        let cats_data: Vec<u16> = cat_array
            .categories
            .iter()
            .flat_map(|s| s.encode_utf16())
            .collect();
        let cats_array = scirs2_core::ndarray::Array1::from_vec(cats_data).into_dyn();
        file.create_dataset_from_array(
            &format!("{}/categories", name),
            &cats_array,
            Some(DatasetOptions::default()),
        )?;

        // Write data indices
        file.create_dataset_from_array(
            &format!("{}/data", name),
            &cat_array.data,
            Some(DatasetOptions::default()),
        )?;

        // Write ordered flag
        file.set_attribute(name, "ordered", AttributeValue::Boolean(cat_array.ordered))?;

        Ok(())
    }

    /// Write a datetime array
    #[cfg(feature = "hdf5")]
    fn write_datetime(
        &self,
        file: &mut HDF5File,
        name: &str,
        dt_array: &DateTimeArray,
    ) -> Result<()> {
        // Create dataset for datetime data
        file.create_dataset_from_array(
            name,
            &dt_array.data,
            Some(DatasetOptions {
                compression: self.compression.clone().unwrap_or_default(),
                ..Default::default()
            }),
        )?;

        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String("datetime".to_string()),
        )?;

        // Write timezone if present
        if let Some(ref tz) = dt_array.timezone {
            file.set_attribute(name, "timezone", AttributeValue::String(tz.clone()))?;
        }

        // Write format
        file.set_attribute(
            name,
            "format",
            AttributeValue::String(dt_array.format.clone()),
        )?;

        Ok(())
    }

    /// Write a string array
    #[cfg(feature = "hdf5")]
    fn write_string_array(
        &self,
        file: &mut HDF5File,
        name: &str,
        strings: &[String],
    ) -> Result<()> {
        // Create a group for the string array
        file.create_group(name)?;
        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String("string".to_string()),
        )?;

        // Write each string as a separate dataset
        for (i, string) in strings.iter().enumerate() {
            let string_data: Vec<u16> = string.encode_utf16().collect();
            let string_array = scirs2_core::ndarray::Array1::from_vec(string_data).into_dyn();
            file.create_dataset_from_array(
                &format!("{}/string_{}", name, i),
                &string_array,
                Some(DatasetOptions::default()),
            )?;
        }

        // Write array size
        file.set_attribute(
            name,
            "size",
            AttributeValue::Array(vec![strings.len() as i64]),
        )?;

        Ok(())
    }

    /// Write a function handle
    #[cfg(feature = "hdf5")]
    fn write_function_handle(
        &self,
        file: &mut HDF5File,
        name: &str,
        func_handle: &FunctionHandle,
    ) -> Result<()> {
        // Create a group for the function _handle
        file.create_group(name)?;
        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String("function_handle".to_string()),
        )?;

        // Write function string
        let func_data: Vec<u16> = func_handle.function.encode_utf16().collect();
        let func_array = scirs2_core::ndarray::Array1::from_vec(func_data).into_dyn();
        file.create_dataset_from_array(
            &format!("{}/function", name),
            &func_array,
            Some(DatasetOptions::default()),
        )?;

        // Write function type
        file.set_attribute(
            name,
            "type",
            AttributeValue::String(func_handle.function_type.clone()),
        )?;

        // Write workspace if present
        if let Some(ref workspace) = func_handle.workspace {
            let ws_group = format!("{}/workspace", name);
            file.create_group(&ws_group)?;

            for (var_name, var_data) in workspace {
                let var_path = format!("{}/{}", ws_group, var_name);
                self.write_standard_type(file, &var_path, var_data)?;
            }
        }

        Ok(())
    }

    /// Write a MATLAB object
    #[cfg(feature = "hdf5")]
    fn write_object(&self, file: &mut HDF5File, name: &str, object: &MatlabObject) -> Result<()> {
        // Create a group for the object
        file.create_group(name)?;
        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String(object.class_name.clone()),
        )?;
        file.set_attribute(name, "MATLAB_object", AttributeValue::Boolean(true))?;

        // Write properties
        let props_group = format!("{}/properties", name);
        file.create_group(&props_group)?;

        for (prop_name, prop_data) in &object.properties {
            let prop_path = format!("{}/{}", props_group, prop_name);
            self.write_standard_type(file, &prop_path, prop_data)?;
        }

        // Write superclass data if present
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
        // Split into real and imaginary parts
        let real_part = array.mapv(|x| x.re);
        let imag_part = array.mapv(|x| x.im);

        // Create a group for the _complex array
        file.create_group(name)?;
        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String("double".to_string()),
        )?;
        file.set_attribute(name, "MATLAB_complex", AttributeValue::Boolean(true))?;

        // Write real and imaginary parts
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
        // Split into real and imaginary parts
        let real_part = array.mapv(|x| x.re);
        let imag_part = array.mapv(|x| x.im);

        // Create a group for the _complex array
        file.create_group(name)?;
        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String("single".to_string()),
        )?;
        file.set_attribute(name, "MATLAB_complex", AttributeValue::Boolean(true))?;

        // Write real and imaginary parts
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

    /// Write standard MatType (helper)
    #[cfg(feature = "hdf5")]
    fn write_standard_type(
        &self,
        file: &mut HDF5File,
        name: &str,
        mat_type: &MatType,
    ) -> Result<()> {
        // Delegate to existing implementation
        // This would use the existing write_mat_type_to_hdf5 logic
        Err(IoError::Other("Not implemented yet".to_string()))
    }

    /// Read an extended type from HDF5
    #[cfg(feature = "hdf5")]
    fn read_extended_type(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        // Check MATLAB_class attribute to determine type
        if let Ok(Some(class_attr)) = file.get_attribute(name, "MATLAB_class") {
            match class_attr {
                AttributeValue::String(class_name) => {
                    match class_name.as_str() {
                        "table" => self.read_table(file, name),
                        "categorical" => self.read_categorical(file, name),
                        "datetime" => self.read_datetime(file, name),
                        "string" => self.read_string_array(file, name),
                        "function_handle" => self.read_function_handle(file, name),
                        _ => {
                            // Check if it's an object
                            if let Ok(Some(AttributeValue::Boolean(true))) =
                                file.get_attribute(name, "MATLAB_object")
                            {
                                self.read_object(file, name)
                            } else {
                                // Try to read as standard type
                                Err(IoError::Other(
                                    "Standard type reading not implemented".to_string(),
                                ))
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

    // Read implementations would follow similar patterns...
    #[cfg(feature = "hdf5")]
    fn read_table(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        Err(IoError::Other(
            "Table reading not implemented yet".to_string(),
        ))
    }

    #[cfg(feature = "hdf5")]
    fn read_categorical(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        Err(IoError::Other(
            "Categorical reading not implemented yet".to_string(),
        ))
    }

    #[cfg(feature = "hdf5")]
    fn read_datetime(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        Err(IoError::Other(
            "DateTime reading not implemented yet".to_string(),
        ))
    }

    #[cfg(feature = "hdf5")]
    fn read_string_array(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        Err(IoError::Other(
            "String array reading not implemented yet".to_string(),
        ))
    }

    #[cfg(feature = "hdf5")]
    fn read_function_handle(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        Err(IoError::Other(
            "Function handle reading not implemented yet".to_string(),
        ))
    }

    #[cfg(feature = "hdf5")]
    fn read_object(&self, file: &HDF5File, name: &str) -> Result<ExtendedMatType> {
        Err(IoError::Other(
            "Object reading not implemented yet".to_string(),
        ))
    }
}

/// Partial I/O support for large variables
pub struct PartialIoSupport;

impl PartialIoSupport {
    /// Read a slice of a large array without loading the entire array
    #[cfg(feature = "hdf5")]
    pub fn read_array_slice<T, P: AsRef<Path>>(
        path: P,
        var_name: &str,
        start: &[usize],
        count: &[usize],
    ) -> Result<ArrayD<T>>
    where
        T: Default + Clone,
    {
        Err(IoError::Other(
            "Partial I/O not implemented yet".to_string(),
        ))
    }

    /// Write to a slice of an existing array
    #[cfg(feature = "hdf5")]
    pub fn write_array_slice<T, P: AsRef<Path>>(
        path: P,
        var_name: &str,
        data: &ArrayD<T>,
        start: &[usize],
    ) -> Result<()>
    where
        T: Default + Clone,
    {
        Err(IoError::Other(
            "Partial I/O not implemented yet".to_string(),
        ))
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

        // Add some data
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
}
