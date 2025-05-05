//! NetCDF file format support
//!
//! This module provides functionality for reading and writing NetCDF files,
//! which are commonly used for storing array-oriented scientific data.
//!
//! NetCDF (Network Common Data Form) is a set of software libraries and
//! machine-independent data formats that support the creation, access, and
//! sharing of array-oriented scientific data.
//!
//! This implementation supports NetCDF3 Classic format and provides:
//! - Reading and writing NetCDF files
//! - Support for dimensions, variables, and attributes
//! - Conversion between NetCDF and ndarray data structures
//! - Memory-efficient access to large datasets

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use ndarray::{Array, ArrayD, Dimension, IxDyn};
use netcdf3::{File as NC3File, Variable, DataType};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use scirs2_core::error::ScirsCoreError;
use crate::error::{IOError, Result};

/// NetCDF file containing dimensions, variables, and attributes
#[derive(Debug)]
pub struct NetCDFFile {
    /// NetCDF file object
    file: NC3File,
    /// File path
    path: String,
    /// File mode ('r' for read, 'w' for write)
    mode: String,
}

/// NetCDF data type mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetCDFDataType {
    /// Byte (8 bits)
    Byte,
    /// Character (8 bits)
    Char,
    /// Short integer (16 bits)
    Short,
    /// Integer (32 bits)
    Int,
    /// Float (32 bits)
    Float,
    /// Double (64 bits)
    Double,
}

impl From<NetCDFDataType> for DataType {
    fn from(dtype: NetCDFDataType) -> Self {
        match dtype {
            NetCDFDataType::Byte => DataType::Byte,
            NetCDFDataType::Char => DataType::Char,
            NetCDFDataType::Short => DataType::Short,
            NetCDFDataType::Int => DataType::Int,
            NetCDFDataType::Float => DataType::Float,
            NetCDFDataType::Double => DataType::Double,
        }
    }
}

impl From<DataType> for NetCDFDataType {
    fn from(dtype: DataType) -> Self {
        match dtype {
            DataType::Byte => NetCDFDataType::Byte,
            DataType::Char => NetCDFDataType::Char,
            DataType::Short => NetCDFDataType::Short,
            DataType::Int => NetCDFDataType::Int,
            DataType::Float => NetCDFDataType::Float,
            DataType::Double => NetCDFDataType::Double,
        }
    }
}

/// Options for opening a NetCDF file
#[derive(Debug, Clone)]
pub struct NetCDFOptions {
    /// Memory mapping enabled (for read operations)
    pub mmap: bool,
    /// Automatically scale variables based on scale_factor and add_offset attributes
    pub auto_scale: bool,
    /// Automatically mask missing values
    pub mask_and_scale: bool,
    /// File mode
    pub mode: String,
}

impl Default for NetCDFOptions {
    fn default() -> Self {
        Self {
            mmap: true,
            auto_scale: true,
            mask_and_scale: true,
            mode: "r".to_string(),
        }
    }
}

impl NetCDFFile {
    /// Open a NetCDF file
    ///
    /// # Arguments
    ///
    /// * `filename` - Path to the NetCDF file
    /// * `options` - Optional NetCDF options
    ///
    /// # Returns
    ///
    /// * `Result<NetCDFFile>` - The opened NetCDF file or an error
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use scirs2_io::netcdf::NetCDFFile;
    ///
    /// // Open a NetCDF file for reading
    /// let nc = NetCDFFile::open("data.nc", None).unwrap();
    ///
    /// // List the dimensions
    /// println!("Dimensions: {:?}", nc.dimensions());
    ///
    /// // List the variables
    /// println!("Variables: {:?}", nc.variables());
    /// ```
    pub fn open<P: AsRef<Path>>(filename: P, options: Option<NetCDFOptions>) -> Result<Self> {
        let opts = options.unwrap_or_default();
        let path_str = filename.as_ref().to_string_lossy().to_string();
        
        let file = match opts.mode.as_str() {
            "r" => {
                // Open for reading
                NC3File::open(filename).map_err(|e| IOError::FileOpenError(
                    format!("Failed to open NetCDF file '{}': {}", path_str, e)))?
            },
            "w" => {
                // Open for writing (create new file)
                NC3File::create(filename).map_err(|e| IOError::FileCreateError(
                    format!("Failed to create NetCDF file '{}': {}", path_str, e)))?
            },
            "a" => {
                // Open for appending
                // If file exists, open for reading and writing
                // If file doesn't exist, create it
                if filename.as_ref().exists() {
                    NC3File::open(filename).map_err(|e| IOError::FileOpenError(
                        format!("Failed to open NetCDF file '{}': {}", path_str, e)))?
                } else {
                    NC3File::create(filename).map_err(|e| IOError::FileCreateError(
                        format!("Failed to create NetCDF file '{}': {}", path_str, e)))?
                }
            },
            _ => {
                return Err(IOError::InvalidArgument(
                    format!("Invalid NetCDF file mode: {}", opts.mode)));
            }
        };
        
        Ok(NetCDFFile {
            file,
            path: path_str,
            mode: opts.mode,
        })
    }
    
    /// List all dimensions in the NetCDF file
    ///
    /// # Returns
    ///
    /// * `HashMap<String, Option<usize>>` - Map of dimension names to sizes (None for unlimited dimensions)
    pub fn dimensions(&self) -> HashMap<String, Option<usize>> {
        self.file.dimensions()
            .map(|(name, len)| {
                let size = if len.is_some() {
                    Some(len.unwrap() as usize)
                } else {
                    None
                };
                (name.to_string(), size)
            })
            .collect()
    }
    
    /// List all variables in the NetCDF file
    ///
    /// # Returns
    ///
    /// * `Vec<String>` - List of variable names
    pub fn variables(&self) -> Vec<String> {
        self.file.variables()
            .map(|(name, _)| name.to_string())
            .collect()
    }
    
    /// Get information about a variable
    ///
    /// # Arguments
    ///
    /// * `name` - Variable name
    ///
    /// # Returns
    ///
    /// * `Result<(NetCDFDataType, Vec<String>, HashMap<String, String>)>` - Tuple of (data type, dimensions, attributes)
    pub fn variable_info(&self, name: &str) -> Result<(NetCDFDataType, Vec<String>, HashMap<String, String>)> {
        let var = self.file.variable(name)
            .ok_or_else(|| IOError::KeyError(format!("Variable '{}' not found", name)))?;
        
        let data_type = var.data_type().into();
        let dimensions = var.dimensions()
            .map(|dim| dim.to_string())
            .collect();
        
        let mut attributes = HashMap::new();
        for (attr_name, attr_value) in var.attributes() {
            let value = format!("{:?}", attr_value);
            attributes.insert(attr_name.to_string(), value);
        }
        
        Ok((data_type, dimensions, attributes))
    }
    
    /// Read a variable as an ndarray Array
    ///
    /// # Arguments
    ///
    /// * `name` - Variable name
    ///
    /// # Returns
    ///
    /// * `Result<ArrayD<T>>` - Array containing the variable data
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use scirs2_io::netcdf::NetCDFFile;
    ///
    /// // Open a NetCDF file for reading
    /// let nc = NetCDFFile::open("data.nc", None).unwrap();
    ///
    /// // Read a variable
    /// let temp: ndarray::ArrayD<f32> = nc.read_variable("temperature").unwrap();
    /// println!("Temperature data shape: {:?}", temp.shape());
    /// ```
    pub fn read_variable<T: netcdf3::FromNetcdf3 + Clone>(&self, name: &str) -> Result<ArrayD<T>> {
        let var = self.file.variable(name)
            .ok_or_else(|| IOError::KeyError(format!("Variable '{}' not found", name)))?;
        
        // Get variable data
        let data = var.values::<T>()
            .map_err(|e| IOError::DataReadError(format!("Failed to read variable '{}': {}", name, e)))?;
        
        // Get dimensions
        let shape: Vec<usize> = var.dimensions()
            .map(|dim| {
                let dim_info = self.file.dimension(dim).unwrap();
                dim_info.len().unwrap_or(0) as usize
            })
            .collect();
        
        // Create ndarray from data and shape
        let array = Array::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| IOError::ConversionError(format!("Failed to reshape variable data: {}", e)))?;
        
        Ok(array)
    }
    
    /// Create a new dimension
    ///
    /// # Arguments
    ///
    /// * `name` - Dimension name
    /// * `length` - Dimension length (None for unlimited dimension)
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or error
    pub fn create_dimension(&self, name: &str, length: Option<usize>) -> Result<()> {
        if self.mode == "r" {
            return Err(IOError::PermissionError("File opened in read-only mode".to_string()));
        }
        
        // Convert usize to i64 for netcdf3 library
        let len = length.map(|l| l as i64);
        
        self.file.add_dimension(name, len)
            .map_err(|e| IOError::IOOperationError(format!("Failed to create dimension '{}': {}", name, e)))?;
        
        Ok(())
    }
    
    /// Create a new variable
    ///
    /// # Arguments
    ///
    /// * `name` - Variable name
    /// * `data_type` - Variable data type
    /// * `dimensions` - Variable dimensions
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or error
    pub fn create_variable(&self, name: &str, data_type: NetCDFDataType, dimensions: &[&str]) -> Result<()> {
        if self.mode == "r" {
            return Err(IOError::PermissionError("File opened in read-only mode".to_string()));
        }
        
        let nc_data_type: DataType = data_type.into();
        
        self.file.add_variable(name, nc_data_type, dimensions)
            .map_err(|e| IOError::IOOperationError(format!("Failed to create variable '{}': {}", name, e)))?;
        
        Ok(())
    }
    
    /// Write data to a variable
    ///
    /// # Arguments
    ///
    /// * `name` - Variable name
    /// * `data` - Data to write
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or error
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ndarray::Array;
    /// use scirs2_io::netcdf::{NetCDFFile, NetCDFOptions, NetCDFDataType};
    ///
    /// // Create a new NetCDF file
    /// let opts = NetCDFOptions {
    ///     mode: "w".to_string(),
    ///     ..Default::default()
    /// };
    /// let nc = NetCDFFile::open("new_data.nc", Some(opts)).unwrap();
    ///
    /// // Create dimensions
    /// nc.create_dimension("x", Some(10)).unwrap();
    /// nc.create_dimension("y", Some(5)).unwrap();
    ///
    /// // Create a variable
    /// nc.create_variable("temperature", NetCDFDataType::Float, &["x", "y"]).unwrap();
    ///
    /// // Create some data
    /// let data = Array::from_elem((10, 5), 25.0f32);
    ///
    /// // Write the data
    /// nc.write_variable("temperature", &data).unwrap();
    ///
    /// // Close the file
    /// nc.close().unwrap();
    /// ```
    pub fn write_variable<T: netcdf3::IntoNetcdf3 + Clone, D: Dimension>(&self, name: &str, data: &Array<T, D>) -> Result<()> {
        if self.mode == "r" {
            return Err(IOError::PermissionError("File opened in read-only mode".to_string()));
        }
        
        let var = self.file.variable(name)
            .ok_or_else(|| IOError::KeyError(format!("Variable '{}' not found", name)))?;
        
        // Convert ndarray to flat vector
        let flat_data = data.iter().cloned().collect::<Vec<T>>();
        
        var.put_values(&flat_data)
            .map_err(|e| IOError::DataWriteError(format!("Failed to write to variable '{}': {}", name, e)))?;
        
        Ok(())
    }
    
    /// Add an attribute to a variable
    ///
    /// # Arguments
    ///
    /// * `var_name` - Variable name
    /// * `attr_name` - Attribute name
    /// * `value` - Attribute value
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or error
    pub fn add_variable_attribute<T: netcdf3::IntoNetcdf3 + Clone>(&self, var_name: &str, attr_name: &str, value: T) -> Result<()> {
        if self.mode == "r" {
            return Err(IOError::PermissionError("File opened in read-only mode".to_string()));
        }
        
        let var = self.file.variable(var_name)
            .ok_or_else(|| IOError::KeyError(format!("Variable '{}' not found", var_name)))?;
        
        var.add_attribute(attr_name, value)
            .map_err(|e| IOError::IOOperationError(
                format!("Failed to add attribute '{}' to variable '{}': {}", attr_name, var_name, e)))?;
        
        Ok(())
    }
    
    /// Add a global attribute
    ///
    /// # Arguments
    ///
    /// * `name` - Attribute name
    /// * `value` - Attribute value
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or error
    pub fn add_global_attribute<T: netcdf3::IntoNetcdf3 + Clone>(&self, name: &str, value: T) -> Result<()> {
        if self.mode == "r" {
            return Err(IOError::PermissionError("File opened in read-only mode".to_string()));
        }
        
        self.file.add_attribute(name, value)
            .map_err(|e| IOError::IOOperationError(
                format!("Failed to add global attribute '{}': {}", name, e)))?;
        
        Ok(())
    }
    
    /// Get all global attributes
    ///
    /// # Returns
    ///
    /// * `HashMap<String, String>` - Map of attribute names to string representations of values
    pub fn global_attributes(&self) -> HashMap<String, String> {
        self.file.attributes()
            .map(|(name, value)| (name.to_string(), format!("{:?}", value)))
            .collect()
    }
    
    /// Sync any changes to disk
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or error
    pub fn sync(&self) -> Result<()> {
        if self.mode == "r" {
            return Ok(());
        }
        
        self.file.sync()
            .map_err(|e| IOError::IOOperationError(format!("Failed to sync NetCDF file: {}", e)))?;
        
        Ok(())
    }
    
    /// Close the NetCDF file
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or error
    pub fn close(&self) -> Result<()> {
        // The netcdf3 library automatically closes the file when it's dropped,
        // but we'll provide a sync method for consistency
        self.sync()
    }
}

/// Helper functions for converting between NetCDF and Rust types
mod convert {
    use super::*;
    use netcdf3::DataValue;
    
    /// Convert a NetCDF data value to a string representation
    pub fn data_value_to_string(value: &DataValue) -> String {
        match value {
            DataValue::Byte(v) => v.to_string(),
            DataValue::Char(v) => format!("{}", *v as char),
            DataValue::Short(v) => v.to_string(),
            DataValue::Int(v) => v.to_string(),
            DataValue::Float(v) => v.to_string(),
            DataValue::Double(v) => v.to_string(),
            DataValue::ByteVec(v) => format!("{:?}", v),
            DataValue::CharVec(v) => {
                if let Ok(s) = std::str::from_utf8(v) {
                    s.trim_end_matches('\0').to_string()
                } else {
                    format!("{:?}", v)
                }
            },
            DataValue::ShortVec(v) => format!("{:?}", v),
            DataValue::IntVec(v) => format!("{:?}", v),
            DataValue::FloatVec(v) => format!("{:?}", v),
            DataValue::DoubleVec(v) => format!("{:?}", v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use tempfile::tempdir;
    
    #[test]
    fn test_create_read_netcdf() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.nc");
        
        // Create a new NetCDF file
        let opts = NetCDFOptions {
            mode: "w".to_string(),
            ..Default::default()
        };
        let nc = NetCDFFile::open(&file_path, Some(opts)).unwrap();
        
        // Create dimensions
        nc.create_dimension("x", Some(3)).unwrap();
        nc.create_dimension("y", Some(2)).unwrap();
        
        // Create variables
        nc.create_variable("temperature", NetCDFDataType::Float, &["x", "y"]).unwrap();
        
        // Write data
        let data = Array2::from_shape_vec((3, 2), vec![20.0f32, 21.0, 22.0, 23.0, 24.0, 25.0]).unwrap();
        nc.write_variable("temperature", &data).unwrap();
        
        // Add attributes
        nc.add_variable_attribute("temperature", "units", "Celsius").unwrap();
        nc.add_global_attribute("title", "Test Data").unwrap();
        
        // Close file
        nc.close().unwrap();
        
        // Reopen for reading
        let nc_read = NetCDFFile::open(&file_path, None).unwrap();
        
        // Check dimensions
        let dims = nc_read.dimensions();
        assert_eq!(dims.len(), 2);
        assert_eq!(dims["x"], Some(3));
        assert_eq!(dims["y"], Some(2));
        
        // Check variables
        let vars = nc_read.variables();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0], "temperature");
        
        // Check variable info
        let (dtype, var_dims, attrs) = nc_read.variable_info("temperature").unwrap();
        assert_eq!(dtype, NetCDFDataType::Float);
        assert_eq!(var_dims, vec!["x", "y"]);
        assert!(attrs.contains_key("units"));
        
        // Read data
        let read_data: ArrayD<f32> = nc_read.read_variable("temperature").unwrap();
        
        assert_eq!(read_data.shape(), &[3, 2]);
        assert_eq!(read_data[[0, 0]], 20.0);
        assert_eq!(read_data[[2, 1]], 25.0);
        
        // Check global attributes
        let global_attrs = nc_read.global_attributes();
        assert!(global_attrs.contains_key("title"));
    }
}