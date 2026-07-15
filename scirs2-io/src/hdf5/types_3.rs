//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::{AttributeValue, DataArray};

/// File access mode
#[derive(Debug, Clone, PartialEq)]
pub enum FileMode {
    /// Read-only mode
    ReadOnly,
    /// Read-write mode
    ReadWrite,
    /// Create new file (fail if exists)
    Create,
    /// Create or truncate existing file
    Truncate,
}
/// String encoding types
#[derive(Debug, Clone, PartialEq)]
pub enum StringEncoding {
    /// UTF-8 encoding
    UTF8,
    /// ASCII encoding
    ASCII,
}
/// File statistics
#[derive(Debug, Clone, Default)]
pub struct FileStats {
    /// Number of groups in the file
    pub num_groups: usize,
    /// Number of datasets in the file
    pub num_datasets: usize,
    /// Number of attributes in the file
    pub num_attributes: usize,
    /// Total data size in bytes
    pub total_data_size: usize,
}
/// HDF5 group
#[derive(Debug, Clone)]
pub struct Group {
    /// Group name
    pub name: String,
    /// Child groups
    pub groups: HashMap<String, Group>,
    /// Datasets in this group
    pub datasets: HashMap<String, Dataset>,
    /// Attributes
    pub attributes: HashMap<String, AttributeValue>,
}
impl Group {
    /// Create a new empty group
    pub fn new(name: String) -> Self {
        Self {
            name,
            groups: HashMap::new(),
            datasets: HashMap::new(),
            attributes: HashMap::new(),
        }
    }
    /// Create a subgroup
    pub fn create_group(&mut self, name: &str) -> &mut Group {
        self.groups
            .entry(name.to_string())
            .or_insert_with(|| Group::new(name.to_string()))
    }
    /// Get a subgroup
    pub fn get_group(&self, name: &str) -> Option<&Group> {
        self.groups.get(name)
    }
    /// Get a mutable subgroup
    pub fn get_group_mut(&mut self, name: &str) -> Option<&mut Group> {
        self.groups.get_mut(name)
    }
    /// Add an attribute
    pub fn set_attribute(&mut self, name: &str, value: AttributeValue) {
        self.attributes.insert(name.to_string(), value);
    }
    /// Get an attribute by name
    pub fn get_attribute(&self, name: &str) -> Option<&AttributeValue> {
        self.attributes.get(name)
    }
    /// Remove an attribute
    pub fn remove_attribute(&mut self, name: &str) -> Option<AttributeValue> {
        self.attributes.remove(name)
    }
    /// List all attribute names
    pub fn attribute_names(&self) -> Vec<&str> {
        self.attributes.keys().map(|s| s.as_str()).collect()
    }
    /// Check if group has a specific attribute
    pub fn has_attribute(&self, name: &str) -> bool {
        self.attributes.contains_key(name)
    }
    /// Get dataset by name
    pub fn get_dataset(&self, name: &str) -> Option<&Dataset> {
        self.datasets.get(name)
    }
    /// Get mutable dataset by name
    pub fn get_dataset_mut(&mut self, name: &str) -> Option<&mut Dataset> {
        self.datasets.get_mut(name)
    }
    /// List all dataset names
    pub fn dataset_names(&self) -> Vec<&str> {
        self.datasets.keys().map(|s| s.as_str()).collect()
    }
    /// List all group names
    pub fn group_names(&self) -> Vec<&str> {
        self.groups.keys().map(|s| s.as_str()).collect()
    }
    /// Check if group has a specific dataset
    pub fn has_dataset(&self, name: &str) -> bool {
        self.datasets.contains_key(name)
    }
    /// Check if group has a specific subgroup
    pub fn has_group(&self, name: &str) -> bool {
        self.groups.contains_key(name)
    }
    /// Remove a dataset
    pub fn remove_dataset(&mut self, name: &str) -> Option<Dataset> {
        self.datasets.remove(name)
    }
    /// Remove a subgroup
    pub fn remove_group(&mut self, name: &str) -> Option<Group> {
        self.groups.remove(name)
    }
}
/// HDF5 dataset
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Dataset name
    pub name: String,
    /// Data type
    pub dtype: HDF5DataType,
    /// Shape
    pub shape: Vec<usize>,
    /// Data (stored as flattened array)
    pub data: DataArray,
    /// Attributes
    pub attributes: HashMap<String, AttributeValue>,
    /// Dataset options
    pub options: DatasetOptions,
}
impl Dataset {
    /// Create a new dataset
    pub fn new(
        name: String,
        dtype: HDF5DataType,
        shape: Vec<usize>,
        data: DataArray,
        options: DatasetOptions,
    ) -> Self {
        Self {
            name,
            dtype,
            shape,
            data,
            attributes: HashMap::new(),
            options,
        }
    }
    /// Set an attribute on the dataset
    pub fn set_attribute(&mut self, name: &str, value: AttributeValue) {
        self.attributes.insert(name.to_string(), value);
    }
    /// Get an attribute by name
    pub fn get_attribute(&self, name: &str) -> Option<&AttributeValue> {
        self.attributes.get(name)
    }
    /// Remove an attribute
    pub fn remove_attribute(&mut self, name: &str) -> Option<AttributeValue> {
        self.attributes.remove(name)
    }
    /// Get the number of elements in the dataset
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    /// Get the total size in bytes (estimate)
    pub fn size_bytes(&self) -> usize {
        let element_size = match &self.dtype {
            HDF5DataType::Integer { size, .. } => *size,
            HDF5DataType::Float { size } => *size,
            HDF5DataType::String { .. } => 8,
            HDF5DataType::Array { .. } => 8,
            HDF5DataType::Compound { .. } => 8,
            HDF5DataType::Enum { .. } => 8,
        };
        self.len() * element_size
    }
    /// Get data as float vector (if possible)
    pub fn as_float_vec(&self) -> Option<Vec<f64>> {
        match &self.data {
            DataArray::Float(data) => Some(data.clone()),
            DataArray::Integer(data) => Some(data.iter().map(|&x| x as f64).collect()),
            _ => None,
        }
    }
    /// Get data as integer vector (if possible)
    pub fn as_integer_vec(&self) -> Option<Vec<i64>> {
        match &self.data {
            DataArray::Integer(data) => Some(data.clone()),
            DataArray::Float(data) => Some(data.iter().map(|&x| x as i64).collect()),
            _ => None,
        }
    }
    /// Get data as string vector (if possible)
    pub fn as_string_vec(&self) -> Option<Vec<String>> {
        match &self.data {
            DataArray::String(data) => Some(data.clone()),
            _ => None,
        }
    }
}
/// HDF5 data type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum HDF5DataType {
    /// Integer types
    Integer {
        /// Size in bytes (1, 2, 4, or 8)
        size: usize,
        /// Whether the integer is signed
        signed: bool,
    },
    /// Floating point types
    Float {
        /// Size in bytes (4 or 8)
        size: usize,
    },
    /// String type
    String {
        /// String encoding (UTF-8 or ASCII)
        encoding: StringEncoding,
    },
    /// Array type
    Array {
        /// Base data type of array elements
        base_type: Box<HDF5DataType>,
        /// Shape of the array
        shape: Vec<usize>,
    },
    /// Compound type
    Compound {
        /// Fields in the compound type (name, type) pairs
        fields: Vec<(String, HDF5DataType)>,
    },
    /// Enum type
    Enum {
        /// Enumeration values (name, value) pairs
        values: Vec<(String, i64)>,
    },
}
/// HDF5 compression options
#[derive(Debug, Clone, Default)]
pub struct CompressionOptions {
    /// Enable gzip compression
    pub gzip: Option<u8>,
    /// Enable szip compression
    pub szip: Option<(u32, u32)>,
    /// Enable LZF compression
    pub lzf: bool,
    /// Enable shuffle filter
    pub shuffle: bool,
}
/// HDF5 dataset creation options
#[derive(Debug, Clone, Default)]
pub struct DatasetOptions {
    /// Chunk size for chunked storage
    pub chunk_size: Option<Vec<usize>>,
    /// Compression options
    pub compression: CompressionOptions,
    /// Fill value for uninitialized elements
    pub fill_value: Option<f64>,
    /// Enable fletcher32 checksum
    pub fletcher32: bool,
}
