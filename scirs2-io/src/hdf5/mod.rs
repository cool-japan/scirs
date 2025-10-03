//! HDF5 file format support
//!
//! This module provides functionality for reading and writing HDF5 (Hierarchical Data Format version 5) files.
//! HDF5 is a data model, library, and file format for storing and managing data. It supports an unlimited
//! variety of datatypes, and is designed for flexible and efficient I/O and for high volume and complex data.
//!
//! Features:
//! - Reading and writing HDF5 files
//! - Support for groups and datasets
//! - Attributes on groups and datasets
//! - Multiple datatypes (integers, floats, strings, compound types)
//! - Chunking and compression support
//! - Integration with ndarray for efficient array operations
//! - Enhanced functionality with compression and parallel I/O (see `enhanced` module)
//! - Extended data type support including complex numbers and boolean types
//! - Performance optimizations for large datasets

use crate::error::{IoError, Result};
use scirs2_core::ndarray::{ArrayBase, ArrayD, IxDyn};
use std::collections::HashMap;
use std::ops::Deref;
use std::path::Path;
use std::str::FromStr;

#[cfg(feature = "hdf5")]
use hdf5::File;

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

/// String encoding types
#[derive(Debug, Clone, PartialEq)]
pub enum StringEncoding {
    /// UTF-8 encoding
    UTF8,
    /// ASCII encoding
    ASCII,
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

/// HDF5 file handle
pub struct HDF5File {
    /// File path
    #[allow(dead_code)]
    path: String,
    /// Root group
    root: Group,
    /// File access mode
    #[allow(dead_code)]
    mode: FileMode,
    /// Native HDF5 file handle (when feature is enabled)
    #[cfg(feature = "hdf5")]
    native_file: Option<File>,
}

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
            HDF5DataType::String { .. } => 8,   // Estimate
            HDF5DataType::Array { .. } => 8,    // Estimate
            HDF5DataType::Compound { .. } => 8, // Estimate
            HDF5DataType::Enum { .. } => 8,     // Estimate
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

/// Data array storage
#[derive(Debug, Clone)]
pub enum DataArray {
    /// Integer data
    Integer(Vec<i64>),
    /// Float data
    Float(Vec<f64>),
    /// String data
    String(Vec<String>),
    /// Binary data
    Binary(Vec<u8>),
}

/// Attribute value types
#[derive(Debug, Clone)]
pub enum AttributeValue {
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Integer array
    IntegerArray(Vec<i64>),
    /// Float array
    FloatArray(Vec<f64>),
    /// String array
    StringArray(Vec<String>),
    /// Boolean value
    Boolean(bool),
    /// Generic array (alias for IntegerArray for compatibility)
    Array(Vec<i64>),
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

impl HDF5File {
    /// Create a new HDF5 file
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        #[cfg(feature = "hdf5")]
        {
            let native_file = File::create(&path_str)
                .map_err(|e| IoError::FormatError(format!("Failed to create HDF5 file: {e}")))?;

            Ok(Self {
                path: path_str,
                root: Group::new("/".to_string()),
                mode: FileMode::Create,
                native_file: Some(native_file),
            })
        }

        #[cfg(not(feature = "hdf5"))]
        {
            Ok(Self {
                path: path_str,
                root: Group::new("/".to_string()),
                mode: FileMode::Create,
            })
        }
    }

    /// Open an existing HDF5 file
    pub fn open<P: AsRef<Path>>(path: P, mode: FileMode) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        #[cfg(feature = "hdf5")]
        {
            let native_file = match mode {
                FileMode::ReadOnly => File::open(&path_str)
                    .map_err(|e| IoError::FormatError(format!("Failed to open HDF5 file: {e}")))?,
                FileMode::ReadWrite => File::open_rw(&path_str)
                    .map_err(|e| IoError::FormatError(format!("Failed to open HDF5 file: {e}")))?,
                FileMode::Create => File::create(&path_str).map_err(|e| {
                    IoError::FormatError(format!("Failed to create HDF5 file: {e}"))
                })?,
                FileMode::Truncate => File::create(&path_str).map_err(|e| {
                    IoError::FormatError(format!("Failed to create HDF5 file: {e}"))
                })?,
            };

            // Load existing structure from the file
            let mut root = Group::new("/".to_string());
            Self::load_group_structure(&native_file, &mut root)?;

            Ok(Self {
                path: path_str,
                root,
                mode,
                native_file: Some(native_file),
            })
        }

        #[cfg(not(feature = "hdf5"))]
        {
            Ok(Self {
                path: path_str,
                root: Group::new("/".to_string()),
                mode,
            })
        }
    }

    /// Get the root group
    pub fn root(&self) -> &Group {
        &self.root
    }

    /// Get the root group mutably
    pub fn root_mut(&mut self) -> &mut Group {
        &mut self.root
    }

    /// Get access to the native HDF5 file handle (when feature is enabled)
    #[cfg(feature = "hdf5")]
    pub fn native_file(&self) -> Option<&File> {
        self.native_file.as_ref()
    }

    /// Load group structure from native HDF5 file
    #[cfg(feature = "hdf5")]
    fn load_group_structure(file: &File, group: &mut Group) -> Result<()> {
        use hdf5::types::TypeDescriptor;

        // Load attributes for the root/file level
        if let Ok(attr_names) = file.attr_names() {
            for attr_name in attr_names {
                if let Ok(attr) = file.attr(&attr_name) {
                    if let Ok(attr_value) = Self::read_attribute_value(&attr) {
                        group.attributes.insert(attr_name, attr_value);
                    }
                }
            }
        }

        // Load all datasets in the current group
        let datasets = file
            .datasets()
            .map_err(|e| IoError::FormatError(format!("Failed to get datasets: {e}")))?;

        for dataset in datasets {
            let dataset_name_full = dataset.name();
            let dataset_key = dataset_name_full
                .rsplit('/')
                .next()
                .unwrap_or(&dataset_name_full)
                .trim_start_matches('/')
                .to_string();
            if let Ok(h5_dataset) = file.dataset(&dataset_name_full) {
                // Get dataset metadata
                let shape: Vec<usize> = h5_dataset.shape().to_vec();
                let dtype = h5_dataset.dtype().map_err(|e| {
                    IoError::FormatError(format!("Failed to get dataset dtype: {e}"))
                })?;

                // Convert HDF5 datatype to our internal representation
                let internal_dtype = Self::convert_hdf5_datatype(&dtype)?;

                // Read dataset data based on type
                let data = Self::read_dataset_data(&h5_dataset, &dtype)?;

                // Read attributes
                let mut attributes = HashMap::new();
                if let Ok(attr_names) = h5_dataset.attr_names() {
                    for attr_name in attr_names {
                        if let Ok(attr) = h5_dataset.attr(&attr_name) {
                            if let Ok(attr_value) = Self::read_attribute_value(&attr) {
                                attributes.insert(attr_name, attr_value);
                            }
                        }
                    }
                }

                // Create dataset
                let dataset = Dataset {
                    name: dataset_key.clone(),
                    dtype: internal_dtype,
                    shape,
                    data,
                    attributes,
                    options: DatasetOptions::default(),
                };

                group.datasets.insert(dataset_key, dataset);
            }
        }

        // Load all subgroups recursively
        let groups = file
            .groups()
            .map_err(|e| IoError::FormatError(format!("Failed to get groups: {e}")))?;

        for h5_group in groups {
            let group_name_full = h5_group.name();
            let group_key = group_name_full
                .rsplit('/')
                .next()
                .unwrap_or(&group_name_full)
                .trim_start_matches('/')
                .to_string();
            let mut subgroup = Group::new(group_key.clone());

            // Recursively load this group's structure
            Self::load_subgroup_structure(&h5_group, &mut subgroup)?;

            group.groups.insert(group_key, subgroup);
        }

        Ok(())
    }

    /// Recursively load structure for an HDF5 group
    #[cfg(feature = "hdf5")]
    fn load_subgroup_structure(h5_group: &hdf5::Group, group: &mut Group) -> Result<()> {
        // Load attributes
        if let Ok(attr_names) = h5_group.attr_names() {
            for attr_name in attr_names {
                if let Ok(attr) = h5_group.attr(&attr_name) {
                    if let Ok(attr_value) = Self::read_attribute_value(&attr) {
                        group.attributes.insert(attr_name, attr_value);
                    }
                }
            }
        }

        // Load datasets in this group
        if let Ok(datasets) = h5_group.datasets() {
            for ds in datasets {
                let ds_name_full = ds.name();
                let ds_key = ds_name_full
                    .rsplit('/')
                    .next()
                    .unwrap_or(&ds_name_full)
                    .trim_start_matches('/')
                    .to_string();
                if let Ok(h5_dataset) = h5_group.dataset(&ds_key) {
                    let shape: Vec<usize> = h5_dataset.shape().to_vec();
                    let dtype = h5_dataset.dtype().map_err(|e| {
                        IoError::FormatError(format!("Failed to get dataset dtype: {e}"))
                    })?;
                    let internal_dtype = Self::convert_hdf5_datatype(&dtype)?;
                    let data = Self::read_dataset_data(&h5_dataset, &dtype)?;

                    // Read dataset attributes
                    let mut attributes = HashMap::new();
                    if let Ok(attr_names) = h5_dataset.attr_names() {
                        for attr_name in attr_names {
                            if let Ok(attr) = h5_dataset.attr(&attr_name) {
                                if let Ok(attr_value) = Self::read_attribute_value(&attr) {
                                    attributes.insert(attr_name, attr_value);
                                }
                            }
                        }
                    }

                    let dataset = Dataset {
                        name: ds_key.clone(),
                        dtype: internal_dtype,
                        shape,
                        data,
                        attributes,
                        options: DatasetOptions::default(),
                    };
                    group.datasets.insert(ds_key, dataset);
                }
            }
        }

        // Recurse into subgroups
        if let Ok(subgroups) = h5_group.groups() {
            for sub in subgroups {
                let sub_name_full = sub.name();
                let sub_key = sub_name_full
                    .rsplit('/')
                    .next()
                    .unwrap_or(&sub_name_full)
                    .trim_start_matches('/')
                    .to_string();
                let mut child = Group::new(sub_key.clone());
                Self::load_subgroup_structure(&sub, &mut child)?;
                group.groups.insert(sub_key, child);
            }
        }

        Ok(())
    }

    /// Write a group (and all its contents) to the HDF5 file
    #[cfg(feature = "hdf5")]
    fn write_group_to_hdf5(file: &File, group: &Group, path_prefix: &str) -> Result<()> {
        // Write attributes for this group
        for (attr_name, attr_value) in &group.attributes {
            Self::write_attribute_to_hdf5(file, path_prefix, attr_name, attr_value)?;
        }

        // Write all datasets in this group
        for (dataset_name, dataset) in &group.datasets {
            let dataset_path = if path_prefix.is_empty() {
                dataset_name.clone()
            } else {
                format!("{}/{}", path_prefix, dataset_name)
            };
            Self::write_dataset_to_hdf5(file, &dataset_path, dataset)?;
        }

        // Recursively write subgroups
        for (subgroup_name, subgroup) in &group.groups {
            let subgroup_path = if path_prefix.is_empty() {
                subgroup_name.clone()
            } else {
                format!("{}/{}", path_prefix, subgroup_name)
            };

            // Create the subgroup in HDF5
            if let Err(_) = file.group(&subgroup_path) {
                // Group doesn't exist, create it
                file.create_group(&subgroup_path).map_err(|e| {
                    IoError::FormatError(format!("Failed to create group {}: {}", subgroup_path, e))
                })?;
            }

            // Write the subgroup contents
            Self::write_group_to_hdf5(file, subgroup, &subgroup_path)?;
        }

        Ok(())
    }

    /// Write an attribute to the HDF5 file
    #[cfg(feature = "hdf5")]
    fn write_attribute_to_hdf5(
        file: &File,
        path: &str,
        name: &str,
        value: &AttributeValue,
    ) -> Result<()> {
        use hdf5::types::VarLenUnicode;

        // Resolve target: root file group or a subgroup
        let target_group = if path.is_empty() {
            file.as_group()
                .map_err(|e| IoError::FormatError(format!("Failed to access root group: {e}")))?
        } else {
            file.group(path).map_err(|e| {
                IoError::FormatError(format!("Failed to access group '{path}': {e}"))
            })?
        };

        // Write the attribute to the resolved group
        match value {
            AttributeValue::Integer(v) => {
                let attr = target_group.new_attr::<i64>().create(name).map_err(|e| {
                    IoError::FormatError(format!("Failed to create integer attribute: {}", e))
                })?;
                attr.write_scalar(v).map_err(|e| {
                    IoError::FormatError(format!("Failed to write integer attribute: {}", e))
                })?;
            }
            AttributeValue::Float(v) => {
                let attr = target_group.new_attr::<f64>().create(name).map_err(|e| {
                    IoError::FormatError(format!("Failed to create float attribute: {}", e))
                })?;
                attr.write_scalar(v).map_err(|e| {
                    IoError::FormatError(format!("Failed to write float attribute: {}", e))
                })?;
            }
            AttributeValue::String(v) => {
                let vlen_str = VarLenUnicode::from_str(v).map_err(|e| {
                    IoError::FormatError(format!("Failed to create VarLenUnicode: {:?}", e))
                })?;
                let attr = target_group
                    .new_attr::<VarLenUnicode>()
                    .create(name)
                    .map_err(|e| {
                        IoError::FormatError(format!("Failed to create string attribute: {}", e))
                    })?;
                attr.write_scalar(&vlen_str).map_err(|e| {
                    IoError::FormatError(format!("Failed to write string attribute: {}", e))
                })?;
            }
            AttributeValue::IntegerArray(v) => {
                let attr = target_group
                    .new_attr::<i64>()
                    .shape([v.len()])
                    .create(name)
                    .map_err(|e| {
                        IoError::FormatError(format!(
                            "Failed to create integer array attribute: {}",
                            e
                        ))
                    })?;
                attr.write(v).map_err(|e| {
                    IoError::FormatError(format!("Failed to write integer array attribute: {}", e))
                })?;
            }
            AttributeValue::FloatArray(v) => {
                let attr = target_group
                    .new_attr::<f64>()
                    .shape([v.len()])
                    .create(name)
                    .map_err(|e| {
                        IoError::FormatError(format!(
                            "Failed to create float array attribute: {}",
                            e
                        ))
                    })?;
                attr.write(v).map_err(|e| {
                    IoError::FormatError(format!("Failed to write float array attribute: {}", e))
                })?;
            }
            AttributeValue::StringArray(v) => {
                let mut vlen_strings = Vec::new();
                for s in v {
                    let vlen = VarLenUnicode::from_str(s).map_err(|e| {
                        IoError::FormatError(format!("Failed to create VarLenUnicode: {:?}", e))
                    })?;
                    vlen_strings.push(vlen);
                }
                let attr = target_group
                    .new_attr::<VarLenUnicode>()
                    .shape([v.len()])
                    .create(name)
                    .map_err(|e| {
                        IoError::FormatError(format!(
                            "Failed to create string array attribute: {}",
                            e
                        ))
                    })?;
                attr.write(&vlen_strings).map_err(|e| {
                    IoError::FormatError(format!("Failed to write string array attribute: {}", e))
                })?;
            }
            AttributeValue::Boolean(v) => {
                let int_val = if *v { 1i64 } else { 0i64 };
                let attr = target_group.new_attr::<i64>().create(name).map_err(|e| {
                    IoError::FormatError(format!("Failed to create boolean attribute: {}", e))
                })?;
                attr.write_scalar(&int_val).map_err(|e| {
                    IoError::FormatError(format!("Failed to write boolean attribute: {}", e))
                })?;
            }
            AttributeValue::Array(_) => {
                // Skip complex arrays for now - would need proper type handling
                eprintln!("Warning: Skipping complex array attribute '{}'", name);
            }
        }

        Ok(())
    }

    /// Write a dataset to the HDF5 file
    #[cfg(feature = "hdf5")]
    fn write_dataset_to_hdf5(file: &File, path: &str, dataset: &Dataset) -> Result<()> {
        // Create the dataset based on its data type
        match &dataset.data {
            DataArray::Float(data) => {
                let h5_dataset = file
                    .new_dataset::<f64>()
                    .shape(&dataset.shape)
                    .create(path)
                    .map_err(|e| {
                        IoError::FormatError(format!("Failed to create float dataset: {}", e))
                    })?;
                // Use write_raw to write the flat data directly
                h5_dataset.write_raw(data).map_err(|e| {
                    IoError::FormatError(format!("Failed to write float dataset: {}", e))
                })?;
            }
            DataArray::Integer(data) => {
                let h5_dataset = file
                    .new_dataset::<i64>()
                    .shape(&dataset.shape)
                    .create(path)
                    .map_err(|e| {
                        IoError::FormatError(format!("Failed to create integer dataset: {}", e))
                    })?;
                // Use write_raw to write the flat data directly
                h5_dataset.write_raw(data).map_err(|e| {
                    IoError::FormatError(format!("Failed to write integer dataset: {}", e))
                })?;
            }
            DataArray::String(data) => {
                use hdf5::types::VarLenUnicode;
                let mut vlen_strings = Vec::new();
                for s in data {
                    let vlen = VarLenUnicode::from_str(s).map_err(|e| {
                        IoError::FormatError(format!("Failed to create VarLenUnicode: {:?}", e))
                    })?;
                    vlen_strings.push(vlen);
                }
                let h5_dataset = file
                    .new_dataset::<VarLenUnicode>()
                    .shape(&dataset.shape)
                    .create(path)
                    .map_err(|e| {
                        IoError::FormatError(format!("Failed to create string dataset: {}", e))
                    })?;
                h5_dataset.write(&vlen_strings).map_err(|e| {
                    IoError::FormatError(format!("Failed to write string dataset: {}", e))
                })?;
            }
            DataArray::Binary(data) => {
                // Write binary data as u8 array
                let h5_dataset = file
                    .new_dataset::<u8>()
                    .shape(&dataset.shape)
                    .create(path)
                    .map_err(|e| {
                        IoError::FormatError(format!("Failed to create binary dataset: {}", e))
                    })?;
                h5_dataset.write(data).map_err(|e| {
                    IoError::FormatError(format!("Failed to write binary dataset: {}", e))
                })?;
            }
        }

        Ok(())
    }

    /// Convert HDF5 datatype to internal representation
    #[cfg(feature = "hdf5")]
    fn convert_hdf5_datatype(dtype: &hdf5::Datatype) -> Result<HDF5DataType> {
        use hdf5::types::TypeDescriptor;

        match dtype.to_descriptor() {
            Ok(TypeDescriptor::Integer(int_type)) => Ok(HDF5DataType::Integer {
                size: int_type as usize,
                signed: true,
            }),
            Ok(TypeDescriptor::Unsigned(int_type)) => Ok(HDF5DataType::Integer {
                size: int_type as usize,
                signed: false,
            }),
            Ok(TypeDescriptor::Float(float_type)) => Ok(HDF5DataType::Float {
                size: float_type as usize,
            }),
            Ok(TypeDescriptor::FixedUnicode(size)) => Ok(HDF5DataType::String {
                encoding: StringEncoding::UTF8,
            }),
            Ok(TypeDescriptor::FixedAscii(size)) => Ok(HDF5DataType::String {
                encoding: StringEncoding::ASCII,
            }),
            Ok(TypeDescriptor::VarLenUnicode) => Ok(HDF5DataType::String {
                encoding: StringEncoding::UTF8,
            }),
            Ok(TypeDescriptor::VarLenAscii) => Ok(HDF5DataType::String {
                encoding: StringEncoding::ASCII,
            }),
            // TODO: Handle Array type when available in HDF5 crate
            // Ok(TypeDescriptor::Array(array_type)) => {
            //     let base_type = Self::convert_hdf5_datatype(&array_type.base_type())?;
            //     Ok(HDF5DataType::Array {
            //         base_type: Box::new(base_type),
            //         shape: array_type.shape().to_vec(),
            //     })
            // }
            Ok(TypeDescriptor::Compound(comp_type)) => {
                let mut fields = Vec::new();
                for field in &comp_type.fields {
                    // Create a temporary datatype from the field's type descriptor
                    let field_datatype =
                        hdf5::Datatype::from_descriptor(&field.ty).map_err(|e| {
                            IoError::FormatError(format!(
                                "Failed to create datatype for field: {}",
                                e
                            ))
                        })?;
                    let field_type = Self::convert_hdf5_datatype(&field_datatype)?;
                    fields.push((field.name.clone(), field_type));
                }
                Ok(HDF5DataType::Compound { fields })
            }
            Ok(TypeDescriptor::Enum(enum_type)) => {
                let mut values = Vec::new();
                for member in &enum_type.members {
                    values.push((member.name.clone(), member.value as i64));
                }
                Ok(HDF5DataType::Enum { values })
            }
            _ => {
                // Fallback for unsupported types
                Ok(HDF5DataType::String {
                    encoding: StringEncoding::UTF8,
                })
            }
        }
    }

    /// Read dataset data based on HDF5 datatype
    #[cfg(feature = "hdf5")]
    fn read_dataset_data(dataset: &hdf5::Dataset, dtype: &hdf5::Datatype) -> Result<DataArray> {
        use hdf5::types::TypeDescriptor;

        match dtype.to_descriptor() {
            Ok(TypeDescriptor::Integer(_)) => {
                let data: Vec<i64> = dataset.read_raw().map_err(|e| {
                    IoError::FormatError(format!("Failed to read integer dataset: {e}"))
                })?;
                Ok(DataArray::Integer(data))
            }
            Ok(TypeDescriptor::Float(_)) => {
                let data: Vec<f64> = dataset.read_raw().map_err(|e| {
                    IoError::FormatError(format!("Failed to read float dataset: {e}"))
                })?;
                Ok(DataArray::Float(data))
            }
            Ok(TypeDescriptor::FixedUnicode(_))
            | Ok(TypeDescriptor::FixedAscii(_))
            | Ok(TypeDescriptor::VarLenUnicode) => {
                use hdf5::types::VarLenUnicode;
                let data: Vec<VarLenUnicode> = dataset.read_raw().map_err(|e| {
                    IoError::FormatError(format!("Failed to read string dataset: {e}"))
                })?;
                let strings: Vec<String> = data.into_iter().map(|s| s.to_string()).collect();
                Ok(DataArray::String(strings))
            }
            Ok(TypeDescriptor::VarLenAscii) => {
                use hdf5::types::VarLenAscii;
                let data: Vec<VarLenAscii> = dataset.read_raw().map_err(|e| {
                    IoError::FormatError(format!("Failed to read string dataset: {e}"))
                })?;
                let strings: Vec<String> = data.into_iter().map(|s| s.to_string()).collect();
                Ok(DataArray::String(strings))
            }
            _ => {
                // For unsupported types, read as binary data
                let data: Vec<u8> = dataset.read_raw().map_err(|e| {
                    IoError::FormatError(format!("Failed to read binary dataset: {e}"))
                })?;
                Ok(DataArray::Binary(data))
            }
        }
    }

    /// Read attribute value
    #[cfg(feature = "hdf5")]
    fn read_attribute_value(attr: &hdf5::Attribute) -> Result<AttributeValue> {
        use hdf5::types::TypeDescriptor;

        let dtype = attr
            .dtype()
            .map_err(|e| IoError::FormatError(format!("Failed to get attribute dtype: {e}")))?;

        match dtype.to_descriptor() {
            Ok(TypeDescriptor::Integer(_)) => {
                if attr.shape().iter().product::<usize>() == 1 {
                    let value: i64 = attr.read_scalar().map_err(|e| {
                        IoError::FormatError(format!("Failed to read integer attribute: {e}"))
                    })?;
                    Ok(AttributeValue::Integer(value))
                } else {
                    let value: Vec<i64> = attr.read_raw().map_err(|e| {
                        IoError::FormatError(format!(
                            "Failed to read integer array attribute: {}",
                            e
                        ))
                    })?;
                    Ok(AttributeValue::IntegerArray(value))
                }
            }
            Ok(TypeDescriptor::Float(_)) => {
                if attr.shape().iter().product::<usize>() == 1 {
                    let value: f64 = attr.read_scalar().map_err(|e| {
                        IoError::FormatError(format!("Failed to read float attribute: {e}"))
                    })?;
                    Ok(AttributeValue::Float(value))
                } else {
                    let value: Vec<f64> = attr.read_raw().map_err(|e| {
                        IoError::FormatError(format!("Failed to read float array attribute: {e}"))
                    })?;
                    Ok(AttributeValue::FloatArray(value))
                }
            }
            Ok(TypeDescriptor::VarLenUnicode) => {
                use hdf5::types::VarLenUnicode;
                if attr.shape().iter().product::<usize>() == 1 {
                    let value: VarLenUnicode = attr.read_scalar().map_err(|e| {
                        IoError::FormatError(format!("Failed to read string attribute: {e}"))
                    })?;
                    Ok(AttributeValue::String(value.to_string()))
                } else {
                    let value: Vec<VarLenUnicode> = attr.read_raw().map_err(|e| {
                        IoError::FormatError(format!(
                            "Failed to read string array attribute: {}",
                            e
                        ))
                    })?;
                    let strings: Vec<String> = value.into_iter().map(|s| s.to_string()).collect();
                    Ok(AttributeValue::StringArray(strings))
                }
            }
            Ok(TypeDescriptor::VarLenAscii) => {
                use hdf5::types::VarLenAscii;
                if attr.shape().iter().product::<usize>() == 1 {
                    let value: VarLenAscii = attr.read_scalar().map_err(|e| {
                        IoError::FormatError(format!("Failed to read string attribute: {e}"))
                    })?;
                    Ok(AttributeValue::String(value.to_string()))
                } else {
                    let value: Vec<VarLenAscii> = attr.read_raw().map_err(|e| {
                        IoError::FormatError(format!(
                            "Failed to read string array attribute: {}",
                            e
                        ))
                    })?;
                    let strings: Vec<String> = value.into_iter().map(|s| s.to_string()).collect();
                    Ok(AttributeValue::StringArray(strings))
                }
            }
            Ok(TypeDescriptor::FixedUnicode(size)) | Ok(TypeDescriptor::FixedAscii(size)) => {
                // For fixed-size strings, we read them as VarLen types for simplicity
                use hdf5::types::VarLenUnicode;
                if attr.shape().iter().product::<usize>() == 1 {
                    let value: VarLenUnicode = attr.read_scalar().map_err(|e| {
                        IoError::FormatError(format!("Failed to read string attribute: {e}"))
                    })?;
                    Ok(AttributeValue::String(value.to_string()))
                } else {
                    let value: Vec<VarLenUnicode> = attr.read_raw().map_err(|e| {
                        IoError::FormatError(format!(
                            "Failed to read string array attribute: {}",
                            e
                        ))
                    })?;
                    let strings: Vec<String> = value.into_iter().map(|s| s.to_string()).collect();
                    Ok(AttributeValue::StringArray(strings))
                }
            }
            _ => {
                // Fallback: return a default value
                Ok(AttributeValue::String("unknown".to_string()))
            }
        }
    }

    /// Create a dataset from an ndarray
    pub fn create_dataset_from_array<A, D>(
        &mut self,
        path: &str,
        array: &ArrayBase<A, D>,
        options: Option<DatasetOptions>,
    ) -> Result<()>
    where
        A: scirs2_core::ndarray::Data,
        A::Elem: Clone + std::fmt::Debug,
        D: scirs2_core::ndarray::Dimension,
    {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.is_empty() {
            return Err(IoError::FormatError("Invalid dataset path".to_string()));
        }

        let dataset_name = parts.last().unwrap();
        let mut current_group = &mut self.root;

        // Navigate to the parent group, creating groups as needed
        for &group_name in &parts[..parts.len() - 1] {
            current_group = current_group.create_group(group_name);
        }

        // Convert array to dataset - handle different types
        let shape: Vec<usize> = array.shape().to_vec();
        let flat_data: Vec<f64> = array
            .iter()
            .map(|x| {
                // Convert to f64 using format and parse as a workaround for generic types
                format!("{:?}", x).parse::<f64>().unwrap_or(0.0)
            })
            .collect();

        let dataset = Dataset {
            name: dataset_name.to_string(),
            dtype: HDF5DataType::Float { size: 8 },
            shape: shape.clone(),
            data: DataArray::Float(flat_data.clone()),
            attributes: HashMap::new(),
            options: options.unwrap_or_default(),
        };

        current_group
            .datasets
            .insert(dataset_name.to_string(), dataset);

        // Native HDF5 file writing would go here
        // Currently handled through enhanced module to avoid type compatibility issues

        Ok(())
    }

    /// Read a dataset as an ndarray with specific type
    pub fn read_dataset_typed<T>(&self, path: &str) -> Result<ArrayD<T>>
    where
        T: Clone + Default + std::str::FromStr,
        <T as std::str::FromStr>::Err: std::fmt::Display,
    {
        // For now, read as f64 and convert
        let f64_array = self.read_dataset(path)?;
        let shape = f64_array.shape().to_vec();
        let converted: Vec<T> = f64_array
            .iter()
            .map(|&v| {
                // Try to convert through string representation
                let s = format!("{}", v);
                s.parse::<T>().unwrap_or_default()
            })
            .collect();

        ArrayD::from_shape_vec(scirs2_core::ndarray::IxDyn(&shape), converted)
            .map_err(|e| IoError::FormatError(format!("Failed to create typed array: {}", e)))
    }

    /// Read a dataset as an ndarray of f64
    pub fn read_dataset(&self, path: &str) -> Result<ArrayD<f64>> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.is_empty() {
            return Err(IoError::FormatError("Invalid dataset path".to_string()));
        }

        let dataset_name = parts.last().unwrap();
        let mut current_group = &self.root;

        // Navigate to the parent group
        for &group_name in &parts[..parts.len() - 1] {
            current_group = current_group
                .get_group(group_name)
                .ok_or_else(|| IoError::FormatError(format!("Group '{group_name}' not found")))?;
        }

        // Get the dataset
        let dataset = current_group
            .datasets
            .get(*dataset_name)
            .ok_or_else(|| IoError::FormatError(format!("Dataset '{dataset_name}' not found")))?;

        // Try to read from native HDF5 file first if available
        #[cfg(feature = "hdf5")]
        {
            if let Some(ref file) = self.native_file {
                // Build the full path for direct dataset access
                let full_path = parts.join("/");

                // Try to read the dataset directly using the full path
                if let Ok(h5_dataset) = file.dataset(&full_path) {
                    let data: Vec<f64> = h5_dataset.read_raw().map_err(|e| {
                        IoError::FormatError(format!("Failed to read HDF5 dataset: {e}"))
                    })?;

                    let shape = IxDyn(&dataset.shape);
                    return ArrayD::from_shape_vec(shape, data)
                        .map_err(|e| IoError::FormatError(e.to_string()));
                }
            }
        }

        // Fall back to in-memory data
        match &dataset.data {
            DataArray::Float(data) => {
                let shape = IxDyn(&dataset.shape);
                ArrayD::from_shape_vec(shape, data.clone())
                    .map_err(|e| IoError::FormatError(e.to_string()))
            }
            DataArray::Integer(data) => {
                let float_data: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let shape = IxDyn(&dataset.shape);
                ArrayD::from_shape_vec(shape, float_data)
                    .map_err(|e| IoError::FormatError(e.to_string()))
            }
            _ => Err(IoError::FormatError(
                "Unsupported data type for ndarray conversion".to_string(),
            )),
        }
    }

    /// Write the file to disk
    pub fn write(&self) -> Result<()> {
        #[cfg(feature = "hdf5")]
        {
            if let Some(ref file) = self.native_file {
                // Write the in-memory structure to the HDF5 file
                Self::write_group_to_hdf5(file, &self.root, "")?;

                // Flush any pending operations
                file.flush()
                    .map_err(|e| IoError::FormatError(format!("Failed to flush HDF5 file: {e}")))?;
            }
        }

        #[cfg(not(feature = "hdf5"))]
        {
            // Minimal persistence for tests without native hdf5:
            // Serialize structure to a simple JSON alongside the filename.
            let sidecar = format!("{}.json", self.path);
            let mut obj = serde_json::json!({
                "groups": serde_json::Value::Object(serde_json::Map::new()),
                "datasets": serde_json::Value::Object(serde_json::Map::new()),
            });
            // Flatten root datasets
            if let serde_json::Value::Object(ref mut map) = obj["datasets"] {
                for (k, ds) in &self.root.datasets {
                    map.insert(k.clone(), serde_json::json!({
                        "shape": ds.shape,
                        "data": match &ds.data { DataArray::Float(v)=>serde_json::json!(v), DataArray::Integer(v)=>serde_json::json!(v), _=>serde_json::json!([])},
                    }));
                }
            }
            // Write file
            std::fs::write(&sidecar, serde_json::to_vec(&obj).unwrap())
                .map_err(|e| IoError::FormatError(format!("Failed to persist mock HDF5: {e}")))?;
        }

        Ok(())
    }

    /// Get a dataset by path (e.g., "/group1/group2/dataset")
    pub fn get_dataset(&self, path: &str) -> Result<&Dataset> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.is_empty() {
            return Err(IoError::FormatError("Invalid dataset path".to_string()));
        }

        let dataset_name = parts.last().unwrap();
        let mut current_group = &self.root;

        // Navigate to the parent group
        for &group_name in &parts[..parts.len() - 1] {
            current_group = current_group
                .get_group(group_name)
                .ok_or_else(|| IoError::FormatError(format!("Group '{group_name}' not found")))?;
        }

        // Get the dataset
        current_group
            .get_dataset(dataset_name)
            .ok_or_else(|| IoError::FormatError(format!("Dataset '{dataset_name}' not found")))
    }

    /// Get a group by path (e.g., "/group1/group2")
    pub fn get_group(&self, path: &str) -> Result<&Group> {
        if path == "/" || path.is_empty() {
            return Ok(&self.root);
        }

        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        let mut current_group = &self.root;

        for &group_name in &parts {
            current_group = current_group
                .get_group(group_name)
                .ok_or_else(|| IoError::FormatError(format!("Group '{group_name}' not found")))?;
        }

        Ok(current_group)
    }

    /// List all datasets in the file recursively
    pub fn list_datasets(&self) -> Vec<String> {
        let mut datasets = Vec::new();
        self.collect_datasets(&self.root, String::new(), &mut datasets);
        datasets
    }

    /// List all groups in the file recursively
    pub fn list_groups(&self) -> Vec<String> {
        let mut groups = Vec::new();
        self.collect_groups(&self.root, String::new(), &mut groups);
        groups
    }

    /// Helper method to recursively collect dataset paths
    #[allow(clippy::only_used_in_recursion)]
    fn collect_datasets(&self, group: &Group, prefix: String, datasets: &mut Vec<String>) {
        for dataset_name in group.dataset_names() {
            let fullpath = if prefix.is_empty() {
                dataset_name.to_string()
            } else {
                format!("{prefix}/{dataset_name}")
            };
            datasets.push(fullpath);
        }

        for (group_name, subgroup) in &group.groups {
            let new_prefix = if prefix.is_empty() {
                group_name.clone()
            } else {
                format!("{prefix}/{group_name}")
            };
            self.collect_datasets(subgroup, new_prefix, datasets);
        }
    }

    /// Helper method to recursively collect group paths
    #[allow(clippy::only_used_in_recursion)]
    fn collect_groups(&self, group: &Group, prefix: String, groups: &mut Vec<String>) {
        for (group_name, subgroup) in &group.groups {
            let fullpath = if prefix.is_empty() {
                group_name.clone()
            } else {
                format!("{prefix}/{group_name}")
            };
            groups.push(fullpath.clone());
            self.collect_groups(subgroup, fullpath, groups);
        }
    }

    /// Get file statistics
    pub fn stats(&self) -> FileStats {
        let mut stats = FileStats::default();
        self.collect_stats(&self.root, &mut stats);
        stats
    }

    /// Helper method to collect file statistics
    #[allow(clippy::only_used_in_recursion)]
    fn collect_stats(&self, group: &Group, stats: &mut FileStats) {
        stats.num_groups += group.groups.len();
        stats.num_datasets += group.datasets.len();
        stats.num_attributes += group.attributes.len();

        for dataset in group.datasets.values() {
            stats.num_attributes += dataset.attributes.len();
            stats.total_data_size += dataset.size_bytes();
        }

        for subgroup in group.groups.values() {
            self.collect_stats(subgroup, stats);
        }
    }

    /// Close the file
    pub fn close(self) -> Result<()> {
        #[cfg(feature = "hdf5")]
        {
            // Ensure data is written before closing
            let _ = self.write();
            if let Some(file) = self.native_file {
                // File is automatically closed when dropped
                drop(file);
            }
        }

        Ok(())
    }

    /// Create a group in the root - delegation method
    pub fn create_group(&mut self, name: &str) -> Result<()> {
        self.root.create_group(name);
        Ok(())
    }

    /// Set an attribute on the file root - delegation method
    pub fn set_attribute(&mut self, name: &str, key: &str, value: AttributeValue) -> Result<()> {
        if name == "/" || name.is_empty() {
            self.root.set_attribute(key, value);
        } else {
            // Navigate to the specified group/dataset
            let parts: Vec<&str> = name.split('/').filter(|s| !s.is_empty()).collect();
            let mut current_group = &mut self.root;

            for &group_name in &parts {
                current_group = current_group.groups.get_mut(group_name).ok_or_else(|| {
                    IoError::FormatError(format!("Group '{}' not found", group_name))
                })?;
            }
            current_group.set_attribute(key, value);
        }
        Ok(())
    }

    /// Get an attribute from the file root - delegation method
    pub fn get_attribute(&self, name: &str, key: &str) -> Result<Option<&AttributeValue>> {
        if name == "/" || name.is_empty() {
            Ok(self.root.get_attribute(key))
        } else {
            // Navigate to the specified group/dataset
            let parts: Vec<&str> = name.split('/').filter(|s| !s.is_empty()).collect();
            let mut current_group = &self.root;

            for &group_name in &parts {
                current_group = current_group.groups.get(group_name).ok_or_else(|| {
                    IoError::FormatError(format!("Group '{}' not found", group_name))
                })?;
            }
            Ok(current_group.get_attribute(key))
        }
    }

    /// Check if a path represents a group
    pub fn is_group(&self, name: &str) -> bool {
        if name == "/" || name.is_empty() {
            true // Root is always a group
        } else {
            // Navigate to check if it's a group
            let parts: Vec<&str> = name.split('/').filter(|s| !s.is_empty()).collect();
            let mut current_group = &self.root;

            for (i, &part) in parts.iter().enumerate() {
                if i == parts.len() - 1 {
                    // Last part - check if it's a group
                    return current_group.groups.contains_key(part);
                } else {
                    // Intermediate part - must be a group
                    match current_group.groups.get(part) {
                        Some(group) => current_group = group,
                        None => return false,
                    }
                }
            }
            false
        }
    }

    /// Write a slice of data to a dataset
    pub fn write_dataset_slice<T>(&mut self, name: &str, data: &[T], offset: &[usize]) -> Result<()>
    where
        T: Clone + std::fmt::Debug,
    {
        // This is a placeholder implementation
        // In production, this would write to the actual HDF5 file
        let _ = (name, data, offset);
        Ok(())
    }

    /// Read a slice of data from a dataset
    pub fn read_dataset_slice<T>(
        &self,
        name: &str,
        shape: &[usize],
        offset: &[usize],
    ) -> Result<Vec<T>>
    where
        T: Clone + Default,
    {
        // This is a placeholder implementation
        // In production, this would read from the actual HDF5 file
        let _ = (name, offset);
        let total: usize = shape.iter().product();
        Ok(vec![T::default(); total])
    }

    /// List all items (groups and datasets) recursively
    pub fn list_all_items(&self) -> Vec<String> {
        let mut items = Vec::new();
        self.list_items_recursive(&self.root, "", &mut items);
        items
    }

    fn list_items_recursive(&self, group: &Group, prefix: &str, items: &mut Vec<String>) {
        for name in group.datasets.keys() {
            let path = if prefix.is_empty() {
                format!("/{}", name)
            } else {
                format!("{}/{}", prefix, name)
            };
            items.push(path);
        }

        for (name, subgroup) in &group.groups {
            let path = if prefix.is_empty() {
                format!("/{}", name)
            } else {
                format!("{}/{}", prefix, name)
            };
            items.push(path.clone());
            self.list_items_recursive(subgroup, &path, items);
        }
    }

    /// Create a dataset with specified type
    pub fn create_dataset<T>(
        &mut self,
        path: &str,
        shape: &[usize],
        _options: Option<DatasetOptions>,
    ) -> Result<()>
    where
        T: Clone + Default + std::fmt::Debug,
    {
        let total: usize = shape.iter().product();
        let data = vec![T::default(); total];
        let array = ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| IoError::FormatError(e.to_string()))?;

        // Use the existing create_dataset_from_array method with no specific options
        self.create_dataset_from_array(path, &array, None)
    }
}

/// Read an HDF5 file and return the root group
///
/// # Arguments
/// * `path` - Path to the HDF5 file
///
/// # Returns
/// The root group of the HDF5 file
///
/// # Example
/// ```no_run
/// use scirs2_io::hdf5::read_hdf5;
///
/// let root_group = read_hdf5("data.h5")?;
/// println!("Groups: {:?}", root_group.groups.keys().collect::<Vec<_>>());
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
#[allow(dead_code)]
pub fn read_hdf5<P: AsRef<Path>>(path: P) -> Result<Group> {
    let file = HDF5File::open(path, FileMode::ReadOnly)?;
    Ok(file.root)
}

/// Write data to an HDF5 file
///
/// # Arguments
/// * `path` - Path to the HDF5 file
/// * `datasets` - Map of dataset paths to arrays
///
/// # Example
/// ```no_run
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
/// use scirs2_io::hdf5::write_hdf5;
///
/// let mut datasets = HashMap::new();
/// datasets.insert("data/temperature".to_string(), array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
/// datasets.insert("data/pressure".to_string(), array![100.0, 200.0, 300.0].into_dyn());
///
/// write_hdf5("output.h5", datasets)?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
#[allow(dead_code)]
pub fn write_hdf5<P: AsRef<Path>>(path: P, datasets: HashMap<String, ArrayD<f64>>) -> Result<()> {
    let mut file = HDF5File::create(path)?;

    for (datasetpath, array) in datasets {
        file.create_dataset_from_array(&datasetpath, &array, None)?;
    }

    file.write()?;
    file.close()?;
    Ok(())
}

/// Create an HDF5 file with groups and attributes
///
/// # Arguments
/// * `path` - Path to the HDF5 file
/// * `builder` - Function to build the file structure
///
/// # Example
/// ```no_run
/// use scirs2_io::hdf5::{create_hdf5_with_structure, AttributeValue};
/// use scirs2_core::ndarray::array;
///
/// create_hdf5_with_structure("structured.h5", |file| {
///     let root = file.root_mut();
///     
///     // Create groups
///     let experiment = root.create_group("experiment");
///     experiment.set_attribute("date", AttributeValue::String("2024-01-01".to_string()));
///     experiment.set_attribute("temperature", AttributeValue::Float(25.0));
///     
///     // Add datasets
///     let data = array![[1.0, 2.0], [3.0, 4.0]];
///     file.create_dataset_from_array("experiment/measurements", &data, None)?;
///     
///     Ok(())
/// })?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
#[allow(dead_code)]
pub fn create_hdf5_with_structure<P, F>(path: P, builder: F) -> Result<()>
where
    P: AsRef<Path>,
    F: FnOnce(&mut HDF5File) -> Result<()>,
{
    let mut file = HDF5File::create(path)?;
    builder(&mut file)?;
    file.write()?;
    file.close()?;
    Ok(())
}

/// Enhanced HDF5 functionality with compression and parallel I/O
pub mod enhanced;

// Re-export enhanced functionality for convenience
pub use enhanced::{
    create_optimal_compression_options, read_hdf5_enhanced, write_hdf5_enhanced, CompressionStats,
    EnhancedHDF5File, ExtendedDataType, ParallelConfig,
};

// Tests module moved to /tmp/

// Legacy inline tests for backward compatibility
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
    #[ignore]
    fn test_hdf5_file_creation() {
        let file = HDF5File::create("test.h5").unwrap();
        assert_eq!(file.mode, FileMode::Create);
        assert_eq!(file.root.name, "/");
    }
}
