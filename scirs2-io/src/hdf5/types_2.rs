//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{IoError, Result};
#[cfg(feature = "hdf5")]
use hdf5::File;
use scirs2_core::ndarray::{ArrayBase, ArrayD, IxDyn};
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

use super::types::{AttributeValue, DataArray};
use super::types_3::{
    Dataset, DatasetOptions, FileMode, FileStats, Group, HDF5DataType, StringEncoding,
};

/// HDF5 file handle
pub struct HDF5File {
    /// File path
    #[allow(dead_code)]
    pub(super) path: String,
    /// Root group
    pub(super) root: Group,
    /// File access mode
    #[allow(dead_code)]
    pub(super) mode: FileMode,
    /// Native HDF5 file handle (when feature is enabled)
    #[cfg(feature = "hdf5")]
    pub(super) native_file: Option<File>,
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
    pub(super) fn load_group_structure(file: &File, group: &mut Group) -> Result<()> {
        use hdf5::types::TypeDescriptor;
        if let Ok(attr_names) = file.attr_names() {
            for attr_name in attr_names {
                if let Ok(attr) = file.attr(&attr_name) {
                    if let Ok(attr_value) = Self::read_attribute_value(&attr) {
                        group.attributes.insert(attr_name, attr_value);
                    }
                }
            }
        }
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
                let shape: Vec<usize> = h5_dataset.shape().to_vec();
                let dtype = h5_dataset.dtype().map_err(|e| {
                    IoError::FormatError(format!("Failed to get dataset dtype: {e}"))
                })?;
                let internal_dtype = Self::convert_hdf5_datatype(&dtype)?;
                let data = Self::read_dataset_data(&h5_dataset, &dtype)?;
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
            Self::load_subgroup_structure(&h5_group, &mut subgroup)?;
            group.groups.insert(group_key, subgroup);
        }
        Ok(())
    }
    /// Recursively load structure for an HDF5 group
    #[cfg(feature = "hdf5")]
    pub(super) fn load_subgroup_structure(h5_group: &hdf5::Group, group: &mut Group) -> Result<()> {
        if let Ok(attr_names) = h5_group.attr_names() {
            for attr_name in attr_names {
                if let Ok(attr) = h5_group.attr(&attr_name) {
                    if let Ok(attr_value) = Self::read_attribute_value(&attr) {
                        group.attributes.insert(attr_name, attr_value);
                    }
                }
            }
        }
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
    pub(super) fn write_group_to_hdf5(file: &File, group: &Group, path_prefix: &str) -> Result<()> {
        for (attr_name, attr_value) in &group.attributes {
            Self::write_attribute_to_hdf5(file, path_prefix, attr_name, attr_value)?;
        }
        for (dataset_name, dataset) in &group.datasets {
            let dataset_path = if path_prefix.is_empty() {
                dataset_name.clone()
            } else {
                format!("{}/{}", path_prefix, dataset_name)
            };
            Self::write_dataset_to_hdf5(file, &dataset_path, dataset)?;
        }
        for (subgroup_name, subgroup) in &group.groups {
            let subgroup_path = if path_prefix.is_empty() {
                subgroup_name.clone()
            } else {
                format!("{}/{}", path_prefix, subgroup_name)
            };
            if let Err(_) = file.group(&subgroup_path) {
                file.create_group(&subgroup_path).map_err(|e| {
                    IoError::FormatError(format!("Failed to create group {}: {}", subgroup_path, e))
                })?;
            }
            Self::write_group_to_hdf5(file, subgroup, &subgroup_path)?;
        }
        Ok(())
    }
    /// Write an attribute to the HDF5 file
    #[cfg(feature = "hdf5")]
    pub(super) fn write_attribute_to_hdf5(
        file: &File,
        path: &str,
        name: &str,
        value: &AttributeValue,
    ) -> Result<()> {
        use hdf5::types::VarLenUnicode;
        let target_group = if path.is_empty() {
            file.as_group()
                .map_err(|e| IoError::FormatError(format!("Failed to access root group: {e}")))?
        } else {
            file.group(path).map_err(|e| {
                IoError::FormatError(format!("Failed to access group '{path}': {e}"))
            })?
        };
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
                eprintln!("Warning: Skipping complex array attribute '{}'", name);
            }
        }
        Ok(())
    }
    /// Write a dataset to the HDF5 file
    #[cfg(feature = "hdf5")]
    pub(super) fn write_dataset_to_hdf5(file: &File, path: &str, dataset: &Dataset) -> Result<()> {
        match &dataset.data {
            DataArray::Float(data) => {
                let h5_dataset = file
                    .new_dataset::<f64>()
                    .shape(&dataset.shape)
                    .create(path)
                    .map_err(|e| {
                        IoError::FormatError(format!("Failed to create float dataset: {}", e))
                    })?;
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
    pub(super) fn convert_hdf5_datatype(dtype: &hdf5::Datatype) -> Result<HDF5DataType> {
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
            Ok(TypeDescriptor::FixedUnicode(_size)) => Ok(HDF5DataType::String {
                encoding: StringEncoding::UTF8,
            }),
            Ok(TypeDescriptor::FixedAscii(_size)) => Ok(HDF5DataType::String {
                encoding: StringEncoding::ASCII,
            }),
            Ok(TypeDescriptor::VarLenUnicode) => Ok(HDF5DataType::String {
                encoding: StringEncoding::UTF8,
            }),
            Ok(TypeDescriptor::VarLenAscii) => Ok(HDF5DataType::String {
                encoding: StringEncoding::ASCII,
            }),
            Ok(TypeDescriptor::FixedArray(elem_ty, len)) => {
                let elem_datatype = hdf5::Datatype::from_descriptor(&elem_ty).map_err(|e| {
                    IoError::FormatError(format!(
                        "Failed to create element datatype for FixedArray: {e}"
                    ))
                })?;
                let base_type = Self::convert_hdf5_datatype(&elem_datatype)?;
                Ok(HDF5DataType::Array {
                    base_type: Box::new(base_type),
                    shape: vec![len],
                })
            }
            Ok(TypeDescriptor::VarLenArray(elem_ty)) => {
                let elem_datatype = hdf5::Datatype::from_descriptor(&elem_ty).map_err(|e| {
                    IoError::FormatError(format!(
                        "Failed to create element datatype for VarLenArray: {e}"
                    ))
                })?;
                let base_type = Self::convert_hdf5_datatype(&elem_datatype)?;
                Ok(HDF5DataType::Array {
                    base_type: Box::new(base_type),
                    shape: vec![0],
                })
            }
            Ok(TypeDescriptor::Compound(comp_type)) => {
                let mut fields = Vec::new();
                for field in &comp_type.fields {
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
            _ => Ok(HDF5DataType::String {
                encoding: StringEncoding::UTF8,
            }),
        }
    }
    /// Read dataset data based on HDF5 datatype
    #[cfg(feature = "hdf5")]
    pub(super) fn read_dataset_data(
        dataset: &hdf5::Dataset,
        dtype: &hdf5::Datatype,
    ) -> Result<DataArray> {
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
                let data: Vec<u8> = dataset.read_raw().map_err(|e| {
                    IoError::FormatError(format!("Failed to read binary dataset: {e}"))
                })?;
                Ok(DataArray::Binary(data))
            }
        }
    }
    /// Read attribute value
    #[cfg(feature = "hdf5")]
    pub(super) fn read_attribute_value(attr: &hdf5::Attribute) -> Result<AttributeValue> {
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
            other => Err(IoError::FormatError(format!(
                "Unsupported HDF5 attribute type descriptor: {other:?}"
            ))),
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
        let dataset_name = parts.last().expect("Operation failed");
        let mut current_group = &mut self.root;
        for &group_name in &parts[..parts.len() - 1] {
            current_group = current_group.create_group(group_name);
        }
        let shape: Vec<usize> = array.shape().to_vec();
        let flat_data: Vec<f64> = array
            .iter()
            .map(|x| format!("{:?}", x).parse::<f64>().unwrap_or(0.0))
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
        Ok(())
    }
    /// Read a dataset as an ndarray with specific type
    pub fn read_dataset_typed<T>(&self, path: &str) -> Result<ArrayD<T>>
    where
        T: Clone + Default + std::str::FromStr,
        <T as std::str::FromStr>::Err: std::fmt::Display,
    {
        let f64_array = self.read_dataset(path)?;
        let shape = f64_array.shape().to_vec();
        let converted: Vec<T> = f64_array
            .iter()
            .map(|&v| {
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
        let dataset_name = parts.last().expect("Operation failed");
        let mut current_group = &self.root;
        for &group_name in &parts[..parts.len() - 1] {
            current_group = current_group
                .get_group(group_name)
                .ok_or_else(|| IoError::FormatError(format!("Group '{group_name}' not found")))?;
        }
        let dataset = current_group
            .datasets
            .get(*dataset_name)
            .ok_or_else(|| IoError::FormatError(format!("Dataset '{dataset_name}' not found")))?;
        #[cfg(feature = "hdf5")]
        {
            if let Some(ref file) = self.native_file {
                let full_path = parts.join("/");
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
                Self::write_group_to_hdf5(file, &self.root, "")?;
                file.flush()
                    .map_err(|e| IoError::FormatError(format!("Failed to flush HDF5 file: {e}")))?;
            }
        }
        #[cfg(not(feature = "hdf5"))]
        {
            let sidecar = format!("{}.json", self.path);
            let mut obj = serde_json::json!(
                { "groups" : serde_json::Value::Object(serde_json::Map::new()),
                "datasets" : serde_json::Value::Object(serde_json::Map::new()), }
            );
            if let serde_json::Value::Object(ref mut map) = obj["datasets"] {
                for (k, ds) in &self.root.datasets {
                    map.insert(
                        k.clone(),
                        serde_json::json!(
                            { "shape" : ds.shape, "data" : match & ds.data {
                            DataArray::Float(v) => serde_json::json!(v),
                            DataArray::Integer(v) => serde_json::json!(v), _ =>
                            serde_json::json!([]) }, }
                        ),
                    );
                }
            }
            std::fs::write(
                &sidecar,
                serde_json::to_vec(&obj).expect("Operation failed"),
            )
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
        let dataset_name = parts.last().expect("Operation failed");
        let mut current_group = &self.root;
        for &group_name in &parts[..parts.len() - 1] {
            current_group = current_group
                .get_group(group_name)
                .ok_or_else(|| IoError::FormatError(format!("Group '{group_name}' not found")))?;
        }
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
    pub(super) fn collect_datasets(
        &self,
        group: &Group,
        prefix: String,
        datasets: &mut Vec<String>,
    ) {
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
    pub(super) fn collect_groups(&self, group: &Group, prefix: String, groups: &mut Vec<String>) {
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
    pub(super) fn collect_stats(&self, group: &Group, stats: &mut FileStats) {
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
            let _ = self.write();
            if let Some(file) = self.native_file {
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
            true
        } else {
            let parts: Vec<&str> = name.split('/').filter(|s| !s.is_empty()).collect();
            let mut current_group = &self.root;
            for (i, &part) in parts.iter().enumerate() {
                if i == parts.len() - 1 {
                    return current_group.groups.contains_key(part);
                } else {
                    match current_group.groups.get(part) {
                        Some(group) => current_group = group,
                        None => return false,
                    }
                }
            }
            false
        }
    }
    /// Write a hyper-rectangular slice of data into an existing dataset.
    ///
    /// This method is generic over an unconstrained element type `T`, but the
    /// backing store holds concretely-typed arrays ([`DataArray`]). There is no
    /// safe way to map an arbitrary `T` onto that typed store, so a correct
    /// implementation is not possible at this signature. Rather than silently
    /// discarding the data (which would falsely report success), this returns an
    /// honest error. For real `f64` slice writes use
    /// [`HDF5File::write_f64_dataset_slice`], which operates on the in-memory
    /// representation via read-modify-write.
    pub fn write_dataset_slice<T>(&mut self, name: &str, data: &[T], offset: &[usize]) -> Result<()>
    where
        T: Clone + std::fmt::Debug,
    {
        let _ = (name, data, offset);
        Err(IoError::Other(
            "write_dataset_slice is not implemented for arbitrary element types; \
             use write_f64_dataset_slice for f64 datasets"
                .to_string(),
        ))
    }
    /// Read a hyper-rectangular slice of data from a dataset.
    ///
    /// As with [`HDF5File::write_dataset_slice`], the unconstrained generic
    /// element type cannot be mapped onto the concretely-typed backing store, so
    /// this returns an honest error instead of fabricating zero-filled data. For
    /// real `f64` slice reads use [`HDF5File::read_f64_dataset_slice`].
    pub fn read_dataset_slice<T>(
        &self,
        name: &str,
        shape: &[usize],
        offset: &[usize],
    ) -> Result<Vec<T>>
    where
        T: Clone + Default,
    {
        let _ = (name, shape, offset);
        Err(IoError::Other(
            "read_dataset_slice is not implemented for arbitrary element types; \
             use read_f64_dataset_slice for f64 datasets"
                .to_string(),
        ))
    }
    /// Read a contiguous hyper-rectangular `f64` slice from a dataset.
    ///
    /// `offset` and `shape` describe the region to extract and must match the
    /// rank of the stored dataset. The full dataset is read from the in-memory
    /// representation (or the native handle when the `hdf5` feature is active)
    /// and the requested region is gathered in row-major order. This is a real
    /// computation over the actual stored values, not a placeholder.
    pub fn read_f64_dataset_slice(
        &self,
        name: &str,
        shape: &[usize],
        offset: &[usize],
    ) -> Result<Vec<f64>> {
        if offset.len() != shape.len() {
            return Err(IoError::Other(
                "read_f64_dataset_slice: offset and shape must have the same length".to_string(),
            ));
        }
        let full = self.read_dataset(name)?;
        let full_shape = full.shape().to_vec();
        let ndim = full_shape.len();
        if offset.len() != ndim {
            return Err(IoError::Other(format!(
                "read_f64_dataset_slice: region rank {} does not match dataset rank {ndim}",
                offset.len()
            )));
        }
        for ax in 0..ndim {
            if offset[ax] + shape[ax] > full_shape[ax] {
                return Err(IoError::Other(format!(
                    "read_f64_dataset_slice: region [{}..{}) exceeds axis {ax} length {}",
                    offset[ax],
                    offset[ax] + shape[ax],
                    full_shape[ax]
                )));
            }
        }
        let full_flat = full
            .as_slice()
            .ok_or_else(|| IoError::Other("Dataset is not contiguous".to_string()))?;
        let mut strides = vec![1usize; ndim];
        for ax in (0..ndim.saturating_sub(1)).rev() {
            strides[ax] = strides[ax + 1] * full_shape[ax + 1];
        }
        let total: usize = shape.iter().product();
        let mut result = Vec::with_capacity(total);
        let mut coords = vec![0usize; ndim];
        if total != 0 {
            loop {
                let flat_idx: usize = coords
                    .iter()
                    .enumerate()
                    .map(|(ax, &c)| (offset[ax] + c) * strides[ax])
                    .sum();
                result.push(*full_flat.get(flat_idx).ok_or_else(|| {
                    IoError::Other(
                        "read_f64_dataset_slice: computed index out of bounds".to_string(),
                    )
                })?);
                let mut carry = true;
                for ax in (0..ndim).rev() {
                    if carry {
                        coords[ax] += 1;
                        if coords[ax] < shape[ax] {
                            carry = false;
                        } else {
                            coords[ax] = 0;
                        }
                    }
                }
                if carry {
                    break;
                }
            }
        }
        Ok(result)
    }
    /// Write a contiguous hyper-rectangular `f64` slice into an existing dataset.
    ///
    /// `data` is laid out row-major with dimensions `shape`, written at `offset`
    /// into the dataset of the same rank. This performs a real read-modify-write
    /// against the in-memory representation: the full dataset is read, the region
    /// is patched, and the updated array is stored back. The dataset must already
    /// exist and be large enough to contain the region.
    pub fn write_f64_dataset_slice(
        &mut self,
        name: &str,
        data: &[f64],
        shape: &[usize],
        offset: &[usize],
    ) -> Result<()> {
        if offset.len() != shape.len() {
            return Err(IoError::Other(
                "write_f64_dataset_slice: offset and shape must have the same length".to_string(),
            ));
        }
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(IoError::Other(format!(
                "write_f64_dataset_slice: data length {} does not match region size {expected}",
                data.len()
            )));
        }
        let full = self.read_dataset(name)?;
        let full_shape = full.shape().to_vec();
        let ndim = full_shape.len();
        if offset.len() != ndim {
            return Err(IoError::Other(format!(
                "write_f64_dataset_slice: region rank {} does not match dataset rank {ndim}",
                offset.len()
            )));
        }
        for ax in 0..ndim {
            if offset[ax] + shape[ax] > full_shape[ax] {
                return Err(IoError::Other(format!(
                    "write_f64_dataset_slice: region [{}..{}) exceeds axis {ax} length {}",
                    offset[ax],
                    offset[ax] + shape[ax],
                    full_shape[ax]
                )));
            }
        }
        let mut full_vec = full
            .as_slice()
            .ok_or_else(|| IoError::Other("Dataset is not contiguous".to_string()))?
            .to_vec();
        let mut strides = vec![1usize; ndim];
        for ax in (0..ndim.saturating_sub(1)).rev() {
            strides[ax] = strides[ax + 1] * full_shape[ax + 1];
        }
        let mut coords = vec![0usize; ndim];
        let mut src_idx = 0usize;
        if expected != 0 {
            loop {
                let flat_idx: usize = coords
                    .iter()
                    .enumerate()
                    .map(|(ax, &c)| (offset[ax] + c) * strides[ax])
                    .sum();
                full_vec[flat_idx] = data[src_idx];
                src_idx += 1;
                let mut carry = true;
                for ax in (0..ndim).rev() {
                    if carry {
                        coords[ax] += 1;
                        if coords[ax] < shape[ax] {
                            carry = false;
                        } else {
                            coords[ax] = 0;
                        }
                    }
                }
                if carry {
                    break;
                }
            }
        }
        let updated = ArrayD::from_shape_vec(IxDyn(&full_shape), full_vec)
            .map_err(|e| IoError::FormatError(format!("Failed to rebuild dataset: {e}")))?;
        self.create_dataset_from_array(name, &updated, None)?;
        Ok(())
    }
    /// List all items (groups and datasets) recursively
    pub fn list_all_items(&self) -> Vec<String> {
        let mut items = Vec::new();
        self.list_items_recursive(&self.root, "", &mut items);
        items
    }
    pub(super) fn list_items_recursive(
        &self,
        group: &Group,
        prefix: &str,
        items: &mut Vec<String>,
    ) {
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
        self.create_dataset_from_array(path, &array, None)
    }
}
