//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{IoError, Result};
use oxih5::{
    AttrView, Dataset as OxiDataset, Dtype, File as OxiFile, FileWriter, Group as OxiGroup,
};
use scirs2_core::ndarray::{ArrayBase, ArrayD, IxDyn};
use std::collections::HashMap;
use std::path::Path;

use super::convert::{convert_dtype, dataset_to_f64, dataset_to_i64, is_floating, is_integral};
use super::types::{AttributeValue, DataArray};
use super::types_3::{Dataset, DatasetOptions, FileMode, FileStats, Group, HDF5DataType};

/// Report a construct the pure-Rust writer cannot express yet.
///
/// oxih5's `FileWriter` is being extended in parallel; until it lands the
/// missing capability the only honest outcome is a named error. Dropping the
/// construct and reporting success is what this migration exists to remove.
fn unsupported_write(construct: &str, requirement: &str) -> IoError {
    IoError::UnsupportedFormat(format!(
        "{construct} cannot be written: oxih5's FileWriter does not yet support {requirement}"
    ))
}

/// Iterate a name-keyed map in a stable order.
///
/// `HashMap` iteration order varies between runs, which would make the bytes of
/// an otherwise identical output file differ run to run.
fn sorted_pairs<V>(map: &HashMap<String, V>) -> Vec<(&String, &V)> {
    let mut pairs: Vec<(&String, &V)> = map.iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    pairs
}

/// HDF5 file handle.
///
/// This is a complete in-memory model of the file: [`HDF5File::open`] walks the
/// whole object tree eagerly and materialises every dataset's data into
/// [`HDF5File::root`]. There is consequently no live file handle to keep — reads
/// are served from `root`, and [`HDF5File::write`] serialises `root` back out in
/// one pass.
pub struct HDF5File {
    /// File path
    #[allow(dead_code)]
    pub(super) path: String,
    /// Root group
    pub(super) root: Group,
    /// File access mode
    #[allow(dead_code)]
    pub(super) mode: FileMode,
}
impl HDF5File {
    /// Create a new HDF5 file.
    ///
    /// The structure is built in memory; nothing reaches `path` until
    /// [`HDF5File::write`] or [`HDF5File::close`] is called.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self {
            path: path.as_ref().to_string_lossy().to_string(),
            root: Group::new("/".to_string()),
            mode: FileMode::Create,
        })
    }
    /// Open an existing HDF5 file.
    ///
    /// The whole object tree — groups, datasets, payloads and attributes — is
    /// read eagerly, after which the file handle is released. [`FileMode::Create`]
    /// and [`FileMode::Truncate`] discard whatever is on disk, so nothing is read
    /// for those modes.
    pub fn open<P: AsRef<Path>>(path: P, mode: FileMode) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let mut root = Group::new("/".to_string());
        if matches!(mode, FileMode::ReadOnly | FileMode::ReadWrite) {
            // Scoped so the mapped file is dropped before we return.
            let file = OxiFile::open(&path_str).map_err(|e| {
                IoError::FormatError(format!("Failed to open HDF5 file '{path_str}': {e}"))
            })?;
            Self::load_group_structure(&file, &mut root)?;
        }
        Ok(Self {
            path: path_str,
            root,
            mode,
        })
    }
    /// Get the root group
    pub fn root(&self) -> &Group {
        &self.root
    }
    /// Get the root group mutably
    pub fn root_mut(&mut self) -> &mut Group {
        &mut self.root
    }
    /// Load the complete object tree of `file` into `group`.
    ///
    /// The recursion is spelled out rather than delegated to
    /// [`oxih5::File::walk`]: `walk` swallows any error hit while descending into
    /// a sub-group, so a partially unreadable file would come back as a silently
    /// truncated tree instead of an error.
    pub(super) fn load_group_structure(file: &OxiFile, group: &mut Group) -> Result<()> {
        let root = file
            .root()
            .map_err(|e| IoError::FormatError(format!("Failed to open root group: {e}")))?;
        Self::load_group_attributes(&root, "/", group)?;
        Self::load_group_contents(file, &root, "", group)
    }
    /// Read the attributes attached to an oxih5 group object into `group`.
    pub(super) fn load_group_attributes(
        oxi_group: &OxiGroup,
        path: &str,
        group: &mut Group,
    ) -> Result<()> {
        let views = oxi_group.attr_views().map_err(|e| {
            IoError::FormatError(format!("Failed to read attributes of group '{path}': {e}"))
        })?;
        for view in &views {
            let value = Self::read_attribute_value(view, path)?;
            group.attributes.insert(view.name().to_string(), value);
        }
        Ok(())
    }
    /// Recursively load the datasets and sub-groups of `oxi_group`.
    ///
    /// `prefix` is the slash-separated path of `oxi_group` relative to the root
    /// (empty for the root itself). It is needed because attribute lookups and
    /// string decoding go through absolute paths on [`oxih5::File`], while
    /// enumeration goes through the group handle.
    pub(super) fn load_group_contents(
        file: &OxiFile,
        oxi_group: &OxiGroup,
        prefix: &str,
        group: &mut Group,
    ) -> Result<()> {
        let dataset_names = oxi_group.datasets().map_err(|e| {
            IoError::FormatError(format!("Failed to list datasets of '{prefix}/': {e}"))
        })?;
        for name in dataset_names {
            let full_path = Self::join_path(prefix, &name);
            let oxi_dataset = oxi_group.dataset(&name).map_err(|e| {
                IoError::FormatError(format!("Failed to read dataset '{full_path}': {e}"))
            })?;
            let dtype = convert_dtype(&oxi_dataset.dtype);
            let data = Self::read_dataset_data(file, &full_path, &oxi_dataset)?;
            let mut attributes = HashMap::new();
            let views = file.attr_views(&full_path).map_err(|e| {
                IoError::FormatError(format!(
                    "Failed to read attributes of dataset '{full_path}': {e}"
                ))
            })?;
            for view in &views {
                let value = Self::read_attribute_value(view, &full_path)?;
                attributes.insert(view.name().to_string(), value);
            }
            group.datasets.insert(
                name.clone(),
                Dataset {
                    name,
                    dtype,
                    shape: oxi_dataset.shape.clone(),
                    data,
                    attributes,
                    options: DatasetOptions::default(),
                },
            );
        }
        let group_names = oxi_group.groups().map_err(|e| {
            IoError::FormatError(format!("Failed to list groups of '{prefix}/': {e}"))
        })?;
        for name in group_names {
            let full_path = Self::join_path(prefix, &name);
            let child = oxi_group.group(&name).map_err(|e| {
                IoError::FormatError(format!("Failed to open group '{full_path}': {e}"))
            })?;
            let mut subgroup = Group::new(name.clone());
            Self::load_group_attributes(&child, &full_path, &mut subgroup)?;
            Self::load_group_contents(file, &child, &full_path, &mut subgroup)?;
            group.groups.insert(name, subgroup);
        }
        Ok(())
    }
    /// Join a group prefix and a child name into an object path.
    fn join_path(prefix: &str, name: &str) -> String {
        if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}/{name}")
        }
    }
    /// Serialise `group` and everything below it into `writer`.
    ///
    /// `prefix` is the group's path from the root, empty for the root itself.
    /// oxih5's writer resolves object paths at any depth and creates the
    /// intermediate groups a dataset path implies, so the SciRS2 tree maps
    /// across one-to-one with no flattening.
    ///
    /// Order matters twice over: a group must exist before an attribute can be
    /// attached to it, and a dataset must exist before its own attributes can.
    fn write_group_to(writer: &mut FileWriter, group: &Group, prefix: &str) -> Result<()> {
        // The root group's attribute path is "/"; "" resolves to it as well but
        // "/" is what the writer documents.
        let group_path = if prefix.is_empty() { "/" } else { prefix };
        Self::write_attributes(writer, group_path, &group.attributes)?;

        for (name, dataset) in sorted_pairs(&group.datasets) {
            let path = Self::join_path(prefix, name);
            Self::write_dataset_to(writer, &path, dataset)?;
            Self::write_attributes(writer, &path, &dataset.attributes)?;
        }
        for (name, subgroup) in sorted_pairs(&group.groups) {
            let path = Self::join_path(prefix, name);
            // Created explicitly rather than left to a dataset path, so that a
            // group holding only sub-groups — or nothing at all — still appears
            // in the file. The parent is always created before its children, so
            // this never collides with an implicitly created intermediate.
            writer.create_group(&path).map_err(|e| {
                IoError::FormatError(format!("Failed to create group '{path}': {e}"))
            })?;
            Self::write_group_to(writer, subgroup, &path)?;
        }
        Ok(())
    }
    /// Attach `attributes` to the dataset or group at `path`.
    ///
    /// `AttributeValue::Boolean` is stored as a 0/1 signed integer, matching
    /// what the C backend wrote; it reads back as
    /// [`AttributeValue::Integer`], since HDF5 has no boolean datatype.
    fn write_attributes(
        writer: &mut FileWriter,
        path: &str,
        attributes: &HashMap<String, AttributeValue>,
    ) -> Result<()> {
        for (key, value) in sorted_pairs(attributes) {
            let outcome = match value {
                AttributeValue::String(v) => writer.write_string_attr(path, key, v),
                AttributeValue::Integer(v) => writer.write_i64_attr(path, key, *v),
                AttributeValue::Float(v) => writer.write_f64_attr(path, key, *v),
                AttributeValue::Boolean(v) => writer.write_i64_attr(path, key, i64::from(*v)),
                AttributeValue::IntegerArray(v) | AttributeValue::Array(v) => {
                    writer.write_i64_array_attr(path, key, v)
                }
                AttributeValue::FloatArray(v) => writer.write_f64_array_attr(path, key, v),
                AttributeValue::StringArray(v) => {
                    let refs: Vec<&str> = v.iter().map(String::as_str).collect();
                    writer.write_string_array_attr(path, key, &refs)
                }
            };
            outcome.map_err(|e| {
                IoError::FormatError(format!(
                    "Failed to write attribute '{key}' on '{path}': {e}"
                ))
            })?;
        }
        Ok(())
    }
    /// Serialise one dataset at `path`.
    fn write_dataset_to(writer: &mut FileWriter, path: &str, dataset: &Dataset) -> Result<()> {
        let shape = &dataset.shape;
        let outcome = match &dataset.data {
            DataArray::Float(data) => writer.write_dataset_f64(path, data, shape).map(|_| ()),
            DataArray::Integer(data) => writer.write_dataset_i64(path, data, shape).map(|_| ()),
            DataArray::Binary(data) => writer.write_dataset_u8(path, data, shape).map(|_| ()),
            DataArray::String(data) => {
                if shape.len() > 1 {
                    return Err(unsupported_write(
                        &format!("{}-D string dataset '{path}'", shape.len()),
                        "multi-dimensional string datasets \
                         (create_vlen_string_dataset lays out a single axis)",
                    ));
                }
                let declared: usize = shape.iter().product();
                if !shape.is_empty() && declared != data.len() {
                    return Err(IoError::FormatError(format!(
                        "String dataset '{path}' declares shape {shape:?} ({declared} elements) \
                         but holds {} strings",
                        data.len()
                    )));
                }
                let refs: Vec<&str> = data.iter().map(String::as_str).collect();
                writer.create_vlen_string_dataset(path, &refs)
            }
        };
        outcome.map_err(|e| IoError::FormatError(format!("Failed to add dataset '{path}': {e}")))
    }
    /// Decode a dataset's payload into SciRS2's [`DataArray`].
    ///
    /// Every numeric read goes through the widening helpers in
    /// [`super::convert`]. The C backend converted implicitly inside
    /// `read_raw::<T>()`; matching only the exact Rust type here would quietly
    /// reduce SciRS2 from "any numeric dataset" to "f64 datasets only".
    pub(super) fn read_dataset_data(
        file: &OxiFile,
        path: &str,
        dataset: &OxiDataset,
    ) -> Result<DataArray> {
        if matches!(dataset.dtype, Dtype::String { .. }) {
            // `File::dataset_strings` handles fixed-length *and* variable-length
            // strings; `Dataset::as_string` returns NotImplemented for vlen,
            // whose elements are global-heap references needing the file bytes.
            let strings = file.dataset_strings(path).map_err(|e| {
                IoError::FormatError(format!("Failed to read string dataset '{path}': {e}"))
            })?;
            return Ok(DataArray::String(strings));
        }
        if is_floating(&dataset.dtype) {
            return Ok(DataArray::Float(dataset_to_f64(dataset)?));
        }
        if is_integral(&dataset.dtype) {
            return Ok(DataArray::Integer(dataset_to_i64(dataset)?));
        }
        // Compound, opaque, reference, vlen-sequence and array datatypes have no
        // scalar counterpart in `DataArray`. Their bytes are kept verbatim rather
        // than reinterpreted as something they are not.
        Ok(DataArray::Binary(dataset.data.clone()))
    }
    /// Re-present an attribute's payload as a dataset so oxih5's element
    /// decoders apply unchanged.
    ///
    /// `AttrView::attr` and every `oxih5_core::Dataset` field are public, so no
    /// decoding logic needs duplicating here. The payload is trimmed to the
    /// element count the dataspace declares: an attribute's data section is
    /// padded out to an 8-byte boundary inside its object-header message, and
    /// decoding that padding would invent trailing elements.
    pub(crate) fn attribute_as_dataset(view: &AttrView<'_>, path: &str) -> Result<OxiDataset> {
        let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
        let n_elems: usize = shape.iter().product();
        let dtype = view.attr.dtype.clone();
        let data = match dtype.size() {
            Some(elem_size) => {
                let needed = n_elems.checked_mul(elem_size).ok_or_else(|| {
                    IoError::FormatError(format!(
                        "Attribute '{}' on '{path}' declares an unrepresentable shape {shape:?}",
                        view.name()
                    ))
                })?;
                view.attr
                    .data
                    .get(..needed)
                    .ok_or_else(|| {
                        IoError::FormatError(format!(
                            "Attribute '{}' on '{path}' is truncated: {} bytes present, {needed} required",
                            view.name(),
                            view.attr.data.len()
                        ))
                    })?
                    .to_vec()
            }
            None => view.attr.data.clone(),
        };
        Ok(OxiDataset {
            data,
            shape,
            dtype,
            attributes: Vec::new(),
            max_dims: None,
        })
    }
    /// Decode an attribute into SciRS2's [`AttributeValue`].
    ///
    /// Takes an [`AttrView`] rather than a bare `Attribute` because that is the
    /// only route that can decode variable-length string attributes: their
    /// elements are 16-byte global-heap references which need the file bytes to
    /// resolve, and an `Attribute` does not carry them.
    pub(super) fn read_attribute_value(view: &AttrView<'_>, path: &str) -> Result<AttributeValue> {
        let name = view.name();
        let scalar = view.is_scalar();
        match &view.attr.dtype {
            Dtype::String { .. } => {
                let strings = view.as_strings().map_err(|e| {
                    IoError::FormatError(format!(
                        "Failed to decode string attribute '{name}' on '{path}': {e}"
                    ))
                })?;
                if scalar {
                    Ok(AttributeValue::String(
                        strings.into_iter().next().unwrap_or_default(),
                    ))
                } else {
                    Ok(AttributeValue::StringArray(strings))
                }
            }
            dtype if is_floating(dtype) => {
                let values = dataset_to_f64(&Self::attribute_as_dataset(view, path)?)?;
                if scalar {
                    let first = values.first().copied().ok_or_else(|| {
                        IoError::FormatError(format!(
                            "Scalar float attribute '{name}' on '{path}' decoded to no value"
                        ))
                    })?;
                    Ok(AttributeValue::Float(first))
                } else {
                    Ok(AttributeValue::FloatArray(values))
                }
            }
            dtype if is_integral(dtype) => {
                let values = dataset_to_i64(&Self::attribute_as_dataset(view, path)?)?;
                if scalar {
                    let first = values.first().copied().ok_or_else(|| {
                        IoError::FormatError(format!(
                            "Scalar integer attribute '{name}' on '{path}' decoded to no value"
                        ))
                    })?;
                    Ok(AttributeValue::Integer(first))
                } else {
                    Ok(AttributeValue::IntegerArray(values))
                }
            }
            other => Err(IoError::UnsupportedFormat(format!(
                "Attribute '{name}' on '{path}' has datatype {other}, which has no \
                 AttributeValue representation"
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
        A::Elem: Clone + Into<f64>,
        D: scirs2_core::ndarray::Dimension,
    {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        let dataset_name = *parts
            .last()
            .ok_or_else(|| IoError::FormatError("Invalid dataset path".to_string()))?;
        let mut current_group = &mut self.root;
        for &group_name in &parts[..parts.len() - 1] {
            current_group = current_group.create_group(group_name);
        }
        let shape: Vec<usize> = array.shape().to_vec();
        // `A::Elem: Into<f64>` gives an exact, total conversion. The previous
        // implementation round-tripped each element through `format!("{:?}")`
        // and `parse::<f64>()`, falling back to `0.0` for anything whose Debug
        // output is not a bare float literal — silently zeroing the data.
        let flat_data: Vec<f64> = array.iter().map(|x| x.clone().into()).collect();
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
        let dataset_name = *parts
            .last()
            .ok_or_else(|| IoError::FormatError("Invalid dataset path".to_string()))?;
        let mut current_group = &self.root;
        for &group_name in &parts[..parts.len() - 1] {
            current_group = current_group
                .get_group(group_name)
                .ok_or_else(|| IoError::FormatError(format!("Group '{group_name}' not found")))?;
        }
        let dataset = current_group
            .datasets
            .get(dataset_name)
            .ok_or_else(|| IoError::FormatError(format!("Dataset '{dataset_name}' not found")))?;
        // `open()` already materialised every payload, and the widening dispatch
        // in `super::convert` ran at that point, so `dataset.data` is the
        // authoritative copy — there is no file handle left to re-read from.
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
    /// Serialise the in-memory tree to `self.path` as a real HDF5 file.
    ///
    /// # Errors
    ///
    /// Refuses to run for [`FileMode::ReadOnly`]. `FileWriter::build` rewrites
    /// the target from scratch rather than patching it, so
    /// `open(.., ReadOnly, ..)` → `close()` would otherwise destroy the very
    /// file that was just read. Under the C backend the same sequence failed
    /// harmlessly because the underlying handle was itself read-only.
    ///
    /// Also returns [`IoError::UnsupportedFormat`], naming the exact construct,
    /// for structures oxih5's writer cannot express yet — see
    /// the private `unsupported_write` helper.
    pub fn write(&self) -> Result<()> {
        if self.mode == FileMode::ReadOnly {
            return Err(IoError::Other(format!(
                "refusing to write '{}': the file was opened read-only, and writing \
                 rebuilds it from scratch, which would destroy its contents",
                self.path
            )));
        }
        let mut writer = FileWriter::new();
        Self::write_group_to(&mut writer, &self.root, "")?;
        writer.build(&self.path).map_err(|e| {
            IoError::FormatError(format!("Failed to write HDF5 file '{}': {e}", self.path))
        })
    }
    /// Serialise the in-memory tree and report how many bytes it occupies.
    ///
    /// Runs the same `FileWriter` pipeline as [`HDF5File::write`] but keeps the
    /// result in memory, so the answer is the real on-disk size rather than an
    /// estimate derived from element counts. Used by
    /// [`super::enhanced::EnhancedHDF5File::get_compression_stats`] to report a
    /// measured compression ratio.
    ///
    /// # Errors
    ///
    /// The same layout failures [`HDF5File::write`] reports; unlike `write` it
    /// does not refuse [`FileMode::ReadOnly`], because nothing is written.
    pub(super) fn serialized_len(&self) -> Result<usize> {
        let mut writer = FileWriter::new();
        Self::write_group_to(&mut writer, &self.root, "")?;
        let bytes = writer.build_to_vec().map_err(|e| {
            IoError::FormatError(format!("Failed to serialise the HDF5 object tree: {e}"))
        })?;
        Ok(bytes.len())
    }
    /// Get a dataset by path (e.g., "/group1/group2/dataset")
    pub fn get_dataset(&self, path: &str) -> Result<&Dataset> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        let dataset_name = *parts
            .last()
            .ok_or_else(|| IoError::FormatError("Invalid dataset path".to_string()))?;
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
    /// Close the file, flushing the in-memory tree to disk first.
    ///
    /// Read-only handles have nothing to flush and close without touching the
    /// file. For every other mode the write is performed and its result is
    /// propagated: the previous implementation discarded it, so a failed flush
    /// looked exactly like a successful close.
    pub fn close(self) -> Result<()> {
        match self.mode {
            FileMode::ReadOnly => Ok(()),
            _ => self.write(),
        }
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
        T: Clone + Default + Into<f64>,
    {
        let total: usize = shape.iter().product();
        let data = vec![T::default(); total];
        let array = ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| IoError::FormatError(e.to_string()))?;
        self.create_dataset_from_array(path, &array, None)
    }
}
