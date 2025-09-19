//! Implementation of record arrays for structured data with named fields
//!
//! Record arrays are useful in scientific computing when working with:
//! - Tabular data with different types per column
//! - Data with named fields
//! - Structured data that mixes different types
//!
//! The implementation is inspired by ``NumPy``'s `RecordArray`.

use ndarray::{Array, Ix1};
use std::collections::HashMap;
use std::fmt;

// Use the array error from masked_array.rs
use super::masked_array::ArrayError;

/// Enum to hold different types of field values
#[derive(Clone, Debug)]
pub enum FieldValue {
    Bool(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    String(String),
    // Could add more types as needed
}

// Implement conversions from common Rust types to FieldValue
impl From<bool> for FieldValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i8> for FieldValue {
    fn from(value: i8) -> Self {
        Self::Int8(value)
    }
}

impl From<i16> for FieldValue {
    fn from(value: i16) -> Self {
        Self::Int16(value)
    }
}

impl From<i32> for FieldValue {
    fn from(value: i32) -> Self {
        Self::Int32(value)
    }
}

impl From<i64> for FieldValue {
    fn from(value: i64) -> Self {
        Self::Int64(value)
    }
}

impl From<u8> for FieldValue {
    fn from(value: u8) -> Self {
        Self::UInt8(value)
    }
}

impl From<u16> for FieldValue {
    fn from(value: u16) -> Self {
        Self::UInt16(value)
    }
}

impl From<u32> for FieldValue {
    fn from(value: u32) -> Self {
        Self::UInt32(value)
    }
}

impl From<u64> for FieldValue {
    fn from(value: u64) -> Self {
        Self::UInt64(value)
    }
}

impl From<f32> for FieldValue {
    fn from(value: f32) -> Self {
        Self::Float32(value)
    }
}

impl From<f64> for FieldValue {
    fn from(value: f64) -> Self {
        Self::Float64(value)
    }
}

impl From<&str> for FieldValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<String> for FieldValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl fmt::Display for FieldValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{v}"),
            Self::Int8(v) => write!(f, "{v}"),
            Self::Int16(v) => write!(f, "{v}"),
            Self::Int32(v) => write!(f, "{v}"),
            Self::Int64(v) => write!(f, "{v}"),
            Self::UInt8(v) => write!(f, "{v}"),
            Self::UInt16(v) => write!(f, "{v}"),
            Self::UInt32(v) => write!(f, "{v}"),
            Self::UInt64(v) => write!(f, "{v}"),
            Self::Float32(v) => write!(f, "{v}"),
            Self::Float64(v) => write!(f, "{v}"),
            Self::String(v) => write!(f, "\"{v}\""),
        }
    }
}

/// Represents a single record (row) in a `RecordArray`
#[derive(Clone, Debug, Default)]
pub struct Record {
    /// Map of field names to values
    fields: HashMap<String, FieldValue>,

    /// Field names in order
    names: Vec<String>,
}

impl Record {
    /// Create a new empty record
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a field to the record
    pub fn add_field(&mut self, name: &str, value: FieldValue) {
        if !self.fields.contains_key(name) {
            self.names.push(name.to_string());
        }
        self.fields.insert(name.to_string(), value);
    }

    /// Get a field value by name
    #[must_use]
    pub fn get_field(&self, name: &str) -> Option<&FieldValue> {
        self.fields.get(name)
    }

    /// Get a mutable reference to a field value by name
    pub fn get_field_mut(&mut self, name: &str) -> Option<&mut FieldValue> {
        self.fields.get_mut(name)
    }

    /// Get the number of fields
    #[must_use]
    pub fn num_fields(&self) -> usize {
        self.fields.len()
    }

    /// Get the field names
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn names(&self) -> &[String] {
        &self.names
    }
    /// Pretty print the record
    #[must_use]
    pub fn pprint(&self) -> String {
        let mut result = String::new();

        let max_name_len = self
            .names
            .iter()
            .map(std::string::String::len)
            .max()
            .unwrap_or(0);

        for name in &self.names {
            if let Some(value) = self.fields.get(name) {
                use std::fmt::Write;
                let _ = writeln!(&mut result, "{name:<max_name_len$}: {value}");
            }
        }

        result
    }
}

impl fmt::Display for Record {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({})",
            self.names
                .iter()
                .filter_map(|name| self.fields.get(name).map(|v| format!("{name}: {v}")))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// A structured array with named fields
#[derive(Clone, Debug)]
pub struct RecordArray {
    /// The array of records
    pub records: Vec<Record>,

    /// The names of the fields (columns)
    pub names: Vec<String>,

    /// Optional titles (aliases) for fields
    pub field_titles: HashMap<String, String>,

    /// Maps field names to their index in `names`
    field_indices: HashMap<String, usize>,

    /// The shape of the array
    shape: Vec<usize>,

    /// Whether fields can be accessed by attribute
    allow_field_attributes: bool,
}

impl RecordArray {
    /// Create a new `RecordArray` from a vector of records
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if records are empty or have inconsistent field structures.
    pub fn new(records: Vec<Record>) -> Result<Self, ArrayError> {
        if records.is_empty() {
            return Err(ArrayError::ValueError(
                "Records cannot be empty".to_string(),
            ));
        }

        // Get field names from the first record
        let names = records[0].names().to_vec();

        // Verify all records have the same fields
        for (i, record) in records.iter().enumerate().skip(1) {
            let record_fields = record.names();
            if record_fields.len() != names.len() {
                return Err(ArrayError::ValueError(format!(
                    "Record {i} has {} fields, but expected {}",
                    record_fields.len(),
                    names.len()
                )));
            }

            for name in &names {
                if !record_fields.contains(name) {
                    return Err(ArrayError::ValueError(format!(
                        "Record {i} is missing field '{name}'"
                    )));
                }
            }
        }

        // Create field index map
        let mut field_indices = HashMap::new();
        for name in names.iter() {
            field_indices.insert(name.clone(), 0);
        }

        // Store the length
        let len = records.len();

        Ok(Self {
            records,
            names,
            field_titles: HashMap::new(),
            field_indices,
            shape: vec![len],
            allow_field_attributes: true,
        })
    }

    /// Create a new `RecordArray` with custom field titles
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if records are invalid or titles reference non-existent fields.
    pub fn with_titles(
        records: Vec<Record>,
        titles: HashMap<String, String>,
    ) -> Result<Self, ArrayError> {
        let mut record_array = Self::new(records)?;

        // Validate titles
        for name in titles.keys() {
            if !record_array.field_indices.contains_key(name) {
                return Err(ArrayError::ValueError(format!(
                    "Cannot add title for non-existent field '{name}'"
                )));
            }
        }

        record_array.field_titles = titles;
        Ok(record_array)
    }

    /// Enable or disable attribute-style field access
    pub const fn set_allow_field_attributes(&mut self, allow: bool) {
        self.allow_field_attributes = allow;
    }

    /// Get whether attribute-style field access is allowed
    #[must_use]
    pub const fn allow_field_attributes(&self) -> bool {
        self.allow_field_attributes
    }

    /// Get the shape of the array
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get all values for a specific field across all records
    pub fn field_values(&self, name: &str) -> Result<Vec<FieldValue>, ArrayError> {
        if !self.names.contains(&name.to_string()) {
            return Err(ArrayError::ValueError(format!(
                "Field '{}' not found",
                name
            )));
        }

        let mut values = Vec::with_capacity(self.records.len());
        for record in &self.records {
            if let Some(value) = record.get_field(name) {
                values.push(value.clone());
            } else {
                return Err(ArrayError::ValueError(format!(
                    "Field '{}' missing in some records",
                    name
                )));
            }
        }

        Ok(values)
    }

    /// Get the number of records
    #[must_use]
    pub fn numrecords(&self) -> usize {
        self.records.len()
    }

    /// Get a reference to a record by index
    #[must_use]
    pub fn get_record(&self, index: usize) -> Option<&Record> {
        self.records.get(index)
    }

    /// Get a mutable reference to a record by index
    pub fn get_record_mut(&mut self, index: usize) -> Option<&mut Record> {
        self.records.get_mut(index)
    }

    /// Get a field values as a vector
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if the field name is not found.
    ///
    /// # Panics
    /// Panics if a record doesn't have the field that should exist based on validation.
    pub fn get_field(&self, name: &str) -> Result<Vec<FieldValue>, ArrayError> {
        if !self.field_indices.contains_key(name) {
            return Err(ArrayError::ValueError(format!("Field '{name}' not found")));
        }

        let values = self
            .records
            .iter()
            .map(|record| {
                record
                    .get_field(name)
                    .expect("Field should exist based on validation")
                    .clone()
            })
            .collect();

        Ok(values)
    }

    /// Get a field as an array of f64 values
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if the field name is not found.
    #[allow(clippy::cast_precision_loss)]
    pub fn field_as_f64_array(&self, name: &str) -> Result<Array<f64, Ix1>, ArrayError> {
        let values = self.field_values(name)?;

        let mut result = Array::zeros(self.records.len());

        for (i, value) in values.iter().enumerate() {
            let val = match value {
                FieldValue::Bool(v) => {
                    if *v {
                        1.0
                    } else {
                        0.0
                    }
                }
                FieldValue::Int8(v) => f64::from(*v),
                FieldValue::Int16(v) => f64::from(*v),
                FieldValue::Int32(v) => f64::from(*v),
                FieldValue::Int64(v) => *v as f64,
                FieldValue::UInt8(v) => f64::from(*v),
                FieldValue::UInt16(v) => f64::from(*v),
                FieldValue::UInt32(v) => f64::from(*v),
                FieldValue::UInt64(v) => *v as f64,
                FieldValue::Float32(v) => f64::from(*v),
                FieldValue::Float64(v) => *v,
                FieldValue::String(_) => {
                    return Err(ArrayError::ValueError(format!(
                        "Cannot convert field '{name}' of type String to f64"
                    )))
                }
            };

            result[i] = val;
        }

        Ok(result)
    }

    /// Get a field as an array of i64 values
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if the field name is not found or contains non-convertible types.
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn field_as_i64_array(&self, name: &str) -> Result<Array<i64, Ix1>, ArrayError> {
        let values = self.field_values(name)?;

        let mut result = Array::zeros(self.records.len());

        for (i, value) in values.iter().enumerate() {
            let val = match value {
                FieldValue::Bool(v) => i64::from(*v),
                FieldValue::Int8(v) => i64::from(*v),
                FieldValue::Int16(v) => i64::from(*v),
                FieldValue::Int32(v) => i64::from(*v),
                FieldValue::Int64(v) => *v,
                FieldValue::UInt8(v) => i64::from(*v),
                FieldValue::UInt16(v) => i64::from(*v),
                FieldValue::UInt32(v) => i64::from(*v),
                FieldValue::UInt64(v) => {
                    if *v > i64::MAX as u64 {
                        return Err(ArrayError::ValueError(format!(
                            "Value {v} in field '{name}' is too large for i64"
                        )));
                    }
                    *v as i64
                }
                FieldValue::Float32(v) => *v as i64,
                FieldValue::Float64(v) => *v as i64,
                FieldValue::String(_) => {
                    return Err(ArrayError::ValueError(format!(
                        "Cannot convert field '{name}' of type String to i64"
                    )))
                }
            };

            result[i] = val;
        }

        Ok(result)
    }

    /// Get a field as an array of String values
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if the field name is not found.
    pub fn name_4(&self, name: &str) -> Result<Vec<String>, ArrayError> {
        let values = self.field_values(name)?;

        let mut result = Vec::with_capacity(self.records.len());

        for value in values {
            let val = match value {
                FieldValue::Bool(v) => v.to_string(),
                FieldValue::Int8(v) => v.to_string(),
                FieldValue::Int16(v) => v.to_string(),
                FieldValue::Int32(v) => v.to_string(),
                FieldValue::Int64(v) => v.to_string(),
                FieldValue::UInt8(v) => v.to_string(),
                FieldValue::UInt16(v) => v.to_string(),
                FieldValue::UInt32(v) => v.to_string(),
                FieldValue::UInt64(v) => v.to_string(),
                FieldValue::Float32(v) => v.to_string(),
                FieldValue::Float64(v) => v.to_string(),
                FieldValue::String(v) => v,
            };

            result.push(val);
        }

        Ok(result)
    }

    /// Get a field by its title (alias) rather than its name
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if the title is not found.
    pub fn get_field_by_title(&self, title: &str) -> Result<Vec<FieldValue>, ArrayError> {
        // Find the field name corresponding to the title
        let name = self
            .field_titles
            .iter()
            .find_map(|(name, t)| if t == title { Some(name) } else { None })
            .ok_or_else(|| ArrayError::ValueError(format!("Title '{title}' not found")))?;

        // Get the field values by name
        self.field_values(name)
    }

    /// Set a field value for a record
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if the field name is not found or record index is out of bounds.
    pub fn name_5(
        &mut self,
        name: &str,
        record_idx: usize,
        value: FieldValue,
    ) -> Result<(), ArrayError> {
        // First check if field exists
        if !self.field_indices.contains_key(name) {
            return Err(ArrayError::ValueError(format!("Field '{name}' not found")));
        }

        // Then get record and set the field
        let record = self.get_record_mut(record_idx).ok_or_else(|| {
            ArrayError::ValueError(format!("Record index {record_idx} out of bounds"))
        })?;

        record.add_field(name, value);
        Ok(())
    }

    /// Adds a new field to all records
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if the field already exists or the number of values doesn't match the number of records.
    pub fn name_6(&mut self, name: &str, values: Vec<FieldValue>) -> Result<(), ArrayError> {
        // Check if field already exists
        if self.field_indices.contains_key(name) {
            return Err(ArrayError::ValueError(format!(
                "Field '{name}' already exists"
            )));
        }

        // Check if the number of values matches the number of records
        if values.len() != self.records.len() {
            return Err(ArrayError::ValueError(format!(
                "Number of values ({}) doesn't match number of records ({})",
                values.len(),
                self.records.len()
            )));
        }

        // Add field to each record
        for (i, record) in self.records.iter_mut().enumerate() {
            record.add_field(name, values[i].clone());
        }

        // Update field names and indices
        let new_index = self.names.len();
        self.names.push(name.to_string());
        self.field_indices.insert(name.to_string(), new_index);

        Ok(())
    }

    /// Removes a field from all records
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if the field name is not found.
    pub fn name_7(&mut self, name: &str) -> Result<(), ArrayError> {
        // Check if field exists
        if !self.field_indices.contains_key(name) {
            return Err(ArrayError::ValueError(format!("Field '{name}' not found")));
        }

        // Remove field from each record
        for record in &mut self.records {
            // Create a new vector of field names without the removed field
            let new_names: Vec<String> = record
                .names
                .iter()
                .filter(|fieldname| *fieldname != name)
                .cloned()
                .collect();

            // Remove the field from the hashmap
            record.fields.remove(name);

            // Update field names
            record.names = new_names;
        }

        // Get the index of the field to remove
        let index_to_remove = self.field_indices[name];

        // Remove field from names
        self.names.remove(index_to_remove);

        // Remove field from field_titles if present
        self.field_titles.remove(name);

        // Rebuild field_indices map
        self.field_indices.clear();
        for (i, fieldname) in self.names.iter().enumerate() {
            self.field_indices.insert(fieldname.clone(), i);
        }

        Ok(())
    }

    /// Rename a field
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if old field is not found or new field already exists.
    pub fn name_8(&mut self, old_name: &str, newname: &str) -> Result<(), ArrayError> {
        // Check if old field exists
        if !self.field_indices.contains_key(old_name) {
            return Err(ArrayError::ValueError(format!(
                "Field '{old_name}' not found"
            )));
        }

        // Check if new field already exists
        if self.field_indices.contains_key(newname) {
            return Err(ArrayError::ValueError(format!(
                "Field '{newname}' already exists"
            )));
        }

        // Rename field in each record
        for record in &mut self.records {
            // Get the value for the field
            if let Some(value) = record.fields.remove(old_name) {
                // Add field with new name
                record.add_field(newname, value);

                // Update names
                let old_index = record
                    .names
                    .iter()
                    .position(|name| name == old_name)
                    .expect("Failed to create RecordArray in test");
                record.names[old_index] = newname.to_string();
            }
        }

        // Update names in RecordArray
        let old_index = self.field_indices[old_name];
        self.names[old_index] = newname.to_string();

        // Update field_indices
        self.field_indices.remove(old_name);
        self.field_indices.insert(newname.to_string(), old_index);

        // Update field_titles if the old name had a title
        if let Some(title) = self.field_titles.remove(old_name) {
            self.field_titles.insert(newname.to_string(), title);
        }

        Ok(())
    }

    /// Create a view of the record array with a subset of records
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if any index is out of bounds.
    pub fn view(&self, indices: &[usize]) -> Result<Self, ArrayError> {
        let mut newrecords = Vec::with_capacity(indices.len());

        // Collect records at specified indices
        for &idx in indices {
            if idx >= self.records.len() {
                return Err(ArrayError::ValueError(format!(
                    "Index {idx} out of bounds for record array of length {}",
                    self.records.len()
                )));
            }

            newrecords.push(self.records[idx].clone());
        }

        // Create a new RecordArray with the selected records
        let result = Self {
            records: newrecords,
            names: self.names.clone(),
            field_titles: self.field_titles.clone(),
            field_indices: self.field_indices.clone(),
            shape: vec![indices.len()],
            allow_field_attributes: self.allow_field_attributes,
        };

        Ok(result)
    }

    /// Filter the record array by a condition on a field
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if the field name is not found.
    pub fn filter<F>(&self, name: &str, condition: F) -> Result<Self, ArrayError>
    where
        F: Fn(&FieldValue) -> bool,
    {
        // Check if field exists
        if !self.field_indices.contains_key(name) {
            return Err(ArrayError::ValueError(format!("Field '{name}' not found")));
        }

        // Get all values for the field
        let values = self.field_values(name)?;

        // Find indices where the condition is true
        let mut indices = Vec::new();
        for (i, value) in values.iter().enumerate() {
            if condition(value) {
                indices.push(i);
            }
        }

        // Create a view with these indices
        self.view(&indices)
    }

    /// Merge two record arrays with compatible fields
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if the arrays have incompatible field structures.
    pub fn merge(&self, other: &Self) -> Result<Self, ArrayError> {
        // Check field compatibility
        if self.names.len() != other.names.len() {
            return Err(ArrayError::ValueError(format!(
                "Cannot merge record arrays with different number of fields ({} vs {})",
                self.names.len(),
                other.names.len()
            )));
        }

        for name in &self.names {
            if !other.field_indices.contains_key(name) {
                return Err(ArrayError::ValueError(format!(
                    "Field '{name}' not found in the second record array"
                )));
            }
        }

        // Combine records
        let mut newrecords = Vec::with_capacity(self.records.len() + other.records.len());
        newrecords.extend_from_slice(&self.records);
        newrecords.extend_from_slice(&other.records);

        // Create merged RecordArray
        let result = Self {
            records: newrecords,
            names: self.names.clone(),
            field_titles: self.field_titles.clone(),
            field_indices: self.field_indices.clone(),
            shape: vec![self.records.len() + other.records.len()],
            allow_field_attributes: self.allow_field_attributes,
        };

        Ok(result)
    }
}

/// Helper function to compare field values
#[allow(dead_code)]
fn compare_field_values(a: &FieldValue, b: &FieldValue) -> Option<std::cmp::Ordering> {
    match (a, b) {
        // Compare same types directly
        (FieldValue::Bool(a), FieldValue::Bool(b)) => Some(a.cmp(b)),
        (FieldValue::Int8(a), FieldValue::Int8(b)) => Some(a.cmp(b)),
        (FieldValue::Int16(a), FieldValue::Int16(b)) => Some(a.cmp(b)),
        (FieldValue::Int32(a), FieldValue::Int32(b)) => Some(a.cmp(b)),
        (FieldValue::Int64(a), FieldValue::Int64(b)) => Some(a.cmp(b)),
        (FieldValue::UInt8(a), FieldValue::UInt8(b)) => Some(a.cmp(b)),
        (FieldValue::UInt16(a), FieldValue::UInt16(b)) => Some(a.cmp(b)),
        (FieldValue::UInt32(a), FieldValue::UInt32(b)) => Some(a.cmp(b)),
        (FieldValue::UInt64(a), FieldValue::UInt64(b)) => Some(a.cmp(b)),
        (FieldValue::Float32(a), FieldValue::Float32(b)) => a.partial_cmp(b),
        (FieldValue::Float64(a), FieldValue::Float64(b)) => a.partial_cmp(b),
        (FieldValue::String(a), FieldValue::String(b)) => Some(a.cmp(b)),

        // Mixed numeric comparisons - convert to f64 when possible
        (FieldValue::Int8(a), FieldValue::Float32(b)) => (*a as f32).partial_cmp(b),
        (FieldValue::Int8(a), FieldValue::Float64(b)) => (*a as f64).partial_cmp(b),
        (FieldValue::Float32(a), FieldValue::Int8(b)) => a.partial_cmp(&(*b as f32)),
        (FieldValue::Float64(a), FieldValue::Int8(b)) => a.partial_cmp(&(*b as f64)),

        // For other mixed types, compare by type name as a fallback
        _ => {
            let type_a = std::any::type_name::<FieldValue>();
            let type_b = std::any::type_name::<FieldValue>();
            Some(type_a.cmp(type_b))
        }
    }
}

impl fmt::Display for RecordArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RecordArray(")?;

        let maxrecords_to_show = 10;
        let numrecords = self.records.len();
        let show_all = numrecords <= maxrecords_to_show;

        let records_to_show = if show_all {
            &self.records[..]
        } else {
            let half = maxrecords_to_show / 2;
            &self.records[..half]
        };

        for record in records_to_show {
            writeln!(f, "  {record},")?;
        }

        if !show_all {
            writeln!(f, "  ...")?;

            let half = maxrecords_to_show / 2;
            let remaining = &self.records[numrecords - half..];

            for record in remaining {
                writeln!(f, "  {record},")?;
            }
        }

        write!(f, ")")
    }
}

/// Create a `RecordArray` from arrays of the same length
///
/// # Errors
/// Returns `ArrayError::ValueError` if field names don't match arrays count or arrays have different lengths.
#[allow(dead_code)]
pub fn from_arrays(names: &[&str], arrays: &[Vec<FieldValue>]) -> Result<RecordArray, ArrayError> {
    if names.len() != arrays.len() {
        return Err(ArrayError::ValueError(format!(
            "Number of field _names ({}) must match number of arrays ({})",
            names.len(),
            arrays.len()
        )));
    }

    if arrays.is_empty() {
        return Err(ArrayError::ValueError("No arrays provided".to_string()));
    }

    let numrecords = arrays[0].len();

    // Check all arrays have the same length
    for (i, array) in arrays.iter().enumerate().skip(1) {
        if array.len() != numrecords {
            return Err(ArrayError::ValueError(format!(
                "Array {i} has length {}, but expected {numrecords}",
                array.len()
            )));
        }
    }

    // Create records
    let mut records = Vec::with_capacity(numrecords);

    for i in 0..numrecords {
        let mut record = Record::new();

        for (name, array) in names.iter().zip(arrays.iter()) {
            record.add_field(name, array[i].clone());
        }

        records.push(record);
    }

    RecordArray::new(records)
}

/// Create a `RecordArray` from arrays with different numeric types
///
/// # Errors
/// Returns `ArrayError::ValueError` if field names don't match 3 arrays or arrays have different lengths.
#[allow(dead_code)]
pub fn record_array_from_typed_arrays<A, B, C>(
    names: &[&str],
    arrays: (&[A], &[B], &[C]),
) -> Result<RecordArray, ArrayError>
where
    A: Clone + Into<FieldValue>,
    B: Clone + Into<FieldValue>,
    C: Clone + Into<FieldValue>,
{
    if names.len() != 3 {
        return Err(ArrayError::ValueError(format!(
            "Number of field _names ({}) must match number of arrays (3)",
            names.len()
        )));
    }

    let a_len = arrays.0.len();
    let b_len = arrays.1.len();
    let c_len = arrays.2.len();

    // Check all arrays have the same length
    if a_len != b_len || a_len != c_len {
        return Err(ArrayError::ValueError(format!(
            "Arrays have different lengths: {a_len}, {b_len}, {c_len}"
        )));
    }

    // Create records
    let mut records = Vec::with_capacity(a_len);

    for i in 0..a_len {
        let mut record = Record::new();

        record.add_field(names[0], arrays.0[i].clone().into());
        record.add_field(names[1], arrays.1[i].clone().into());
        record.add_field(names[2], arrays.2[i].clone().into());

        records.push(record);
    }

    RecordArray::new(records)
}

/// Create a `RecordArray` from a sequence of tuples
///
/// # Errors
/// Returns `ArrayError::ValueError` if field names don't match 3 tuple elements.
#[allow(dead_code)]
pub fn record_array_fromrecords<A, B, C>(
    names: &[&str],
    tuples: &[(A, B, C)],
) -> Result<RecordArray, ArrayError>
where
    A: Clone + Into<FieldValue>,
    B: Clone + Into<FieldValue>,
    C: Clone + Into<FieldValue>,
{
    if names.len() != 3 {
        return Err(ArrayError::ValueError(format!(
            "Number of field _names ({}) must match number of tuple elements (3)",
            names.len()
        )));
    }

    // Create records
    let mut records = Vec::with_capacity(tuples.len());

    for tuple in tuples {
        let mut record = Record::new();

        record.add_field(names[0], tuple.0.clone().into());
        record.add_field(names[1], tuple.1.clone().into());
        record.add_field(names[2], tuple.2.clone().into());

        records.push(record);
    }

    RecordArray::new(records)
}
