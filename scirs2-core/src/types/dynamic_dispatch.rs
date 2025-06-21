//! # Dynamic Type Dispatch for Heterogeneous Collections
//!
//! This module provides a system for runtime type dispatch and management of heterogeneous
//! collections, allowing safe and efficient operations on collections containing multiple types.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::RwLock;

/// A type-erased value that can hold any type implementing the necessary traits
pub struct DynamicValue {
    /// The type-erased value
    value: Box<dyn Any + Send + Sync>,
    /// Type information
    type_info: TypeInfo,
    /// Optional metadata
    metadata: HashMap<String, String>,
}

/// Type information for runtime dispatch
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// Type ID for runtime checking
    pub type_id: TypeId,
    /// Human-readable type name
    pub type_name: String,
    /// Type category for grouping related types
    pub category: TypeCategory,
    /// Size in bytes (if known)
    pub size_hint: Option<usize>,
}

/// Categories of types for high-level classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeCategory {
    /// Scalar numeric types (f32, f64, i32, etc.)
    Scalar,
    /// Complex numeric types
    Complex,
    /// Vector types (arrays, vectors)
    Vector,
    /// Matrix types (2D arrays, matrices)
    Matrix,
    /// Tensor types (multi-dimensional arrays)
    Tensor,
    /// String types
    String,
    /// Boolean types
    Boolean,
    /// Date/time types
    DateTime,
    /// Custom user-defined types
    Custom(String),
}

impl fmt::Display for TypeCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeCategory::Scalar => write!(f, "Scalar"),
            TypeCategory::Complex => write!(f, "Complex"),
            TypeCategory::Vector => write!(f, "Vector"),
            TypeCategory::Matrix => write!(f, "Matrix"),
            TypeCategory::Tensor => write!(f, "Tensor"),
            TypeCategory::String => write!(f, "String"),
            TypeCategory::Boolean => write!(f, "Boolean"),
            TypeCategory::DateTime => write!(f, "DateTime"),
            TypeCategory::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

impl DynamicValue {
    /// Create a new dynamic value
    pub fn new<T: Any + Send + Sync>(value: T) -> Self {
        let type_info = TypeInfo {
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>().to_string(),
            category: Self::infer_category::<T>(),
            size_hint: Some(std::mem::size_of::<T>()),
        };

        Self {
            value: Box::new(value),
            type_info,
            metadata: HashMap::new(),
        }
    }

    /// Create a new dynamic value with custom type information
    pub fn with_type_info<T: Any + Send + Sync>(value: T, type_info: TypeInfo) -> Self {
        Self {
            value: Box::new(value),
            type_info,
            metadata: HashMap::new(),
        }
    }

    /// Try to downcast to a specific type
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.value.downcast_ref::<T>()
    }

    /// Try to downcast to a mutable reference of a specific type
    pub fn downcast_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.value.downcast_mut::<T>()
    }

    /// Consume and downcast to a specific type
    #[allow(clippy::result_large_err)]
    pub fn downcast<T: Any>(self) -> Result<T, Self> {
        match self.value.downcast::<T>() {
            Ok(value) => Ok(*value),
            Err(value) => Err(Self {
                value,
                type_info: self.type_info,
                metadata: self.metadata,
            }),
        }
    }

    /// Check if the value is of a specific type
    pub fn is_type<T: Any>(&self) -> bool {
        self.type_info.type_id == TypeId::of::<T>()
    }

    /// Get type information
    pub const fn type_info(&self) -> &TypeInfo {
        &self.type_info
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Get all metadata
    pub const fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Infer type category from the Rust type
    fn infer_category<T: Any>() -> TypeCategory {
        let type_name = std::any::type_name::<T>();

        // Basic pattern matching on type names
        if type_name.contains("f32")
            || type_name.contains("f64")
            || type_name.contains("i8")
            || type_name.contains("i16")
            || type_name.contains("i32")
            || type_name.contains("i64")
            || type_name.contains("u8")
            || type_name.contains("u16")
            || type_name.contains("u32")
            || type_name.contains("u64")
        {
            TypeCategory::Scalar
        } else if type_name.contains("Complex") || type_name.contains("complex") {
            TypeCategory::Complex
        } else if type_name.contains("Vec") || type_name.contains("Array1") {
            TypeCategory::Vector
        } else if type_name.contains("Array2") || type_name.contains("Matrix") {
            TypeCategory::Matrix
        } else if type_name.contains("Array") || type_name.contains("Tensor") {
            TypeCategory::Tensor
        } else if type_name.contains("String") || type_name.contains("str") {
            TypeCategory::String
        } else if type_name.contains("bool") {
            TypeCategory::Boolean
        } else if type_name.contains("DateTime") || type_name.contains("Time") {
            TypeCategory::DateTime
        } else {
            TypeCategory::Custom(type_name.to_string())
        }
    }
}

impl fmt::Debug for DynamicValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicValue")
            .field("type_info", &self.type_info)
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// A collection that can hold heterogeneous types with runtime dispatch
pub struct HeterogeneousCollection {
    /// The stored values
    values: Vec<DynamicValue>,
    /// Index for fast lookup by type
    type_index: HashMap<TypeId, Vec<usize>>,
    /// Index for fast lookup by category
    category_index: HashMap<TypeCategory, Vec<usize>>,
    /// Optional name for the collection
    name: Option<String>,
}

impl HeterogeneousCollection {
    /// Create a new heterogeneous collection
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            type_index: HashMap::new(),
            category_index: HashMap::new(),
            name: None,
        }
    }

    /// Create a new named heterogeneous collection
    pub fn with_name(name: String) -> Self {
        Self {
            values: Vec::new(),
            type_index: HashMap::new(),
            category_index: HashMap::new(),
            name: Some(name),
        }
    }

    /// Add a value to the collection
    pub fn push<T: Any + Send + Sync>(&mut self, value: T) -> usize {
        let dynamic_value = DynamicValue::new(value);
        let index = self.values.len();
        let type_id = dynamic_value.type_info.type_id;
        let category = dynamic_value.type_info.category.clone();

        // Update indices
        self.type_index.entry(type_id).or_default().push(index);
        self.category_index.entry(category).or_default().push(index);

        self.values.push(dynamic_value);
        index
    }

    /// Add a dynamic value to the collection
    pub fn push_dynamic(&mut self, value: DynamicValue) -> usize {
        let index = self.values.len();
        let type_id = value.type_info.type_id;
        let category = value.type_info.category.clone();

        // Update indices
        self.type_index.entry(type_id).or_default().push(index);
        self.category_index.entry(category).or_default().push(index);

        self.values.push(value);
        index
    }

    /// Get a value by index
    pub fn get(&self, index: usize) -> Option<&DynamicValue> {
        self.values.get(index)
    }

    /// Get a mutable value by index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut DynamicValue> {
        self.values.get_mut(index)
    }

    /// Get all values of a specific type
    pub fn get_by_type<T: Any>(&self) -> Vec<&DynamicValue> {
        let type_id = TypeId::of::<T>();
        if let Some(indices) = self.type_index.get(&type_id) {
            indices.iter().filter_map(|&i| self.values.get(i)).collect()
        } else {
            Vec::new()
        }
    }

    /// Get all values of a specific category
    pub fn get_by_category(&self, category: &TypeCategory) -> Vec<&DynamicValue> {
        if let Some(indices) = self.category_index.get(category) {
            indices.iter().filter_map(|&i| self.values.get(i)).collect()
        } else {
            Vec::new()
        }
    }

    /// Try to downcast and collect all values of a specific type
    pub fn collect_type<T: Any + Clone>(&self) -> Vec<T> {
        self.get_by_type::<T>()
            .into_iter()
            .filter_map(|dv| dv.downcast_ref::<T>().cloned())
            .collect()
    }

    /// Get the number of values in the collection
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get all unique types in the collection
    pub fn get_types(&self) -> Vec<&TypeInfo> {
        self.values.iter().map(|v| &v.type_info).collect()
    }

    /// Get all unique categories in the collection
    pub fn get_categories(&self) -> Vec<&TypeCategory> {
        self.category_index.keys().collect()
    }

    /// Get statistics about the collection
    pub fn statistics(&self) -> CollectionStatistics {
        let mut type_counts = HashMap::new();
        let mut category_counts = HashMap::new();

        for value in &self.values {
            *type_counts
                .entry(value.type_info.type_name.clone())
                .or_insert(0) += 1;
            *category_counts
                .entry(value.type_info.category.clone())
                .or_insert(0) += 1;
        }

        CollectionStatistics {
            total_count: self.values.len(),
            type_counts,
            category_counts,
            unique_types: self.type_index.len(),
            unique_categories: self.category_index.len(),
        }
    }

    /// Apply a function to all values of a specific type
    pub fn apply_to_type<T: Any, F>(&mut self, mut f: F) -> CoreResult<()>
    where
        F: FnMut(&mut T) -> CoreResult<()>,
    {
        let type_id = TypeId::of::<T>();
        if let Some(indices) = self.type_index.get(&type_id).cloned() {
            for index in indices {
                if let Some(value) = self.values.get_mut(index) {
                    if let Some(typed_value) = value.downcast_mut::<T>() {
                        f(typed_value)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Filter values by a predicate
    pub fn filter<F>(&self, mut predicate: F) -> Vec<&DynamicValue>
    where
        F: FnMut(&DynamicValue) -> bool,
    {
        self.values.iter().filter(|v| predicate(v)).collect()
    }

    /// Remove values by index (returns the removed value)
    pub fn remove(&mut self, index: usize) -> Option<DynamicValue> {
        if index >= self.values.len() {
            return None;
        }

        let removed = self.values.remove(index);

        // Rebuild indices (inefficient but correct)
        self.rebuild_indices();

        Some(removed)
    }

    /// Clear all values
    pub fn clear(&mut self) {
        self.values.clear();
        self.type_index.clear();
        self.category_index.clear();
    }

    /// Rebuild internal indices after modifications
    fn rebuild_indices(&mut self) {
        self.type_index.clear();
        self.category_index.clear();

        for (index, value) in self.values.iter().enumerate() {
            let type_id = value.type_info.type_id;
            let category = value.type_info.category.clone();

            self.type_index.entry(type_id).or_default().push(index);
            self.category_index.entry(category).or_default().push(index);
        }
    }

    /// Get the collection name
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    /// Set the collection name
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }
}

impl Default for HeterogeneousCollection {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a heterogeneous collection
#[derive(Debug, Clone)]
pub struct CollectionStatistics {
    /// Total number of values
    pub total_count: usize,
    /// Count of each type by name
    pub type_counts: HashMap<String, usize>,
    /// Count of each category
    pub category_counts: HashMap<TypeCategory, usize>,
    /// Number of unique types
    pub unique_types: usize,
    /// Number of unique categories
    pub unique_categories: usize,
}

impl fmt::Display for CollectionStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Collection Statistics:")?;
        writeln!(f, "  Total values: {}", self.total_count)?;
        writeln!(f, "  Unique types: {}", self.unique_types)?;
        writeln!(f, "  Unique categories: {}", self.unique_categories)?;

        writeln!(f, "  Type distribution:")?;
        for (type_name, count) in &self.type_counts {
            writeln!(f, "    {}: {}", type_name, count)?;
        }

        writeln!(f, "  Category distribution:")?;
        for (category, count) in &self.category_counts {
            writeln!(f, "    {}: {}", category, count)?;
        }

        Ok(())
    }
}

/// Trait for types that can be stored in heterogeneous collections
pub trait DynamicDispatchable: Any + Send + Sync {
    /// Get the type category for this type
    fn type_category() -> TypeCategory;

    /// Get the type name for this type
    fn type_name() -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Create type information for this type
    fn type_info() -> TypeInfo
    where
        Self: Sized,
    {
        TypeInfo {
            type_id: TypeId::of::<Self>(),
            type_name: Self::type_name().to_string(),
            category: Self::type_category(),
            size_hint: Some(std::mem::size_of::<Self>()),
        }
    }
}

// Implement DynamicDispatchable for common types
impl DynamicDispatchable for f32 {
    fn type_category() -> TypeCategory {
        TypeCategory::Scalar
    }
}

impl DynamicDispatchable for f64 {
    fn type_category() -> TypeCategory {
        TypeCategory::Scalar
    }
}

impl DynamicDispatchable for i32 {
    fn type_category() -> TypeCategory {
        TypeCategory::Scalar
    }
}

impl DynamicDispatchable for i64 {
    fn type_category() -> TypeCategory {
        TypeCategory::Scalar
    }
}

impl DynamicDispatchable for String {
    fn type_category() -> TypeCategory {
        TypeCategory::String
    }
}

impl DynamicDispatchable for bool {
    fn type_category() -> TypeCategory {
        TypeCategory::Boolean
    }
}

impl<T: DynamicDispatchable> DynamicDispatchable for Vec<T> {
    fn type_category() -> TypeCategory {
        TypeCategory::Vector
    }
}

/// A registry for managing type operations and conversions
pub struct TypeRegistry {
    /// Registered type converters
    converters: RwLock<HashMap<(TypeId, TypeId), Box<dyn TypeConverter + Send + Sync>>>,
    /// Registered type operations
    operations: RwLock<HashMap<TypeId, Box<dyn TypeOperations + Send + Sync>>>,
}

/// Trait for converting between types
pub trait TypeConverter {
    /// Convert a dynamic value to another type
    fn convert(&self, value: &DynamicValue) -> CoreResult<DynamicValue>;

    /// Check if conversion is possible without actually converting
    fn can_convert(&self, value: &DynamicValue) -> bool;
}

/// Trait for operations on a specific type
pub trait TypeOperations {
    /// Clone a dynamic value
    fn clone_value(&self, value: &DynamicValue) -> CoreResult<DynamicValue>;

    /// Compare two dynamic values for equality
    fn equals(&self, a: &DynamicValue, b: &DynamicValue) -> CoreResult<bool>;

    /// Get a string representation of the value
    fn to_string(&self, value: &DynamicValue) -> CoreResult<String>;

    /// Get the size in bytes
    fn size_of(&self, value: &DynamicValue) -> usize;
}

impl TypeRegistry {
    /// Create a new type registry
    pub fn new() -> Self {
        Self {
            converters: RwLock::new(HashMap::new()),
            operations: RwLock::new(HashMap::new()),
        }
    }

    /// Register a type converter
    pub fn register_converter<From: Any, To: Any>(
        &self,
        converter: Box<dyn TypeConverter + Send + Sync>,
    ) -> CoreResult<()> {
        let from_id = TypeId::of::<From>();
        let to_id = TypeId::of::<To>();

        let mut converters = self.converters.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire write lock"))
        })?;

        converters.insert((from_id, to_id), converter);
        Ok(())
    }

    /// Register type operations
    pub fn register_operations<T: Any>(
        &self,
        operations: Box<dyn TypeOperations + Send + Sync>,
    ) -> CoreResult<()> {
        let type_id = TypeId::of::<T>();

        let mut ops = self.operations.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire write lock"))
        })?;

        ops.insert(type_id, operations);
        Ok(())
    }

    /// Convert a value to another type
    pub fn convert<From: Any, To: Any>(&self, value: &DynamicValue) -> CoreResult<DynamicValue> {
        let from_id = TypeId::of::<From>();
        let to_id = TypeId::of::<To>();

        let converters = self.converters.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire read lock"))
        })?;

        if let Some(converter) = converters.get(&(from_id, to_id)) {
            converter.convert(value)
        } else {
            Err(CoreError::TypeError(ErrorContext::new(format!(
                "No converter registered for {} -> {}",
                std::any::type_name::<From>(),
                std::any::type_name::<To>()
            ))))
        }
    }

    /// Check if operations are registered for a type
    pub fn has_operations<T: Any>(&self) -> CoreResult<bool> {
        let type_id = TypeId::of::<T>();

        let operations = self.operations.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire read lock"))
        })?;

        Ok(operations.contains_key(&type_id))
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global type registry instance
static GLOBAL_TYPE_REGISTRY: std::sync::LazyLock<TypeRegistry> =
    std::sync::LazyLock::new(TypeRegistry::new);

/// Get the global type registry
pub fn global_type_registry() -> &'static TypeRegistry {
    &GLOBAL_TYPE_REGISTRY
}

/// Utility functions for common operations
pub mod utils {
    use super::*;

    /// Create a heterogeneous collection from a list of values
    pub fn collect_heterogeneous<I>(values: I) -> HeterogeneousCollection
    where
        I: IntoIterator<Item = DynamicValue>,
    {
        let mut collection = HeterogeneousCollection::new();
        for value in values {
            collection.push_dynamic(value);
        }
        collection
    }

    /// Group values by category
    pub fn group_by_category(
        collection: &HeterogeneousCollection,
    ) -> HashMap<TypeCategory, Vec<&DynamicValue>> {
        let mut groups = HashMap::new();

        for value in &collection.values {
            groups
                .entry(value.type_info.category.clone())
                .or_insert_with(Vec::new)
                .push(value);
        }

        groups
    }

    /// Find the most common type in a collection
    pub fn most_common_type(collection: &HeterogeneousCollection) -> Option<TypeId> {
        collection
            .type_index
            .iter()
            .max_by_key(|(_, indices)| indices.len())
            .map(|(type_id, _)| *type_id)
    }

    /// Check if all values in a collection are of the same category
    pub fn is_homogeneous_category(collection: &HeterogeneousCollection) -> bool {
        collection.category_index.len() <= 1
    }

    /// Check if all values in a collection are of the same type
    pub fn is_homogeneous_type(collection: &HeterogeneousCollection) -> bool {
        collection.type_index.len() <= 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_value() {
        let value = DynamicValue::new(42i32);
        assert!(value.is_type::<i32>());
        assert!(!value.is_type::<f64>());

        assert_eq!(value.downcast_ref::<i32>(), Some(&42));
        assert_eq!(value.downcast_ref::<f64>(), None);

        assert_eq!(value.type_info().category, TypeCategory::Scalar);
    }

    #[test]
    fn test_heterogeneous_collection() {
        let mut collection = HeterogeneousCollection::new();

        collection.push(42i32);
        collection.push(std::f64::consts::PI);
        collection.push("hello".to_string());
        collection.push(true);

        assert_eq!(collection.len(), 4);
        assert_eq!(collection.get_types().len(), 4);

        let integers = collection.collect_type::<i32>();
        assert_eq!(integers, vec![42]);

        let scalars = collection.get_by_category(&TypeCategory::Scalar);
        assert_eq!(scalars.len(), 2); // i32 and f64

        let strings = collection.get_by_category(&TypeCategory::String);
        assert_eq!(strings.len(), 1);
    }

    #[test]
    fn test_collection_statistics() {
        let mut collection = HeterogeneousCollection::new();

        collection.push(1i32);
        collection.push(2i32);
        collection.push(std::f64::consts::PI);
        collection.push("test".to_string());

        let stats = collection.statistics();
        assert_eq!(stats.total_count, 4);
        assert_eq!(stats.unique_types, 3);
        assert_eq!(stats.type_counts.get("i32"), Some(&2));
        assert_eq!(stats.type_counts.get("f64"), Some(&1));
        assert_eq!(stats.type_counts.get("alloc::string::String"), Some(&1));
    }

    #[test]
    fn test_dynamic_dispatchable() {
        assert_eq!(f32::type_category(), TypeCategory::Scalar);
        assert_eq!(String::type_category(), TypeCategory::String);
        assert_eq!(Vec::<f64>::type_category(), TypeCategory::Vector);

        let type_info = f64::type_info();
        assert_eq!(type_info.type_id, TypeId::of::<f64>());
        assert_eq!(type_info.category, TypeCategory::Scalar);
    }

    #[test]
    fn test_collection_operations() {
        let mut collection = HeterogeneousCollection::new();

        collection.push(1i32);
        collection.push(2i32);
        collection.push(3i32);

        let mut sum = 0i32;
        collection
            .apply_to_type::<i32, _>(|value| {
                sum += *value;
                Ok(())
            })
            .unwrap();

        assert_eq!(sum, 6);
    }

    #[test]
    fn test_type_inference() {
        let value1 = DynamicValue::new(42f64);
        assert_eq!(value1.type_info().category, TypeCategory::Scalar);

        let value2 = DynamicValue::new(vec![1, 2, 3]);
        assert_eq!(value2.type_info().category, TypeCategory::Vector);

        let value3 = DynamicValue::new("hello".to_string());
        assert_eq!(value3.type_info().category, TypeCategory::String);
    }

    #[test]
    fn test_metadata() {
        let mut value = DynamicValue::new(42i32);
        value.add_metadata("source".to_string(), "user_input".to_string());
        value.add_metadata("validated".to_string(), "true".to_string());

        assert_eq!(
            value.get_metadata("source"),
            Some(&"user_input".to_string())
        );
        assert_eq!(value.get_metadata("validated"), Some(&"true".to_string()));
        assert_eq!(value.get_metadata("nonexistent"), None);
    }

    #[test]
    fn test_collection_filtering() {
        let mut collection = HeterogeneousCollection::new();

        collection.push(1i32);
        collection.push(2.0f64);
        collection.push(3i32);
        collection.push(4.0f64);

        let integers = collection
            .filter(|v| v.type_info().category == TypeCategory::Scalar && v.is_type::<i32>());
        assert_eq!(integers.len(), 2);

        let floats = collection.filter(|v| v.is_type::<f64>());
        assert_eq!(floats.len(), 2);
    }

    #[test]
    fn test_collection_utils() {
        let mut collection = HeterogeneousCollection::new();
        collection.push(1i32);
        collection.push(2.0f64);
        collection.push("test".to_string());

        assert!(!utils::is_homogeneous_type(&collection));
        assert!(!utils::is_homogeneous_category(&collection));

        let grouped = utils::group_by_category(&collection);
        assert_eq!(grouped.len(), 2); // Scalar and String categories
        assert_eq!(grouped.get(&TypeCategory::Scalar).unwrap().len(), 2);
        assert_eq!(grouped.get(&TypeCategory::String).unwrap().len(), 1);
    }
}
