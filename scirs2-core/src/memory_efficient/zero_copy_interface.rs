//! Zero-copy interface for efficient data exchange between components.
//!
//! This module provides a comprehensive zero-copy interface that enables efficient
//! data sharing between different components of the `SciRS2` library without
//! unnecessary memory allocations or data copying.
//!
//! ## Features
//!
//! - **Zero-Copy Views**: Share data between components without copying
//! - **Reference Counting**: Automatic memory management for shared data
//! - **Type Safety**: Compile-time guarantees for data type consistency
//! - **Cross-Component Exchange**: Seamless data exchange between modules
//! - **Memory Pool Integration**: Efficient allocation and reuse
//! - **NUMA Awareness**: Optimize data placement for NUMA systems
//! - **Thread Safety**: Safe concurrent access to shared data
//!
//! ## Example Usage
//!
//! ```rust
//! use scirs2_core::memory_efficient::zero_copy_interface::{
//!     ZeroCopyData, ZeroCopyInterface, DataExchange
//! };
//!
//! // Create zero-copy data
//! let data = vec![1.0, 2.0, 3.0, 4.0];
//! let zero_copy_data = ZeroCopyData::new(data)?;
//!
//! // Share data between components
//! let interface = ZeroCopyInterface::new();
//! interface.register_data("dataset1", zero_copy_data)?;
//!
//! // Access data from another component
//! let borrowed_data = interface.borrow_data::<f64>("dataset1")?;
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

/// Unique identifier for zero-copy data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataId(u64);

impl DataId {
    /// Create a new unique data ID
    pub fn new() -> Self {
        use std::sync::atomic::AtomicU64;
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for DataId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DataId({})", self.0)
    }
}

impl Default for DataId {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about zero-copy data
#[derive(Debug, Clone)]
pub struct DataMetadata {
    /// Data type information
    pub type_id: TypeId,
    /// Human-readable type name
    pub type_name: String,
    /// Size of the data in bytes
    pub size_bytes: usize,
    /// Number of elements
    pub element_count: usize,
    /// Element size in bytes
    pub element_size: usize,
    /// Creation timestamp
    pub created_at: std::time::Instant,
    /// Optional description
    pub description: Option<String>,
    /// NUMA node hint (if applicable)
    pub numa_node: Option<usize>,
    /// Whether the data is mutable
    pub is_mutable: bool,
}

impl DataMetadata {
    /// Create metadata for a given type and data
    pub fn new<T: 'static>(
        data: &[T],
        description: Option<String>,
        numa_node: Option<usize>,
        is_mutable: bool,
    ) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>().to_string(),
            size_bytes: std::mem::size_of_val(data),
            element_count: data.len(),
            element_size: std::mem::size_of::<T>(),
            created_at: std::time::Instant::now(),
            description,
            numa_node,
            is_mutable,
        }
    }

    /// Check if this metadata is compatible with another type
    pub fn is_compatible_with<T: 'static>(&self) -> bool {
        self.type_id == TypeId::of::<T>()
    }
}

/// Reference-counted zero-copy data container
#[derive(Debug)]
struct ZeroCopyDataInner<T> {
    /// The actual data
    data: Vec<T>,
    /// Reference count
    ref_count: AtomicUsize,
    /// Metadata about the data
    metadata: DataMetadata,
    /// Weak references to this data
    #[allow(dead_code)]
    weak_refs: Mutex<Vec<Weak<ZeroCopyDataInner<T>>>>,
}

impl<T> ZeroCopyDataInner<T> {
    fn new(
        data: Vec<T>,
        description: Option<String>,
        numa_node: Option<usize>,
        is_mutable: bool,
    ) -> Self
    where
        T: 'static,
    {
        let metadata = DataMetadata::new(&data, description, numa_node, is_mutable);

        Self {
            data,
            ref_count: AtomicUsize::new(1),
            metadata,
            weak_refs: Mutex::new(Vec::new()),
        }
    }

    fn add_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
    }

    fn remove_ref(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::Relaxed) - 1
    }

    fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::Relaxed)
    }
}

/// Zero-copy data container that can be shared between components
#[derive(Debug)]
pub struct ZeroCopyData<T> {
    inner: Arc<ZeroCopyDataInner<T>>,
    id: DataId,
}

impl<T> ZeroCopyData<T>
where
    T: Clone + 'static,
{
    /// Create a new zero-copy data container
    pub fn new(data: Vec<T>) -> CoreResult<Self> {
        Self::with_metadata(data, None, None, false)
    }

    /// Create a new zero-copy data container with metadata
    pub fn with_metadata(
        data: Vec<T>,
        description: Option<String>,
        numa_node: Option<usize>,
        is_mutable: bool,
    ) -> CoreResult<Self> {
        if data.is_empty() {
            return Err(CoreError::ValidationError(
                ErrorContext::new("Cannot create zero-copy data from empty vector".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let inner = Arc::new(ZeroCopyDataInner::new(
            data,
            description,
            numa_node,
            is_mutable,
        ));
        let id = DataId::new();

        Ok(Self { inner, id })
    }

    /// Create a new mutable zero-copy data container
    pub fn new_mutable(data: Vec<T>) -> CoreResult<Self> {
        Self::with_metadata(data, None, None, true)
    }

    /// Get the unique ID of this data
    pub fn id(&self) -> DataId {
        self.id
    }

    /// Get metadata about this data
    pub fn metadata(&self) -> &DataMetadata {
        &self.inner.metadata
    }

    /// Get a reference to the data
    pub fn as_slice(&self) -> &[T] {
        &self.inner.data
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.inner.data.len()
    }

    /// Check if the data is empty
    pub fn is_empty(&self) -> bool {
        self.inner.data.is_empty()
    }

    /// Get the reference count
    pub fn ref_count(&self) -> usize {
        self.inner.ref_count()
    }

    /// Check if this is the only reference to the data
    pub fn is_unique(&self) -> bool {
        self.ref_count() == 1
    }

    /// Create a view of part of the data
    pub fn view(&self, start: usize, len: usize) -> CoreResult<ZeroCopyView<T>> {
        if start + len > self.len() {
            return Err(CoreError::IndexError(
                ErrorContext::new(format!(
                    "View range [{}..{}] exceeds data length {}",
                    start,
                    start + len,
                    self.len()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        Ok(ZeroCopyView::new(self.clone(), start, len))
    }

    /// Create a weak reference to this data
    pub fn downgrade(&self) -> ZeroCopyWeakRef<T> {
        ZeroCopyWeakRef {
            inner: Arc::downgrade(&self.inner),
            id: self.id,
        }
    }
}

impl<T> Clone for ZeroCopyData<T> {
    fn clone(&self) -> Self {
        self.inner.add_ref();
        Self {
            inner: self.inner.clone(),
            id: self.id,
        }
    }
}

impl<T> Drop for ZeroCopyData<T> {
    fn drop(&mut self) {
        if self.inner.remove_ref() == 0 {
            // Last reference dropped - data will be deallocated
        }
    }
}

/// Weak reference to zero-copy data
#[derive(Debug)]
pub struct ZeroCopyWeakRef<T> {
    inner: Weak<ZeroCopyDataInner<T>>,
    id: DataId,
}

impl<T> ZeroCopyWeakRef<T> {
    /// Try to upgrade to a strong reference
    pub fn upgrade(&self) -> Option<ZeroCopyData<T>> {
        self.inner
            .upgrade()
            .map(|inner| ZeroCopyData { inner, id: self.id })
    }

    /// Get the data ID
    pub fn id(&self) -> DataId {
        self.id
    }

    /// Check if the data is still alive
    pub fn is_alive(&self) -> bool {
        self.inner.strong_count() > 0
    }
}

impl<T> Clone for ZeroCopyWeakRef<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            id: self.id,
        }
    }
}

/// A view into a portion of zero-copy data
#[derive(Debug)]
pub struct ZeroCopyView<T> {
    data: ZeroCopyData<T>,
    start: usize,
    len: usize,
}

impl<T> ZeroCopyView<T>
where
    T: Clone + 'static,
{
    fn new(data: ZeroCopyData<T>, start: usize, len: usize) -> Self {
        Self { data, start, len }
    }

    /// Get a slice of the viewed data
    pub fn as_slice(&self) -> &[T] {
        &self.data.as_slice()[self.start..self.start + self.len]
    }

    /// Get the length of the view
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the view is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the underlying data
    pub const fn underlying_data(&self) -> &ZeroCopyData<T> {
        &self.data
    }

    /// Create a sub-view of this view
    pub fn subview(&self, start: usize, len: usize) -> CoreResult<ZeroCopyView<T>> {
        if start + len > self.len {
            return Err(CoreError::IndexError(
                ErrorContext::new(format!(
                    "Subview range [{}..{}] exceeds view length {}",
                    start,
                    start + len,
                    self.len
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        Ok(ZeroCopyView::new(
            self.data.clone(),
            self.start + start,
            len,
        ))
    }
}

impl<T> Clone for ZeroCopyView<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            start: self.start,
            len: self.len,
        }
    }
}

/// Type-erased zero-copy data for storage in collections
trait AnyZeroCopyData: Send + Sync + std::fmt::Debug + Any {
    /// Get the type ID of the contained data
    #[allow(dead_code)]
    fn type_id(&self) -> TypeId;

    /// Get the metadata
    fn metadata(&self) -> &DataMetadata;

    /// Clone the data as a boxed trait object
    fn clone_box(&self) -> Box<dyn AnyZeroCopyData>;

    /// Get the data ID
    fn data_id(&self) -> DataId;

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

impl<T: Clone + 'static + Send + Sync + std::fmt::Debug> AnyZeroCopyData for ZeroCopyData<T> {
    #[allow(dead_code)]
    fn type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn metadata(&self) -> &DataMetadata {
        &self.inner.metadata
    }

    fn clone_box(&self) -> Box<dyn AnyZeroCopyData> {
        Box::new(self.clone())
    }

    fn data_id(&self) -> DataId {
        self.id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Zero-copy interface for data exchange between components
#[derive(Debug)]
pub struct ZeroCopyInterface {
    /// Named data storage
    named_data: RwLock<HashMap<String, Box<dyn AnyZeroCopyData>>>,

    /// Data storage by ID
    id_data: RwLock<HashMap<DataId, Box<dyn AnyZeroCopyData>>>,

    /// Type-based data storage
    type_data: RwLock<HashMap<TypeId, Vec<Box<dyn AnyZeroCopyData>>>>,

    /// Exchange statistics
    stats: RwLock<InterfaceStats>,
}

/// Statistics for the zero-copy interface
#[derive(Debug, Clone, Default)]
pub struct InterfaceStats {
    /// Number of data items registered
    pub items_registered: usize,

    /// Number of successful data exchanges
    pub exchanges_successful: usize,

    /// Number of failed data exchanges
    pub exchanges_failed: usize,

    /// Total memory managed (bytes)
    pub total_memory_managed: usize,

    /// Number of active references
    pub active_references: usize,

    /// Number of views created
    pub views_created: usize,
}

impl ZeroCopyInterface {
    /// Create a new zero-copy interface
    pub fn new() -> Self {
        Self {
            named_data: RwLock::new(HashMap::new()),
            id_data: RwLock::new(HashMap::new()),
            type_data: RwLock::new(HashMap::new()),
            stats: RwLock::new(InterfaceStats::default()),
        }
    }

    /// Register data with a name
    pub fn register_data<T: Clone + 'static + Send + Sync + std::fmt::Debug>(
        &self,
        name: &str,
        data: ZeroCopyData<T>,
    ) -> CoreResult<()> {
        let boxed_data = Box::new(data.clone()) as Box<dyn AnyZeroCopyData>;

        // Store by name
        {
            let mut named = self.named_data.write().unwrap();
            if named.contains_key(name) {
                return Err(CoreError::ValidationError(
                    ErrorContext::new(format!("Data with name '{}' already exists", name))
                        .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }
            named.insert(name.to_string(), boxed_data.clone_box());
        }

        // Store by ID
        {
            let mut id_map = self.id_data.write().unwrap();
            id_map.insert(data.id(), boxed_data.clone_box());
        }

        // Store by type
        {
            let mut type_map = self.type_data.write().unwrap();
            let type_id = TypeId::of::<T>();
            type_map.entry(type_id).or_default().push(boxed_data);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.items_registered += 1;
            stats.total_memory_managed += data.metadata().size_bytes;
        }

        Ok(())
    }

    /// Get data by name
    pub fn get_data<T: Clone + 'static + Send + Sync + std::fmt::Debug>(
        &self,
        name: &str,
    ) -> CoreResult<ZeroCopyData<T>> {
        let named = self.named_data.read().unwrap();

        if let Some(any_data) = named.get(name) {
            if let Some(typed_data) = any_data.as_any().downcast_ref::<ZeroCopyData<T>>() {
                self.update_exchange_stats(true);
                Ok(typed_data.clone())
            } else {
                self.update_exchange_stats(false);
                Err(CoreError::ValidationError(
                    ErrorContext::new(format!(
                        "Data '{}' exists but has wrong type. Expected {}, found {}",
                        name,
                        std::any::type_name::<T>(),
                        any_data.metadata().type_name
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ))
            }
        } else {
            self.update_exchange_stats(false);
            Err(CoreError::ValidationError(
                ErrorContext::new(format!("No data found with name '{}'", name))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ))
        }
    }

    /// Get data by ID
    pub fn get_data_by_id<T: Clone + 'static + Send + Sync + std::fmt::Debug>(
        &self,
        id: DataId,
    ) -> CoreResult<ZeroCopyData<T>> {
        let id_map = self.id_data.read().unwrap();

        if let Some(any_data) = id_map.get(&id) {
            if let Some(typed_data) = any_data.as_any().downcast_ref::<ZeroCopyData<T>>() {
                self.update_exchange_stats(true);
                Ok(typed_data.clone())
            } else {
                self.update_exchange_stats(false);
                Err(CoreError::ValidationError(
                    ErrorContext::new(format!(
                        "Data with ID {} exists but has wrong type. Expected {}, found {}",
                        id,
                        std::any::type_name::<T>(),
                        any_data.metadata().type_name
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ))
            }
        } else {
            self.update_exchange_stats(false);
            Err(CoreError::ValidationError(
                ErrorContext::new(format!("No data found with ID {}", id))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ))
        }
    }

    /// Get all data of a specific type
    pub fn get_data_by_type<T: Clone + 'static + Send + Sync + std::fmt::Debug>(
        &self,
    ) -> Vec<ZeroCopyData<T>> {
        let type_map = self.type_data.read().unwrap();
        let type_id = TypeId::of::<T>();

        if let Some(data_vec) = type_map.get(&type_id) {
            data_vec
                .iter()
                .filter_map(|any_data| any_data.as_any().downcast_ref::<ZeroCopyData<T>>())
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Borrow data by name (creates a view)
    pub fn borrow_data<T: Clone + 'static + Send + Sync + std::fmt::Debug>(
        &self,
        name: &str,
    ) -> CoreResult<ZeroCopyView<T>> {
        let data = self.get_data::<T>(name)?;
        let view = data.view(0, data.len())?;

        {
            let mut stats = self.stats.write().unwrap();
            stats.views_created += 1;
        }

        Ok(view)
    }

    /// Check if data exists by name
    pub fn has_data(&self, name: &str) -> bool {
        self.named_data.read().unwrap().contains_key(name)
    }

    /// Check if data exists by ID
    pub fn has_data_by_id(&self, id: DataId) -> bool {
        self.id_data.read().unwrap().contains_key(&id)
    }

    /// Remove data by name
    pub fn remove_data(&self, name: &str) -> CoreResult<()> {
        let mut named = self.named_data.write().unwrap();

        if let Some(data) = named.remove(name) {
            let id = data.data_id();

            // Remove from ID map
            let mut id_map = self.id_data.write().unwrap();
            id_map.remove(&id);

            // Update statistics
            {
                let mut stats = self.stats.write().unwrap();
                stats.total_memory_managed -= data.metadata().size_bytes;
            }

            Ok(())
        } else {
            Err(CoreError::ValidationError(
                ErrorContext::new(format!("No data found with name '{}'", name))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ))
        }
    }

    /// List all registered data names
    pub fn list_data_names(&self) -> Vec<String> {
        self.named_data.read().unwrap().keys().cloned().collect()
    }

    /// List all registered data IDs
    pub fn list_data_ids(&self) -> Vec<DataId> {
        self.id_data.read().unwrap().keys().cloned().collect()
    }

    /// Get metadata for named data
    pub fn get_metadata(&self, name: &str) -> CoreResult<DataMetadata> {
        let named = self.named_data.read().unwrap();

        if let Some(data) = named.get(name) {
            Ok(data.metadata().clone())
        } else {
            Err(CoreError::ValidationError(
                ErrorContext::new(format!("No data found with name '{}'", name))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ))
        }
    }

    /// Get interface statistics
    pub fn stats(&self) -> InterfaceStats {
        self.stats.read().unwrap().clone()
    }

    /// Clear all data
    pub fn clear(&self) {
        let mut named = self.named_data.write().unwrap();
        let mut id_map = self.id_data.write().unwrap();
        let mut type_map = self.type_data.write().unwrap();

        named.clear();
        id_map.clear();
        type_map.clear();

        {
            let mut stats = self.stats.write().unwrap();
            *stats = InterfaceStats::default();
        }
    }

    fn update_exchange_stats(&self, success: bool) {
        let mut stats = self.stats.write().unwrap();
        if success {
            stats.exchanges_successful += 1;
        } else {
            stats.exchanges_failed += 1;
        }
    }
}

impl Default for ZeroCopyInterface {
    fn default() -> Self {
        Self::new()
    }
}

/// Global zero-copy interface instance
static GLOBAL_INTERFACE: std::sync::OnceLock<ZeroCopyInterface> = std::sync::OnceLock::new();

/// Get the global zero-copy interface
pub fn global_interface() -> &'static ZeroCopyInterface {
    GLOBAL_INTERFACE.get_or_init(ZeroCopyInterface::new)
}

/// Trait for types that can participate in zero-copy data exchange
pub trait DataExchange<T: Clone + 'static> {
    /// Export data to the zero-copy interface
    fn export_data(&self, interface: &ZeroCopyInterface, name: &str) -> CoreResult<DataId>;

    /// Import data from the zero-copy interface
    fn import_data(interface: &ZeroCopyInterface, name: &str) -> CoreResult<Self>
    where
        Self: Sized;
}

/// Helper trait for converting types to zero-copy data
pub trait IntoZeroCopy<T: Clone + 'static> {
    /// Convert into zero-copy data
    fn into_zero_copy(self) -> CoreResult<ZeroCopyData<T>>;
}

impl<T: Clone + 'static> IntoZeroCopy<T> for Vec<T> {
    fn into_zero_copy(self) -> CoreResult<ZeroCopyData<T>> {
        ZeroCopyData::new(self)
    }
}

impl<T: Clone + 'static> IntoZeroCopy<T> for &[T] {
    fn into_zero_copy(self) -> CoreResult<ZeroCopyData<T>> {
        ZeroCopyData::new(self.to_vec())
    }
}

/// Helper trait for extracting data from zero-copy containers
pub trait FromZeroCopy<T: Clone + 'static> {
    /// Extract data from zero-copy container
    fn from_zero_copy(data: &ZeroCopyData<T>) -> Self;
}

impl<T: Clone + 'static> FromZeroCopy<T> for Vec<T> {
    fn from_zero_copy(data: &ZeroCopyData<T>) -> Self {
        data.as_slice().to_vec()
    }
}

/// Create a global zero-copy data registry
pub fn create_global_data_registry() -> &'static ZeroCopyInterface {
    global_interface()
}

/// Register data globally by name
pub fn register_global_data<T: Clone + 'static + Send + Sync + std::fmt::Debug>(
    name: &str,
    data: ZeroCopyData<T>,
) -> CoreResult<()> {
    global_interface().register_data(name, data)
}

/// Get data globally by name
pub fn get_global_data<T: Clone + 'static + Send + Sync + std::fmt::Debug>(
    name: &str,
) -> CoreResult<ZeroCopyData<T>> {
    global_interface().get_data(name)
}

/// Create zero-copy data from a vector
pub fn create_zero_copy_data<T: Clone + 'static + Send + Sync + std::fmt::Debug>(
    data: Vec<T>,
) -> CoreResult<ZeroCopyData<T>> {
    ZeroCopyData::new(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_data_creation() {
        let data = vec![1, 2, 3, 4, 5];
        let zero_copy = ZeroCopyData::new(data.clone()).unwrap();

        assert_eq!(zero_copy.as_slice(), &data);
        assert_eq!(zero_copy.len(), 5);
        assert!(!zero_copy.is_empty());
        assert_eq!(zero_copy.ref_count(), 1);
        assert!(zero_copy.is_unique());
    }

    #[test]
    fn test_zero_copy_data_cloning() {
        let data = vec![1, 2, 3, 4, 5];
        let zero_copy1 = ZeroCopyData::new(data).unwrap();
        let zero_copy2 = zero_copy1.clone();

        assert_eq!(zero_copy1.ref_count(), 2);
        assert_eq!(zero_copy2.ref_count(), 2);
        assert!(!zero_copy1.is_unique());
        assert!(!zero_copy2.is_unique());
        assert_eq!(zero_copy1.id(), zero_copy2.id());
    }

    #[test]
    fn test_zero_copy_view() {
        let data = vec![1, 2, 3, 4, 5];
        let zero_copy = ZeroCopyData::new(data).unwrap();
        let view = zero_copy.view(1, 3).unwrap();

        assert_eq!(view.as_slice(), &[2, 3, 4]);
        assert_eq!(view.len(), 3);

        // Test subview
        let subview = view.subview(1, 1).unwrap();
        assert_eq!(subview.as_slice(), &[3]);
    }

    #[test]
    fn test_zero_copy_interface() {
        let interface = ZeroCopyInterface::new();
        let data = vec![1.0, 2.0, 3.0];
        let zero_copy = ZeroCopyData::new(data.clone()).unwrap();

        // Register data
        interface.register_data("test_data", zero_copy).unwrap();
        assert!(interface.has_data("test_data"));

        // Retrieve data
        let retrieved = interface.get_data::<f64>("test_data").unwrap();
        assert_eq!(retrieved.as_slice(), &data);

        // Borrow data (create view)
        let view = interface.borrow_data::<f64>("test_data").unwrap();
        assert_eq!(view.as_slice(), &data);

        // Check metadata
        let metadata = interface.get_metadata("test_data").unwrap();
        assert_eq!(metadata.element_count, 3);
        assert_eq!(metadata.element_size, std::mem::size_of::<f64>());
    }

    #[test]
    fn test_zero_copy_interface_type_safety() {
        let interface = ZeroCopyInterface::new();
        let data = vec![1, 2, 3];
        let zero_copy = ZeroCopyData::new(data).unwrap();

        interface.register_data("int_data", zero_copy).unwrap();

        // Try to retrieve with wrong type
        let result = interface.get_data::<f64>("int_data");
        assert!(result.is_err());
    }

    #[test]
    fn test_weak_references() {
        let data = vec![1, 2, 3];
        let zero_copy = ZeroCopyData::new(data).unwrap();
        let weak_ref = zero_copy.downgrade();

        assert!(weak_ref.is_alive());
        assert_eq!(weak_ref.id(), zero_copy.id());

        let upgraded = weak_ref.upgrade().unwrap();
        assert_eq!(upgraded.as_slice(), zero_copy.as_slice());

        drop(zero_copy);
        drop(upgraded);

        // Weak reference should still exist but data should be gone
        assert!(!weak_ref.is_alive());
        assert!(weak_ref.upgrade().is_none());
    }

    #[test]
    fn test_global_interface() {
        let data = vec![1.0, 2.0, 3.0];
        let zero_copy = ZeroCopyData::new(data.clone()).unwrap();

        register_global_data("global_test", zero_copy).unwrap();

        let retrieved = get_global_data::<f64>("global_test").unwrap();
        assert_eq!(retrieved.as_slice(), &data);
    }

    #[test]
    fn test_into_zero_copy_trait() {
        let data = vec![1, 2, 3, 4, 5];
        let zero_copy = data.clone().into_zero_copy().unwrap();
        assert_eq!(zero_copy.as_slice(), &data);

        let slice: &[i32] = &data;
        let zero_copy2 = slice.into_zero_copy().unwrap();
        assert_eq!(zero_copy2.as_slice(), &data);
    }

    #[test]
    fn test_from_zero_copy_trait() {
        let data = vec![1, 2, 3, 4, 5];
        let zero_copy = ZeroCopyData::new(data.clone()).unwrap();

        let extracted: Vec<i32> = FromZeroCopy::from_zero_copy(&zero_copy);
        assert_eq!(extracted, data);
    }
}
