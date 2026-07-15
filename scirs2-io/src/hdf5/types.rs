//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

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
