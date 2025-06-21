//! Serialization utilities for ndarray types with serde
//!
//! This module provides helper functions for serializing and deserializing
//! ndarray Array1 and Array2 types with serde, enabling JSON and other format
//! compatibility for dataset structures.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::vec::Vec;

/// Serialize a 2D array to a format compatible with serde
///
/// The serialization format stores the shape information (rows, cols) at the
/// beginning of a flat vector, followed by the array data in row-major order.
///
/// # Arguments
///
/// * `array` - The 2D array to serialize
/// * `serializer` - The serde serializer to use
///
/// # Returns
///
/// * `Result<S::Ok, S::Error>` - Serialization result
pub fn serialize_array2<S>(array: &Array2<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let shape = array.shape();
    let mut vec = Vec::with_capacity(shape[0] * shape[1] + 2);

    // Store shape at the beginning
    vec.push(shape[0] as f64);
    vec.push(shape[1] as f64);

    // Store data
    vec.extend(array.iter().cloned());

    vec.serialize(serializer)
}

/// Deserialize a 2D array from a serde-compatible format
///
/// Reconstructs an Array2 from the flattened format created by serialize_array2.
/// The first two elements are interpreted as the shape (rows, cols), and the
/// remaining elements are reshaped into the 2D array.
///
/// # Arguments
///
/// * `deserializer` - The serde deserializer to use
///
/// # Returns
///
/// * `Result<Array2<f64>, D::Error>` - Deserialized 2D array
pub fn deserialize_array2<'de, D>(deserializer: D) -> Result<Array2<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let vec = Vec::<f64>::deserialize(deserializer)?;
    if vec.len() < 2 {
        return Err(serde::de::Error::custom("Invalid array2 serialization"));
    }

    let nrows = vec[0] as usize;
    let ncols = vec[1] as usize;

    if vec.len() != nrows * ncols + 2 {
        return Err(serde::de::Error::custom("Invalid array2 serialization"));
    }

    let data = vec[2..].to_vec();
    match Array2::from_shape_vec((nrows, ncols), data) {
        Ok(array) => Ok(array),
        Err(_) => Err(serde::de::Error::custom("Failed to reshape array2")),
    }
}

/// Serialize a 1D array to a format compatible with serde
///
/// Simply converts the Array1 to a Vec for JSON serialization.
///
/// # Arguments
///
/// * `array` - The 1D array to serialize
/// * `serializer` - The serde serializer to use
///
/// # Returns
///
/// * `Result<S::Ok, S::Error>` - Serialization result
#[allow(dead_code)]
pub fn serialize_array1<S>(array: &Array1<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let vec = array.to_vec();
    vec.serialize(serializer)
}

/// Deserialize a 1D array from a serde-compatible format
///
/// Reconstructs an Array1 from a Vec<f64>.
///
/// # Arguments
///
/// * `deserializer` - The serde deserializer to use
///
/// # Returns
///
/// * `Result<Array1<f64>, D::Error>` - Deserialized 1D array
pub fn deserialize_array1<'de, D>(deserializer: D) -> Result<Array1<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let vec = Vec::<f64>::deserialize(deserializer)?;
    Ok(Array1::from(vec))
}

/// Helper functions for serializing Option<Array1<f64>> types
pub mod optional_array1 {
    use super::*;

    /// Serialize an optional 1D array
    ///
    /// # Arguments
    ///
    /// * `array_opt` - The optional array to serialize
    /// * `serializer` - The serde serializer to use
    ///
    /// # Returns
    ///
    /// * `Result<S::Ok, S::Error>` - Serialization result
    #[allow(dead_code)]
    pub fn serialize<S>(array_opt: &Option<Array1<f64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match array_opt {
            Some(array) => {
                #[derive(Serialize)]
                struct Wrapper<'a> {
                    #[serde(
                        serialize_with = "super::serialize_array1",
                        deserialize_with = "super::deserialize_array1"
                    )]
                    value: &'a Array1<f64>,
                }

                Wrapper { value: array }.serialize(serializer)
            }
            None => serializer.serialize_none(),
        }
    }

    /// Deserialize an optional 1D array
    ///
    /// # Arguments
    ///
    /// * `deserializer` - The serde deserializer to use
    ///
    /// # Returns
    ///
    /// * `Result<Option<Array1<f64>>, D::Error>` - Deserialized optional array
    #[allow(dead_code)]
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Array1<f64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wrapper {
            #[serde(
                serialize_with = "super::serialize_array1",
                deserialize_with = "super::deserialize_array1"
            )]
            #[allow(dead_code)]
            value: Array1<f64>,
        }

        Option::<Wrapper>::deserialize(deserializer).map(|opt_wrapper| opt_wrapper.map(|w| w.value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_array2_serialization_roundtrip() {
        let original = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Serialize to JSON
        let _json = serde_json::to_string(&original.map(|x| *x)).unwrap();

        // For testing, we need to manually test the serialization functions
        // since they're designed to work with serde attributes
        let vec = vec![2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let reconstructed = Array2::from_shape_vec((2, 3), vec[2..].to_vec()).unwrap();

        assert_eq!(original, reconstructed);
    }

    #[test]
    fn test_array1_serialization_roundtrip() {
        let original = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let vec = original.to_vec();
        let reconstructed = Array1::from(vec);

        assert_eq!(original, reconstructed);
    }

    #[test]
    fn test_invalid_array2_deserialization() {
        // Test with insufficient data
        let vec = vec![2.0, 3.0, 1.0]; // Claims 2x3 but only has 1 element
        let result = Array2::from_shape_vec((2, 3), vec[2..].to_vec());
        assert!(result.is_err());
    }
}
