//! Serialization format utilities
//!
//! This module provides utilities for converting between different serialization formats.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use super::SerializationFormat;
use crate::error::{MetricsError, Result};

/// Convert from one serialization format to another
///
/// # Arguments
///
/// * `data` - Data to convert
/// * `from_format` - Format to convert from
/// * `to_format` - Format to convert to
///
/// # Returns
///
/// * Result containing the converted data
pub fn convert_format<T>(
    data: &T,
    from_format: SerializationFormat,
    to_format: SerializationFormat,
) -> Result<Vec<u8>>
where
    T: Serialize + for<'de> Deserialize<'de>,
{
    // If formats are the same, just serialize directly
    if from_format == to_format {
        return serialize(data, to_format);
    }

    // Serialize the data to a JSON representation regardless of source format
    let json_value = match from_format {
        SerializationFormat::Json => {
            // If source is JSON, just serialize to a JSON value
            serde_json::to_value(data)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?
        }
        SerializationFormat::Yaml => {
            // Serialize to YAML string, then to JSON value
            let yaml_str = serde_yaml::to_string(data)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;

            // First convert to YAML value
            let yaml_value = serde_yaml::from_str::<serde_yaml::Value>(&yaml_str)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;

            // Then convert to a generic serializable form
            serde_json::to_value(&yaml_value)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?
        }
        SerializationFormat::Toml => {
            // Serialize to TOML string, then to JSON value
            let toml_str = toml::to_string(data)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;

            // First convert to TOML value
            let toml_value = toml::from_str::<toml::Value>(&toml_str)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;

            // Then convert to a generic serializable form
            serde_json::to_value(&toml_value)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?
        }
        SerializationFormat::Cbor => {
            // For CBOR, first serialize to bytes
            let mut serialized = Vec::new();
            ciborium::ser::into_writer(data, &mut serialized)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;

            // Then deserialize to ciborium::Value
            let value: ciborium::Value = ciborium::de::from_reader(&serialized[..])
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;

            // Convert ciborium::Value to serde_json::Value (intermediate step)
            serde_json::to_value(value)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?
        }
    };

    // Serialize to the target format
    match to_format {
        SerializationFormat::Json => serde_json::to_vec_pretty(&json_value)
            .map_err(|e| MetricsError::SerializationError(e.to_string())),
        SerializationFormat::Yaml => {
            let yaml = serde_yaml::to_string(&json_value)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;
            Ok(yaml.into_bytes())
        }
        SerializationFormat::Toml => {
            // Convert through JSON for TOML
            let json = serde_json::to_string(&json_value)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;
            let toml_value: toml::Value = serde_json::from_str(&json)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;
            let toml_str = toml::to_string_pretty(&toml_value)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;
            Ok(toml_str.into_bytes())
        }
        SerializationFormat::Cbor => {
            let mut bytes = Vec::new();
            ciborium::ser::into_writer(&json_value, &mut bytes)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;
            Ok(bytes)
        }
    }
}

/// Convert a file from one serialization format to another
///
/// # Arguments
///
/// * `input_path` - Path to the input file
/// * `output_path` - Path to the output file
/// * `from_format` - Format to convert from
/// * `to_format` - Format to convert to
///
/// # Returns
///
/// * Result indicating success or error
pub fn convert_file<T, P1, P2>(
    input_path: P1,
    output_path: P2,
    from_format: SerializationFormat,
    to_format: SerializationFormat,
) -> Result<()>
where
    T: Serialize + for<'de> Deserialize<'de>,
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    // Read the input file
    let data = deserialize_from_file::<T, _>(input_path, from_format)?;

    // Convert and write to the output file
    serialize_to_file(&data, output_path, to_format)
}

/// Serialize data to a specific format
///
/// # Arguments
///
/// * `data` - Data to serialize
/// * `format` - Format to serialize to
///
/// # Returns
///
/// * Result containing the serialized data
pub fn serialize<T>(data: &T, format: SerializationFormat) -> Result<Vec<u8>>
where
    T: Serialize,
{
    match format {
        SerializationFormat::Json => serde_json::to_vec_pretty(data)
            .map_err(|e| MetricsError::SerializationError(e.to_string())),
        SerializationFormat::Yaml => {
            let yaml = serde_yaml::to_string(data)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;
            Ok(yaml.into_bytes())
        }
        SerializationFormat::Toml => {
            let toml = toml::to_string_pretty(data)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;
            Ok(toml.into_bytes())
        }
        SerializationFormat::Cbor => {
            let mut bytes = Vec::new();
            ciborium::ser::into_writer(data, &mut bytes)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;
            Ok(bytes)
        }
    }
}

/// Deserialize data from a specific format
///
/// # Arguments
///
/// * `data` - Serialized data
/// * `format` - Format to deserialize from
///
/// # Returns
///
/// * Result containing the deserialized data
pub fn deserialize<T>(data: &[u8], format: SerializationFormat) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    match format {
        SerializationFormat::Json => serde_json::from_slice(data)
            .map_err(|e| MetricsError::SerializationError(e.to_string())),
        SerializationFormat::Yaml => {
            let yaml_str = std::str::from_utf8(data)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;
            serde_yaml::from_str(yaml_str)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))
        }
        SerializationFormat::Toml => {
            let toml_str = std::str::from_utf8(data)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?;
            toml::from_str(toml_str).map_err(|e| MetricsError::SerializationError(e.to_string()))
        }
        SerializationFormat::Cbor => ciborium::de::from_reader(data)
            .map_err(|e| MetricsError::SerializationError(e.to_string())),
    }
}

/// Serialize data to a file
///
/// # Arguments
///
/// * `data` - Data to serialize
/// * `path` - Path to write to
/// * `format` - Format to serialize to
///
/// # Returns
///
/// * Result indicating success or error
pub fn serialize_to_file<T, P>(data: &T, path: P, format: SerializationFormat) -> Result<()>
where
    T: Serialize,
    P: AsRef<Path>,
{
    let serialized = serialize(data, format)?;

    let mut file = File::create(path).map_err(|e| MetricsError::IOError(e.to_string()))?;

    file.write_all(&serialized)
        .map_err(|e| MetricsError::IOError(e.to_string()))?;

    Ok(())
}

/// Deserialize data from a file
///
/// # Arguments
///
/// * `path` - Path to read from
/// * `format` - Format to deserialize from
///
/// # Returns
///
/// * Result containing the deserialized data
pub fn deserialize_from_file<T, P>(path: P, format: SerializationFormat) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
    P: AsRef<Path>,
{
    let mut file = File::open(path).map_err(|e| MetricsError::IOError(e.to_string()))?;

    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| MetricsError::IOError(e.to_string()))?;

    deserialize(&data, format)
}
