//! Enhanced TOML utilities: typed configuration, deep merging, flattening,
//! and schema validation.
//!
//! This module extends the standard [`toml`] crate with higher-level tools
//! that are frequently needed in scientific computing applications:
//!
//! - [`TomlConfig`]: typed wrapper with convenient `get_*` helpers.
//! - [`merge_tomls`]: recursive / deep-merge two TOML trees.
//! - [`flatten_toml`]: flatten a nested TOML value into `HashMap<String, String>`.
//! - [`TomlSchema`] / [`validate_toml_schema`]: declarative schema validation.
//!
//! # Example
//!
//! ```rust
//! use scirs2_io::toml_ext::{TomlConfig, merge_tomls, flatten_toml};
//! use toml::Value;
//!
//! let base: Value = toml::from_str(r#"
//! [server]
//! host = "localhost"
//! port = 8080
//! "#).unwrap();
//!
//! let overlay: Value = toml::from_str(r#"
//! [server]
//! port = 9090
//! [server.tls]
//! enabled = true
//! "#).unwrap();
//!
//! let merged = merge_tomls(&base, &overlay);
//! let cfg = TomlConfig::new(merged);
//!
//! assert_eq!(cfg.get_str("server.host").unwrap(), "localhost");
//! assert_eq!(cfg.get_i64("server.port").unwrap(), 9090);
//! assert_eq!(cfg.get_bool("server.tls.enabled").unwrap(), true);
//!
//! let flat = flatten_toml(cfg.value(), ".");
//! assert_eq!(flat["server.host"], "localhost");
//! ```

use std::collections::HashMap;

use toml::Value;

use crate::error::IoError;

/// Result type used throughout this module.
pub type TomlResult<T> = Result<T, IoError>;

// ─────────────────────────── TomlConfig ──────────────────────────────────────

/// A typed wrapper around a [`toml::Value`] providing dot-separated path access
/// and ergonomic type conversions.
///
/// # Example
///
/// ```rust
/// use scirs2_io::toml_ext::TomlConfig;
///
/// let src = r#"
/// [database]
/// host = "db.example.com"
/// port = 5432
/// debug = false
/// timeout = 30.5
/// tags = ["web", "api"]
/// "#;
/// let cfg = TomlConfig::from_str(src).unwrap();
/// assert_eq!(cfg.get_str("database.host").unwrap(), "db.example.com");
/// assert_eq!(cfg.get_i64("database.port").unwrap(), 5432);
/// assert_eq!(cfg.get_bool("database.debug").unwrap(), false);
/// assert!((cfg.get_f64("database.timeout").unwrap() - 30.5).abs() < 1e-9);
/// ```
#[derive(Debug, Clone)]
pub struct TomlConfig {
    root: Value,
}

impl TomlConfig {
    /// Wrap an existing `toml::Value`.
    pub fn new(root: Value) -> Self {
        Self { root }
    }

    /// Parse TOML text and wrap the result.
    pub fn from_str(src: &str) -> TomlResult<Self> {
        let root: Value =
            toml::from_str(src).map_err(|e| IoError::ParseError(format!("TOML parse: {e}")))?;
        Ok(Self { root })
    }

    /// Read a file and parse it as TOML.
    pub fn from_file(path: &std::path::Path) -> TomlResult<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| IoError::FileError(e.to_string()))?;
        Self::from_str(&text)
    }

    /// Return a reference to the underlying `toml::Value`.
    pub fn value(&self) -> &Value {
        &self.root
    }

    /// Consume the config and return the underlying `toml::Value`.
    pub fn into_value(self) -> Value {
        self.root
    }

    /// Retrieve a nested value by a dot-separated path string.
    ///
    /// Returns `None` if the path does not exist.
    pub fn get(&self, path: &str) -> Option<&Value> {
        get_by_path(&self.root, path)
    }

    /// Get a string value at `path`.
    pub fn get_str(&self, path: &str) -> Option<&str> {
        self.get(path)?.as_str()
    }

    /// Get an integer value at `path`.
    pub fn get_i64(&self, path: &str) -> Option<i64> {
        self.get(path)?.as_integer()
    }

    /// Get a floating-point value at `path`.
    pub fn get_f64(&self, path: &str) -> Option<f64> {
        self.get(path)?.as_float()
    }

    /// Get a boolean value at `path`.
    pub fn get_bool(&self, path: &str) -> Option<bool> {
        self.get(path)?.as_bool()
    }

    /// Get an array value at `path`.
    pub fn get_array(&self, path: &str) -> Option<&Vec<Value>> {
        self.get(path)?.as_array()
    }

    /// Get a nested table at `path`.
    pub fn get_table(&self, path: &str) -> Option<&toml::map::Map<String, Value>> {
        self.get(path)?.as_table()
    }

    /// Check whether a key exists at `path`.
    pub fn contains(&self, path: &str) -> bool {
        self.get(path).is_some()
    }

    /// Return all top-level keys (if the root is a table).
    pub fn keys(&self) -> Vec<&str> {
        match &self.root {
            Value::Table(t) => t.keys().map(|s| s.as_str()).collect(),
            _ => vec![],
        }
    }

    /// Serialise the configuration back to a TOML string.
    pub fn to_toml_string(&self) -> TomlResult<String> {
        toml::to_string_pretty(&self.root)
            .map_err(|e| IoError::SerializationError(format!("TOML serialize: {e}")))
    }
}

/// Navigate a `toml::Value` tree via a dot-separated path.
fn get_by_path<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = root;
    for segment in path.split('.') {
        match current {
            Value::Table(table) => {
                current = table.get(segment)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

// ─────────────────────────── merge_tomls ─────────────────────────────────────

/// Recursively merge two TOML values.
///
/// When both `base` and `overlay` are tables, their keys are merged recursively.
/// For all other combinations, `overlay` takes precedence.
///
/// # Example
///
/// ```rust
/// use scirs2_io::toml_ext::merge_tomls;
/// use toml::Value;
///
/// let base: Value = toml::from_str(r#"
/// [a]
/// x = 1
/// y = 2
/// "#).unwrap();
///
/// let overlay: Value = toml::from_str(r#"
/// [a]
/// y = 99
/// z = 3
/// "#).unwrap();
///
/// let merged = merge_tomls(&base, &overlay);
/// let cfg = merged.as_table().unwrap();
/// let a = cfg["a"].as_table().unwrap();
/// assert_eq!(a["x"].as_integer(), Some(1));
/// assert_eq!(a["y"].as_integer(), Some(99));
/// assert_eq!(a["z"].as_integer(), Some(3));
/// ```
pub fn merge_tomls(base: &Value, overlay: &Value) -> Value {
    match (base, overlay) {
        (Value::Table(base_map), Value::Table(overlay_map)) => {
            let mut merged = base_map.clone();
            for (key, overlay_val) in overlay_map {
                let new_val = if let Some(base_val) = merged.get(key) {
                    merge_tomls(base_val, overlay_val)
                } else {
                    overlay_val.clone()
                };
                merged.insert(key.clone(), new_val);
            }
            Value::Table(merged)
        }
        // For non-table values overlay wins
        (_, overlay_val) => overlay_val.clone(),
    }
}

// ─────────────────────────── flatten_toml ────────────────────────────────────

/// Flatten a nested TOML value into a `HashMap<String, String>`.
///
/// Nested keys are joined with `sep`. Leaf values (integers, floats, booleans,
/// strings, datetimes) are converted to their string representation. Arrays are
/// serialised as JSON arrays.
///
/// # Example
///
/// ```rust
/// use scirs2_io::toml_ext::flatten_toml;
/// use toml::Value;
///
/// let val: Value = toml::from_str(r#"
/// [server]
/// host = "localhost"
/// port = 8080
///
/// [server.tls]
/// enabled = true
/// cert = "/etc/cert.pem"
/// "#).unwrap();
///
/// let flat = flatten_toml(&val, ".");
/// assert_eq!(flat["server.host"], "localhost");
/// assert_eq!(flat["server.port"], "8080");
/// assert_eq!(flat["server.tls.enabled"], "true");
/// assert_eq!(flat["server.tls.cert"], "/etc/cert.pem");
/// ```
pub fn flatten_toml(value: &Value, sep: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    flatten_recursive(value, "", sep, &mut out);
    out
}

fn flatten_recursive(value: &Value, prefix: &str, sep: &str, out: &mut HashMap<String, String>) {
    match value {
        Value::Table(table) => {
            for (key, val) in table {
                let new_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}{sep}{key}")
                };
                flatten_recursive(val, &new_prefix, sep, out);
            }
        }
        Value::Array(arr) => {
            // Serialise array as a JSON-like list for display
            let items: Vec<String> = arr.iter().map(|v| value_to_string(v)).collect();
            let repr = format!("[{}]", items.join(", "));
            out.insert(prefix.to_string(), repr);
        }
        leaf => {
            out.insert(prefix.to_string(), value_to_string(leaf));
        }
    }
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::Datetime(dt) => dt.to_string(),
        Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(|a| value_to_string(a)).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Table(t) => {
            // Inline table
            let pairs: Vec<String> = t
                .iter()
                .map(|(k, v)| format!("{k} = {}", value_to_string(v)))
                .collect();
            format!("{{{}}}", pairs.join(", "))
        }
    }
}

// ─────────────────────────── TomlSchema ──────────────────────────────────────

/// Expected type of a TOML value in schema validation.
#[derive(Debug, Clone, PartialEq)]
pub enum TomlValueType {
    /// Any string.
    String,
    /// Integer.
    Integer,
    /// Float.
    Float,
    /// Boolean.
    Boolean,
    /// Array (elements are not further validated).
    Array,
    /// Table / inline table.
    Table,
    /// Datetime value.
    Datetime,
    /// Accept any type.
    Any,
}

/// A single field rule inside a [`TomlSchema`].
#[derive(Debug, Clone)]
pub struct TomlFieldRule {
    /// Dot-separated path to the field.
    pub path: String,
    /// Whether the field must be present.
    pub required: bool,
    /// Expected type (when `Some`).
    pub expected_type: Option<TomlValueType>,
    /// Optional human-readable description for error messages.
    pub description: Option<String>,
}

impl TomlFieldRule {
    /// Create a required field rule with a type constraint.
    pub fn required(path: impl Into<String>, ty: TomlValueType) -> Self {
        Self {
            path: path.into(),
            required: true,
            expected_type: Some(ty),
            description: None,
        }
    }

    /// Create an optional field rule with a type constraint.
    pub fn optional(path: impl Into<String>, ty: TomlValueType) -> Self {
        Self {
            path: path.into(),
            required: false,
            expected_type: Some(ty),
            description: None,
        }
    }

    /// Attach a description for richer error messages.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// A declarative schema for validating TOML documents.
///
/// # Example
///
/// ```rust
/// use scirs2_io::toml_ext::{TomlSchema, TomlFieldRule, TomlValueType, validate_toml_schema};
///
/// let schema = TomlSchema::new(vec![
///     TomlFieldRule::required("name", TomlValueType::String),
///     TomlFieldRule::required("version", TomlValueType::String),
///     TomlFieldRule::optional("debug", TomlValueType::Boolean),
/// ]);
///
/// let val: toml::Value = toml::from_str(r#"
/// name = "my-app"
/// version = "1.0.0"
/// "#).unwrap();
///
/// validate_toml_schema(&val, &schema).expect("validation failed");
/// ```
#[derive(Debug, Clone, Default)]
pub struct TomlSchema {
    /// Ordered list of field rules.
    pub rules: Vec<TomlFieldRule>,
}

impl TomlSchema {
    /// Construct a schema from a list of rules.
    pub fn new(rules: Vec<TomlFieldRule>) -> Self {
        Self { rules }
    }

    /// Add a rule to the schema.
    pub fn add_rule(&mut self, rule: TomlFieldRule) {
        self.rules.push(rule);
    }
}

/// Validate a TOML value against a [`TomlSchema`].
///
/// Returns `Ok(())` if all required fields are present and all typed fields
/// match their declared type.  Returns the *first* validation error encountered.
///
/// # Example
///
/// ```rust
/// use scirs2_io::toml_ext::{
///     TomlSchema, TomlFieldRule, TomlValueType, validate_toml_schema,
/// };
///
/// let schema = TomlSchema::new(vec![
///     TomlFieldRule::required("host", TomlValueType::String),
///     TomlFieldRule::required("port", TomlValueType::Integer),
/// ]);
///
/// let good: toml::Value = toml::from_str(r#"host = "localhost" \nport = 3000"#
///     .replace("\\n", "\n").as_str()).unwrap();
/// assert!(validate_toml_schema(&good, &schema).is_ok());
///
/// let bad: toml::Value = toml::from_str(r#"host = "localhost""#).unwrap();
/// assert!(validate_toml_schema(&bad, &schema).is_err());
/// ```
pub fn validate_toml_schema(value: &Value, schema: &TomlSchema) -> TomlResult<()> {
    for rule in &schema.rules {
        let found = get_by_path(value, &rule.path);

        match found {
            None => {
                if rule.required {
                    let desc = rule
                        .description
                        .as_deref()
                        .unwrap_or("required field missing");
                    return Err(IoError::ValidationError(format!(
                        "TOML schema: required field '{}' not found ({desc})",
                        rule.path
                    )));
                }
            }
            Some(val) => {
                if let Some(ref expected) = rule.expected_type {
                    if !type_matches(val, expected) {
                        let actual = type_name_of(val);
                        let exp_name = format!("{expected:?}");
                        let desc = rule.description.as_deref().unwrap_or("");
                        return Err(IoError::ValidationError(format!(
                            "TOML schema: field '{}' has type '{actual}' but expected '{exp_name}'{suffix}",
                            rule.path,
                            suffix = if desc.is_empty() { String::new() } else { format!(" ({desc})") }
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

fn type_matches(value: &Value, expected: &TomlValueType) -> bool {
    match expected {
        TomlValueType::Any => true,
        TomlValueType::String => matches!(value, Value::String(_)),
        TomlValueType::Integer => matches!(value, Value::Integer(_)),
        TomlValueType::Float => matches!(value, Value::Float(_)),
        TomlValueType::Boolean => matches!(value, Value::Boolean(_)),
        TomlValueType::Array => matches!(value, Value::Array(_)),
        TomlValueType::Table => matches!(value, Value::Table(_)),
        TomlValueType::Datetime => matches!(value, Value::Datetime(_)),
    }
}

fn type_name_of(value: &Value) -> &'static str {
    match value {
        Value::String(_) => "String",
        Value::Integer(_) => "Integer",
        Value::Float(_) => "Float",
        Value::Boolean(_) => "Boolean",
        Value::Array(_) => "Array",
        Value::Table(_) => "Table",
        Value::Datetime(_) => "Datetime",
    }
}

// ─────────────────────────── Tests ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> Value {
        toml::from_str(src).expect("parse failed")
    }

    // ── TomlConfig ──────────────────────────────────────────────────────────

    #[test]
    fn test_config_get_str() {
        let cfg = TomlConfig::from_str(r#"name = "scirs2""#).unwrap();
        assert_eq!(cfg.get_str("name"), Some("scirs2"));
        assert_eq!(cfg.get_str("missing"), None);
    }

    #[test]
    fn test_config_get_nested() {
        let src = r#"
[server]
host = "127.0.0.1"
port = 443
debug = true
latency = 1.5
"#;
        let cfg = TomlConfig::from_str(src).unwrap();
        assert_eq!(cfg.get_str("server.host"), Some("127.0.0.1"));
        assert_eq!(cfg.get_i64("server.port"), Some(443));
        assert_eq!(cfg.get_bool("server.debug"), Some(true));
        assert!((cfg.get_f64("server.latency").unwrap() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_config_contains_and_keys() {
        let cfg = TomlConfig::from_str(r#"a = 1\nb = 2"#.replace("\\n", "\n").as_str()).unwrap();
        assert!(cfg.contains("a"));
        assert!(!cfg.contains("z"));
        let keys = cfg.keys();
        assert!(keys.contains(&"a"));
        assert!(keys.contains(&"b"));
    }

    #[test]
    fn test_config_to_toml_string_roundtrip() {
        let src = "[db]\nhost = \"localhost\"\nport = 5432\n";
        let cfg = TomlConfig::from_str(src).unwrap();
        let out = cfg.to_toml_string().unwrap();
        let reparsed = TomlConfig::from_str(&out).unwrap();
        assert_eq!(reparsed.get_i64("db.port"), Some(5432));
    }

    #[test]
    fn test_config_get_array() {
        let cfg = TomlConfig::from_str("tags = [\"a\", \"b\", \"c\"]").unwrap();
        let arr = cfg.get_array("tags").expect("should have array");
        assert_eq!(arr.len(), 3);
    }

    // ── merge_tomls ──────────────────────────────────────────────────────────

    #[test]
    fn test_merge_disjoint_keys() {
        let base = parse("[a]\nx = 1");
        let overlay = parse("[b]\ny = 2");
        let merged = merge_tomls(&base, &overlay);
        let t = merged.as_table().unwrap();
        assert!(t.contains_key("a"));
        assert!(t.contains_key("b"));
    }

    #[test]
    fn test_merge_overlay_wins_scalar() {
        let base = parse("x = 1");
        let overlay = parse("x = 99");
        let merged = merge_tomls(&base, &overlay);
        assert_eq!(merged.as_table().unwrap()["x"].as_integer(), Some(99));
    }

    #[test]
    fn test_merge_deep_nested() {
        let base = parse("[a]\n[a.b]\nx = 1\ny = 2");
        let overlay = parse("[a]\n[a.b]\ny = 999\nz = 3");
        let merged = merge_tomls(&base, &overlay);
        let cfg = TomlConfig::new(merged);
        assert_eq!(cfg.get_i64("a.b.x"), Some(1)); // preserved from base
        assert_eq!(cfg.get_i64("a.b.y"), Some(999)); // overridden
        assert_eq!(cfg.get_i64("a.b.z"), Some(3)); // added from overlay
    }

    #[test]
    fn test_merge_empty_base() {
        let base = parse("");
        let overlay = parse("key = \"value\"");
        let merged = merge_tomls(&base, &overlay);
        let cfg = TomlConfig::new(merged);
        assert_eq!(cfg.get_str("key"), Some("value"));
    }

    // ── flatten_toml ─────────────────────────────────────────────────────────

    #[test]
    fn test_flatten_simple() {
        let val = parse("x = 1\ny = 2.5\nz = \"hello\"");
        let flat = flatten_toml(&val, ".");
        assert_eq!(flat["x"], "1");
        assert_eq!(flat["z"], "hello");
    }

    #[test]
    fn test_flatten_nested() {
        let val = parse("[server]\nhost = \"localhost\"\n[server.tls]\nenabled = true");
        let flat = flatten_toml(&val, ".");
        assert_eq!(flat["server.host"], "localhost");
        assert_eq!(flat["server.tls.enabled"], "true");
    }

    #[test]
    fn test_flatten_custom_separator() {
        let val = parse("[a]\n[a.b]\nkey = 42");
        let flat = flatten_toml(&val, "/");
        assert!(flat.contains_key("a/b/key"));
        assert_eq!(flat["a/b/key"], "42");
    }

    #[test]
    fn test_flatten_array() {
        let val = parse("tags = [\"alpha\", \"beta\"]");
        let flat = flatten_toml(&val, ".");
        // Array is serialised as "[alpha, beta]"
        assert!(flat["tags"].contains("alpha"));
        assert!(flat["tags"].contains("beta"));
    }

    // ── validate_toml_schema ─────────────────────────────────────────────────

    #[test]
    fn test_validate_required_missing_fails() {
        let schema = TomlSchema::new(vec![TomlFieldRule::required("name", TomlValueType::String)]);
        let val = parse("other = 1");
        assert!(validate_toml_schema(&val, &schema).is_err());
    }

    #[test]
    fn test_validate_type_mismatch_fails() {
        let schema = TomlSchema::new(vec![TomlFieldRule::required(
            "port",
            TomlValueType::Integer,
        )]);
        let val = parse("port = \"not-an-int\"");
        assert!(validate_toml_schema(&val, &schema).is_err());
    }

    #[test]
    fn test_validate_optional_absent_succeeds() {
        let schema = TomlSchema::new(vec![
            TomlFieldRule::required("host", TomlValueType::String),
            TomlFieldRule::optional("port", TomlValueType::Integer),
        ]);
        let val = parse("host = \"localhost\"");
        assert!(validate_toml_schema(&val, &schema).is_ok());
    }

    #[test]
    fn test_validate_full_pass() {
        let schema = TomlSchema::new(vec![
            TomlFieldRule::required("name", TomlValueType::String),
            TomlFieldRule::required("version", TomlValueType::String),
            TomlFieldRule::optional("debug", TomlValueType::Boolean),
            TomlFieldRule::optional("workers", TomlValueType::Integer),
        ]);
        let val = parse("name = \"app\"\nversion = \"1.0\"\ndebug = true\nworkers = 4");
        assert!(validate_toml_schema(&val, &schema).is_ok());
    }

    #[test]
    fn test_validate_nested_path() {
        let schema = TomlSchema::new(vec![
            TomlFieldRule::required("server.host", TomlValueType::String),
            TomlFieldRule::required("server.port", TomlValueType::Integer),
        ]);
        let val = parse("[server]\nhost = \"127.0.0.1\"\nport = 8080");
        assert!(validate_toml_schema(&val, &schema).is_ok());

        let missing_port = parse("[server]\nhost = \"127.0.0.1\"");
        assert!(validate_toml_schema(&missing_port, &schema).is_err());
    }

    #[test]
    fn test_validate_description_in_error() {
        let schema = TomlSchema::new(vec![TomlFieldRule::required(
            "api_key",
            TomlValueType::String,
        )
        .with_description("API authentication key")]);
        let val = parse("other = 1");
        let err = validate_toml_schema(&val, &schema).unwrap_err();
        assert!(err.to_string().contains("api_key"));
        assert!(err.to_string().contains("API authentication key"));
    }

    #[test]
    fn test_validate_any_type_always_passes() {
        let schema = TomlSchema::new(vec![TomlFieldRule::required("value", TomlValueType::Any)]);
        for src in &[
            "value = 1",
            "value = \"str\"",
            "value = true",
            "value = 1.5",
        ] {
            let val = parse(src);
            assert!(
                validate_toml_schema(&val, &schema).is_ok(),
                "failed for: {src}"
            );
        }
    }
}
