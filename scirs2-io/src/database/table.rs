//! Pure Rust in-memory columnar table with join, group-by, filter, sort, project.
//!
//! Provides a DataFrame-like in-memory table that operates entirely in pure Rust
//! without any external SQL dependencies.
//!
//! # Example
//!
//! ```rust
//! use scirs2_io::database::table::{InMemoryTable, ColumnValue, TableFilter, Predicate};
//!
//! let mut table = InMemoryTable::new(vec![
//!     ("id".to_string(),    scirs2_io::database::table::ColumnType::Int64),
//!     ("name".to_string(),  scirs2_io::database::table::ColumnType::Utf8),
//!     ("score".to_string(), scirs2_io::database::table::ColumnType::Float64),
//! ]);
//!
//! table.push_row(&[
//!     ColumnValue::Int(1),
//!     ColumnValue::Utf8("Alice".to_string()),
//!     ColumnValue::Float(95.0),
//! ]).unwrap();
//!
//! table.push_row(&[
//!     ColumnValue::Int(2),
//!     ColumnValue::Utf8("Bob".to_string()),
//!     ColumnValue::Float(82.5),
//! ]).unwrap();
//!
//! let filtered = TableFilter::new(&table)
//!     .predicate(Predicate::Greater("score".to_string(), ColumnValue::Float(90.0)))
//!     .apply()
//!     .unwrap();
//! assert_eq!(filtered.row_count(), 1);
//! ```

#![allow(missing_docs)]

use crate::error::{IoError, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

// ─── Column types and values ─────────────────────────────────────────────────

/// The type of a column in an [`InMemoryTable`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColumnType {
    /// 64-bit signed integer
    Int64,
    /// 64-bit floating point
    Float64,
    /// UTF-8 string
    Utf8,
    /// Boolean
    Boolean,
    /// Nullable wrapper
    Nullable(Box<ColumnType>),
}

impl ColumnType {
    /// Return a string identifier.
    pub fn as_str(&self) -> &str {
        match self {
            ColumnType::Int64 => "int64",
            ColumnType::Float64 => "float64",
            ColumnType::Utf8 => "utf8",
            ColumnType::Boolean => "boolean",
            ColumnType::Nullable(_) => "nullable",
        }
    }
}

/// A single cell value in an [`InMemoryTable`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColumnValue {
    /// NULL / missing
    Null,
    /// 64-bit signed integer
    Int(i64),
    /// 64-bit floating point
    Float(f64),
    /// UTF-8 string
    Utf8(String),
    /// Boolean
    Boolean(bool),
}

impl ColumnValue {
    /// Return the column type for this value.
    pub fn column_type(&self) -> ColumnType {
        match self {
            ColumnValue::Null => ColumnType::Nullable(Box::new(ColumnType::Utf8)),
            ColumnValue::Int(_) => ColumnType::Int64,
            ColumnValue::Float(_) => ColumnType::Float64,
            ColumnValue::Utf8(_) => ColumnType::Utf8,
            ColumnValue::Boolean(_) => ColumnType::Boolean,
        }
    }

    /// Try to extract as f64 (for aggregation).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ColumnValue::Float(f) => Some(*f),
            ColumnValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to extract as i64.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ColumnValue::Int(i) => Some(*i),
            ColumnValue::Float(f) if f.fract() == 0.0 => Some(*f as i64),
            _ => None,
        }
    }

    /// Partial comparison for ordering.
    pub fn partial_cmp_value(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (ColumnValue::Int(a), ColumnValue::Int(b)) => a.partial_cmp(b),
            (ColumnValue::Float(a), ColumnValue::Float(b)) => a.partial_cmp(b),
            (ColumnValue::Int(a), ColumnValue::Float(b)) => (*a as f64).partial_cmp(b),
            (ColumnValue::Float(a), ColumnValue::Int(b)) => a.partial_cmp(&(*b as f64)),
            (ColumnValue::Utf8(a), ColumnValue::Utf8(b)) => a.partial_cmp(b),
            (ColumnValue::Boolean(a), ColumnValue::Boolean(b)) => a.partial_cmp(b),
            (ColumnValue::Null, ColumnValue::Null) => Some(Ordering::Equal),
            (ColumnValue::Null, _) => Some(Ordering::Less),
            (_, ColumnValue::Null) => Some(Ordering::Greater),
            _ => None,
        }
    }

    /// Convert to JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            ColumnValue::Null => serde_json::Value::Null,
            ColumnValue::Int(i) => serde_json::json!(i),
            ColumnValue::Float(f) => serde_json::json!(f),
            ColumnValue::Utf8(s) => serde_json::json!(s),
            ColumnValue::Boolean(b) => serde_json::json!(b),
        }
    }

    /// Try to convert from a JSON value.
    pub fn from_json(v: &serde_json::Value, col_type: &ColumnType) -> Self {
        match (col_type, v) {
            (_, serde_json::Value::Null) => ColumnValue::Null,
            (ColumnType::Int64, serde_json::Value::Number(n)) => {
                ColumnValue::Int(n.as_i64().unwrap_or_default())
            }
            (ColumnType::Float64, serde_json::Value::Number(n)) => {
                ColumnValue::Float(n.as_f64().unwrap_or_default())
            }
            (ColumnType::Utf8, serde_json::Value::String(s)) => ColumnValue::Utf8(s.clone()),
            (ColumnType::Boolean, serde_json::Value::Bool(b)) => ColumnValue::Boolean(*b),
            (ColumnType::Nullable(inner), val) => Self::from_json(val, inner),
            _ => ColumnValue::Null,
        }
    }
}

impl std::fmt::Display for ColumnValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColumnValue::Null => write!(f, "NULL"),
            ColumnValue::Int(i) => write!(f, "{i}"),
            ColumnValue::Float(v) => write!(f, "{v}"),
            ColumnValue::Utf8(s) => write!(f, "{s}"),
            ColumnValue::Boolean(b) => write!(f, "{b}"),
        }
    }
}

// ─── Column schema ───────────────────────────────────────────────────────────

/// Describes a column in an [`InMemoryTable`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSchema {
    /// Column name
    pub name: String,
    /// Column type
    pub col_type: ColumnType,
}

// ─── InMemoryTable ───────────────────────────────────────────────────────────

/// A columnar in-memory table.
///
/// Data is stored in row-major form (`Vec<Vec<ColumnValue>>`) to simplify row
/// operations like filtering and joining, while still supporting column projection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InMemoryTable {
    /// Column schemas (ordered)
    pub columns: Vec<ColumnSchema>,
    /// Row data: each row is `Vec<ColumnValue>` with one entry per column
    pub rows: Vec<Vec<ColumnValue>>,
    /// Optional table name
    pub name: Option<String>,
}

impl InMemoryTable {
    /// Create a new empty table with the given column name/type pairs.
    pub fn new(columns: Vec<(String, ColumnType)>) -> Self {
        Self {
            columns: columns
                .into_iter()
                .map(|(n, t)| ColumnSchema {
                    name: n,
                    col_type: t,
                })
                .collect(),
            rows: Vec::new(),
            name: None,
        }
    }

    /// Set the table name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Return the number of rows.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Return the number of columns.
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Return column index by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Add a row. Returns an error if the row length doesn't match the schema.
    pub fn push_row(&mut self, row: &[ColumnValue]) -> Result<()> {
        if row.len() != self.columns.len() {
            return Err(IoError::ValidationError(format!(
                "Row has {} values but table has {} columns",
                row.len(),
                self.columns.len()
            )));
        }
        self.rows.push(row.to_vec());
        Ok(())
    }

    /// Add a row from a map. Missing columns get NULL.
    pub fn push_row_map(&mut self, map: HashMap<String, ColumnValue>) -> Result<()> {
        let row: Vec<ColumnValue> = self
            .columns
            .iter()
            .map(|col| map.get(&col.name).cloned().unwrap_or(ColumnValue::Null))
            .collect();
        self.rows.push(row);
        Ok(())
    }

    /// Get a cell value.
    pub fn get(&self, row: usize, col: usize) -> Option<&ColumnValue> {
        self.rows.get(row)?.get(col)
    }

    /// Get a column as a slice of values.
    pub fn get_column(&self, name: &str) -> Option<Vec<&ColumnValue>> {
        let idx = self.column_index(name)?;
        Some(self.rows.iter().map(|r| &r[idx]).collect())
    }

    /// Return a row as a HashMap.
    pub fn row_as_map(&self, row_idx: usize) -> Option<HashMap<String, ColumnValue>> {
        let row = self.rows.get(row_idx)?;
        Some(
            self.columns
                .iter()
                .zip(row.iter())
                .map(|(col, val)| (col.name.clone(), val.clone()))
                .collect(),
        )
    }

    /// Return rows as a Vec of JSON objects.
    pub fn to_json_rows(&self) -> Vec<serde_json::Value> {
        self.rows
            .iter()
            .map(|row| {
                let obj: serde_json::Map<String, serde_json::Value> = self
                    .columns
                    .iter()
                    .zip(row.iter())
                    .map(|(col, val)| (col.name.clone(), val.to_json()))
                    .collect();
                serde_json::Value::Object(obj)
            })
            .collect()
    }

    /// Append all rows from another table (must have compatible schemas).
    pub fn append(&mut self, other: &InMemoryTable) -> Result<()> {
        if self.columns.len() != other.columns.len() {
            return Err(IoError::ValidationError(
                "Column count mismatch in append".to_string(),
            ));
        }
        for (a, b) in self.columns.iter().zip(other.columns.iter()) {
            if a.name != b.name {
                return Err(IoError::ValidationError(format!(
                    "Column name mismatch: '{}' vs '{}'",
                    a.name, b.name
                )));
            }
        }
        self.rows.extend(other.rows.clone());
        Ok(())
    }
}

// ─── TableFilter ─────────────────────────────────────────────────────────────

/// A filter predicate for rows.
#[derive(Debug, Clone)]
pub enum Predicate {
    /// column == value
    Eq(String, ColumnValue),
    /// column != value
    Ne(String, ColumnValue),
    /// column > value
    Greater(String, ColumnValue),
    /// column >= value
    GreaterEq(String, ColumnValue),
    /// column < value
    Less(String, ColumnValue),
    /// column <= value
    LessEq(String, ColumnValue),
    /// column IS NULL
    IsNull(String),
    /// column IS NOT NULL
    IsNotNull(String),
    /// String column LIKE pattern (% = any chars, _ = single char)
    Like(String, String),
    /// column value is one of
    In(String, Vec<ColumnValue>),
    /// Both predicates must hold
    And(Box<Predicate>, Box<Predicate>),
    /// Either predicate must hold
    Or(Box<Predicate>, Box<Predicate>),
    /// Predicate must not hold
    Not(Box<Predicate>),
}

impl Predicate {
    fn eval(&self, row: &[ColumnValue], columns: &[ColumnSchema]) -> bool {
        match self {
            Predicate::Eq(col, val) => get_col_val(row, columns, col)
                .map(|v| v == val)
                .unwrap_or(false),
            Predicate::Ne(col, val) => get_col_val(row, columns, col)
                .map(|v| v != val)
                .unwrap_or(false),
            Predicate::Greater(col, val) => get_col_val(row, columns, col)
                .and_then(|v| v.partial_cmp_value(val))
                .map(|o| o == Ordering::Greater)
                .unwrap_or(false),
            Predicate::GreaterEq(col, val) => get_col_val(row, columns, col)
                .and_then(|v| v.partial_cmp_value(val))
                .map(|o| o != Ordering::Less)
                .unwrap_or(false),
            Predicate::Less(col, val) => get_col_val(row, columns, col)
                .and_then(|v| v.partial_cmp_value(val))
                .map(|o| o == Ordering::Less)
                .unwrap_or(false),
            Predicate::LessEq(col, val) => get_col_val(row, columns, col)
                .and_then(|v| v.partial_cmp_value(val))
                .map(|o| o != Ordering::Greater)
                .unwrap_or(false),
            Predicate::IsNull(col) => get_col_val(row, columns, col)
                .map(|v| matches!(v, ColumnValue::Null))
                .unwrap_or(true),
            Predicate::IsNotNull(col) => get_col_val(row, columns, col)
                .map(|v| !matches!(v, ColumnValue::Null))
                .unwrap_or(false),
            Predicate::Like(col, pattern) => get_col_val(row, columns, col)
                .and_then(|v| {
                    if let ColumnValue::Utf8(s) = v {
                        Some(like_match(s, pattern))
                    } else {
                        None
                    }
                })
                .unwrap_or(false),
            Predicate::In(col, values) => get_col_val(row, columns, col)
                .map(|v| values.contains(v))
                .unwrap_or(false),
            Predicate::And(a, b) => a.eval(row, columns) && b.eval(row, columns),
            Predicate::Or(a, b) => a.eval(row, columns) || b.eval(row, columns),
            Predicate::Not(p) => !p.eval(row, columns),
        }
    }
}

fn get_col_val<'a>(
    row: &'a [ColumnValue],
    columns: &[ColumnSchema],
    name: &str,
) -> Option<&'a ColumnValue> {
    let idx = columns.iter().position(|c| c.name == name)?;
    row.get(idx)
}

/// Simple LIKE pattern matching (% = any, _ = one char).
fn like_match(s: &str, pattern: &str) -> bool {
    like_match_recursive(s.as_bytes(), pattern.as_bytes())
}

fn like_match_recursive(s: &[u8], p: &[u8]) -> bool {
    match (s, p) {
        (_, []) => s.is_empty(),
        (_, [b'%', rest @ ..]) => {
            // % matches zero or more characters
            for i in 0..=s.len() {
                if like_match_recursive(&s[i..], rest) {
                    return true;
                }
            }
            false
        }
        ([], _) => false,
        ([sc, s_rest @ ..], [b'_', p_rest @ ..]) => {
            like_match_recursive(s_rest, p_rest)
                || (sc.is_ascii() && like_match_recursive(s_rest, p_rest))
        }
        ([sc, s_rest @ ..], [pc, p_rest @ ..]) => {
            sc.eq_ignore_ascii_case(pc) && like_match_recursive(s_rest, p_rest)
        }
    }
}

/// Predicate-based row filtering.
pub struct TableFilter<'a> {
    table: &'a InMemoryTable,
    predicates: Vec<Predicate>,
}

impl<'a> TableFilter<'a> {
    /// Create a new filter builder.
    pub fn new(table: &'a InMemoryTable) -> Self {
        Self {
            table,
            predicates: Vec::new(),
        }
    }

    /// Add a predicate (combined with AND).
    pub fn predicate(mut self, p: Predicate) -> Self {
        self.predicates.push(p);
        self
    }

    /// Apply all predicates and return a new table with matching rows.
    pub fn apply(&self) -> Result<InMemoryTable> {
        let mut result = InMemoryTable {
            columns: self.table.columns.clone(),
            rows: Vec::new(),
            name: self.table.name.clone(),
        };
        for row in &self.table.rows {
            let matches = self
                .predicates
                .iter()
                .all(|p| p.eval(row, &self.table.columns));
            if matches {
                result.rows.push(row.clone());
            }
        }
        Ok(result)
    }
}

// ─── TableSort ───────────────────────────────────────────────────────────────

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    /// Ascending order
    Asc,
    /// Descending order
    Desc,
}

/// A column + direction sort key.
#[derive(Debug, Clone)]
pub struct SortKey {
    /// Column name
    pub column: String,
    /// Sort direction
    pub direction: SortDirection,
    /// Whether NULLs sort first
    pub nulls_first: bool,
}

impl SortKey {
    /// Ascending sort on a column.
    pub fn asc(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            direction: SortDirection::Asc,
            nulls_first: false,
        }
    }

    /// Descending sort on a column.
    pub fn desc(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            direction: SortDirection::Desc,
            nulls_first: false,
        }
    }
}

/// Multi-column sort.
pub struct TableSort;

impl TableSort {
    /// Sort `table` by the given keys and return a new sorted table.
    pub fn sort(table: &InMemoryTable, keys: &[SortKey]) -> Result<InMemoryTable> {
        // Validate that all sort columns exist
        for key in keys {
            if table.column_index(&key.column).is_none() {
                return Err(IoError::ValidationError(format!(
                    "Sort column '{}' not found in table",
                    key.column
                )));
            }
        }

        let mut rows = table.rows.clone();
        rows.sort_by(|a, b| {
            for key in keys {
                let idx = table
                    .columns
                    .iter()
                    .position(|c| c.name == key.column)
                    .unwrap_or(0);
                let va = &a[idx];
                let vb = &b[idx];

                let ord = match (va, vb) {
                    (ColumnValue::Null, ColumnValue::Null) => Ordering::Equal,
                    (ColumnValue::Null, _) => {
                        if key.nulls_first {
                            Ordering::Less
                        } else {
                            Ordering::Greater
                        }
                    }
                    (_, ColumnValue::Null) => {
                        if key.nulls_first {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    }
                    _ => va.partial_cmp_value(vb).unwrap_or(Ordering::Equal),
                };

                let ord = if key.direction == SortDirection::Desc {
                    ord.reverse()
                } else {
                    ord
                };

                if ord != Ordering::Equal {
                    return ord;
                }
            }
            Ordering::Equal
        });

        Ok(InMemoryTable {
            columns: table.columns.clone(),
            rows,
            name: table.name.clone(),
        })
    }
}

// ─── TableProjection ────────────────────────────────────────────────────────

/// Column selection and renaming.
pub struct TableProjection<'a> {
    table: &'a InMemoryTable,
    selections: Vec<(String, Option<String>)>, // (original_name, alias)
}

impl<'a> TableProjection<'a> {
    /// Create a projection builder.
    pub fn new(table: &'a InMemoryTable) -> Self {
        Self {
            table,
            selections: Vec::new(),
        }
    }

    /// Select a column (keep original name).
    pub fn column(mut self, name: impl Into<String>) -> Self {
        self.selections.push((name.into(), None));
        self
    }

    /// Select a column with an alias.
    pub fn column_as(mut self, name: impl Into<String>, alias: impl Into<String>) -> Self {
        self.selections.push((name.into(), Some(alias.into())));
        self
    }

    /// Apply the projection and return a new table.
    pub fn apply(&self) -> Result<InMemoryTable> {
        // Resolve column indices
        let mut indices: Vec<(usize, String)> = Vec::new();
        for (orig, alias) in &self.selections {
            let idx = self.table.column_index(orig).ok_or_else(|| {
                IoError::ValidationError(format!("Projection column '{}' not found in table", orig))
            })?;
            let out_name = alias.as_deref().unwrap_or(orig.as_str()).to_string();
            indices.push((idx, out_name));
        }

        let new_columns: Vec<ColumnSchema> = indices
            .iter()
            .map(|(idx, name)| ColumnSchema {
                name: name.clone(),
                col_type: self.table.columns[*idx].col_type.clone(),
            })
            .collect();

        let new_rows: Vec<Vec<ColumnValue>> = self
            .table
            .rows
            .iter()
            .map(|row| indices.iter().map(|(idx, _)| row[*idx].clone()).collect())
            .collect();

        Ok(InMemoryTable {
            columns: new_columns,
            rows: new_rows,
            name: self.table.name.clone(),
        })
    }
}

// ─── GroupBy ─────────────────────────────────────────────────────────────────

/// Aggregation function for group-by.
#[derive(Debug, Clone)]
pub enum AggFunc {
    /// Count rows in group
    Count,
    /// Sum of column values
    Sum(String),
    /// Arithmetic mean
    Mean(String),
    /// Minimum value
    Min(String),
    /// Maximum value
    Max(String),
    /// Sample standard deviation
    Std(String),
    /// Collect distinct count
    CountDistinct(String),
}

impl AggFunc {
    /// Return the output column name for this aggregation.
    pub fn output_name(&self) -> String {
        match self {
            AggFunc::Count => "count".to_string(),
            AggFunc::Sum(col) => format!("sum_{col}"),
            AggFunc::Mean(col) => format!("mean_{col}"),
            AggFunc::Min(col) => format!("min_{col}"),
            AggFunc::Max(col) => format!("max_{col}"),
            AggFunc::Std(col) => format!("std_{col}"),
            AggFunc::CountDistinct(col) => format!("count_distinct_{col}"),
        }
    }

    fn compute(&self, rows: &[&Vec<ColumnValue>], columns: &[ColumnSchema]) -> ColumnValue {
        match self {
            AggFunc::Count => ColumnValue::Int(rows.len() as i64),
            AggFunc::Sum(col) => {
                let sum: f64 = rows
                    .iter()
                    .filter_map(|r| get_col_val(r, columns, col)?.as_f64())
                    .sum();
                ColumnValue::Float(sum)
            }
            AggFunc::Mean(col) => {
                let vals: Vec<f64> = rows
                    .iter()
                    .filter_map(|r| get_col_val(r, columns, col)?.as_f64())
                    .collect();
                if vals.is_empty() {
                    ColumnValue::Null
                } else {
                    ColumnValue::Float(vals.iter().sum::<f64>() / vals.len() as f64)
                }
            }
            AggFunc::Min(col) => rows
                .iter()
                .filter_map(|r| get_col_val(r, columns, col))
                .filter(|v| !matches!(v, ColumnValue::Null))
                .min_by(|a, b| a.partial_cmp_value(b).unwrap_or(Ordering::Equal))
                .cloned()
                .unwrap_or(ColumnValue::Null),
            AggFunc::Max(col) => rows
                .iter()
                .filter_map(|r| get_col_val(r, columns, col))
                .filter(|v| !matches!(v, ColumnValue::Null))
                .max_by(|a, b| a.partial_cmp_value(b).unwrap_or(Ordering::Equal))
                .cloned()
                .unwrap_or(ColumnValue::Null),
            AggFunc::Std(col) => {
                let vals: Vec<f64> = rows
                    .iter()
                    .filter_map(|r| get_col_val(r, columns, col)?.as_f64())
                    .collect();
                if vals.len() < 2 {
                    ColumnValue::Float(0.0)
                } else {
                    let n = vals.len() as f64;
                    let mean = vals.iter().sum::<f64>() / n;
                    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
                    ColumnValue::Float(variance.sqrt())
                }
            }
            AggFunc::CountDistinct(col) => {
                let mut seen: Vec<String> = Vec::new();
                for r in rows {
                    if let Some(v) = get_col_val(r, columns, col) {
                        let s = v.to_string();
                        if !seen.contains(&s) {
                            seen.push(s);
                        }
                    }
                }
                ColumnValue::Int(seen.len() as i64)
            }
        }
    }
}

/// Group-by aggregation operator.
pub struct GroupBy<'a> {
    table: &'a InMemoryTable,
    group_cols: Vec<String>,
    agg_funcs: Vec<AggFunc>,
}

impl<'a> GroupBy<'a> {
    /// Create a group-by builder.
    pub fn new(table: &'a InMemoryTable, group_cols: Vec<String>) -> Self {
        Self {
            table,
            group_cols,
            agg_funcs: Vec::new(),
        }
    }

    /// Add an aggregation.
    pub fn agg(mut self, func: AggFunc) -> Self {
        self.agg_funcs.push(func);
        self
    }

    /// Execute the group-by and return an aggregated table.
    pub fn apply(&self) -> Result<InMemoryTable> {
        // Validate group columns exist
        let group_indices: Vec<usize> = self
            .group_cols
            .iter()
            .map(|col| {
                self.table.column_index(col).ok_or_else(|| {
                    IoError::ValidationError(format!("Group-by column '{col}' not found"))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Group rows by key
        let mut groups: HashMap<Vec<String>, Vec<&Vec<ColumnValue>>> = HashMap::new();
        for row in &self.table.rows {
            let key: Vec<String> = group_indices.iter().map(|&i| row[i].to_string()).collect();
            groups.entry(key).or_default().push(row);
        }

        // Build output schema
        let mut out_columns: Vec<ColumnSchema> = self
            .group_cols
            .iter()
            .map(|name| {
                let idx = self.table.column_index(name).unwrap_or(0);
                ColumnSchema {
                    name: name.clone(),
                    col_type: self.table.columns[idx].col_type.clone(),
                }
            })
            .collect();
        for agg in &self.agg_funcs {
            out_columns.push(ColumnSchema {
                name: agg.output_name(),
                col_type: ColumnType::Float64,
            });
        }

        // Compute results
        let mut out_rows: Vec<Vec<ColumnValue>> = Vec::new();
        // Sort groups for determinism
        let mut keys: Vec<Vec<String>> = groups.keys().cloned().collect();
        keys.sort();

        for key in &keys {
            let group_rows = &groups[key];
            // Get representative row for group key values
            let first_row = group_rows[0];
            let mut out_row: Vec<ColumnValue> = group_indices
                .iter()
                .map(|&i| first_row[i].clone())
                .collect();
            for agg in &self.agg_funcs {
                out_row.push(agg.compute(group_rows, &self.table.columns));
            }
            out_rows.push(out_row);
        }

        Ok(InMemoryTable {
            columns: out_columns,
            rows: out_rows,
            name: None,
        })
    }
}

// ─── TableJoin ───────────────────────────────────────────────────────────────

/// Join type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join (only matching rows)
    Inner,
    /// Left outer join (all left rows, NULLs for unmatched right)
    Left,
    /// Right outer join (all right rows, NULLs for unmatched left)
    Right,
    /// Cross join (cartesian product)
    Cross,
}

/// Table join operations.
pub struct TableJoin;

impl TableJoin {
    /// Hash join two tables on the given key columns.
    pub fn hash_join(
        left: &InMemoryTable,
        right: &InMemoryTable,
        left_key: &str,
        right_key: &str,
        join_type: JoinType,
    ) -> Result<InMemoryTable> {
        let left_key_idx = left.column_index(left_key).ok_or_else(|| {
            IoError::ValidationError(format!("Left join key '{left_key}' not found"))
        })?;
        let right_key_idx = right.column_index(right_key).ok_or_else(|| {
            IoError::ValidationError(format!("Right join key '{right_key}' not found"))
        })?;

        // Build output schema: left columns + right columns (excluding join key)
        let mut out_columns: Vec<ColumnSchema> = left.columns.clone();
        for (i, col) in right.columns.iter().enumerate() {
            if i != right_key_idx {
                // Avoid duplicate name
                let name = if left.column_index(&col.name).is_some() {
                    format!("right_{}", col.name)
                } else {
                    col.name.clone()
                };
                out_columns.push(ColumnSchema {
                    name,
                    col_type: col.col_type.clone(),
                });
            }
        }

        let null_left: Vec<ColumnValue> = left.columns.iter().map(|_| ColumnValue::Null).collect();
        let null_right_no_key: Vec<ColumnValue> = right
            .columns
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != right_key_idx)
            .map(|_| ColumnValue::Null)
            .collect();

        match join_type {
            JoinType::Cross => {
                let mut rows = Vec::new();
                for l_row in &left.rows {
                    for r_row in &right.rows {
                        let mut out_row = l_row.clone();
                        for (i, v) in r_row.iter().enumerate() {
                            if i != right_key_idx {
                                out_row.push(v.clone());
                            }
                        }
                        rows.push(out_row);
                    }
                }
                return Ok(InMemoryTable {
                    columns: out_columns,
                    rows,
                    name: None,
                });
            }
            JoinType::Inner | JoinType::Left | JoinType::Right => {}
        }

        // Build hash map from right key → rows
        let mut right_map: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, r_row) in right.rows.iter().enumerate() {
            let key = r_row[right_key_idx].to_string();
            right_map.entry(key).or_default().push(i);
        }

        let mut rows: Vec<Vec<ColumnValue>> = Vec::new();
        let mut right_matched: Vec<bool> = vec![false; right.rows.len()];

        for l_row in &left.rows {
            let key = l_row[left_key_idx].to_string();
            match right_map.get(&key) {
                Some(r_indices) => {
                    for &ri in r_indices {
                        right_matched[ri] = true;
                        let mut out_row = l_row.clone();
                        for (i, v) in right.rows[ri].iter().enumerate() {
                            if i != right_key_idx {
                                out_row.push(v.clone());
                            }
                        }
                        rows.push(out_row);
                    }
                }
                None => {
                    if join_type == JoinType::Left {
                        let mut out_row = l_row.clone();
                        out_row.extend(null_right_no_key.iter().cloned());
                        rows.push(out_row);
                    }
                }
            }
        }

        // Right outer join: include unmatched right rows
        if join_type == JoinType::Right {
            for (i, r_row) in right.rows.iter().enumerate() {
                if !right_matched[i] {
                    let mut out_row = null_left.clone();
                    for (j, v) in r_row.iter().enumerate() {
                        if j != right_key_idx {
                            out_row.push(v.clone());
                        }
                    }
                    rows.push(out_row);
                }
            }
        }

        Ok(InMemoryTable {
            columns: out_columns,
            rows,
            name: None,
        })
    }

    /// Merge join two pre-sorted tables on the given key columns.
    /// Both tables must already be sorted ascending by their key columns.
    pub fn merge_join(
        left: &InMemoryTable,
        right: &InMemoryTable,
        left_key: &str,
        right_key: &str,
    ) -> Result<InMemoryTable> {
        let left_key_idx = left.column_index(left_key).ok_or_else(|| {
            IoError::ValidationError(format!("Left merge key '{left_key}' not found"))
        })?;
        let right_key_idx = right.column_index(right_key).ok_or_else(|| {
            IoError::ValidationError(format!("Right merge key '{right_key}' not found"))
        })?;

        let mut out_columns: Vec<ColumnSchema> = left.columns.clone();
        for (i, col) in right.columns.iter().enumerate() {
            if i != right_key_idx {
                let name = if left.column_index(&col.name).is_some() {
                    format!("right_{}", col.name)
                } else {
                    col.name.clone()
                };
                out_columns.push(ColumnSchema {
                    name,
                    col_type: col.col_type.clone(),
                });
            }
        }

        let mut rows = Vec::new();
        let mut li = 0usize;
        let mut ri = 0usize;

        while li < left.rows.len() && ri < right.rows.len() {
            let lk = &left.rows[li][left_key_idx];
            let rk = &right.rows[ri][right_key_idx];

            match lk.partial_cmp_value(rk).unwrap_or(Ordering::Equal) {
                Ordering::Equal => {
                    // Collect all matching right rows
                    let mut rj = ri;
                    while rj < right.rows.len() {
                        let rk2 = &right.rows[rj][right_key_idx];
                        if lk.partial_cmp_value(rk2) != Some(Ordering::Equal) {
                            break;
                        }
                        let mut out_row = left.rows[li].clone();
                        for (k, v) in right.rows[rj].iter().enumerate() {
                            if k != right_key_idx {
                                out_row.push(v.clone());
                            }
                        }
                        rows.push(out_row);
                        rj += 1;
                    }
                    li += 1;
                }
                Ordering::Less => li += 1,
                Ordering::Greater => ri += 1,
            }
        }

        Ok(InMemoryTable {
            columns: out_columns,
            rows,
            name: None,
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_table() -> InMemoryTable {
        let mut t = InMemoryTable::new(vec![
            ("id".to_string(), ColumnType::Int64),
            ("name".to_string(), ColumnType::Utf8),
            ("score".to_string(), ColumnType::Float64),
            ("dept".to_string(), ColumnType::Utf8),
        ]);
        t.push_row(&[
            ColumnValue::Int(1),
            ColumnValue::Utf8("Alice".to_string()),
            ColumnValue::Float(95.0),
            ColumnValue::Utf8("eng".to_string()),
        ])
        .unwrap();
        t.push_row(&[
            ColumnValue::Int(2),
            ColumnValue::Utf8("Bob".to_string()),
            ColumnValue::Float(82.5),
            ColumnValue::Utf8("eng".to_string()),
        ])
        .unwrap();
        t.push_row(&[
            ColumnValue::Int(3),
            ColumnValue::Utf8("Carol".to_string()),
            ColumnValue::Float(91.0),
            ColumnValue::Utf8("hr".to_string()),
        ])
        .unwrap();
        t.push_row(&[
            ColumnValue::Int(4),
            ColumnValue::Utf8("Dave".to_string()),
            ColumnValue::Float(78.0),
            ColumnValue::Utf8("hr".to_string()),
        ])
        .unwrap();
        t
    }

    #[test]
    fn test_filter_greater() {
        let t = make_table();
        let filtered = TableFilter::new(&t)
            .predicate(Predicate::Greater(
                "score".to_string(),
                ColumnValue::Float(90.0),
            ))
            .apply()
            .unwrap();
        assert_eq!(filtered.row_count(), 2); // Alice(95) and Carol(91)
    }

    #[test]
    fn test_filter_eq() {
        let t = make_table();
        let filtered = TableFilter::new(&t)
            .predicate(Predicate::Eq(
                "dept".to_string(),
                ColumnValue::Utf8("eng".to_string()),
            ))
            .apply()
            .unwrap();
        assert_eq!(filtered.row_count(), 2);
    }

    #[test]
    fn test_sort_asc() {
        let t = make_table();
        let sorted = TableSort::sort(&t, &[SortKey::asc("score")]).unwrap();
        let scores: Vec<f64> = sorted
            .get_column("score")
            .unwrap()
            .into_iter()
            .filter_map(|v| v.as_f64())
            .collect();
        assert_eq!(scores, vec![78.0, 82.5, 91.0, 95.0]);
    }

    #[test]
    fn test_sort_desc() {
        let t = make_table();
        let sorted = TableSort::sort(&t, &[SortKey::desc("score")]).unwrap();
        let scores: Vec<f64> = sorted
            .get_column("score")
            .unwrap()
            .into_iter()
            .filter_map(|v| v.as_f64())
            .collect();
        assert_eq!(scores, vec![95.0, 91.0, 82.5, 78.0]);
    }

    #[test]
    fn test_projection() {
        let t = make_table();
        let projected = TableProjection::new(&t)
            .column("id")
            .column_as("name", "full_name")
            .apply()
            .unwrap();
        assert_eq!(projected.column_count(), 2);
        assert_eq!(projected.columns[1].name, "full_name");
        assert_eq!(projected.row_count(), 4);
    }

    #[test]
    fn test_group_by_sum_mean() {
        let t = make_table();
        let grouped = GroupBy::new(&t, vec!["dept".to_string()])
            .agg(AggFunc::Count)
            .agg(AggFunc::Sum("score".to_string()))
            .agg(AggFunc::Mean("score".to_string()))
            .apply()
            .unwrap();

        assert_eq!(grouped.row_count(), 2); // eng, hr
                                            // eng has Alice(95) + Bob(82.5) = 177.5
        let eng_row = grouped
            .rows
            .iter()
            .find(|r| r[0] == ColumnValue::Utf8("eng".to_string()))
            .expect("eng group missing");
        assert_eq!(eng_row[1], ColumnValue::Int(2)); // count
        if let ColumnValue::Float(sum) = eng_row[2] {
            assert!((sum - 177.5).abs() < 1e-9);
        } else {
            panic!("Expected float sum");
        }
    }

    #[test]
    fn test_inner_join() {
        let mut left = InMemoryTable::new(vec![
            ("id".to_string(), ColumnType::Int64),
            ("val".to_string(), ColumnType::Float64),
        ]);
        left.push_row(&[ColumnValue::Int(1), ColumnValue::Float(1.0)])
            .unwrap();
        left.push_row(&[ColumnValue::Int(2), ColumnValue::Float(2.0)])
            .unwrap();
        left.push_row(&[ColumnValue::Int(3), ColumnValue::Float(3.0)])
            .unwrap();

        let mut right = InMemoryTable::new(vec![
            ("id".to_string(), ColumnType::Int64),
            ("label".to_string(), ColumnType::Utf8),
        ]);
        right
            .push_row(&[ColumnValue::Int(1), ColumnValue::Utf8("one".to_string())])
            .unwrap();
        right
            .push_row(&[ColumnValue::Int(2), ColumnValue::Utf8("two".to_string())])
            .unwrap();

        let joined = TableJoin::hash_join(&left, &right, "id", "id", JoinType::Inner).unwrap();
        assert_eq!(joined.row_count(), 2);
    }

    #[test]
    fn test_left_join() {
        let mut left = InMemoryTable::new(vec![("id".to_string(), ColumnType::Int64)]);
        left.push_row(&[ColumnValue::Int(1)]).unwrap();
        left.push_row(&[ColumnValue::Int(2)]).unwrap();
        left.push_row(&[ColumnValue::Int(3)]).unwrap(); // no match in right

        let mut right = InMemoryTable::new(vec![
            ("id".to_string(), ColumnType::Int64),
            ("x".to_string(), ColumnType::Float64),
        ]);
        right
            .push_row(&[ColumnValue::Int(1), ColumnValue::Float(10.0)])
            .unwrap();
        right
            .push_row(&[ColumnValue::Int(2), ColumnValue::Float(20.0)])
            .unwrap();

        let joined = TableJoin::hash_join(&left, &right, "id", "id", JoinType::Left).unwrap();
        assert_eq!(joined.row_count(), 3);
        // Row for id=3 should have NULL for 'x'
        let row3 = joined
            .rows
            .iter()
            .find(|r| r[0] == ColumnValue::Int(3))
            .expect("row 3 missing");
        assert_eq!(row3[1], ColumnValue::Null);
    }

    #[test]
    fn test_cross_join() {
        let mut a = InMemoryTable::new(vec![("a".to_string(), ColumnType::Int64)]);
        a.push_row(&[ColumnValue::Int(1)]).unwrap();
        a.push_row(&[ColumnValue::Int(2)]).unwrap();

        let mut b = InMemoryTable::new(vec![("b".to_string(), ColumnType::Int64)]);
        b.push_row(&[ColumnValue::Int(10)]).unwrap();
        b.push_row(&[ColumnValue::Int(20)]).unwrap();
        b.push_row(&[ColumnValue::Int(30)]).unwrap();

        let crossed = TableJoin::hash_join(&a, &b, "a", "b", JoinType::Cross).unwrap();
        assert_eq!(crossed.row_count(), 6); // 2 × 3
    }

    #[test]
    fn test_like_match() {
        assert!(like_match("hello world", "%world"));
        assert!(like_match("hello world", "hello%"));
        assert!(like_match("hello world", "%lo w%"));
        assert!(!like_match("hello world", "xyz%"));
        assert!(like_match("abc", "a_c"));
        assert!(!like_match("axyz", "a_c"));
    }
}
