//! SQL-like query interface for data files.
//!
//! Provides a builder-pattern query engine that can operate over in-memory rows
//! or CSV-backed data sources without any external SQL engine dependency.
//!
//! # Example
//!
//! ```rust
//! use scirs2_io::query::{DataQuery, DataSource, ColumnValue};
//!
//! // Build rows manually
//! let rows = vec![
//!     vec![("name".to_string(), ColumnValue::Text("Alice".to_string())),
//!          ("age".to_string(),  ColumnValue::Integer(30))],
//!     vec![("name".to_string(), ColumnValue::Text("Bob".to_string())),
//!          ("age".to_string(),  ColumnValue::Integer(25))],
//! ];
//! let source = DataSource::InMemoryRows(rows);
//! let result = DataQuery::from(source)
//!     .select(&["name", "age"])
//!     .order_by("age", true)
//!     .execute()
//!     .expect("query failed");
//!
//! assert_eq!(result.n_rows, 2);
//! ```

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::{IoError, Result};

// ──────────────────────────────────────────────────────────────────────────────
// Column value type
// ──────────────────────────────────────────────────────────────────────────────

/// A single typed cell value within a [`Row`].
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnValue {
    /// Null / missing value.
    Null,
    /// 64-bit signed integer.
    Integer(i64),
    /// 64-bit floating-point number.
    Float(f64),
    /// Boolean.
    Boolean(bool),
    /// UTF-8 text string.
    Text(String),
}

impl ColumnValue {
    /// Return a best-effort `f64` representation (for aggregations).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ColumnValue::Integer(i) => Some(*i as f64),
            ColumnValue::Float(f) => Some(*f),
            ColumnValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Return a sortable string key representation.
    fn sort_key(&self) -> String {
        match self {
            ColumnValue::Null => "\x00".to_string(),
            ColumnValue::Integer(i) => format!("{:020}", i),
            ColumnValue::Float(f) => format!("{:030.15}", f),
            ColumnValue::Boolean(b) => if *b { "1" } else { "0" }.to_string(),
            ColumnValue::Text(s) => s.clone(),
        }
    }
}

impl std::fmt::Display for ColumnValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColumnValue::Null => write!(f, "NULL"),
            ColumnValue::Integer(i) => write!(f, "{}", i),
            ColumnValue::Float(v) => write!(f, "{}", v),
            ColumnValue::Boolean(b) => write!(f, "{}", b),
            ColumnValue::Text(s) => write!(f, "{}", s),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Row
// ──────────────────────────────────────────────────────────────────────────────

/// A single data row with named columns.
///
/// Columns are stored in insertion order; lookup by name performs a linear scan
/// which is fast for the typical column counts in tabular data.
#[derive(Debug, Clone)]
pub struct Row {
    /// Column names in order.
    pub columns: Vec<String>,
    /// Values aligned with `columns`.
    pub values: Vec<ColumnValue>,
}

impl Row {
    /// Construct a row from a `Vec<(name, value)>` list.
    pub fn from_pairs(pairs: Vec<(String, ColumnValue)>) -> Self {
        let mut columns = Vec::with_capacity(pairs.len());
        let mut values = Vec::with_capacity(pairs.len());
        for (k, v) in pairs {
            columns.push(k);
            values.push(v);
        }
        Row { columns, values }
    }

    /// Look up a value by column name. Returns `None` when the column is absent.
    pub fn get(&self, column: &str) -> Option<&ColumnValue> {
        self.columns
            .iter()
            .position(|c| c == column)
            .map(|idx| &self.values[idx])
    }

    /// Returns `true` when the row contains no columns.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Number of columns in this row.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Return an iterator over `(column_name, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ColumnValue)> {
        self.columns
            .iter()
            .map(|s| s.as_str())
            .zip(self.values.iter())
    }

    /// Project to a subset of columns (in the given order).  Missing columns
    /// produce `ColumnValue::Null`.
    fn project(&self, cols: &[String]) -> Row {
        let values = cols
            .iter()
            .map(|c| self.get(c).cloned().unwrap_or(ColumnValue::Null))
            .collect();
        Row {
            columns: cols.to_vec(),
            values,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Data source
// ──────────────────────────────────────────────────────────────────────────────

/// Backing data for a [`DataQuery`].
pub enum DataSource {
    /// A CSV file path.  The first line is treated as the header row.
    CsvFile(String),
    /// Rows already loaded into memory as `(column_name, value)` pairs.
    InMemoryRows(Vec<Vec<(String, ColumnValue)>>),
    /// An already-constructed `Vec<Row>`.
    Rows(Vec<Row>),
}

// ──────────────────────────────────────────────────────────────────────────────
// Query result
// ──────────────────────────────────────────────────────────────────────────────

/// The result of executing a [`DataQuery`].
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// All matching rows after projection, filtering, ordering and limiting.
    pub rows: Vec<Row>,
    /// Column names present in the result rows (in projection order).
    pub columns: Vec<String>,
    /// Total number of result rows (equals `rows.len()`).
    pub n_rows: usize,
}

impl QueryResult {
    /// Iterate over the rows.
    pub fn iter(&self) -> impl Iterator<Item = &Row> {
        self.rows.iter()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Group-by result
// ──────────────────────────────────────────────────────────────────────────────

/// A set of rows grouped by the value of one column.
#[derive(Debug, Clone)]
pub struct GroupedResult {
    /// Group key → list of rows in that group.
    pub groups: HashMap<String, Vec<Row>>,
    /// The column that was used as the grouping key.
    pub group_column: String,
}

impl GroupedResult {
    /// Count the number of rows in each group.
    pub fn count(&self) -> HashMap<String, usize> {
        self.groups
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect()
    }

    /// Sum of a numeric column for each group.
    pub fn sum(&self, column: &str) -> HashMap<String, f64> {
        self.groups
            .iter()
            .map(|(k, rows)| {
                let s = rows
                    .iter()
                    .filter_map(|r| r.get(column).and_then(|v| v.as_f64()))
                    .sum();
                (k.clone(), s)
            })
            .collect()
    }

    /// Arithmetic mean of a numeric column for each group.
    pub fn mean(&self, column: &str) -> HashMap<String, f64> {
        self.groups
            .iter()
            .map(|(k, rows)| {
                let vals: Vec<f64> = rows
                    .iter()
                    .filter_map(|r| r.get(column).and_then(|v| v.as_f64()))
                    .collect();
                let mean = if vals.is_empty() {
                    0.0
                } else {
                    vals.iter().sum::<f64>() / vals.len() as f64
                };
                (k.clone(), mean)
            })
            .collect()
    }

    /// Minimum value of a numeric column for each group.
    pub fn min(&self, column: &str) -> HashMap<String, f64> {
        self.groups
            .iter()
            .map(|(k, rows)| {
                let min = rows
                    .iter()
                    .filter_map(|r| r.get(column).and_then(|v| v.as_f64()))
                    .fold(f64::INFINITY, f64::min);
                (k.clone(), min)
            })
            .collect()
    }

    /// Maximum value of a numeric column for each group.
    pub fn max(&self, column: &str) -> HashMap<String, f64> {
        self.groups
            .iter()
            .map(|(k, rows)| {
                let max = rows
                    .iter()
                    .filter_map(|r| r.get(column).and_then(|v| v.as_f64()))
                    .fold(f64::NEG_INFINITY, f64::max);
                (k.clone(), max)
            })
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Aggregate helper (global, not grouped)
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregation functions over all rows in a [`QueryResult`].
pub struct Aggregations<'a> {
    result: &'a QueryResult,
}

impl<'a> Aggregations<'a> {
    /// Number of rows.
    pub fn count(&self) -> usize {
        self.result.n_rows
    }

    /// Sum of a numeric column across all rows.
    pub fn sum(&self, column: &str) -> f64 {
        self.result
            .rows
            .iter()
            .filter_map(|r| r.get(column).and_then(|v| v.as_f64()))
            .sum()
    }

    /// Arithmetic mean of a numeric column across all rows.
    pub fn mean(&self, column: &str) -> f64 {
        let vals: Vec<f64> = self
            .result
            .rows
            .iter()
            .filter_map(|r| r.get(column).and_then(|v| v.as_f64()))
            .collect();
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    }

    /// Minimum of a numeric column across all rows.
    pub fn min(&self, column: &str) -> f64 {
        self.result
            .rows
            .iter()
            .filter_map(|r| r.get(column).and_then(|v| v.as_f64()))
            .fold(f64::INFINITY, f64::min)
    }

    /// Maximum of a numeric column across all rows.
    pub fn max(&self, column: &str) -> f64 {
        self.result
            .rows
            .iter()
            .filter_map(|r| r.get(column).and_then(|v| v.as_f64()))
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

impl QueryResult {
    /// Return an [`Aggregations`] helper bound to this result.
    pub fn agg(&self) -> Aggregations<'_> {
        Aggregations { result: self }
    }

    /// Group rows by the string representation of the given column's values.
    pub fn group_by(self, column: &str) -> GroupedResult {
        let mut groups: HashMap<String, Vec<Row>> = HashMap::new();
        let col_owned = column.to_string();
        for row in self.rows {
            let key = row
                .get(column)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "NULL".to_string());
            groups.entry(key).or_default().push(row);
        }
        GroupedResult {
            groups,
            group_column: col_owned,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DataQuery builder
// ──────────────────────────────────────────────────────────────────────────────

/// Order-by specification.
struct OrderSpec {
    column: String,
    ascending: bool,
}

/// A lazy query builder.
///
/// Build the query using the fluent API, then call [`execute`] to materialise
/// the results.
///
/// [`execute`]: DataQuery::execute
pub struct DataQuery {
    source: DataSource,
    select_cols: Option<Vec<String>>,
    predicates: Vec<Box<dyn Fn(&Row) -> bool + Send + Sync>>,
    order: Option<OrderSpec>,
    limit: Option<usize>,
}

impl DataQuery {
    /// Create a query from a [`DataSource`].
    pub fn from(source: DataSource) -> Self {
        DataQuery {
            source,
            select_cols: None,
            predicates: Vec::new(),
            order: None,
            limit: None,
        }
    }

    /// Restrict the projected columns.  Pass an empty slice to select all.
    pub fn select(mut self, columns: &[&str]) -> Self {
        if !columns.is_empty() {
            self.select_cols = Some(columns.iter().map(|s| s.to_string()).collect());
        }
        self
    }

    /// Add a row-level filter predicate.  Multiple calls are ANDed together.
    pub fn filter(mut self, predicate: impl Fn(&Row) -> bool + Send + Sync + 'static) -> Self {
        self.predicates.push(Box::new(predicate));
        self
    }

    /// Limit the number of result rows.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Sort the result by a column.  `ascending = true` → smallest first.
    pub fn order_by(mut self, column: &str, ascending: bool) -> Self {
        self.order = Some(OrderSpec {
            column: column.to_string(),
            ascending,
        });
        self
    }

    /// Execute the query and return the materialised result.
    pub fn execute(self) -> Result<QueryResult> {
        let DataQuery {
            source,
            select_cols,
            predicates,
            order,
            limit,
        } = self;

        // 1. Load rows from the source
        let mut rows = load_rows(source)?;

        // 2. Apply predicates
        rows.retain(|row| predicates.iter().all(|p| p(row)));

        // 3. Sort
        if let Some(ord) = order {
            let col = ord.column.clone();
            let asc = ord.ascending;
            rows.sort_by(|a, b| {
                let ka = a.get(&col).map(|v| v.sort_key()).unwrap_or_default();
                let kb = b.get(&col).map(|v| v.sort_key()).unwrap_or_default();
                if asc {
                    ka.cmp(&kb)
                } else {
                    kb.cmp(&ka)
                }
            });
        }

        // 4. Limit
        if let Some(n) = limit {
            rows.truncate(n);
        }

        // 5. Projection
        let columns: Vec<String> = match &select_cols {
            Some(cols) => cols.clone(),
            None => {
                // Collect union of all column names preserving first-seen order
                let mut seen = std::collections::LinkedList::new();
                let mut seen_set = std::collections::HashSet::new();
                for row in &rows {
                    for col in &row.columns {
                        if seen_set.insert(col.clone()) {
                            seen.push_back(col.clone());
                        }
                    }
                }
                seen.into_iter().collect()
            }
        };

        let projected: Vec<Row> = rows.iter().map(|r| r.project(&columns)).collect();
        let n_rows = projected.len();

        Ok(QueryResult {
            rows: projected,
            columns,
            n_rows,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Parse a CSV cell string into a [`ColumnValue`].
fn parse_cell(s: &str) -> ColumnValue {
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") || trimmed == "NA" {
        return ColumnValue::Null;
    }
    if let Ok(i) = trimmed.parse::<i64>() {
        return ColumnValue::Integer(i);
    }
    if let Ok(f) = trimmed.parse::<f64>() {
        return ColumnValue::Float(f);
    }
    if trimmed.eq_ignore_ascii_case("true") {
        return ColumnValue::Boolean(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return ColumnValue::Boolean(false);
    }
    ColumnValue::Text(trimmed.to_string())
}

/// Split a CSV line respecting double-quoted fields.
fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if in_quotes {
                    // Check for escaped quote ("")
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        cur.push('"');
                    } else {
                        in_quotes = false;
                    }
                } else {
                    in_quotes = true;
                }
            }
            ',' if !in_quotes => {
                fields.push(cur.clone());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    fields.push(cur);
    fields
}

/// Load all rows from the given [`DataSource`].
fn load_rows(source: DataSource) -> Result<Vec<Row>> {
    match source {
        DataSource::Rows(rows) => Ok(rows),

        DataSource::InMemoryRows(pairs_list) => {
            Ok(pairs_list.into_iter().map(Row::from_pairs).collect())
        }

        DataSource::CsvFile(path) => {
            let file = std::fs::File::open(Path::new(&path))
                .map_err(|e| IoError::FileError(format!("cannot open '{}': {}", path, e)))?;
            let reader = BufReader::new(file);
            let mut lines = reader.lines();

            // First line → headers
            let header_line = lines
                .next()
                .ok_or_else(|| IoError::FormatError("CSV file is empty".to_string()))?
                .map_err(|e| IoError::Io(e))?;
            let headers: Vec<String> = split_csv_line(&header_line)
                .into_iter()
                .map(|s| s.trim().to_string())
                .collect();

            let mut rows = Vec::new();
            for line_result in lines {
                let line = line_result.map_err(|e| IoError::Io(e))?;
                if line.trim().is_empty() {
                    continue;
                }
                let cells = split_csv_line(&line);
                let pairs: Vec<(String, ColumnValue)> = headers
                    .iter()
                    .enumerate()
                    .map(|(i, h)| {
                        let val = cells
                            .get(i)
                            .map(|s| parse_cell(s))
                            .unwrap_or(ColumnValue::Null);
                        (h.clone(), val)
                    })
                    .collect();
                rows.push(Row::from_pairs(pairs));
            }
            Ok(rows)
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// execute() free function
// ──────────────────────────────────────────────────────────────────────────────

/// Execute a [`DataQuery`] and return the [`QueryResult`].
///
/// This is a convenience wrapper identical to calling [`DataQuery::execute`].
pub fn execute(query: DataQuery) -> Result<QueryResult> {
    query.execute()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rows() -> Vec<Row> {
        vec![
            Row::from_pairs(vec![
                ("name".to_string(), ColumnValue::Text("Alice".to_string())),
                ("age".to_string(), ColumnValue::Integer(30)),
                ("score".to_string(), ColumnValue::Float(95.5)),
                ("dept".to_string(), ColumnValue::Text("eng".to_string())),
            ]),
            Row::from_pairs(vec![
                ("name".to_string(), ColumnValue::Text("Bob".to_string())),
                ("age".to_string(), ColumnValue::Integer(25)),
                ("score".to_string(), ColumnValue::Float(80.0)),
                ("dept".to_string(), ColumnValue::Text("hr".to_string())),
            ]),
            Row::from_pairs(vec![
                ("name".to_string(), ColumnValue::Text("Carol".to_string())),
                ("age".to_string(), ColumnValue::Integer(35)),
                ("score".to_string(), ColumnValue::Float(88.0)),
                ("dept".to_string(), ColumnValue::Text("eng".to_string())),
            ]),
        ]
    }

    #[test]
    fn test_select_and_count() {
        let result = DataQuery::from(DataSource::Rows(sample_rows()))
            .select(&["name", "age"])
            .execute()
            .expect("execute");
        assert_eq!(result.n_rows, 3);
        assert_eq!(result.columns, vec!["name", "age"]);
        // score should be absent
        assert!(result.rows[0]
            .get("score")
            .map(|v| matches!(v, ColumnValue::Null))
            .unwrap_or(true));
    }

    #[test]
    fn test_filter() {
        let result = DataQuery::from(DataSource::Rows(sample_rows()))
            .filter(|row| matches!(row.get("age"), Some(ColumnValue::Integer(a)) if *a > 28))
            .execute()
            .expect("execute");
        assert_eq!(result.n_rows, 2); // Alice (30) and Carol (35)
    }

    #[test]
    fn test_order_by_ascending() {
        let result = DataQuery::from(DataSource::Rows(sample_rows()))
            .order_by("age", true)
            .execute()
            .expect("execute");
        let ages: Vec<i64> = result
            .rows
            .iter()
            .filter_map(|r| {
                if let Some(ColumnValue::Integer(a)) = r.get("age") {
                    Some(*a)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(ages, vec![25, 30, 35]);
    }

    #[test]
    fn test_order_by_descending() {
        let result = DataQuery::from(DataSource::Rows(sample_rows()))
            .order_by("age", false)
            .execute()
            .expect("execute");
        let ages: Vec<i64> = result
            .rows
            .iter()
            .filter_map(|r| {
                if let Some(ColumnValue::Integer(a)) = r.get("age") {
                    Some(*a)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(ages, vec![35, 30, 25]);
    }

    #[test]
    fn test_limit() {
        let result = DataQuery::from(DataSource::Rows(sample_rows()))
            .limit(2)
            .execute()
            .expect("execute");
        assert_eq!(result.n_rows, 2);
    }

    #[test]
    fn test_aggregations() {
        let result = DataQuery::from(DataSource::Rows(sample_rows()))
            .execute()
            .expect("execute");
        let agg = result.agg();
        assert_eq!(agg.count(), 3);
        assert!((agg.sum("age") - 90.0).abs() < 1e-9);
        assert!((agg.mean("age") - 30.0).abs() < 1e-9);
        assert!((agg.min("age") - 25.0).abs() < 1e-9);
        assert!((agg.max("age") - 35.0).abs() < 1e-9);
    }

    #[test]
    fn test_group_by() {
        let result = DataQuery::from(DataSource::Rows(sample_rows()))
            .execute()
            .expect("execute");
        let grouped = result.group_by("dept");
        let counts = grouped.count();
        assert_eq!(*counts.get("eng").unwrap_or(&0), 2);
        assert_eq!(*counts.get("hr").unwrap_or(&0), 1);
    }

    #[test]
    fn test_group_by_sum() {
        let result = DataQuery::from(DataSource::Rows(sample_rows()))
            .execute()
            .expect("execute");
        let grouped = result.group_by("dept");
        let sums = grouped.sum("age");
        // eng: Alice(30) + Carol(35) = 65
        assert!((sums.get("eng").copied().unwrap_or(0.0) - 65.0).abs() < 1e-9);
    }

    #[test]
    fn test_csv_round_trip() {
        use std::io::Write;

        let dir = std::env::temp_dir();
        let path = dir.join("test_query_csv.csv");
        {
            let mut f = std::fs::File::create(&path).expect("create csv");
            writeln!(f, "id,val").expect("write header");
            writeln!(f, "1,10.5").expect("write row 1");
            writeln!(f, "2,20.0").expect("write row 2");
        }

        let result = DataQuery::from(DataSource::CsvFile(
            path.to_str().expect("path").to_string(),
        ))
        .execute()
        .expect("execute");

        assert_eq!(result.n_rows, 2);
        assert!((result.agg().sum("val") - 30.5).abs() < 1e-9);

        // cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_in_memory_rows_source() {
        let pairs = vec![
            vec![
                ("x".to_string(), ColumnValue::Integer(1)),
                ("y".to_string(), ColumnValue::Float(1.1)),
            ],
            vec![
                ("x".to_string(), ColumnValue::Integer(2)),
                ("y".to_string(), ColumnValue::Float(2.2)),
            ],
        ];
        let result = DataQuery::from(DataSource::InMemoryRows(pairs))
            .execute()
            .expect("execute");
        assert_eq!(result.n_rows, 2);
    }

    #[test]
    fn test_combined_filter_order_limit() {
        let result = DataQuery::from(DataSource::Rows(sample_rows()))
            .filter(|row| matches!(row.get("dept"), Some(ColumnValue::Text(d)) if d == "eng"))
            .order_by("age", true)
            .limit(1)
            .execute()
            .expect("execute");
        // eng: Alice(30), Carol(35) → sorted asc → Alice → limit 1
        assert_eq!(result.n_rows, 1);
        assert_eq!(
            result.rows[0].get("name"),
            Some(&ColumnValue::Text("Alice".to_string()))
        );
    }
}
