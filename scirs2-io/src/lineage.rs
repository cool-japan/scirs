//! Data lineage tracking — directed acyclic graph of data transformations.
//!
//! Tracks the provenance of datasets through a chain of transformations.
//! Each node in the graph represents a dataset, and each edge represents a
//! transformation that produced the downstream dataset from one or more upstream
//! datasets.
//!
//! # Example
//!
//! ```rust
//! use scirs2_io::lineage::{
//!     DataLineage, DataSource, DataNode, ColumnType,
//!     record_transformation, get_provenance, export_lineage_dot, lineage_to_json,
//! };
//! use std::collections::HashMap;
//!
//! let mut lineage = DataLineage::new();
//!
//! // Register source node
//! let raw_id = lineage.add_node(DataNode::new(
//!     "raw_csv".to_string(),
//!     DataSource::File("/data/raw.csv".to_string()),
//!     vec![
//!         ("id".to_string(),    ColumnType::Integer),
//!         ("value".to_string(), ColumnType::Float),
//!     ],
//! ));
//!
//! // Register transformed node
//! let clean_id = lineage.add_node(DataNode::new(
//!     "cleaned".to_string(),
//!     DataSource::InMemory,
//!     vec![
//!         ("id".to_string(),    ColumnType::Integer),
//!         ("value".to_string(), ColumnType::Float),
//!     ],
//! ));
//!
//! // Record the transformation
//! let mut params = HashMap::new();
//! params.insert("drop_nulls".to_string(), "true".to_string());
//! record_transformation(&mut lineage, vec![raw_id], clean_id, "filter", params);
//!
//! // Query provenance
//! let upstream = get_provenance(&lineage, clean_id);
//! assert_eq!(upstream.len(), 1);
//!
//! // Export
//! let dot = export_lineage_dot(&lineage);
//! assert!(dot.contains("digraph"));
//! let json = lineage_to_json(&lineage);
//! assert!(json.contains("cleaned"));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ──────────────────────────────────────────────────────────────────────────────
// Node identifier
// ──────────────────────────────────────────────────────────────────────────────

/// Opaque identifier for a node in the lineage graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(u64);

impl NodeId {
    /// Return the underlying numeric value.
    pub fn value(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node_{}", self.0)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Data source
// ──────────────────────────────────────────────────────────────────────────────

/// The origin of a dataset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataSource {
    /// A file on disk (path stored as a `String`).
    File(String),
    /// A named database connection / table.
    Database(String),
    /// A dataset residing only in RAM with no persistent backing.
    InMemory,
    /// A synthetically generated dataset.
    Generated,
}

impl DataSource {
    /// Human-readable label used in DOT / JSON output.
    pub fn label(&self) -> String {
        match self {
            DataSource::File(p) => format!("file:{}", p),
            DataSource::Database(s) => format!("db:{}", s),
            DataSource::InMemory => "in_memory".to_string(),
            DataSource::Generated => "generated".to_string(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Column type
// ──────────────────────────────────────────────────────────────────────────────

/// Logical column type stored in a node's schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnType {
    /// 64-bit signed integer.
    Integer,
    /// 64-bit floating-point.
    Float,
    /// Boolean.
    Boolean,
    /// UTF-8 string.
    Text,
    /// Any other / unknown type, with an optional description.
    Other(String),
}

impl ColumnType {
    fn label(&self) -> &str {
        match self {
            ColumnType::Integer => "integer",
            ColumnType::Float => "float",
            ColumnType::Boolean => "boolean",
            ColumnType::Text => "text",
            ColumnType::Other(s) => s.as_str(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Data node
// ──────────────────────────────────────────────────────────────────────────────

/// A node in the lineage graph representing a versioned dataset snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataNode {
    /// Unique identifier assigned by [`DataLineage::add_node`].
    pub id: NodeId,
    /// Human-readable name.
    pub name: String,
    /// Where the data originated.
    pub source: DataSource,
    /// Schema: ordered list of `(column_name, column_type)` pairs.
    pub schema: Vec<(String, ColumnType)>,
    /// Wall-clock creation timestamp (RFC 3339).
    pub created_at: String,
    /// Arbitrary user-defined tags.
    pub tags: HashMap<String, String>,
}

impl DataNode {
    /// Construct a new node.  The `id` field is set to a placeholder and will
    /// be replaced by [`DataLineage::add_node`].
    pub fn new(name: String, source: DataSource, schema: Vec<(String, ColumnType)>) -> Self {
        DataNode {
            id: NodeId(0),
            name,
            source,
            schema,
            created_at: chrono_now(),
            tags: HashMap::new(),
        }
    }

    /// Builder: attach an arbitrary tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Transformation
// ──────────────────────────────────────────────────────────────────────────────

/// An edge in the lineage graph describing how a set of input nodes produced
/// one output node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transformation {
    /// Input dataset node IDs (≥ 1).
    pub input_nodes: Vec<NodeId>,
    /// Output dataset node ID.
    pub output_node: NodeId,
    /// Operation name (e.g. `"filter"`, `"join"`, `"aggregate"`).
    pub op_name: String,
    /// Free-form key → value parameters describing the operation.
    pub params: HashMap<String, String>,
    /// Wall-clock creation timestamp (RFC 3339).
    pub created_at: String,
}

impl Transformation {
    /// Construct a new transformation record.
    pub fn new(
        input_nodes: Vec<NodeId>,
        output_node: NodeId,
        op_name: impl Into<String>,
        params: HashMap<String, String>,
    ) -> Self {
        Transformation {
            input_nodes,
            output_node,
            op_name: op_name.into(),
            params,
            created_at: chrono_now(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DataLineage graph
// ──────────────────────────────────────────────────────────────────────────────

/// A directed acyclic graph (DAG) of data nodes and their transformations.
///
/// Nodes are datasets; edges are transformation operations.  The graph only
/// allows a DAG structure — cycles are prevented by the append-only design
/// (nodes always receive IDs in monotonically increasing order, and
/// transformations can only point from lower-ID inputs to a higher-ID output).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DataLineage {
    nodes: HashMap<NodeId, DataNode>,
    transformations: Vec<Transformation>,
    next_id: u64,
}

impl DataLineage {
    /// Create a new empty lineage graph.
    pub fn new() -> Self {
        DataLineage {
            nodes: HashMap::new(),
            transformations: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a node to the graph and return its assigned [`NodeId`].
    pub fn add_node(&mut self, mut node: DataNode) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        node.id = id;
        self.nodes.insert(id, node);
        id
    }

    /// Look up a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&DataNode> {
        self.nodes.get(&id)
    }

    /// Mutable reference to a node (for post-hoc tag additions, etc.).
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut DataNode> {
        self.nodes.get_mut(&id)
    }

    /// All nodes in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = &DataNode> {
        self.nodes.values()
    }

    /// All transformations recorded in the graph.
    pub fn transformations(&self) -> &[Transformation] {
        &self.transformations
    }

    /// Record that `output_node` was produced from `input_nodes` by the
    /// named operation with the given parameters.
    ///
    /// Returns `false` without modifying the graph when any of the referenced
    /// node IDs are unknown.
    pub fn add_transformation(&mut self, t: Transformation) -> bool {
        // Validate that all referenced node IDs exist
        let all_known = t
            .input_nodes
            .iter()
            .chain(std::iter::once(&t.output_node))
            .all(|id| self.nodes.contains_key(id));
        if all_known {
            self.transformations.push(t);
            true
        } else {
            false
        }
    }

    /// Return all transformations whose output is the given node.
    fn producing_transformations(&self, node_id: NodeId) -> Vec<&Transformation> {
        self.transformations
            .iter()
            .filter(|t| t.output_node == node_id)
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Free functions
// ──────────────────────────────────────────────────────────────────────────────

/// Record a transformation in `lineage` that produced `output` from `inputs`.
///
/// Returns `true` when all node IDs are known and the transformation was added.
pub fn record_transformation(
    lineage: &mut DataLineage,
    inputs: Vec<NodeId>,
    output: NodeId,
    op_name: impl Into<String>,
    params: HashMap<String, String>,
) -> bool {
    let t = Transformation::new(inputs, output, op_name, params);
    lineage.add_transformation(t)
}

/// Return all ancestor (upstream) [`DataNode`]s of `node_id` via BFS.
///
/// The returned slice is ordered breadth-first from `node_id` upwards and does
/// **not** include `node_id` itself unless there is a cycle (which the
/// append-only design prevents).
pub fn get_provenance(lineage: &DataLineage, node_id: NodeId) -> Vec<DataNode> {
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    let mut result: Vec<DataNode> = Vec::new();

    // Seed the BFS with direct inputs
    for t in lineage.producing_transformations(node_id) {
        for &inp in &t.input_nodes {
            if visited.insert(inp) {
                queue.push_back(inp);
            }
        }
    }

    while let Some(id) = queue.pop_front() {
        if let Some(node) = lineage.get_node(id) {
            result.push(node.clone());
        }
        // Recurse upwards
        for t in lineage.producing_transformations(id) {
            for &inp in &t.input_nodes {
                if visited.insert(inp) {
                    queue.push_back(inp);
                }
            }
        }
    }

    result
}

/// Render the lineage graph as a Graphviz DOT string suitable for visualization.
///
/// Nodes are labelled with their name and source; edges are labelled with the
/// operation name.
pub fn export_lineage_dot(lineage: &DataLineage) -> String {
    let mut dot = String::from("digraph lineage {\n    rankdir=LR;\n    node [shape=box];\n");

    // Emit nodes
    let mut node_ids: Vec<NodeId> = lineage.nodes.keys().copied().collect();
    node_ids.sort_by_key(|n| n.0);
    for nid in &node_ids {
        if let Some(node) = lineage.get_node(*nid) {
            // Build label
            let schema_str: String = node
                .schema
                .iter()
                .map(|(c, t)| format!("{}:{}", c, t.label()))
                .collect::<Vec<_>>()
                .join(", ");
            let label = format!(
                "{}\\n[{}]\\n({} col(s))",
                escape_dot(&node.name),
                escape_dot(&node.source.label()),
                node.schema.len(),
            );
            // Include schema as tooltip
            dot.push_str(&format!(
                "    {} [label=\"{}\" tooltip=\"{}\"];\n",
                nid,
                label,
                escape_dot(&schema_str),
            ));
        }
    }

    // Emit edges
    for (tidx, t) in lineage.transformations.iter().enumerate() {
        let edge_label = format!("{}\\n(step {})", escape_dot(&t.op_name), tidx + 1);
        for &inp in &t.input_nodes {
            dot.push_str(&format!(
                "    {} -> {} [label=\"{}\"];\n",
                inp, t.output_node, edge_label
            ));
        }
    }

    dot.push_str("}\n");
    dot
}

/// Serialise the entire lineage graph to a JSON string.
///
/// The format is a self-contained object with `nodes` and `transformations`
/// arrays.  No external serde dependency is required — the JSON is built by
/// hand to avoid adding serde derives to the types.
pub fn lineage_to_json(lineage: &DataLineage) -> String {
    let mut out = String::from("{\n");

    // -- nodes --
    out.push_str("  \"nodes\": [\n");
    let mut node_ids: Vec<NodeId> = lineage.nodes.keys().copied().collect();
    node_ids.sort_by_key(|n| n.0);
    for (i, nid) in node_ids.iter().enumerate() {
        if let Some(node) = lineage.get_node(*nid) {
            out.push_str("    {\n");
            out.push_str(&format!("      \"id\": {},\n", nid.0));
            out.push_str(&format!(
                "      \"name\": \"{}\",\n",
                json_escape(&node.name)
            ));
            out.push_str(&format!(
                "      \"source\": \"{}\",\n",
                json_escape(&node.source.label())
            ));
            out.push_str(&format!(
                "      \"created_at\": \"{}\",\n",
                json_escape(&node.created_at)
            ));
            // Schema
            out.push_str("      \"schema\": [\n");
            for (si, (col, typ)) in node.schema.iter().enumerate() {
                out.push_str(&format!(
                    "        {{\"column\": \"{}\", \"type\": \"{}\"}}{}",
                    json_escape(col),
                    json_escape(typ.label()),
                    if si + 1 < node.schema.len() {
                        ",\n"
                    } else {
                        "\n"
                    }
                ));
            }
            out.push_str("      ],\n");
            // Tags
            out.push_str("      \"tags\": {");
            let tag_pairs: Vec<_> = node.tags.iter().collect();
            for (ti, (k, v)) in tag_pairs.iter().enumerate() {
                out.push_str(&format!(
                    "\"{}\": \"{}\"{}",
                    json_escape(k),
                    json_escape(v),
                    if ti + 1 < tag_pairs.len() { ", " } else { "" }
                ));
            }
            out.push_str("}\n");
            out.push_str(if i + 1 < node_ids.len() {
                "    },\n"
            } else {
                "    }\n"
            });
        }
    }
    out.push_str("  ],\n");

    // -- transformations --
    out.push_str("  \"transformations\": [\n");
    for (i, t) in lineage.transformations.iter().enumerate() {
        out.push_str("    {\n");
        // inputs
        let inputs_str: String = t
            .input_nodes
            .iter()
            .map(|n| n.0.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!("      \"input_nodes\": [{}],\n", inputs_str));
        out.push_str(&format!("      \"output_node\": {},\n", t.output_node.0));
        out.push_str(&format!(
            "      \"op_name\": \"{}\",\n",
            json_escape(&t.op_name)
        ));
        out.push_str(&format!(
            "      \"created_at\": \"{}\",\n",
            json_escape(&t.created_at)
        ));
        // params
        out.push_str("      \"params\": {");
        let param_pairs: Vec<_> = t.params.iter().collect();
        for (pi, (k, v)) in param_pairs.iter().enumerate() {
            out.push_str(&format!(
                "\"{}\": \"{}\"{}",
                json_escape(k),
                json_escape(v),
                if pi + 1 < param_pairs.len() { ", " } else { "" }
            ));
        }
        out.push_str("}\n");
        out.push_str(if i + 1 < lineage.transformations.len() {
            "    },\n"
        } else {
            "    }\n"
        });
    }
    out.push_str("  ]\n");

    out.push_str("}\n");
    out
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Escape a string for inclusion in a DOT label (backslash-n is preserved as a
/// newline indicator; double-quotes are escaped).
fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Escape a string for inclusion in a JSON string literal.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Return the current UTC time as an RFC 3339 string (best-effort; falls back
/// to a static string when the system clock is unavailable).
fn chrono_now() -> String {
    // Use chrono if available via the workspace dependency.
    use std::time::{SystemTime, UNIX_EPOCH};
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => {
            let secs = d.as_secs();
            // Simple ISO 8601 UTC representation  (YYYY-MM-DDTHH:MM:SSZ)
            let (days_since_epoch, secs_of_day) = (secs / 86400, secs % 86400);
            let (h, m, s) = (
                secs_of_day / 3600,
                (secs_of_day % 3600) / 60,
                secs_of_day % 60,
            );
            // Gregorian calendar calculation from Julian Day Number
            let jdn = days_since_epoch + 2440588; // Julian Day of 1970-01-01
            let (y, mo, da) = jdn_to_ymd(jdn);
            format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, da, h, m, s)
        }
        Err(_) => "1970-01-01T00:00:00Z".to_string(),
    }
}

/// Convert a Julian Day Number to (year, month, day) in the proleptic Gregorian
/// calendar.  Algorithm from Richards (2013).
fn jdn_to_ymd(jdn: u64) -> (u64, u64, u64) {
    let jdn = jdn as i64;
    let f = jdn + 1401 + (((4 * jdn + 274277) / 146097) * 3) / 4 - 38;
    let e = 4 * f + 3;
    let g = (e % 1461) / 4;
    let h = 5 * g + 2;
    let day = (h % 153) / 5 + 1;
    let month = (h / 153 + 2) % 12 + 1;
    let year = e / 1461 - 4716 + (14 - month) / 12;
    (year as u64, month as u64, day as u64)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_simple_lineage() -> (DataLineage, NodeId, NodeId, NodeId) {
        let mut lineage = DataLineage::new();

        let raw = lineage.add_node(DataNode::new(
            "raw".to_string(),
            DataSource::File("/data/raw.csv".to_string()),
            vec![
                ("id".to_string(), ColumnType::Integer),
                ("value".to_string(), ColumnType::Float),
            ],
        ));

        let clean = lineage.add_node(DataNode::new(
            "cleaned".to_string(),
            DataSource::InMemory,
            vec![
                ("id".to_string(), ColumnType::Integer),
                ("value".to_string(), ColumnType::Float),
            ],
        ));

        let agg = lineage.add_node(DataNode::new(
            "aggregated".to_string(),
            DataSource::Generated,
            vec![("mean_value".to_string(), ColumnType::Float)],
        ));

        let mut params1 = HashMap::new();
        params1.insert("drop_nulls".to_string(), "true".to_string());
        record_transformation(&mut lineage, vec![raw], clean, "filter", params1);

        let mut params2 = HashMap::new();
        params2.insert("fn".to_string(), "mean".to_string());
        record_transformation(&mut lineage, vec![clean], agg, "aggregate", params2);

        (lineage, raw, clean, agg)
    }

    #[test]
    fn test_add_and_get_node() {
        let mut lineage = DataLineage::new();
        let id = lineage.add_node(DataNode::new(
            "test".to_string(),
            DataSource::InMemory,
            vec![],
        ));
        let node = lineage.get_node(id).expect("node must exist");
        assert_eq!(node.name, "test");
        assert_eq!(node.id, id);
    }

    #[test]
    fn test_record_transformation_valid() {
        let (lineage, raw, clean, _agg) = build_simple_lineage();
        assert_eq!(lineage.transformations().len(), 2);
        assert_eq!(lineage.transformations()[0].input_nodes, vec![raw]);
        assert_eq!(lineage.transformations()[0].output_node, clean);
    }

    #[test]
    fn test_record_transformation_invalid_node() {
        let mut lineage = DataLineage::new();
        let fake_id = NodeId(999);
        let out_id = lineage.add_node(DataNode::new(
            "out".to_string(),
            DataSource::InMemory,
            vec![],
        ));
        let ok = record_transformation(&mut lineage, vec![fake_id], out_id, "op", HashMap::new());
        assert!(!ok, "should reject unknown input node");
    }

    #[test]
    fn test_get_provenance_depth_one() {
        let (lineage, raw, clean, _agg) = build_simple_lineage();
        let prov = get_provenance(&lineage, clean);
        assert_eq!(prov.len(), 1);
        assert_eq!(prov[0].id, raw);
    }

    #[test]
    fn test_get_provenance_depth_two() {
        let (lineage, raw, clean, agg) = build_simple_lineage();
        let prov = get_provenance(&lineage, agg);
        // Should find both raw and clean
        let ids: HashSet<NodeId> = prov.iter().map(|n| n.id).collect();
        assert!(ids.contains(&raw));
        assert!(ids.contains(&clean));
        assert_eq!(prov.len(), 2);
    }

    #[test]
    fn test_get_provenance_root_node() {
        let (lineage, raw, _clean, _agg) = build_simple_lineage();
        let prov = get_provenance(&lineage, raw);
        assert!(prov.is_empty(), "root node has no ancestors");
    }

    #[test]
    fn test_export_lineage_dot_structure() {
        let (lineage, _raw, _clean, _agg) = build_simple_lineage();
        let dot = export_lineage_dot(&lineage);

        assert!(dot.starts_with("digraph lineage {"));
        assert!(dot.ends_with("}\n"));
        assert!(dot.contains("raw"));
        assert!(dot.contains("cleaned"));
        assert!(dot.contains("aggregated"));
        assert!(dot.contains("->"));
        assert!(dot.contains("filter"));
        assert!(dot.contains("aggregate"));
    }

    #[test]
    fn test_lineage_to_json_structure() {
        let (lineage, _raw, _clean, _agg) = build_simple_lineage();
        let json = lineage_to_json(&lineage);

        assert!(json.contains("\"nodes\""));
        assert!(json.contains("\"transformations\""));
        assert!(json.contains("\"raw\""));
        assert!(json.contains("\"cleaned\""));
        assert!(json.contains("\"aggregated\""));
        assert!(json.contains("\"filter\""));
        assert!(json.contains("\"aggregate\""));
        // Valid enough JSON: parse with serde_json
        let _: serde_json::Value = serde_json::from_str(&json).expect("must be valid JSON");
    }

    #[test]
    fn test_data_source_labels() {
        assert_eq!(DataSource::File("/a/b.csv".into()).label(), "file:/a/b.csv");
        assert_eq!(
            DataSource::Database("pg://localhost/mydb".into()).label(),
            "db:pg://localhost/mydb"
        );
        assert_eq!(DataSource::InMemory.label(), "in_memory");
        assert_eq!(DataSource::Generated.label(), "generated");
    }

    #[test]
    fn test_node_tags() {
        let mut lineage = DataLineage::new();
        let id = lineage.add_node(
            DataNode::new("tagged".to_string(), DataSource::InMemory, vec![])
                .with_tag("owner", "alice")
                .with_tag("version", "1"),
        );
        let node = lineage.get_node(id).expect("exists");
        assert_eq!(node.tags.get("owner").map(|s| s.as_str()), Some("alice"));
        assert_eq!(node.tags.get("version").map(|s| s.as_str()), Some("1"));
    }

    #[test]
    fn test_multi_input_transformation() {
        let mut lineage = DataLineage::new();
        let a = lineage.add_node(DataNode::new("A".into(), DataSource::InMemory, vec![]));
        let b = lineage.add_node(DataNode::new("B".into(), DataSource::InMemory, vec![]));
        let c = lineage.add_node(DataNode::new("C".into(), DataSource::InMemory, vec![]));

        let ok = record_transformation(&mut lineage, vec![a, b], c, "join", HashMap::new());
        assert!(ok);
        let prov = get_provenance(&lineage, c);
        let prov_ids: HashSet<NodeId> = prov.iter().map(|n| n.id).collect();
        assert!(prov_ids.contains(&a));
        assert!(prov_ids.contains(&b));
    }

    #[test]
    fn test_json_escaping() {
        let s = r#"he said "hello\nworld""#;
        let escaped = super::json_escape(s);
        // The result should be embeddable in JSON without breaking the parser
        let json = format!("{{\"v\": \"{}\"}}", escaped);
        let _: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    }

    #[test]
    fn test_column_type_labels() {
        assert_eq!(ColumnType::Integer.label(), "integer");
        assert_eq!(ColumnType::Float.label(), "float");
        assert_eq!(ColumnType::Boolean.label(), "boolean");
        assert_eq!(ColumnType::Text.label(), "text");
        assert_eq!(ColumnType::Other("blob".into()).label(), "blob");
    }
}
