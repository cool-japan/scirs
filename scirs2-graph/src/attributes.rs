//! Node and edge attribute system
//!
//! This module provides a flexible system for associating arbitrary metadata
//! with nodes and edges in graphs. Attributes can store any serializable data
//! and provide type-safe access patterns.

use crate::base::{DiGraph, EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
#[cfg(test)]
use serde::Deserialize;
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

/// A type-erased attribute value that can store any serializable data
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    /// String attribute
    String(String),
    /// Integer attribute  
    Integer(i64),
    /// Float attribute
    Float(f64),
    /// Boolean attribute
    Boolean(bool),
    /// JSON-serialized arbitrary data
    Json(serde_json::Value),
}

impl AttributeValue {
    /// Create a string attribute
    pub fn string<S: Into<String>>(value: S) -> Self {
        AttributeValue::String(value.into())
    }

    /// Create an integer attribute
    pub fn integer(value: i64) -> Self {
        AttributeValue::Integer(value)
    }

    /// Create a float attribute
    pub fn float(value: f64) -> Self {
        AttributeValue::Float(value)
    }

    /// Create a boolean attribute
    pub fn boolean(value: bool) -> Self {
        AttributeValue::Boolean(value)
    }

    /// Create a JSON attribute from any serializable type
    pub fn json<T: Serialize>(value: &T) -> Result<Self> {
        let json_value = serde_json::to_value(value).map_err(|_| {
            GraphError::SerializationError("Failed to serialize to JSON".to_string())
        })?;
        Ok(AttributeValue::Json(json_value))
    }

    /// Get the attribute as a string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            AttributeValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get the attribute as an integer
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            AttributeValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Get the attribute as a float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            AttributeValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Get the attribute as a boolean
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            AttributeValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Get the attribute as a typed value from JSON
    pub fn as_json<T: DeserializeOwned>(&self) -> Result<T> {
        match self {
            AttributeValue::Json(json) => serde_json::from_value(json.clone()).map_err(|_| {
                GraphError::SerializationError("Failed to deserialize from JSON".to_string())
            }),
            _ => Err(GraphError::InvalidAttribute(
                "Attribute is not JSON".to_string(),
            )),
        }
    }

    /// Convert any attribute value to a string representation
    pub fn to_string_repr(&self) -> String {
        match self {
            AttributeValue::String(s) => s.clone(),
            AttributeValue::Integer(i) => i.to_string(),
            AttributeValue::Float(f) => f.to_string(),
            AttributeValue::Boolean(b) => b.to_string(),
            AttributeValue::Json(json) => json.to_string(),
        }
    }
}

impl Eq for AttributeValue {}

impl Hash for AttributeValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            AttributeValue::String(s) => {
                0u8.hash(state);
                s.hash(state);
            }
            AttributeValue::Integer(i) => {
                1u8.hash(state);
                i.hash(state);
            }
            AttributeValue::Float(f) => {
                2u8.hash(state);
                // Use the bit representation for consistent hashing
                f.to_bits().hash(state);
            }
            AttributeValue::Boolean(b) => {
                3u8.hash(state);
                b.hash(state);
            }
            AttributeValue::Json(json) => {
                4u8.hash(state);
                // Use string representation for consistent hashing
                json.to_string().hash(state);
            }
        }
    }
}

/// A collection of attributes (key-value pairs)
pub type Attributes = HashMap<String, AttributeValue>;

/// Graph with node and edge attributes
pub struct AttributedGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    /// The underlying graph structure
    graph: Graph<N, E, Ix>,
    /// Node attributes indexed by node
    node_attributes: HashMap<N, Attributes>,
    /// Edge attributes indexed by (source, target) pair
    edge_attributes: HashMap<(N, N), Attributes>,
    /// Graph-level attributes
    graph_attributes: Attributes,
}

/// Directed graph with node and edge attributes
pub struct AttributedDiGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    /// The underlying directed graph structure
    graph: DiGraph<N, E, Ix>,
    /// Node attributes indexed by node
    node_attributes: HashMap<N, Attributes>,
    /// Edge attributes indexed by (source, target) pair
    edge_attributes: HashMap<(N, N), Attributes>,
    /// Graph-level attributes
    graph_attributes: Attributes,
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for AttributedGraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> AttributedGraph<N, E, Ix> {
    /// Create a new empty attributed graph
    pub fn new() -> Self {
        AttributedGraph {
            graph: Graph::new(),
            node_attributes: HashMap::new(),
            edge_attributes: HashMap::new(),
            graph_attributes: HashMap::new(),
        }
    }

    /// Create an attributed graph from an existing graph
    pub fn from_graph(graph: Graph<N, E, Ix>) -> Self {
        AttributedGraph {
            graph,
            node_attributes: HashMap::new(),
            edge_attributes: HashMap::new(),
            graph_attributes: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: N) {
        self.graph.add_node(node.clone());
        self.node_attributes.entry(node).or_default();
    }

    /// Add a node with initial attributes
    pub fn add_node_with_attributes(&mut self, node: N, attributes: Attributes) {
        self.graph.add_node(node.clone());
        self.node_attributes.insert(node, attributes);
    }

    /// Add an edge between two nodes with a given weight
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> Result<()> {
        self.graph
            .add_edge(source.clone(), target.clone(), weight)?;
        self.edge_attributes.entry((source, target)).or_default();
        Ok(())
    }

    /// Add an edge with initial attributes
    pub fn add_edge_with_attributes(
        &mut self,
        source: N,
        target: N,
        weight: E,
        attributes: Attributes,
    ) -> Result<()> {
        self.graph
            .add_edge(source.clone(), target.clone(), weight)?;
        self.edge_attributes.insert((source, target), attributes);
        Ok(())
    }

    /// Set a node attribute
    pub fn set_node_attribute<K: Into<String>>(&mut self, node: &N, key: K, value: AttributeValue) {
        self.node_attributes
            .entry(node.clone())
            .or_default()
            .insert(key.into(), value);
    }

    /// Get a node attribute
    pub fn get_node_attribute(&self, node: &N, key: &str) -> Option<&AttributeValue> {
        self.node_attributes.get(node)?.get(key)
    }

    /// Get all node attributes
    pub fn get_node_attributes(&self, node: &N) -> Option<&Attributes> {
        self.node_attributes.get(node)
    }

    /// Get mutable reference to node attributes
    pub fn get_node_attributes_mut(&mut self, node: &N) -> Option<&mut Attributes> {
        self.node_attributes.get_mut(node)
    }

    /// Remove a node attribute
    pub fn remove_node_attribute(&mut self, node: &N, key: &str) -> Option<AttributeValue> {
        self.node_attributes.get_mut(node)?.remove(key)
    }

    /// Set an edge attribute
    pub fn set_edge_attribute<K: Into<String>>(
        &mut self,
        source: &N,
        target: &N,
        key: K,
        value: AttributeValue,
    ) -> Result<()> {
        if !self.graph.has_edge(source, target) {
            return Err(GraphError::EdgeNotFound);
        }
        self.edge_attributes
            .entry((source.clone(), target.clone()))
            .or_default()
            .insert(key.into(), value);
        Ok(())
    }

    /// Get an edge attribute
    pub fn get_edge_attribute(&self, source: &N, target: &N, key: &str) -> Option<&AttributeValue> {
        self.edge_attributes
            .get(&(source.clone(), target.clone()))?
            .get(key)
    }

    /// Get all edge attributes
    pub fn get_edge_attributes(&self, source: &N, target: &N) -> Option<&Attributes> {
        self.edge_attributes.get(&(source.clone(), target.clone()))
    }

    /// Get mutable reference to edge attributes
    pub fn get_edge_attributes_mut(&mut self, source: &N, target: &N) -> Option<&mut Attributes> {
        self.edge_attributes
            .get_mut(&(source.clone(), target.clone()))
    }

    /// Remove an edge attribute
    pub fn remove_edge_attribute(
        &mut self,
        source: &N,
        target: &N,
        key: &str,
    ) -> Option<AttributeValue> {
        self.edge_attributes
            .get_mut(&(source.clone(), target.clone()))?
            .remove(key)
    }

    /// Set a graph-level attribute
    pub fn set_graph_attribute<K: Into<String>>(&mut self, key: K, value: AttributeValue) {
        self.graph_attributes.insert(key.into(), value);
    }

    /// Get a graph-level attribute
    pub fn get_graph_attribute(&self, key: &str) -> Option<&AttributeValue> {
        self.graph_attributes.get(key)
    }

    /// Get all graph-level attributes
    pub fn get_graph_attributes(&self) -> &Attributes {
        &self.graph_attributes
    }

    /// Remove a graph-level attribute
    pub fn remove_graph_attribute(&mut self, key: &str) -> Option<AttributeValue> {
        self.graph_attributes.remove(key)
    }

    /// Get nodes with a specific attribute
    pub fn nodes_with_attribute(&self, key: &str) -> Vec<&N> {
        self.node_attributes
            .iter()
            .filter_map(|(node, attrs)| {
                if attrs.contains_key(key) {
                    Some(node)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get nodes where an attribute matches a specific value
    pub fn nodes_with_attribute_value(&self, key: &str, value: &AttributeValue) -> Vec<&N> {
        self.node_attributes
            .iter()
            .filter_map(|(node, attrs)| {
                if let Some(attr_value) = attrs.get(key) {
                    if matches_attribute_value(attr_value, value) {
                        Some(node)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get edges with a specific attribute
    pub fn edges_with_attribute(&self, key: &str) -> Vec<(&N, &N)> {
        self.edge_attributes
            .iter()
            .filter_map(|((source, target), attrs)| {
                if attrs.contains_key(key) {
                    Some((source, target))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get edges where an attribute matches a specific value
    pub fn edges_with_attribute_value(&self, key: &str, value: &AttributeValue) -> Vec<(&N, &N)> {
        self.edge_attributes
            .iter()
            .filter_map(|((source, target), attrs)| {
                if let Some(attr_value) = attrs.get(key) {
                    if matches_attribute_value(attr_value, value) {
                        Some((source, target))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get a reference to the underlying graph
    pub fn graph(&self) -> &Graph<N, E, Ix> {
        &self.graph
    }

    /// Get a mutable reference to the underlying graph
    pub fn graph_mut(&mut self) -> &mut Graph<N, E, Ix> {
        &mut self.graph
    }

    /// Convert back to a regular graph (losing attributes)
    pub fn into_graph(self) -> Graph<N, E, Ix> {
        self.graph
    }

    /// Create a subgraph containing only nodes with specific attributes
    pub fn filter_nodes_by_attribute(&self, key: &str, value: &AttributeValue) -> Self
    where
        N: Clone,
        E: Clone,
    {
        let matching_nodes: std::collections::HashSet<N> = self
            .nodes_with_attribute_value(key, value)
            .into_iter()
            .cloned()
            .collect();

        let mut new_graph = AttributedGraph::new();

        // Add matching nodes with their attributes
        for node in &matching_nodes {
            if let Some(attrs) = self.get_node_attributes(node) {
                new_graph.add_node_with_attributes(node.clone(), attrs.clone());
            } else {
                new_graph.add_node(node.clone());
            }
        }

        // Add edges between matching nodes
        for edge in self.graph.edges() {
            if matching_nodes.contains(&edge.source) && matching_nodes.contains(&edge.target) {
                if let Some(attrs) = self.get_edge_attributes(&edge.source, &edge.target) {
                    new_graph
                        .add_edge_with_attributes(
                            edge.source.clone(),
                            edge.target.clone(),
                            edge.weight.clone(),
                            attrs.clone(),
                        )
                        .unwrap();
                } else {
                    new_graph
                        .add_edge(
                            edge.source.clone(),
                            edge.target.clone(),
                            edge.weight.clone(),
                        )
                        .unwrap();
                }
            }
        }

        // Copy graph-level attributes
        new_graph.graph_attributes = self.graph_attributes.clone();

        new_graph
    }

    /// Get summary statistics about attributes
    pub fn attribute_summary(&self) -> AttributeSummary {
        let mut node_attribute_keys = std::collections::HashSet::new();
        let mut edge_attribute_keys = std::collections::HashSet::new();

        for attrs in self.node_attributes.values() {
            for key in attrs.keys() {
                node_attribute_keys.insert(key.clone());
            }
        }

        for attrs in self.edge_attributes.values() {
            for key in attrs.keys() {
                edge_attribute_keys.insert(key.clone());
            }
        }

        AttributeSummary {
            nodes_with_attributes: self.node_attributes.len(),
            edges_with_attributes: self.edge_attributes.len(),
            unique_node_attribute_keys: node_attribute_keys.len(),
            unique_edge_attribute_keys: edge_attribute_keys.len(),
            graph_attribute_keys: self.graph_attributes.len(),
            node_attribute_keys: node_attribute_keys.into_iter().collect(),
            edge_attribute_keys: edge_attribute_keys.into_iter().collect(),
        }
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for AttributedDiGraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> AttributedDiGraph<N, E, Ix> {
    /// Create a new empty attributed directed graph
    pub fn new() -> Self {
        AttributedDiGraph {
            graph: DiGraph::new(),
            node_attributes: HashMap::new(),
            edge_attributes: HashMap::new(),
            graph_attributes: HashMap::new(),
        }
    }

    /// Create an attributed directed graph from an existing directed graph
    pub fn from_digraph(graph: DiGraph<N, E, Ix>) -> Self {
        AttributedDiGraph {
            graph,
            node_attributes: HashMap::new(),
            edge_attributes: HashMap::new(),
            graph_attributes: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: N) {
        self.graph.add_node(node.clone());
        self.node_attributes.entry(node).or_default();
    }

    /// Add a node with initial attributes
    pub fn add_node_with_attributes(&mut self, node: N, attributes: Attributes) {
        self.graph.add_node(node.clone());
        self.node_attributes.insert(node, attributes);
    }

    /// Add an edge between two nodes with a given weight
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> Result<()> {
        self.graph
            .add_edge(source.clone(), target.clone(), weight)?;
        self.edge_attributes.entry((source, target)).or_default();
        Ok(())
    }

    /// Add an edge with initial attributes
    pub fn add_edge_with_attributes(
        &mut self,
        source: N,
        target: N,
        weight: E,
        attributes: Attributes,
    ) -> Result<()> {
        self.graph
            .add_edge(source.clone(), target.clone(), weight)?;
        self.edge_attributes.insert((source, target), attributes);
        Ok(())
    }

    /// Set a node attribute
    pub fn set_node_attribute<K: Into<String>>(&mut self, node: &N, key: K, value: AttributeValue) {
        self.node_attributes
            .entry(node.clone())
            .or_default()
            .insert(key.into(), value);
    }

    /// Get a node attribute
    pub fn get_node_attribute(&self, node: &N, key: &str) -> Option<&AttributeValue> {
        self.node_attributes.get(node)?.get(key)
    }

    /// Set an edge attribute
    pub fn set_edge_attribute<K: Into<String>>(
        &mut self,
        source: &N,
        target: &N,
        key: K,
        value: AttributeValue,
    ) -> Result<()> {
        if !self.graph.has_edge(source, target) {
            return Err(GraphError::EdgeNotFound);
        }
        self.edge_attributes
            .entry((source.clone(), target.clone()))
            .or_default()
            .insert(key.into(), value);
        Ok(())
    }

    /// Get an edge attribute
    pub fn get_edge_attribute(&self, source: &N, target: &N, key: &str) -> Option<&AttributeValue> {
        self.edge_attributes
            .get(&(source.clone(), target.clone()))?
            .get(key)
    }

    /// Set a graph-level attribute
    pub fn set_graph_attribute<K: Into<String>>(&mut self, key: K, value: AttributeValue) {
        self.graph_attributes.insert(key.into(), value);
    }

    /// Get a graph-level attribute
    pub fn get_graph_attribute(&self, key: &str) -> Option<&AttributeValue> {
        self.graph_attributes.get(key)
    }

    /// Get predecessors of a node
    pub fn predecessors(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        self.graph.predecessors(node)
    }

    /// Get successors of a node
    pub fn successors(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        self.graph.successors(node)
    }

    /// Get the underlying directed graph
    pub fn graph(&self) -> &DiGraph<N, E, Ix> {
        &self.graph
    }

    /// Get a mutable reference to the underlying directed graph
    pub fn graph_mut(&mut self) -> &mut DiGraph<N, E, Ix> {
        &mut self.graph
    }

    /// Convert back to a regular directed graph (losing attributes)
    pub fn into_digraph(self) -> DiGraph<N, E, Ix> {
        self.graph
    }
}

/// Summary information about attributes in a graph
#[derive(Debug, Clone)]
pub struct AttributeSummary {
    /// Number of nodes that have attributes
    pub nodes_with_attributes: usize,
    /// Number of edges that have attributes
    pub edges_with_attributes: usize,
    /// Number of unique node attribute keys
    pub unique_node_attribute_keys: usize,
    /// Number of unique edge attribute keys
    pub unique_edge_attribute_keys: usize,
    /// Number of graph-level attribute keys
    pub graph_attribute_keys: usize,
    /// List of all node attribute keys
    pub node_attribute_keys: Vec<String>,
    /// List of all edge attribute keys
    pub edge_attribute_keys: Vec<String>,
}

/// Helper function to compare attribute values with flexible type matching
fn matches_attribute_value(attr_value: &AttributeValue, target_value: &AttributeValue) -> bool {
    match (attr_value, target_value) {
        (AttributeValue::String(a), AttributeValue::String(b)) => a == b,
        (AttributeValue::Integer(a), AttributeValue::Integer(b)) => a == b,
        (AttributeValue::Float(a), AttributeValue::Float(b)) => (a - b).abs() < f64::EPSILON,
        (AttributeValue::Boolean(a), AttributeValue::Boolean(b)) => a == b,
        (AttributeValue::Json(a), AttributeValue::Json(b)) => a == b,
        // Type conversion attempts
        (AttributeValue::Integer(a), AttributeValue::Float(b)) => {
            (*a as f64 - b).abs() < f64::EPSILON
        }
        (AttributeValue::Float(a), AttributeValue::Integer(b)) => {
            (a - *b as f64).abs() < f64::EPSILON
        }
        _ => false,
    }
}

/// Attribute view for efficient querying and filtering
pub struct AttributeView<'a, N: Node> {
    attributes: &'a HashMap<N, Attributes>,
}

impl<'a, N: Node> AttributeView<'a, N> {
    /// Create a new attribute view
    pub fn new(attributes: &'a HashMap<N, Attributes>) -> Self {
        AttributeView { attributes }
    }

    /// Find nodes with numeric attributes in a range
    pub fn nodes_in_numeric_range(&self, key: &str, min: f64, max: f64) -> Vec<&N> {
        self.attributes
            .iter()
            .filter_map(|(node, attrs)| {
                if let Some(attr_value) = attrs.get(key) {
                    let numeric_value = match attr_value {
                        AttributeValue::Integer(i) => Some(*i as f64),
                        AttributeValue::Float(f) => Some(*f),
                        _ => None,
                    };

                    if let Some(value) = numeric_value {
                        if value >= min && value <= max {
                            Some(node)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find nodes with string attributes matching a pattern
    pub fn nodes_matching_pattern(&self, key: &str, pattern: &str) -> Vec<&N> {
        self.attributes
            .iter()
            .filter_map(|(node, attrs)| {
                if let Some(AttributeValue::String(s)) = attrs.get(key) {
                    if s.contains(pattern) {
                        Some(node)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all unique values for a specific attribute key
    pub fn unique_values(&self, key: &str) -> Vec<&AttributeValue> {
        let mut values = std::collections::HashSet::new();
        for attrs in self.attributes.values() {
            if let Some(value) = attrs.get(key) {
                values.insert(value);
            }
        }
        values.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_value_creation() {
        let str_attr = AttributeValue::string("test");
        assert_eq!(str_attr.as_string(), Some("test"));

        let int_attr = AttributeValue::integer(42);
        assert_eq!(int_attr.as_integer(), Some(42));

        let float_attr = AttributeValue::float(3.15);
        assert_eq!(float_attr.as_float(), Some(3.15));

        let bool_attr = AttributeValue::boolean(true);
        assert_eq!(bool_attr.as_boolean(), Some(true));
    }

    #[test]
    fn test_attribute_value_json() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct TestData {
            name: String,
            value: i32,
        }

        let data = TestData {
            name: "test".to_string(),
            value: 123,
        };

        let json_attr = AttributeValue::json(&data).unwrap();
        let recovered: TestData = json_attr.as_json().unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_attributed_graph_basic_operations() {
        let mut graph: AttributedGraph<&str, f64> = AttributedGraph::new();

        // Add nodes with attributes
        let mut node_attrs = HashMap::new();
        node_attrs.insert("type".to_string(), AttributeValue::string("person"));
        node_attrs.insert("age".to_string(), AttributeValue::integer(30));

        graph.add_node_with_attributes("Alice", node_attrs);
        graph.add_node("Bob");

        // Set attributes
        graph.set_node_attribute(&"Bob", "type", AttributeValue::string("person"));
        graph.set_node_attribute(&"Bob", "age", AttributeValue::integer(25));

        // Add edges with attributes
        graph.add_edge("Alice", "Bob", 1.0).unwrap();
        graph
            .set_edge_attribute(
                &"Alice",
                &"Bob",
                "relationship",
                AttributeValue::string("friend"),
            )
            .unwrap();

        // Test retrieval
        assert_eq!(
            graph
                .get_node_attribute(&"Alice", "type")
                .unwrap()
                .as_string(),
            Some("person")
        );
        assert_eq!(
            graph
                .get_node_attribute(&"Alice", "age")
                .unwrap()
                .as_integer(),
            Some(30)
        );
        assert_eq!(
            graph
                .get_edge_attribute(&"Alice", &"Bob", "relationship")
                .unwrap()
                .as_string(),
            Some("friend")
        );
    }

    #[test]
    fn test_attributed_graph_filtering() {
        let mut graph: AttributedGraph<i32, f64> = AttributedGraph::new();

        // Add nodes with different types
        graph.add_node(1);
        graph.set_node_attribute(&1, "type", AttributeValue::string("server"));

        graph.add_node(2);
        graph.set_node_attribute(&2, "type", AttributeValue::string("client"));

        graph.add_node(3);
        graph.set_node_attribute(&3, "type", AttributeValue::string("server"));

        // Add edges
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(1, 3, 2.0).unwrap();

        // Filter nodes by attribute
        let servers = graph.nodes_with_attribute_value("type", &AttributeValue::string("server"));
        assert_eq!(servers.len(), 2);
        assert!(servers.contains(&&1));
        assert!(servers.contains(&&3));

        // Create subgraph
        let server_subgraph =
            graph.filter_nodes_by_attribute("type", &AttributeValue::string("server"));
        assert_eq!(server_subgraph.graph().node_count(), 2);
        assert_eq!(server_subgraph.graph().edge_count(), 1); // Only edge between servers
    }

    #[test]
    fn test_attribute_summary() {
        let mut graph: AttributedGraph<&str, f64> = AttributedGraph::new();

        graph.add_node("A");
        graph.set_node_attribute(&"A", "category", AttributeValue::string("important"));

        graph.add_node("B");
        graph.set_node_attribute(&"B", "category", AttributeValue::string("normal"));
        graph.set_node_attribute(&"B", "weight", AttributeValue::float(1.5));

        graph.add_edge("A", "B", 1.0).unwrap();
        graph
            .set_edge_attribute(&"A", &"B", "type", AttributeValue::string("connection"))
            .unwrap();

        graph.set_graph_attribute("name", AttributeValue::string("test_graph"));

        let summary = graph.attribute_summary();
        assert_eq!(summary.nodes_with_attributes, 2);
        assert_eq!(summary.edges_with_attributes, 1);
        assert_eq!(summary.unique_node_attribute_keys, 2); // "category" and "weight"
        assert_eq!(summary.unique_edge_attribute_keys, 1); // "type"
        assert_eq!(summary.graph_attribute_keys, 1); // "name"
    }

    #[test]
    fn test_attribute_view() {
        let mut attributes = HashMap::new();

        let mut attrs1 = HashMap::new();
        attrs1.insert("score".to_string(), AttributeValue::float(85.5));
        attrs1.insert("name".to_string(), AttributeValue::string("Alice"));
        attributes.insert("person1", attrs1);

        let mut attrs2 = HashMap::new();
        attrs2.insert("score".to_string(), AttributeValue::float(92.0));
        attrs2.insert("name".to_string(), AttributeValue::string("Bob"));
        attributes.insert("person2", attrs2);

        let mut attrs3 = HashMap::new();
        attrs3.insert("score".to_string(), AttributeValue::integer(88));
        attrs3.insert("name".to_string(), AttributeValue::string("Charlie"));
        attributes.insert("person3", attrs3);

        let view = AttributeView::new(&attributes);

        // Test numeric range
        let high_scorers = view.nodes_in_numeric_range("score", 90.0, 100.0);
        assert_eq!(high_scorers.len(), 1);
        assert!(high_scorers.contains(&&"person2"));

        // Test pattern matching (case-sensitive)
        let names_with_a = view.nodes_matching_pattern("name", "a");
        assert_eq!(names_with_a.len(), 1); // Only Charlie has lowercase 'a'
        assert!(names_with_a.contains(&&"person3"));

        // Test pattern matching for uppercase A
        let names_with_capital_a = view.nodes_matching_pattern("name", "A");
        assert_eq!(names_with_capital_a.len(), 1); // Only Alice has uppercase 'A'
        assert!(names_with_capital_a.contains(&&"person1"));

        // Test unique values
        let unique_scores = view.unique_values("score");
        assert_eq!(unique_scores.len(), 3);
    }

    #[test]
    fn test_attribute_value_matching() {
        let int_val = AttributeValue::integer(42);
        let float_val = AttributeValue::float(42.0);
        let string_val = AttributeValue::string("42");

        // Test type-aware matching
        assert!(matches_attribute_value(
            &int_val,
            &AttributeValue::integer(42)
        ));
        assert!(matches_attribute_value(&int_val, &float_val)); // int-float conversion
        assert!(!matches_attribute_value(&int_val, &string_val)); // no string conversion

        assert!(matches_attribute_value(
            &float_val,
            &AttributeValue::float(42.0)
        ));
        assert!(matches_attribute_value(&float_val, &int_val)); // float-int conversion

        assert!(matches_attribute_value(
            &string_val,
            &AttributeValue::string("42")
        ));
    }

    #[test]
    fn test_graph_level_attributes() {
        let mut graph: AttributedGraph<i32, f64> = AttributedGraph::new();

        graph.set_graph_attribute("title", AttributeValue::string("Test Network"));
        graph.set_graph_attribute("created", AttributeValue::string("2024"));
        graph.set_graph_attribute("version", AttributeValue::float(1.0));

        assert_eq!(
            graph.get_graph_attribute("title").unwrap().as_string(),
            Some("Test Network")
        );
        assert_eq!(
            graph.get_graph_attribute("version").unwrap().as_float(),
            Some(1.0)
        );

        assert_eq!(graph.get_graph_attributes().len(), 3);

        // Remove attribute
        let removed = graph.remove_graph_attribute("created");
        assert!(removed.is_some());
        assert_eq!(graph.get_graph_attributes().len(), 2);
    }

    #[test]
    fn test_attributed_digraph() {
        let mut digraph: AttributedDiGraph<&str, f64> = AttributedDiGraph::new();

        digraph.add_node("A");
        digraph.add_node("B");
        digraph.add_edge("A", "B", 1.0).unwrap();

        assert_eq!(digraph.graph().node_count(), 2);
        assert_eq!(digraph.graph().edge_count(), 1);
        assert!(digraph.graph().has_edge(&"A", &"B"));
        assert!(!digraph.graph().has_edge(&"B", &"A")); // Directed
    }
}
