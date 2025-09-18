//! Data types for community detection algorithms

use crate::base::Node;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Represents a community structure in a graph
#[derive(Debug, Clone)]
pub struct CommunityStructure<N: Node> {
    /// Map from node to community ID
    pub node_communities: HashMap<N, usize>,
    /// The modularity score of this community structure
    pub modularity: f64,
}

/// Standardized result type for community detection algorithms
///
/// This type provides a unified interface for all community detection
/// algorithms, allowing for consistent API usage and easy conversion
/// between different representations.
#[derive(Debug, Clone)]
pub struct CommunityResult<N: Node> {
    /// Map from node to community ID
    pub node_communities: HashMap<N, usize>,
    /// The communities as sets of nodes
    pub communities: Vec<HashSet<N>>,
    /// Number of communities found
    pub num_communities: usize,
    /// Quality metric for the community structure (e.g., modularity)
    pub quality_score: Option<f64>,
    /// Additional metadata about the communities
    pub metadata: HashMap<String, f64>,
}

impl<N: Node + Clone + Hash + Eq> CommunityResult<N> {
    /// Create a new CommunityResult from a node-to-community mapping
    pub fn from_node_map(_nodecommunities: HashMap<N, usize>) -> Self {
        let mut _communities: HashMap<usize, HashSet<N>> = HashMap::new();

        for (node, comm_id) in &_nodecommunities {
            _communities
                .entry(*comm_id)
                .or_default()
                .insert(node.clone());
        }

        let mut communities_vec: Vec<HashSet<N>> = _communities.into_values().collect();
        communities_vec.sort_by_key(|c| c.len());
        communities_vec.reverse(); // Largest communities first

        let num_communities = communities_vec.len();

        Self {
            node_communities: _nodecommunities,
            communities: communities_vec,
            num_communities,
            quality_score: None,
            metadata: HashMap::new(),
        }
    }

    /// Create from a CommunityStructure (for backward compatibility)
    pub fn from_community_structure(cs: CommunityStructure<N>) -> Self {
        let mut result = Self::from_node_map(cs.node_communities);
        result.quality_score = Some(cs.modularity);
        result
            .metadata
            .insert("modularity".to_string(), cs.modularity);
        result
    }

    /// Convert to the legacy CommunityStructure format
    pub fn to_community_structure(self) -> CommunityStructure<N> {
        CommunityStructure {
            node_communities: self.node_communities,
            modularity: self.quality_score.unwrap_or(0.0),
        }
    }

    /// Get communities as a vector of sets (NetworkX-style)
    pub fn as_community_sets(&self) -> &Vec<HashSet<N>> {
        &self.communities
    }

    /// Get the community assignment for a specific node
    pub fn get_community(&self, node: &N) -> Option<usize> {
        self.node_communities.get(node).copied()
    }

    /// Get all nodes in a specific community
    pub fn get_community_members(&self, communityid: usize) -> Option<&HashSet<N>> {
        self.communities.get(communityid)
    }
}
