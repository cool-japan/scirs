//! Graph connectivity algorithms
//!
//! This module contains algorithms for analyzing graph connectivity properties
//! including connected components, articulation points, bridges, and bipartite checking.

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};

/// Each connected component is represented as a set of nodes
pub type Component<N> = HashSet<N>;

/// Finds all connected components in an undirected graph
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// * A vector of connected components, where each component is a set of nodes
pub fn connected_components<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Component<N>>
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let mut components: Vec<Component<N>> = Vec::new();
    let mut visited = HashSet::new();

    // For each node in the graph
    for node_idx in graph.inner().node_indices() {
        // Skip if already visited
        if visited.contains(&node_idx) {
            continue;
        }

        // BFS to find all nodes in this component
        let mut component = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(node_idx);
        visited.insert(node_idx);

        while let Some(current) = queue.pop_front() {
            component.insert(graph.inner()[current].clone());

            // Visit all unvisited neighbors
            for neighbor in graph.inner().neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        components.push(component);
    }

    components
}

/// Finds all strongly connected components in a directed graph using Tarjan's algorithm
///
/// # Arguments
/// * `graph` - The directed graph to analyze
///
/// # Returns
/// * A vector of strongly connected components
pub fn strongly_connected_components<N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> Vec<Component<N>>
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    struct TarjanState<Ix: petgraph::graph::IndexType> {
        index: usize,
        stack: Vec<petgraph::graph::NodeIndex<Ix>>,
        indices: HashMap<petgraph::graph::NodeIndex<Ix>, usize>,
        lowlinks: HashMap<petgraph::graph::NodeIndex<Ix>, usize>,
        on_stack: HashSet<petgraph::graph::NodeIndex<Ix>>,
    }

    fn strongconnect<N, E, Ix>(
        v: petgraph::graph::NodeIndex<Ix>,
        graph: &DiGraph<N, E, Ix>,
        state: &mut TarjanState<Ix>,
        components: &mut Vec<Component<N>>,
    ) where
        N: Node,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        state.indices.insert(v, state.index);
        state.lowlinks.insert(v, state.index);
        state.index += 1;
        state.stack.push(v);
        state.on_stack.insert(v);

        // Consider successors of v
        for w in graph.inner().neighbors_directed(v, Direction::Outgoing) {
            if !state.indices.contains_key(&w) {
                // Successor w has not yet been visited; recurse on it
                strongconnect(w, graph, state, components);
                let w_lowlink = *state.lowlinks.get(&w).unwrap();
                let v_lowlink = *state.lowlinks.get(&v).unwrap();
                state.lowlinks.insert(v, v_lowlink.min(w_lowlink));
            } else if state.on_stack.contains(&w) {
                // Successor w is in stack S and hence in the current SCC
                let w_index = *state.indices.get(&w).unwrap();
                let v_lowlink = *state.lowlinks.get(&v).unwrap();
                state.lowlinks.insert(v, v_lowlink.min(w_index));
            }
        }

        // If v is a root node, pop the stack and create an SCC
        if state.lowlinks.get(&v) == state.indices.get(&v) {
            let mut component = HashSet::new();
            loop {
                let w = state.stack.pop().unwrap();
                state.on_stack.remove(&w);
                component.insert(graph.inner()[w].clone());
                if w == v {
                    break;
                }
            }
            if !component.is_empty() {
                components.push(component);
            }
        }
    }

    let mut state = TarjanState {
        index: 0,
        stack: Vec::new(),
        indices: HashMap::new(),
        lowlinks: HashMap::new(),
        on_stack: HashSet::new(),
    };
    let mut components = Vec::new();

    for v in graph.inner().node_indices() {
        if !state.indices.contains_key(&v) {
            strongconnect(v, graph, &mut state, &mut components);
        }
    }

    components
}

/// Finds all articulation points (cut vertices) in an undirected graph
///
/// An articulation point is a vertex whose removal increases the number of connected components.
///
/// # Arguments
/// * `graph` - The undirected graph to analyze
///
/// # Returns
/// * A set of articulation points
pub fn articulation_points<N, E, Ix>(graph: &Graph<N, E, Ix>) -> HashSet<N>
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    struct DfsState<Ix: petgraph::graph::IndexType> {
        time: usize,
        disc: HashMap<petgraph::graph::NodeIndex<Ix>, usize>,
        low: HashMap<petgraph::graph::NodeIndex<Ix>, usize>,
        parent: HashMap<petgraph::graph::NodeIndex<Ix>, Option<petgraph::graph::NodeIndex<Ix>>>,
        visited: HashSet<petgraph::graph::NodeIndex<Ix>>,
    }

    fn dfs<N, E, Ix>(
        u: petgraph::graph::NodeIndex<Ix>,
        graph: &Graph<N, E, Ix>,
        state: &mut DfsState<Ix>,
        articulation_points: &mut HashSet<N>,
    ) where
        N: Node,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        state.visited.insert(u);
        state.disc.insert(u, state.time);
        state.low.insert(u, state.time);
        state.time += 1;

        let mut children = 0;

        for v in graph.inner().neighbors(u) {
            if !state.visited.contains(&v) {
                children += 1;
                state.parent.insert(v, Some(u));
                dfs(v, graph, state, articulation_points);

                // Check if the subtree rooted at v has a back edge to an ancestor of u
                let v_low = *state.low.get(&v).unwrap();
                let u_low = *state.low.get(&u).unwrap();
                state.low.insert(u, u_low.min(v_low));

                // u is an articulation point if:
                // (1) u is root and has more than one child
                // (2) u is not root and low[v] >= disc[u]
                let u_disc = *state.disc.get(&u).unwrap();
                if (state.parent.get(&u).unwrap().is_none() && children > 1)
                    || (state.parent.get(&u).unwrap().is_some() && v_low >= u_disc)
                {
                    articulation_points.insert(graph.inner()[u].clone());
                }
            } else if state.parent.get(&u).unwrap() != &Some(v) {
                // Update low[u] for back edge
                let v_disc = *state.disc.get(&v).unwrap();
                let u_low = *state.low.get(&u).unwrap();
                state.low.insert(u, u_low.min(v_disc));
            }
        }
    }

    let mut articulation_points = HashSet::new();
    let mut state = DfsState {
        time: 0,
        disc: HashMap::new(),
        low: HashMap::new(),
        parent: HashMap::new(),
        visited: HashSet::new(),
    };

    for node in graph.inner().node_indices() {
        if !state.visited.contains(&node) {
            state.parent.insert(node, None);
            dfs(node, graph, &mut state, &mut articulation_points);
        }
    }

    articulation_points
}

/// Result of bipartite checking
#[derive(Debug, Clone)]
pub struct BipartiteResult<N: Node> {
    /// Whether the graph is bipartite
    pub is_bipartite: bool,
    /// Node coloring (0 or 1) if bipartite, empty if not
    pub coloring: HashMap<N, u8>,
}

/// Checks if a graph is bipartite and returns the coloring if it is
///
/// A graph is bipartite if its vertices can be divided into two disjoint sets
/// such that no two vertices within the same set are adjacent.
///
/// # Arguments
/// * `graph` - The graph to check
///
/// # Returns
/// * A BipartiteResult indicating if the graph is bipartite and the coloring
pub fn is_bipartite<N, E, Ix>(graph: &Graph<N, E, Ix>) -> BipartiteResult<N>
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let mut coloring: HashMap<petgraph::graph::NodeIndex<Ix>, u8> = HashMap::new();
    let mut queue = VecDeque::new();

    // Check each connected component
    for start_idx in graph.inner().node_indices() {
        if coloring.contains_key(&start_idx) {
            continue;
        }

        // Start BFS coloring from this node
        queue.push_back(start_idx);
        coloring.insert(start_idx, 0);

        while let Some(node_idx) = queue.pop_front() {
            let node_color = *coloring.get(&node_idx).unwrap();
            let next_color = 1 - node_color;

            for neighbor in graph.inner().neighbors(node_idx) {
                if let Some(&neighbor_color) = coloring.get(&neighbor) {
                    // If neighbor has same color, not bipartite
                    if neighbor_color == node_color {
                        return BipartiteResult {
                            is_bipartite: false,
                            coloring: HashMap::new(),
                        };
                    }
                } else {
                    // Color the neighbor and add to queue
                    coloring.insert(neighbor, next_color);
                    queue.push_back(neighbor);
                }
            }
        }
    }

    // Convert node indices to actual nodes
    let node_coloring: HashMap<N, u8> = coloring
        .into_iter()
        .map(|(idx, color)| (graph.inner()[idx].clone(), color))
        .collect();

    BipartiteResult {
        is_bipartite: true,
        coloring: node_coloring,
    }
}

/// Finds all bridges (cut edges) in an undirected graph
///
/// A bridge is an edge whose removal increases the number of connected components.
///
/// # Arguments
/// * `graph` - The undirected graph to analyze
///
/// # Returns
/// * A vector of bridges, where each bridge is represented as a tuple of nodes
pub fn bridges<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<(N, N)>
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    struct DfsState<Ix: petgraph::graph::IndexType> {
        time: usize,
        disc: HashMap<petgraph::graph::NodeIndex<Ix>, usize>,
        low: HashMap<petgraph::graph::NodeIndex<Ix>, usize>,
        parent: HashMap<petgraph::graph::NodeIndex<Ix>, Option<petgraph::graph::NodeIndex<Ix>>>,
        visited: HashSet<petgraph::graph::NodeIndex<Ix>>,
    }

    fn dfs<N, E, Ix>(
        u: petgraph::graph::NodeIndex<Ix>,
        graph: &Graph<N, E, Ix>,
        state: &mut DfsState<Ix>,
        bridges: &mut Vec<(N, N)>,
    ) where
        N: Node,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        state.visited.insert(u);
        state.disc.insert(u, state.time);
        state.low.insert(u, state.time);
        state.time += 1;

        for v in graph.inner().neighbors(u) {
            if !state.visited.contains(&v) {
                state.parent.insert(v, Some(u));
                dfs(v, graph, state, bridges);

                // Check if the subtree rooted at v has a back edge to an ancestor of u
                let v_low = *state.low.get(&v).unwrap();
                let u_low = *state.low.get(&u).unwrap();
                state.low.insert(u, u_low.min(v_low));

                // If low[v] > disc[u], then (u, v) is a bridge
                let u_disc = *state.disc.get(&u).unwrap();
                if v_low > u_disc {
                    bridges.push((graph.inner()[u].clone(), graph.inner()[v].clone()));
                }
            } else if state.parent.get(&u).unwrap() != &Some(v) {
                // Update low[u] for back edge
                let v_disc = *state.disc.get(&v).unwrap();
                let u_low = *state.low.get(&u).unwrap();
                state.low.insert(u, u_low.min(v_disc));
            }
        }
    }

    let mut bridges = Vec::new();
    let mut state = DfsState {
        time: 0,
        disc: HashMap::new(),
        low: HashMap::new(),
        parent: HashMap::new(),
        visited: HashSet::new(),
    };

    for node in graph.inner().node_indices() {
        if !state.visited.contains(&node) {
            state.parent.insert(node, None);
            dfs(node, graph, &mut state, &mut bridges);
        }
    }

    bridges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_connected_components() -> GraphResult<()> {
        // Create a graph with two components: {0, 1, 2} and {3, 4}
        let mut graph = create_graph::<i32, ()>();

        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(3, 4, ())?;

        let components = connected_components(&graph);
        assert_eq!(components.len(), 2);

        // Check component sizes
        let sizes: Vec<usize> = components.iter().map(|c| c.len()).collect();
        assert!(sizes.contains(&3));
        assert!(sizes.contains(&2));

        Ok(())
    }

    #[test]
    fn test_strongly_connected_components() -> GraphResult<()> {
        // Create a directed graph with SCCs
        let mut graph = crate::base::DiGraph::<&str, ()>::new();

        // Create a cycle A -> B -> C -> A
        graph.add_edge("A", "B", ())?;
        graph.add_edge("B", "C", ())?;
        graph.add_edge("C", "A", ())?;

        // Add isolated node D by adding an edge from D to E
        graph.add_edge("D", "E", ())?;

        let sccs = strongly_connected_components(&graph);
        assert_eq!(sccs.len(), 3);

        // One SCC should have 3 nodes (A, B, C), two should have 1 (D and E)
        let sizes: Vec<usize> = sccs.iter().map(|c| c.len()).collect();
        assert!(sizes.contains(&3));
        assert_eq!(sizes.iter().filter(|&&s| s == 1).count(), 2);

        Ok(())
    }

    #[test]
    fn test_articulation_points() -> GraphResult<()> {
        // Create a graph where node 1 is an articulation point
        let mut graph = create_graph::<i32, ()>();

        // Structure: 0 - 1 - 2
        //                |
        //                3
        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(1, 3, ())?;

        let aps = articulation_points(&graph);
        assert_eq!(aps.len(), 1);
        assert!(aps.contains(&1));

        Ok(())
    }

    #[test]
    fn test_is_bipartite() -> GraphResult<()> {
        // Create a bipartite graph (square)
        let mut bipartite = create_graph::<i32, ()>();

        bipartite.add_edge(0, 1, ())?;
        bipartite.add_edge(1, 2, ())?;
        bipartite.add_edge(2, 3, ())?;
        bipartite.add_edge(3, 0, ())?;

        let result = is_bipartite(&bipartite);
        assert!(result.is_bipartite);
        assert_eq!(result.coloring.len(), 4);

        // Create a non-bipartite graph (triangle)
        let mut non_bipartite = create_graph::<i32, ()>();

        non_bipartite.add_edge(0, 1, ())?;
        non_bipartite.add_edge(1, 2, ())?;
        non_bipartite.add_edge(2, 0, ())?;

        let result = is_bipartite(&non_bipartite);
        assert!(!result.is_bipartite);

        Ok(())
    }

    #[test]
    fn test_bridges() -> GraphResult<()> {
        // Create a graph with a bridge
        let mut graph = create_graph::<i32, ()>();

        // Create two triangles connected by a bridge
        // Triangle 1: 0-1-2
        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(2, 0, ())?;

        // Bridge: 2-3
        graph.add_edge(2, 3, ())?;

        // Add another triangle to make bridge more obvious
        graph.add_edge(3, 4, ())?;
        graph.add_edge(4, 5, ())?;
        graph.add_edge(5, 3, ())?;

        let bridges_found = bridges(&graph);
        assert_eq!(bridges_found.len(), 1);

        // The bridge should be (2, 3) or (3, 2)
        let bridge = &bridges_found[0];
        assert!((bridge.0 == 2 && bridge.1 == 3) || (bridge.0 == 3 && bridge.1 == 2));

        Ok(())
    }
}
