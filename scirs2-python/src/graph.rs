//! Python bindings for scirs2-graph
//!
//! This module provides Python bindings for graph algorithms,
//! including traversal, shortest paths, centrality, and community detection.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};

// NumPy types for Python array interface
use scirs2_numpy::{IntoPyArray, PyArray2};

// ndarray types from scirs2-core
#[allow(unused_imports)]
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;

// Direct imports from scirs2-graph
use scirs2_graph::{
    articulation_points,
    // Centrality
    betweenness_centrality,
    // Traversal
    breadth_first_search,
    bridges,
    closeness_centrality,
    // Connectivity
    connected_components,
    depth_first_search,
    // Graph properties
    diameter,
    // Shortest paths
    dijkstra_path,
    floyd_warshall,
    is_bipartite,
    label_propagation_result,
    // Community detection
    louvain_communities_result,
    // Spanning trees
    minimum_spanning_tree,
    modularity,
    strongly_connected_components,
    DiGraph,
    // Graph types
    Graph,
};
// Additional imports from submodules
use scirs2_graph::generators::{
    barabasi_albert_graph, complete_graph, cycle_graph, erdos_renyi_graph, path_graph, star_graph,
    watts_strogatz_graph,
};
use scirs2_graph::measures::{clustering_coefficient, pagerank_centrality};

// ========================================
// GRAPH CREATION
// ========================================

/// Create an undirected graph from edge list
#[pyfunction]
fn graph_from_edges_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    num_nodes: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();
    let mut max_node = 0usize;

    // Add edges
    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        let weight: f64 = if tuple.len() > 2 {
            tuple.get_item(2)?.extract()?
        } else {
            1.0
        };

        max_node = max_node.max(u).max(v);
        graph.add_edge(u, v, weight);
    }

    let n = num_nodes.unwrap_or(max_node + 1);

    let edges_vec = graph.edges();
    let dict = PyDict::new(py);
    dict.set_item("num_nodes", n)?;
    dict.set_item("num_edges", edges_vec.len())?;
    dict.set_item("directed", false)?;

    // Convert to adjacency list representation
    let mut adj_list: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for edge in &edges_vec {
        adj_list[edge.source].push((edge.target, edge.weight));
        adj_list[edge.target].push((edge.source, edge.weight)); // Undirected
    }

    let adj_py: Vec<Vec<(usize, f64)>> = adj_list;
    dict.set_item("adjacency", adj_py)?;

    Ok(dict.into())
}

/// Create a directed graph from edge list
#[pyfunction]
fn digraph_from_edges_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    num_nodes: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let mut graph = DiGraph::<usize, f64, u32>::new();
    let mut max_node = 0usize;

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        let weight: f64 = if tuple.len() > 2 {
            tuple.get_item(2)?.extract()?
        } else {
            1.0
        };

        max_node = max_node.max(u).max(v);
        graph.add_edge(u, v, weight);
    }

    let n = num_nodes.unwrap_or(max_node + 1);

    let edges_vec = graph.edges();
    let dict = PyDict::new(py);
    dict.set_item("num_nodes", n)?;
    dict.set_item("num_edges", edges_vec.len())?;
    dict.set_item("directed", true)?;

    // Convert to adjacency list
    let mut adj_list: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for edge in &edges_vec {
        adj_list[edge.source].push((edge.target, edge.weight));
    }

    dict.set_item("adjacency", adj_list)?;

    Ok(dict.into())
}

// ========================================
// GRAPH GENERATORS
// ========================================

/// Generate Erdős-Rényi random graph
#[pyfunction]
fn erdos_renyi_graph_py(py: Python, n: usize, p: f64) -> PyResult<Py<PyAny>> {
    let mut rng = thread_rng();
    let graph = erdos_renyi_graph(n, p, &mut rng)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to generate graph: {}", e)))?;

    let edges_vec = graph.edges();
    let dict = PyDict::new(py);
    dict.set_item("num_nodes", n)?;
    dict.set_item("num_edges", edges_vec.len())?;
    dict.set_item("directed", false)?;

    // Extract edges
    let edges: Vec<(usize, usize, f64)> = edges_vec
        .iter()
        .map(|e| (e.source, e.target, e.weight))
        .collect();
    dict.set_item("edges", edges)?;

    Ok(dict.into())
}

/// Generate Barabási-Albert preferential attachment graph
#[pyfunction]
fn barabasi_albert_graph_py(py: Python, n: usize, m: usize) -> PyResult<Py<PyAny>> {
    let mut rng = thread_rng();
    let graph = barabasi_albert_graph(n, m, &mut rng)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to generate graph: {}", e)))?;

    let edges_vec = graph.edges();
    let dict = PyDict::new(py);
    dict.set_item("num_nodes", n)?;
    dict.set_item("num_edges", edges_vec.len())?;
    dict.set_item("directed", false)?;

    let edges: Vec<(usize, usize, f64)> = edges_vec
        .iter()
        .map(|e| (e.source, e.target, e.weight))
        .collect();
    dict.set_item("edges", edges)?;

    Ok(dict.into())
}

/// Generate Watts-Strogatz small-world graph
#[pyfunction]
fn watts_strogatz_graph_py(py: Python, n: usize, k: usize, p: f64) -> PyResult<Py<PyAny>> {
    let mut rng = thread_rng();
    let graph = watts_strogatz_graph(n, k, p, &mut rng)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to generate graph: {}", e)))?;

    let edges_vec = graph.edges();
    let dict = PyDict::new(py);
    dict.set_item("num_nodes", n)?;
    dict.set_item("num_edges", edges_vec.len())?;
    dict.set_item("directed", false)?;

    let edges: Vec<(usize, usize, f64)> = edges_vec
        .iter()
        .map(|e| (e.source, e.target, e.weight))
        .collect();
    dict.set_item("edges", edges)?;

    Ok(dict.into())
}

/// Generate complete graph
#[pyfunction]
fn complete_graph_py(py: Python, n: usize) -> PyResult<Py<PyAny>> {
    let graph = complete_graph(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to generate graph: {}", e)))?;

    let edges_vec = graph.edges();
    let dict = PyDict::new(py);
    dict.set_item("num_nodes", n)?;
    dict.set_item("num_edges", edges_vec.len())?;
    dict.set_item("directed", false)?;

    let edges: Vec<(usize, usize, f64)> = edges_vec
        .iter()
        .map(|e| (e.source, e.target, e.weight))
        .collect();
    dict.set_item("edges", edges)?;

    Ok(dict.into())
}

/// Generate path graph
#[pyfunction]
fn path_graph_py(py: Python, n: usize) -> PyResult<Py<PyAny>> {
    let graph = path_graph(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to generate graph: {}", e)))?;

    let edges_vec = graph.edges();
    let dict = PyDict::new(py);
    dict.set_item("num_nodes", n)?;
    dict.set_item("num_edges", edges_vec.len())?;
    dict.set_item("directed", false)?;

    let edges: Vec<(usize, usize, f64)> = edges_vec
        .iter()
        .map(|e| (e.source, e.target, e.weight))
        .collect();
    dict.set_item("edges", edges)?;

    Ok(dict.into())
}

/// Generate cycle graph
#[pyfunction]
fn cycle_graph_py(py: Python, n: usize) -> PyResult<Py<PyAny>> {
    let graph = cycle_graph(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to generate graph: {}", e)))?;

    let edges_vec = graph.edges();
    let dict = PyDict::new(py);
    dict.set_item("num_nodes", n)?;
    dict.set_item("num_edges", edges_vec.len())?;
    dict.set_item("directed", false)?;

    let edges: Vec<(usize, usize, f64)> = edges_vec
        .iter()
        .map(|e| (e.source, e.target, e.weight))
        .collect();
    dict.set_item("edges", edges)?;

    Ok(dict.into())
}

/// Generate star graph
#[pyfunction]
fn star_graph_py(py: Python, n: usize) -> PyResult<Py<PyAny>> {
    let graph = star_graph(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to generate graph: {}", e)))?;

    let edges_vec = graph.edges();
    let dict = PyDict::new(py);
    dict.set_item("num_nodes", n)?;
    dict.set_item("num_edges", edges_vec.len())?;
    dict.set_item("directed", false)?;

    let edges: Vec<(usize, usize, f64)> = edges_vec
        .iter()
        .map(|e| (e.source, e.target, e.weight))
        .collect();
    dict.set_item("edges", edges)?;

    Ok(dict.into())
}

// ========================================
// TRAVERSAL ALGORITHMS
// ========================================

/// Breadth-first search from a source node
#[pyfunction]
fn bfs_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    source: usize,
    num_nodes: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();
    let mut max_node = 0usize;

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        max_node = max_node.max(u).max(v);
        graph.add_edge(u, v, 1.0);
    }

    let _n = num_nodes.unwrap_or(max_node + 1);

    let result = breadth_first_search(&graph, &source)
        .map_err(|e| PyRuntimeError::new_err(format!("BFS failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("order", result)?;

    Ok(dict.into())
}

/// Depth-first search from a source node
#[pyfunction]
fn dfs_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    source: usize,
    num_nodes: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();
    let mut max_node = 0usize;

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        max_node = max_node.max(u).max(v);
        graph.add_edge(u, v, 1.0);
    }

    let _n = num_nodes.unwrap_or(max_node + 1);

    let result = depth_first_search(&graph, &source)
        .map_err(|e| PyRuntimeError::new_err(format!("DFS failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("order", result)?;

    Ok(dict.into())
}

// ========================================
// SHORTEST PATH ALGORITHMS
// ========================================

/// Dijkstra's shortest path algorithm
#[pyfunction]
fn dijkstra_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    source: usize,
    target: usize,
) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple: &Bound<'_, PyTuple> = edge.cast()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        let weight: f64 = if tuple.len() > 2 {
            tuple.get_item(2)?.extract()?
        } else {
            1.0
        };
        graph.add_edge(u, v, weight);
    }

    let result = dijkstra_path(&graph, &source, &target)
        .map_err(|e| PyRuntimeError::new_err(format!("Dijkstra failed: {}", e)))?;

    let dict = PyDict::new(py);
    if let Some(path) = result {
        dict.set_item("path", path.nodes)?;
        dict.set_item("distance", path.total_weight)?;
    } else {
        return Err(PyRuntimeError::new_err("No path found"));
    }

    Ok(dict.into())
}

/// Floyd-Warshall all-pairs shortest paths
#[pyfunction]
#[allow(unused_variables)]
fn floyd_warshall_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    num_nodes: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple: &Bound<'_, PyTuple> = edge.cast()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        let weight: f64 = if tuple.len() > 2 {
            tuple.get_item(2)?.extract()?
        } else {
            1.0
        };
        graph.add_edge(u, v, weight);
    }

    let result = floyd_warshall(&graph)
        .map_err(|e| PyRuntimeError::new_err(format!("Floyd-Warshall failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

// ========================================
// CONNECTIVITY
// ========================================

/// Find connected components
#[pyfunction]
#[allow(unused_variables)]
fn connected_components_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    num_nodes: usize,
) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple: &Bound<'_, PyTuple> = edge.cast()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        graph.add_edge(u, v, 1.0);
    }

    let components = connected_components(&graph);

    let dict = PyDict::new(py);
    dict.set_item("n_components", components.len())?;

    // Convert components to list of lists (Component<N> = HashSet<N>)
    let comp_list: Vec<Vec<usize>> = components
        .into_iter()
        .map(|c: std::collections::HashSet<usize>| c.into_iter().collect())
        .collect();
    dict.set_item("components", comp_list)?;

    Ok(dict.into())
}

/// Find strongly connected components (directed graphs)
#[pyfunction]
#[allow(unused_variables)]
fn strongly_connected_components_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    num_nodes: usize,
) -> PyResult<Py<PyAny>> {
    let mut graph = DiGraph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple: &Bound<'_, PyTuple> = edge.cast()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        graph.add_edge(u, v, 1.0);
    }

    let components = strongly_connected_components(&graph);

    let dict = PyDict::new(py);
    dict.set_item("n_components", components.len())?;

    let comp_list: Vec<Vec<usize>> = components
        .into_iter()
        .map(|c: std::collections::HashSet<usize>| c.into_iter().collect())
        .collect();
    dict.set_item("components", comp_list)?;

    Ok(dict.into())
}

/// Find articulation points (cut vertices)
#[pyfunction]
#[allow(unused_variables)]
fn articulation_points_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    num_nodes: usize,
) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        graph.add_edge(u, v, 1.0);
    }

    // articulation_points takes only graph, returns HashSet (not Result)
    let points = articulation_points(&graph);

    let dict = PyDict::new(py);
    let points_vec: Vec<usize> = points.into_iter().collect();
    dict.set_item("articulation_points", points_vec)?;

    Ok(dict.into())
}

/// Find bridges (cut edges)
#[pyfunction]
#[allow(unused_variables)]
fn bridges_py(py: Python, edges: &Bound<'_, PyList>, num_nodes: usize) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        graph.add_edge(u, v, 1.0);
    }

    // bridges takes only graph, returns Vec<(N, N)> (not Result)
    let bridge_list = bridges(&graph);

    let dict = PyDict::new(py);
    let bridges_vec: Vec<(usize, usize)> = bridge_list;
    dict.set_item("bridges", bridges_vec)?;

    Ok(dict.into())
}

/// Check if graph is bipartite
#[pyfunction]
#[allow(unused_variables)]
fn is_bipartite_py(py: Python, edges: &Bound<'_, PyList>, num_nodes: usize) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        graph.add_edge(u, v, 1.0);
    }

    // is_bipartite takes only graph, returns BipartiteResult (not Result)
    let result = is_bipartite(&graph);

    let dict = PyDict::new(py);
    dict.set_item("is_bipartite", result.is_bipartite)?;

    if result.is_bipartite {
        // BipartiteResult has coloring: HashMap<N, u8> where 0 is left and 1 is right
        let left: Vec<usize> = result
            .coloring
            .iter()
            .filter(|(_, &color)| color == 0)
            .map(|(&node, _)| node)
            .collect();
        let right: Vec<usize> = result
            .coloring
            .iter()
            .filter(|(_, &color)| color == 1)
            .map(|(&node, _)| node)
            .collect();
        dict.set_item("left", left)?;
        dict.set_item("right", right)?;
    }

    Ok(dict.into())
}

// ========================================
// CENTRALITY MEASURES
// ========================================

/// Calculate betweenness centrality
#[pyfunction]
#[pyo3(signature = (edges, num_nodes, normalized=true))]
fn betweenness_centrality_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    num_nodes: usize,
    normalized: bool,
) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        graph.add_edge(u, v, 1.0);
    }

    // betweenness_centrality takes (graph, normalized), returns HashMap (not Result)
    let centrality = betweenness_centrality(&graph, normalized);

    let dict = PyDict::new(py);
    let centrality_vec: Vec<f64> = (0..num_nodes)
        .map(|i| *centrality.get(&i).unwrap_or(&0.0))
        .collect();
    dict.set_item("centrality", centrality_vec)?;

    Ok(dict.into())
}

/// Calculate closeness centrality
#[pyfunction]
#[pyo3(signature = (edges, num_nodes, normalized=true))]
fn closeness_centrality_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    num_nodes: usize,
    normalized: bool,
) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        graph.add_edge(u, v, 1.0);
    }

    // closeness_centrality takes (graph, normalized), returns HashMap (not Result)
    let centrality = closeness_centrality(&graph, normalized);

    let dict = PyDict::new(py);
    let centrality_vec: Vec<f64> = (0..num_nodes)
        .map(|i| *centrality.get(&i).unwrap_or(&0.0))
        .collect();
    dict.set_item("centrality", centrality_vec)?;

    Ok(dict.into())
}

/// Calculate PageRank
#[pyfunction]
#[pyo3(signature = (edges, num_nodes, damping=0.85, _max_iter=100, tol=1e-6))]
#[allow(unused_variables)]
fn pagerank_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    num_nodes: usize,
    damping: f64,
    _max_iter: usize,
    tol: f64,
) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        graph.add_edge(u, v, 1.0);
    }

    // pagerank_centrality takes (graph, damping, tolerance) - max_iter is hardcoded internally
    let pr = pagerank_centrality(&graph, damping, tol)
        .map_err(|e| PyRuntimeError::new_err(format!("PageRank failed: {}", e)))?;

    let dict = PyDict::new(py);
    let pr_vec: Vec<f64> = (0..num_nodes)
        .map(|i| *pr.get(&i).unwrap_or(&0.0))
        .collect();
    dict.set_item("pagerank", pr_vec)?;

    Ok(dict.into())
}

// ========================================
// COMMUNITY DETECTION
// ========================================

/// Louvain community detection
#[pyfunction]
#[allow(unused_variables)]
fn louvain_communities_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    num_nodes: usize,
) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        let weight: f64 = if tuple.len() > 2 {
            tuple.get_item(2)?.extract()?
        } else {
            1.0
        };
        graph.add_edge(u, v, weight);
    }

    // louvain_communities_result takes only graph, returns CommunityResult (not Result)
    let result = louvain_communities_result(&graph);

    let dict = PyDict::new(py);
    dict.set_item("n_communities", result.communities.len())?;
    // CommunityResult uses quality_score instead of modularity
    dict.set_item("modularity", result.quality_score)?;

    // Convert communities (Vec<HashSet<N>>) to list of lists
    let comm_list: Vec<Vec<usize>> = result
        .communities
        .into_iter()
        .map(|c: std::collections::HashSet<usize>| c.into_iter().collect())
        .collect();
    dict.set_item("communities", comm_list)?;

    Ok(dict.into())
}

/// Label propagation community detection
#[pyfunction]
#[pyo3(signature = (edges, num_nodes, max_iterations=100))]
#[allow(unused_variables)]
fn label_propagation_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    num_nodes: usize,
    max_iterations: usize,
) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        graph.add_edge(u, v, 1.0);
    }

    // label_propagation_result takes graph and max_iterations, returns CommunityResult
    let result = label_propagation_result(&graph, max_iterations);

    let dict = PyDict::new(py);
    dict.set_item("n_communities", result.communities.len())?;

    let comm_list: Vec<Vec<usize>> = result
        .communities
        .into_iter()
        .map(|c: std::collections::HashSet<usize>| c.into_iter().collect())
        .collect();
    dict.set_item("communities", comm_list)?;

    Ok(dict.into())
}

/// Calculate modularity of a partition
#[pyfunction]
#[allow(unused_variables)]
fn modularity_py(
    edges: &Bound<'_, PyList>,
    communities: &Bound<'_, PyList>,
    num_nodes: usize,
) -> PyResult<f64> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        let weight: f64 = if tuple.len() > 2 {
            tuple.get_item(2)?.extract()?
        } else {
            1.0
        };
        graph.add_edge(u, v, weight);
    }

    // Convert communities from Python - modularity expects HashMap<N, usize> (node -> community_id)
    let mut comm_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (comm_id, comm) in communities.iter().enumerate() {
        let nodes: Vec<usize> = comm.extract()?;
        for node in nodes {
            comm_map.insert(node, comm_id);
        }
    }

    // modularity returns f64 directly (no Result wrapper)
    Ok(modularity(&graph, &comm_map))
}

// ========================================
// GRAPH PROPERTIES
// ========================================

/// Calculate graph diameter
#[pyfunction]
#[allow(unused_variables)]
fn diameter_py(edges: &Bound<'_, PyList>, num_nodes: usize) -> PyResult<f64> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        graph.add_edge(u, v, 1.0);
    }

    // diameter returns Option<f64>
    diameter(&graph).ok_or_else(|| PyRuntimeError::new_err("Graph is not connected or empty"))
}

/// Calculate clustering coefficient
#[pyfunction]
fn clustering_coefficient_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    num_nodes: usize,
) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        graph.add_edge(u, v, 1.0);
    }

    // clustering_coefficient takes only graph, returns Result<HashMap<N, f64>>
    let coeffs = clustering_coefficient(&graph)
        .map_err(|e| PyRuntimeError::new_err(format!("Clustering coefficient failed: {}", e)))?;

    let dict = PyDict::new(py);
    let coeff_vec: Vec<f64> = (0..num_nodes)
        .map(|i| *coeffs.get(&i).unwrap_or(&0.0))
        .collect();

    // Calculate average before moving coeff_vec
    let avg: f64 = coeff_vec.iter().sum::<f64>() / num_nodes as f64;
    dict.set_item("coefficients", coeff_vec)?;
    dict.set_item("average", avg)?;

    Ok(dict.into())
}

/// Calculate graph density
#[pyfunction]
fn graph_density_py(edges: &Bound<'_, PyList>, num_nodes: usize, directed: bool) -> PyResult<f64> {
    let num_edges = edges.len();

    if directed {
        let max_edges = num_nodes * (num_nodes - 1);
        Ok(num_edges as f64 / max_edges as f64)
    } else {
        let max_edges = num_nodes * (num_nodes - 1) / 2;
        Ok(num_edges as f64 / max_edges as f64)
    }
}

// ========================================
// SPANNING TREES
// ========================================

/// Compute minimum spanning tree
#[pyfunction]
#[allow(unused_variables)]
fn minimum_spanning_tree_py(
    py: Python,
    edges: &Bound<'_, PyList>,
    num_nodes: usize,
) -> PyResult<Py<PyAny>> {
    let mut graph = Graph::<usize, f64, u32>::new();

    for edge in edges.iter() {
        let tuple = edge.cast::<PyTuple>()?;
        let u: usize = tuple.get_item(0)?.extract()?;
        let v: usize = tuple.get_item(1)?.extract()?;
        let weight: f64 = if tuple.len() > 2 {
            tuple.get_item(2)?.extract()?
        } else {
            1.0
        };
        graph.add_edge(u, v, weight);
    }

    // minimum_spanning_tree returns Result<Vec<Edge<N, E>>>
    let mst = minimum_spanning_tree(&graph)
        .map_err(|e| PyRuntimeError::new_err(format!("MST failed: {}", e)))?;

    let dict = PyDict::new(py);
    // mst is a Vec<Edge>, where Edge has source, target, weight fields
    let mst_edges: Vec<(usize, usize, f64)> =
        mst.iter().map(|e| (e.source, e.target, e.weight)).collect();
    dict.set_item("edges", mst_edges.clone())?;

    let total_weight: f64 = mst_edges.iter().map(|(_, _, w)| *w).sum();
    dict.set_item("total_weight", total_weight)?;

    Ok(dict.into())
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Graph creation
    m.add_function(wrap_pyfunction!(graph_from_edges_py, m)?)?;
    m.add_function(wrap_pyfunction!(digraph_from_edges_py, m)?)?;

    // Graph generators
    m.add_function(wrap_pyfunction!(erdos_renyi_graph_py, m)?)?;
    m.add_function(wrap_pyfunction!(barabasi_albert_graph_py, m)?)?;
    m.add_function(wrap_pyfunction!(watts_strogatz_graph_py, m)?)?;
    m.add_function(wrap_pyfunction!(complete_graph_py, m)?)?;
    m.add_function(wrap_pyfunction!(path_graph_py, m)?)?;
    m.add_function(wrap_pyfunction!(cycle_graph_py, m)?)?;
    m.add_function(wrap_pyfunction!(star_graph_py, m)?)?;

    // Traversal
    m.add_function(wrap_pyfunction!(bfs_py, m)?)?;
    m.add_function(wrap_pyfunction!(dfs_py, m)?)?;

    // Shortest paths
    m.add_function(wrap_pyfunction!(dijkstra_py, m)?)?;
    m.add_function(wrap_pyfunction!(floyd_warshall_py, m)?)?;

    // Connectivity
    m.add_function(wrap_pyfunction!(connected_components_py, m)?)?;
    m.add_function(wrap_pyfunction!(strongly_connected_components_py, m)?)?;
    m.add_function(wrap_pyfunction!(articulation_points_py, m)?)?;
    m.add_function(wrap_pyfunction!(bridges_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_bipartite_py, m)?)?;

    // Centrality
    m.add_function(wrap_pyfunction!(betweenness_centrality_py, m)?)?;
    m.add_function(wrap_pyfunction!(closeness_centrality_py, m)?)?;
    m.add_function(wrap_pyfunction!(pagerank_py, m)?)?;

    // Community detection
    m.add_function(wrap_pyfunction!(louvain_communities_py, m)?)?;
    m.add_function(wrap_pyfunction!(label_propagation_py, m)?)?;
    m.add_function(wrap_pyfunction!(modularity_py, m)?)?;

    // Graph properties
    m.add_function(wrap_pyfunction!(diameter_py, m)?)?;
    m.add_function(wrap_pyfunction!(clustering_coefficient_py, m)?)?;
    m.add_function(wrap_pyfunction!(graph_density_py, m)?)?;

    // Spanning trees
    m.add_function(wrap_pyfunction!(minimum_spanning_tree_py, m)?)?;

    Ok(())
}
