//! Tarjan's iterative strongly connected components (SCC) algorithm
//!
//! This module provides a fully iterative (non-recursive) implementation of
//! Tarjan's SCC algorithm. It is used as a component of the Pantelides
//! structural analysis for DAE systems.

/// Tarjan's iterative strongly connected components algorithm.
///
/// `adj[i]` is the list of neighbors of node `i` in a directed graph with
/// `adj.len()` nodes. Returns a list of SCCs; each SCC is a `Vec<usize>` of
/// node indices. SCCs are returned in reverse topological order (sinks first).
///
/// # Algorithm
///
/// Simulates the classic recursive Tarjan DFS iteratively using an explicit
/// stack of `(node, edge_cursor)` frames. Maintains `index`, `lowlink`, and
/// `on_stack` arrays, plus a stack of nodes for SCC extraction.
///
/// # Complexity
///
/// O(V + E) time and O(V) extra space.
///
/// # Examples
///
/// ```
/// use scirs2_integrate::dae::tarjan_scc::tarjan_scc;
///
/// // Graph with a single cycle 0→1→2→0
/// let adj = vec![vec![1usize], vec![2], vec![0]];
/// let sccs = tarjan_scc(&adj);
/// assert_eq!(sccs.len(), 1);
/// assert_eq!(sccs[0].len(), 3);
/// ```
pub fn tarjan_scc(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }

    let mut index_counter: usize = 0;
    let mut index = vec![usize::MAX; n]; // usize::MAX == "undefined"
    let mut lowlink = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut scc_stack: Vec<usize> = Vec::new(); // the Tarjan "S" stack
    let mut result: Vec<Vec<usize>> = Vec::new();

    // DFS stack frame: (node, edge_cursor)
    let mut dfs_stack: Vec<(usize, usize)> = Vec::new();

    for start in 0..n {
        if index[start] != usize::MAX {
            continue; // Already visited
        }

        // Push starting node
        index[start] = index_counter;
        lowlink[start] = index_counter;
        index_counter += 1;
        on_stack[start] = true;
        scc_stack.push(start);
        dfs_stack.push((start, 0));

        while let Some(frame) = dfs_stack.last_mut() {
            let (u, ref mut ei) = *frame;
            if *ei < adj[u].len() {
                let v = adj[u][*ei];
                *ei += 1;

                if index[v] == usize::MAX {
                    // v not yet visited — recurse
                    index[v] = index_counter;
                    lowlink[v] = index_counter;
                    index_counter += 1;
                    on_stack[v] = true;
                    scc_stack.push(v);
                    dfs_stack.push((v, 0));
                } else if on_stack[v] {
                    // v is on stack — it's an ancestor
                    let lv = lowlink[v];
                    let lu = &mut lowlink[u];
                    if lv < *lu {
                        *lu = lv;
                    }
                }
                // else: v is already finalized (in a completed SCC) — ignore
            } else {
                // All edges from u are explored
                dfs_stack.pop();

                // Propagate lowlink to parent (if any)
                if let Some(parent_frame) = dfs_stack.last() {
                    let p = parent_frame.0;
                    let lu = lowlink[u];
                    let lp = &mut lowlink[p];
                    if lu < *lp {
                        *lp = lu;
                    }
                }

                // Check if u is the root of an SCC
                if lowlink[u] == index[u] {
                    let mut scc = Vec::new();
                    loop {
                        let w = scc_stack
                            .pop()
                            .expect("SCC stack must be non-empty when root found");
                        on_stack[w] = false;
                        scc.push(w);
                        if w == u {
                            break;
                        }
                    }
                    result.push(scc);
                }
            }
        }
    }

    result
}

/// Returns the condensation (DAG of SCCs) as an adjacency list.
///
/// Each node in the returned graph corresponds to one SCC. An edge `a → b`
/// is added if any node in SCC `a` has an edge to any node in SCC `b`
/// (and `a != b`).
///
/// `sccs` is the result of `tarjan_scc`. `adj` is the original graph.
pub fn scc_condensation(
    n_nodes: usize,
    adj: &[Vec<usize>],
    sccs: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    let mut node_to_scc = vec![0usize; n_nodes];
    for (scc_idx, scc) in sccs.iter().enumerate() {
        for &node in scc {
            node_to_scc[node] = scc_idx;
        }
    }

    let n_sccs = sccs.len();
    let mut condensed: Vec<std::collections::HashSet<usize>> =
        vec![std::collections::HashSet::new(); n_sccs];

    for u in 0..n_nodes.min(adj.len()) {
        let su = node_to_scc[u];
        for &v in &adj[u] {
            if v < n_nodes {
                let sv = node_to_scc[v];
                if su != sv {
                    condensed[su].insert(sv);
                }
            }
        }
    }

    condensed
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_cycle() {
        let adj = vec![vec![1usize], vec![2], vec![0]];
        let sccs = tarjan_scc(&adj);
        assert_eq!(sccs.len(), 1);
        let mut nodes = sccs[0].clone();
        nodes.sort_unstable();
        assert_eq!(nodes, vec![0, 1, 2]);
    }

    #[test]
    fn test_two_components() {
        // 0→1→2  and  3→4
        let adj = vec![vec![1usize], vec![2], vec![], vec![4], vec![]];
        let sccs = tarjan_scc(&adj);
        assert_eq!(sccs.len(), 5, "Each node is its own SCC in a DAG");
    }

    #[test]
    fn test_single_node() {
        let adj = vec![vec![]];
        let sccs = tarjan_scc(&adj);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0], vec![0]);
    }

    #[test]
    fn test_empty_graph() {
        let adj: Vec<Vec<usize>> = Vec::new();
        let sccs = tarjan_scc(&adj);
        assert!(sccs.is_empty());
    }

    #[test]
    fn test_self_loop() {
        let adj = vec![vec![0usize]]; // single self-loop
        let sccs = tarjan_scc(&adj);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0], vec![0]);
    }

    #[test]
    fn test_two_sccs_with_bridge() {
        // SCC1: 0→1→0,  bridge: 1→2,  SCC2: 2→3→2
        let adj = vec![
            vec![1usize], // 0→1
            vec![0, 2],   // 1→0, 1→2
            vec![3],      // 2→3
            vec![2],      // 3→2
        ];
        let sccs = tarjan_scc(&adj);
        assert_eq!(sccs.len(), 2);
        // The two SCCs should be {0,1} and {2,3}
        let sizes: Vec<usize> = {
            let mut s: Vec<usize> = sccs.iter().map(|c| c.len()).collect();
            s.sort_unstable();
            s
        };
        assert_eq!(sizes, vec![2, 2]);
    }
}
