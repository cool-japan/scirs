use scirs2_graph::algorithms::decomposition::k_core_decomposition;
use scirs2_graph::generators::create_graph;

fn main() {
    let mut graph = create_graph::<&str, ()>();

    // Build the same graph as in the test
    graph.add_edge("A", "B", ()).unwrap();
    graph.add_edge("B", "C", ()).unwrap();
    graph.add_edge("C", "A", ()).unwrap();
    graph.add_edge("D", "A", ()).unwrap();
    graph.add_edge("D", "B", ()).unwrap();
    graph.add_edge("E", "D", ()).unwrap();

    // Print degrees
    println!("Node degrees:");
    for node in graph.nodes() {
        let degree = graph.neighbors(node).unwrap().len();
        println!("{}: {}", node, degree);
    }

    // Run k-core decomposition
    let core_numbers = k_core_decomposition(&graph);

    println!("\nK-core numbers:");
    for (node, core) in &core_numbers {
        println!("{}: {}", node, core);
    }
}
