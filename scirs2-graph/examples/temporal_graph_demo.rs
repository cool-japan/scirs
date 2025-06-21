//! Demonstration of temporal graph functionality
//!
//! This example shows how to work with temporal graphs where edges
//! and nodes have time-dependent properties.

use scirs2_graph::{
    temporal_betweenness_centrality, temporal_reachability, TemporalGraph, TimeInstant,
    TimeInterval,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Temporal Graph Demo ===\n");

    // Create a temporal graph representing a social network over time
    let mut tgraph: TemporalGraph<&str, f64> = TemporalGraph::new();

    println!("1. Creating temporal graph with time-dependent relationships:");

    // Add nodes with their activity periods
    let alice_period = TimeInterval::new(0, 300)?;
    let bob_period = TimeInterval::new(50, 250)?;
    let charlie_period = TimeInterval::new(100, 400)?;
    let diana_period = TimeInterval::new(0, 200)?;
    let eve_period = TimeInterval::new(150, 350)?;

    tgraph.add_node("Alice", alice_period);
    tgraph.add_node("Bob", bob_period);
    tgraph.add_node("Charlie", charlie_period);
    tgraph.add_node("Diana", diana_period);
    tgraph.add_node("Eve", eve_period);

    println!("   Added 5 people with different activity periods");

    println!("\n2. Adding temporal relationships:");

    // Early friendship: Alice and Diana
    tgraph.add_edge("Alice", "Diana", 1.0, TimeInterval::new(10, 80)?)?;
    println!("   Alice-Diana friendship: t=10 to t=80");

    // Alice meets Bob
    tgraph.add_edge("Alice", "Bob", 0.8, TimeInterval::new(60, 120)?)?;
    println!("   Alice-Bob connection: t=60 to t=120");

    // Bob introduces Charlie
    tgraph.add_edge("Bob", "Charlie", 1.2, TimeInterval::new(110, 180)?)?;
    println!("   Bob-Charlie connection: t=110 to t=180");

    // Charlie and Eve become friends
    tgraph.add_edge("Charlie", "Eve", 1.5, TimeInterval::new(160, 250)?)?;
    println!("   Charlie-Eve friendship: t=160 to t=250");

    // Late connection: Alice and Eve
    tgraph.add_edge("Alice", "Eve", 0.9, TimeInterval::new(200, 280)?)?;
    println!("   Alice-Eve late connection: t=200 to t=280");

    println!("\n3. Analyzing graph snapshots at different times:");

    let time_points = vec![30, 70, 130, 190, 270];
    for &time in &time_points {
        let snapshot = tgraph.snapshot_at(TimeInstant::new(time));
        println!(
            "   t={}: {} nodes, {} edges",
            time,
            snapshot.node_count(),
            snapshot.edge_count()
        );
    }

    println!("\n4. Graph structure change times:");
    let change_times = tgraph.change_times();
    println!(
        "   Change times: {:?}",
        change_times.iter().map(|t| t.value()).collect::<Vec<_>>()
    );

    println!("\n5. Active interval analysis:");
    if let Some(active_interval) = tgraph.active_interval() {
        println!(
            "   Graph is active from t={} to t={} (duration: {})",
            active_interval.start.value(),
            active_interval.end.value(),
            active_interval.duration()
        );
    }

    println!("\n6. Temporal connectivity analysis:");

    // Check who Alice can reach at different times with a time budget
    let reachability_times = vec![50, 100, 200];
    for &start_time in &reachability_times {
        let reachable = temporal_reachability(
            &tgraph,
            &"Alice",
            TimeInstant::new(start_time),
            100, // time budget
        );
        println!(
            "   From Alice at t={} (budget=100): can reach {:?}",
            start_time,
            reachable.iter().collect::<Vec<_>>()
        );
    }

    println!("\n7. Temporal paths analysis:");

    // Find temporal paths from Alice to Eve
    let paths = tgraph.temporal_paths(&"Alice", &"Eve", TimeInstant::new(0), 300);

    println!("   Temporal paths from Alice to Eve:");
    for (i, path) in paths.iter().enumerate() {
        println!("     Path {}: {:?}", i + 1, path.nodes);
        println!(
            "       Duration: {}, Hops: {}",
            path.duration(),
            path.hop_count()
        );
        println!(
            "       From t={} to t={}",
            path.start_time.value(),
            path.end_time.value()
        );
    }

    println!("\n8. Time-specific connectivity:");

    let test_times = vec![40, 100, 170, 230];
    for &time in &test_times {
        let connected_ab = tgraph.are_connected_at(&"Alice", &"Bob", TimeInstant::new(time));
        let connected_ce = tgraph.are_connected_at(&"Charlie", &"Eve", TimeInstant::new(time));
        println!(
            "   t={}: Alice-Bob connected: {}, Charlie-Eve connected: {}",
            time, connected_ab, connected_ce
        );
    }

    println!("\n9. Temporal centrality analysis:");

    // Compute temporal betweenness centrality over the middle period
    let analysis_window = TimeInterval::new(100, 200)?;
    let centrality = temporal_betweenness_centrality(&tgraph, analysis_window);

    println!("   Temporal betweenness centrality (t=100 to t=200):");
    let mut centrality_vec: Vec<_> = centrality.iter().collect();
    centrality_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    for (node, score) in centrality_vec {
        println!("     {}: {:.3}", node, score);
    }

    println!("\n10. Edge activity analysis:");

    let analysis_interval = TimeInterval::new(50, 250)?;
    let active_edges = tgraph.edges_in_interval(analysis_interval);

    println!("   Edges active during t=50 to t=250:");
    for edge in active_edges {
        println!(
            "     {} -- {} (weight: {:.1}, active: t={} to t={})",
            edge.source,
            edge.target,
            edge.weight,
            edge.interval.start.value(),
            edge.interval.end.value()
        );
    }

    println!("\n11. Dynamic analysis - network evolution:");

    let sample_times = vec![25, 75, 125, 175, 225, 275];
    println!("   Network size evolution:");
    for &time in &sample_times {
        let node_count = tgraph.node_count_at(TimeInstant::new(time));
        let edge_count = tgraph.edge_count_at(TimeInstant::new(time));
        let density = if node_count > 1 {
            2.0 * edge_count as f64 / (node_count * (node_count - 1)) as f64
        } else {
            0.0
        };
        println!(
            "     t={}: {} nodes, {} edges, density: {:.3}",
            time, node_count, edge_count, density
        );
    }

    println!("\n12. Temporal edge overlap analysis:");

    let temporal_edges = tgraph.temporal_edges();
    println!("   Total temporal edges: {}", temporal_edges.len());

    let mut overlapping_pairs = 0;
    for i in 0..temporal_edges.len() {
        for j in (i + 1)..temporal_edges.len() {
            if temporal_edges[i]
                .interval
                .overlaps(&temporal_edges[j].interval)
            {
                overlapping_pairs += 1;
            }
        }
    }
    println!("   Overlapping edge pairs: {}", overlapping_pairs);

    println!("\n=== Demo Complete ===");
    println!("\nTemporal graphs enable modeling of:");
    println!("- Time-dependent relationships");
    println!("- Evolution of network structure");
    println!("- Temporal reachability and paths");
    println!("- Dynamic centrality measures");
    println!("- Event-driven network analysis");

    Ok(())
}
