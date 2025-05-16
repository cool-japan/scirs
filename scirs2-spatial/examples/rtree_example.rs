use ndarray::array;
use scirs2_spatial::rtree::RTree;

fn main() {
    // Create a new 2D R-tree with min entries 2 and max entries 5
    let mut rtree: RTree<String> = RTree::new(2, 2, 5).unwrap();

    // Define some city locations (longitude, latitude) and names
    let cities = vec![
        (array![13.4050, 52.5200], "Berlin".to_string()), // Berlin
        (array![2.3522, 48.8566], "Paris".to_string()),   // Paris
        (array![-0.1278, 51.5074], "London".to_string()), // London
        (array![12.4964, 41.9028], "Rome".to_string()),   // Rome
        (array![4.9041, 52.3676], "Amsterdam".to_string()), // Amsterdam
        (array![18.0686, 59.3293], "Stockholm".to_string()), // Stockholm
        (array![21.0122, 52.2297], "Warsaw".to_string()), // Warsaw
        (array![9.1900, 45.4642], "Milan".to_string()),   // Milan
        (array![16.3738, 48.2082], "Vienna".to_string()), // Vienna
        (array![14.4378, 50.0755], "Prague".to_string()), // Prague
    ];

    // Insert cities into the R-tree
    println!("Inserting cities into R-tree...");
    for (location, name) in cities {
        rtree.insert(location, name).unwrap();
    }

    println!("R-tree size: {}", rtree.size());

    // Search for cities within a geographic bounding box (roughly central Europe)
    println!("\nCities in Central Europe (longitude 5-20, latitude 45-55):");
    let central_europe_min = array![5.0, 45.0];
    let central_europe_max = array![20.0, 55.0];

    let results = rtree
        .search_range(&central_europe_min.view(), &central_europe_max.view())
        .unwrap();

    for (_idx, city) in results {
        println!("- {}", city);
    }

    // Find the 3 nearest cities to Vienna
    println!("\nFinding the 3 nearest cities to Vienna:");
    let vienna_coords = array![16.3738, 48.2082];

    let nearest_results = rtree.nearest(&vienna_coords.view(), 3).unwrap();

    for (_idx, city, distance) in nearest_results {
        println!("- {} (distance: {:.2})", city, distance);
    }

    // Find the 3 nearest cities to Paris
    println!("\nFinding the 3 nearest cities to Paris:");
    let paris_coords = array![2.3522, 48.8566];

    let nearest_results = rtree.nearest(&paris_coords.view(), 3).unwrap();

    for (_idx, city, distance) in nearest_results {
        println!("- {} (distance: {:.2})", city, distance);
    }

    // Delete a city from the R-tree
    println!("\nDeleting London from the R-tree...");
    let london_coords = array![-0.1278, 51.5074];
    let deleted = rtree
        .delete::<fn(&String) -> bool>(&london_coords.view(), None)
        .unwrap();

    if deleted {
        println!("London successfully deleted.");
        println!("R-tree size: {}", rtree.size());
    } else {
        println!("Failed to delete London.");
    }

    // Show nearest cities to London again (London should no longer appear)
    println!("\nFinding the 3 nearest cities to London's location after deletion:");

    let nearest_results = rtree.nearest(&london_coords.view(), 3).unwrap();

    for (_idx, city, distance) in nearest_results {
        println!("- {} (distance: {:.2})", city, distance);
    }
}
