use scirs2_datasets::loaders;
use scirs2_datasets::utils::{train_test_split, Dataset};
use std::env;
use std::path::Path;

fn main() {
    // Check if a CSV file is provided as a command-line argument
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <path_to_csv_file>", args[0]);
        println!("Example: {} examples/sample_data.csv", args[0]);
        return;
    }

    let file_path = &args[1];

    // Verify the file exists
    if !Path::new(file_path).exists() {
        println!("Error: File '{}' does not exist", file_path);
        return;
    }

    // Load CSV file
    println!("Loading CSV file: {}", file_path);
    match loaders::load_csv(file_path, true, None) {
        Ok(dataset) => {
            print_dataset_info(&dataset, "Loaded CSV");

            // Split the dataset for demonstration
            println!("\nDemonstrating train-test split...");
            match train_test_split(&dataset, 0.2, Some(42)) {
                Ok((train, test)) => {
                    println!("Training set: {} samples", train.n_samples());
                    println!("Test set: {} samples", test.n_samples());

                    // Save as JSON for demonstration
                    let json_path = format!("{}.json", file_path);
                    println!("\nSaving training dataset to JSON: {}", json_path);
                    if let Err(e) = loaders::save_json(&train, &json_path) {
                        println!("Error saving JSON: {}", e);
                    } else {
                        println!("Successfully saved JSON file");

                        // Load back the JSON file
                        println!("\nLoading back from JSON file...");
                        match loaders::load_json(&json_path) {
                            Ok(loaded) => {
                                print_dataset_info(&loaded, "Loaded JSON");
                            }
                            Err(e) => println!("Error loading JSON: {}", e),
                        }
                    }
                }
                Err(e) => println!("Error splitting dataset: {}", e),
            }
        }
        Err(e) => println!("Error loading CSV: {}", e),
    }
}

fn print_dataset_info(dataset: &Dataset, name: &str) {
    println!("=== {} Dataset ===", name);
    println!("Number of samples: {}", dataset.n_samples());
    println!("Number of features: {}", dataset.n_features());

    if let Some(feature_names) = &dataset.feature_names {
        println!(
            "Features: {:?}",
            &feature_names[0..std::cmp::min(5, feature_names.len())]
        );
        if feature_names.len() > 5 {
            println!("... and {} more", feature_names.len() - 5);
        }
    }

    if let Some(target) = &dataset.target {
        println!("Target shape: {}", target.len());

        if let Some(target_names) = &dataset.target_names {
            println!("Target classes: {:?}", target_names);
        }
    }

    for (key, value) in &dataset.metadata {
        println!("Metadata - {}: {}", key, value);
    }
}
