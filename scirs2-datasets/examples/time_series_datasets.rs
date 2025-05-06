use ndarray::s;
use scirs2_datasets::time_series::electrocardiogram;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading time series datasets...\n");

    // Load the electrocardiogram dataset
    let ecg = electrocardiogram()?;

    println!("Electrocardiogram dataset:");
    println!("  Time steps: {}", ecg.n_samples());
    println!("  Features: {}", ecg.n_features());
    println!(
        "  Sampling rate: {} Hz",
        ecg.metadata
            .get("sampling_rate")
            .unwrap_or(&"unknown".to_string())
    );
    println!(
        "  Duration: {}",
        ecg.metadata
            .get("duration")
            .unwrap_or(&"unknown".to_string())
    );

    // Get a slice of the data and display basic statistics
    let ecg_slice = ecg.data.slice(s![0..10, 0]);
    println!("  First 10 data points: {:?}", ecg_slice);

    // Calculate some basic statistics
    let ecg_data = ecg.data.column(0);
    let min = ecg_data.fold(f64::INFINITY, |a, &b| a.min(b));
    let max = ecg_data.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mean = ecg_data.sum() / ecg_data.len() as f64;

    println!("  Min: {:.3} mV", min);
    println!("  Max: {:.3} mV", max);
    println!("  Mean: {:.3} mV", mean);

    // Note: Stock market and weather datasets are commented out because their source data
    // is not yet available.

    /*
    // Load the stock market dataset
    println!("\nStock market dataset:");

    // Get price changes (returns)
    let stock_returns = stock_market(true)?;
    println!("  Time steps: {}", stock_returns.n_samples());
    println!("  Companies: {}", stock_returns.n_features());

    // Print companies
    if let Some(feature_names) = &stock_returns.feature_names {
        println!("  Companies: {}", feature_names.join(", "));
    }

    // Load the weather dataset
    println!("\nWeather dataset:");
    let temp_data = weather(Some("temperature"))?;

    println!("  Time steps: {}", temp_data.n_samples());
    println!("  Locations: {}", temp_data.n_features());
    */

    println!("\nTime series dataset loaded successfully!");

    Ok(())
}
