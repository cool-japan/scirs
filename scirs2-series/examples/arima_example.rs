//! Example of using ARIMA models for time series analysis

use scirs2_core::ndarray::array;
use scirs2_series::arima_models::{
    auto_arima, ArimaModel, ArimaSelectionOptions, SelectionCriterion,
};

#[allow(dead_code)]
fn main() {
    // Example 1: Manual ARIMA fitting
    println!("Example 1: Manual ARIMA(1,1,1) model");
    let data = array![
        100.0, 102.5, 103.8, 106.2, 108.1, 110.5, 113.2, 115.8, 118.1, 120.7, 123.5, 126.2, 129.1,
        132.5, 135.8, 139.2, 142.1, 145.5, 148.8, 152.2
    ];

    let mut arima_model = ArimaModel::new(1, 1, 1).expect("Failed to create ARIMA model");
    arima_model.fit(&data).expect("Failed to fit ARIMA model");

    println!("Model fitted successfully!");
    println!("AIC: {:.2}", arima_model.aic());
    println!("BIC: {:.2}", arima_model.bic());

    // Forecast 5 steps ahead
    let forecasts = arima_model.forecast(5, &data).expect("Failed to forecast");
    println!("Forecasts: {:?}", forecasts);

    // Example 2: Automatic ARIMA model selection
    println!("\nExample 2: Automatic ARIMA model selection");
    let options = ArimaSelectionOptions {
        max_p: 3,
        max_d: 2,
        max_q: 3,
        criterion: SelectionCriterion::AIC,
        stepwise: true,
        ..Default::default()
    };

    let (best_model, params) = auto_arima(&data, &options).expect("Failed to select model");
    println!(
        "Best model: ARIMA({},{},{})",
        params.pdq.0, params.pdq.1, params.pdq.2
    );
    println!("AIC: {:.2}", best_model.aic());

    // Example 3: Seasonal ARIMA
    println!("\nExample 3: Seasonal ARIMA model");
    let seasonal_data = array![
        100.0, 95.0, 90.0, 110.0, 120.0, 115.0, 105.0, 125.0, 130.0, 125.0, 115.0, 135.0, 140.0,
        135.0, 125.0, 145.0, 150.0, 145.0, 135.0, 155.0, 160.0, 155.0, 145.0, 165.0
    ];

    let seasonal_options = ArimaSelectionOptions {
        max_p: 2,
        max_d: 1,
        max_q: 2,
        seasonal: true,
        seasonal_period: Some(4),
        max_seasonal_p: 1,
        max_seasonal_d: 1,
        max_seasonal_q: 1,
        criterion: SelectionCriterion::BIC,
        ..Default::default()
    };

    let (_seasonal_model, seasonal_params) =
        auto_arima(&seasonal_data, &seasonal_options).expect("Failed to select seasonal model");

    println!(
        "Best seasonal model: ARIMA({},{},{})x({},{},{}){}",
        seasonal_params.pdq.0,
        seasonal_params.pdq.1,
        seasonal_params.pdq.2,
        seasonal_params.seasonal_pdq.0,
        seasonal_params.seasonal_pdq.1,
        seasonal_params.seasonal_pdq.2,
        seasonal_params.seasonal_period
    );
}
