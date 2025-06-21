use ndarray::array;
use scirs2_series::forecasting::{auto_arima, auto_arima_with_options, AutoArimaOptions};

#[test]
fn test_auto_arima_basic() {
    // Create a simple time series
    let ts = array![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0
    ];

    // Test the basic auto_arima function
    let params = auto_arima(&ts, 2, 1, 2, false, None).unwrap();

    // Verify that parameters are within expected ranges
    assert!(params.p <= 2);
    assert!(params.d <= 1);
    assert!(params.q <= 2);
    assert!(params.seasonal_p.is_none());
    assert!(params.seasonal_d.is_none());
    assert!(params.seasonal_q.is_none());
    assert!(params.seasonal_period.is_none());
}

#[test]
fn test_auto_arima_with_options() {
    // Create a time series with a trend
    let ts = array![
        1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5,
        10.0, 10.5
    ];

    // Configure auto ARIMA options
    let options = AutoArimaOptions {
        max_p: 3,
        max_q: 3,
        information_criterion: "bic".to_string(),
        auto_diff: true,
        ..Default::default()
    };

    // Test the advanced auto_arima_with_options function
    let params = auto_arima_with_options(&ts, &options).unwrap();

    // Verify that parameters are within expected ranges
    assert!(params.p <= 3);
    assert!(params.q <= 3);
    // For a trending series, the model should either:
    // - Use differencing (d > 0)
    // - Use autoregressive terms (p > 0)
    // - Or possibly use a constant term (fit_intercept = true)
    assert!(params.d > 0 || params.p > 0 || params.fit_intercept);
}

#[test]
fn test_auto_arima_seasonal() {
    // Create a seasonal time series
    let ts = array![
        1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5, 2.0, 3.0,
        4.0, 3.0, 2.0, 3.0, 4.0, 3.0
    ];

    // Test auto_arima with seasonal components
    let params = auto_arima(&ts, 1, 1, 1, true, Some(4)).unwrap();

    // Verify that seasonal parameters are set
    assert!(params.seasonal_p.is_some());
    assert!(params.seasonal_d.is_some());
    assert!(params.seasonal_q.is_some());
    assert_eq!(params.seasonal_period, Some(4));
}

#[test]
fn test_auto_arima_with_seasonal_options() {
    // Create a seasonal time series with trend
    let ts = array![
        1.0, 2.0, 3.0, 2.0, 1.5, 2.5, 3.5, 2.5, 2.0, 3.0, 4.0, 3.0, 2.5, 3.5, 4.5, 3.5, 3.0, 4.0,
        5.0, 4.0, 3.5, 4.5, 5.5, 4.5
    ];

    // Configure auto ARIMA options with seasonality
    let options = AutoArimaOptions {
        max_p: 2,
        max_q: 2,
        seasonal: true,
        seasonal_period: Some(4),
        max_seasonal_p: 1,
        max_seasonal_q: 1,
        information_criterion: "aic".to_string(),
        ..Default::default()
    };

    // Test the advanced auto_arima_with_options function with seasonality
    let params = auto_arima_with_options(&ts, &options).unwrap();

    // Verify that parameters are correctly set
    assert!(params.p <= 2);
    assert!(params.q <= 2);
    assert!(params.seasonal_p.unwrap() <= 1);
    assert!(params.seasonal_q.unwrap() <= 1);
    assert_eq!(params.seasonal_period, Some(4));
}
