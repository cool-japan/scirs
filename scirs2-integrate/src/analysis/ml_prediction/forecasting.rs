//! Time Series Forecasting for Bifurcation Prediction
//!
//! This module contains structures and configurations for time series forecasting,
//! anomaly detection, and trend analysis in dynamical systems.

/// Time series forecasting for bifurcation prediction
#[derive(Debug, Clone)]
pub struct TimeSeriesBifurcationForecaster {
    /// Base time series model
    pub base_model: TimeSeriesModel,
    /// Bifurcation detection threshold
    pub detection_threshold: f64,
    /// Forecast horizon
    pub forecast_horizon: usize,
    /// Multi-step forecasting strategy
    pub multistep_strategy: MultiStepStrategy,
    /// Anomaly detection configuration
    pub anomaly_detection: AnomalyDetectionConfig,
    /// Trend analysis configuration
    pub trend_analysis: TrendAnalysisConfig,
}

/// Time series model types
#[derive(Debug, Clone)]
pub enum TimeSeriesModel {
    /// LSTM-based model
    LSTM {
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
    },
    /// GRU-based model
    GRU {
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
    },
    /// Transformer-based model
    Transformer {
        d_model: usize,
        nhead: usize,
        num_layers: usize,
        positional_encoding: bool,
    },
    /// Conv1D-based model
    Conv1D {
        channels: Vec<usize>,
        kernel_sizes: Vec<usize>,
        dilations: Vec<usize>,
    },
    /// Hybrid CNN-RNN model
    HybridCNNRNN {
        cnn_channels: Vec<usize>,
        rnn_hidden_size: usize,
        rnn_layers: usize,
    },
}

/// Multi-step forecasting strategies
#[derive(Debug, Clone, Copy)]
pub enum MultiStepStrategy {
    /// Recursive one-step ahead
    Recursive,
    /// Direct multi-step
    Direct,
    /// Multi-input multi-output
    MIMO,
    /// Ensemble of strategies
    Ensemble,
}

/// Anomaly detection configuration
#[derive(Debug, Clone)]
pub struct AnomalyDetectionConfig {
    /// Anomaly detection method
    pub method: AnomalyDetectionMethod,
    /// Threshold for anomaly detection
    pub threshold: f64,
    /// Window size for anomaly detection
    pub window_size: usize,
    /// Minimum anomaly duration
    pub min_duration: usize,
}

/// Anomaly detection methods
#[derive(Debug, Clone, Copy)]
pub enum AnomalyDetectionMethod {
    /// Statistical outlier detection
    StatisticalOutlier,
    /// Isolation forest
    IsolationForest,
    /// One-class SVM
    OneClassSVM,
    /// Autoencoder-based detection
    Autoencoder,
    /// LSTM-based prediction error
    LSTMPredictionError,
}

/// Trend analysis configuration
#[derive(Debug, Clone)]
pub struct TrendAnalysisConfig {
    /// Trend detection method
    pub method: TrendDetectionMethod,
    /// Trend analysis window size
    pub window_size: usize,
    /// Significance level for trend tests
    pub significance_level: f64,
    /// Change point detection
    pub change_point_detection: bool,
}

/// Trend detection methods
#[derive(Debug, Clone, Copy)]
pub enum TrendDetectionMethod {
    /// Linear regression slope
    LinearRegression,
    /// Mann-Kendall test
    MannKendall,
    /// Sen's slope estimator
    SensSlope,
    /// Seasonal Mann-Kendall
    SeasonalMannKendall,
    /// CUSUM test
    CUSUM,
}
