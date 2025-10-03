//! Financial time series analysis demonstration
//!
//! This example showcases specialized financial time series analysis capabilities,
//! including GARCH volatility modeling, technical indicators, and risk metrics.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_series::{
    correlation::CorrelationAnalyzer,
    // features::{FeatureConfig, FeatureExtractor}, // TODO: Fix when needed
    financial::{
        models::{Distribution, MeanModel},
        GarchConfig, GarchModel,
    },
    transformations::NormalizationMethod,
};
use statrs::statistics::Statistics;
use std::time::Instant;

// TODO: Fix imports and re-enable this example
/*
#[allow(dead_code)]
fn main() {
    println!("=== Financial Time Series Analysis Demo ===\n");

    // Generate synthetic financial data
    let (prices, returns) = generate_financial_data();
    println!("Generated {} price observations", prices.len());

    // Demo 1: Basic financial transformations
    println!("\n1. Financial Data Transformations");
    financial_transformations_demo(&prices);

    // Demo 2: GARCH volatility modeling
    println!("\n2. GARCH Volatility Modeling");
    garch_modeling_demo(&returns);

    // Demo 3: Technical indicators
    println!("\n3. Technical Indicators");
    technical_indicators_demo(&prices);

    // Demo 4: Risk metrics calculation
    println!("\n4. Risk Metrics");
    risk_metrics_demo(&returns);

    // Demo 5: Portfolio analysis
    println!("\n5. Portfolio Analysis");
    portfolio_analysis_demo();

    // Demo 6: Value at Risk (VaR) calculation
    println!("\n6. Value at Risk Analysis");
    var_analysis_demo(&returns);

    // Demo 7: Backtesting strategies
    println!("\n7. Strategy Backtesting");
    backtesting_demo(&prices);

    // Demo 8: Advanced financial features
    println!("\n8. Advanced Financial Features");
    advanced_features_demo(&prices, &returns);

    println!("\n=== Financial Analysis Complete ===");
}

#[allow(dead_code)]
fn generate_financial_data() -> (Array1<f64>, Array1<f64>) {
    let n = 1000;
    let mut prices = Array1::zeros(n);
    let mut returns = Array1::zeros(n - 1);

    // Initialize
    prices[0] = 100.0;

    // Generate price series with volatility clustering
    let mut volatility = 0.02;

    for i in 1..n {
        let t = i as f64;

        // Time-varying volatility (GARCH-like)
        let vol_innovation = 0.001 * (t * 0.01).sin();
        volatility = 0.9 * volatility + 0.1 * 0.02 + vol_innovation.abs();

        // Generate return with fat tails
        let z = generate_student_t(3.0, i as u64); // t-distribution with 3 df
        let return_val = volatility * z;

        // Add market trend and mean reversion
        let trend = -0.0001 * (prices[i - 1] - 100.0) / 100.0; // Mean reversion to 100
        let market_trend = 0.0002 * (1.0 + (t / 200.0).sin()); // Cyclical market trend

        let total_return = return_val + trend + market_trend;
        returns[i - 1] = total_return;
        prices[i] = prices[i - 1] * (1.0 + total_return);
    }

    (prices, returns)
}

#[allow(dead_code)]
fn generate_student_t(df: f64, seed: u64) -> f64 {
    // Simple approximation to t-distribution using Box-Muller + scaling
    let u1 = ((seed * 7919 + 1013) % 10000) as f64 / 10000.0;
    let u2 = ((seed * 7907 + 1019) % 10000) as f64 / 10000.0;

    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

    // Approximate scaling for t-distribution
    let scale_factor = if _df > 2.0 {
        (_df / (_df - 2.0)).sqrt()
    } else {
        2.0
    };
    z * scale_factor
}

#[allow(dead_code)]
fn financial_transformations_demo(prices: &Array1<f64>) {
    println!("  Applying financial transformations...");

    // Log transformation
    let mut log_transformer = LogTransformer::new();
    match log_transformer.fit_transform(_prices) {
        Ok(log_prices) => {
            println!(
                "    Log transformation: {} -> {} (range: {:.4} to {:.4})",
                prices.len(),
                log_prices.len(),
                log_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                log_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            );
        }
        Err(e) => println!("    Log transformation failed: {}", e),
    }

    // Returns transformation
    let mut returns_transformer = ReturnsTransformer::new(false); // Simple returns
    match returns_transformer.fit_transform(_prices) {
        Ok(simple_returns) => {
            let mean_return = simple_returns.mean().unwrap_or(0.0);
            let return_volatility = simple_returns.variance().sqrt();
            println!(
                "    Simple returns: mean={:.4}, volatility={:.4}",
                mean_return, return_volatility
            );
        }
        Err(e) => println!("    Returns transformation failed: {}", e),
    }

    // Log returns
    let mut log_returns_transformer = ReturnsTransformer::new(true); // Log returns
    match log_returns_transformer.fit_transform(_prices) {
        Ok(log_returns) => {
            let mean_log_return = log_returns.mean().unwrap_or(0.0);
            let log_return_volatility = log_returns.variance().sqrt();
            println!(
                "    Log returns: mean={:.4}, volatility={:.4}",
                mean_log_return, log_return_volatility
            );

            // Annualized statistics (assuming daily data)
            let annualized_return = mean_log_return * 252.0;
            let annualized_volatility = log_return_volatility * (252.0_f64).sqrt();
            println!(
                "    Annualized: return={:.2}%, volatility={:.2}%",
                annualized_return * 100.0,
                annualized_volatility * 100.0
            );
        }
        Err(e) => println!("    Log returns transformation failed: {}", e),
    }
}

#[allow(dead_code)]
fn garch_modeling_demo(returns: &Array1<f64>) {
    println!("  GARCH volatility modeling...");

    // Test different GARCH configurations
    let configs = vec![
        (
            "GARCH(1,1)",
            GarchConfig {
                p: 1,
                q: 1,
                mean_model: MeanModel::Constant,
                distribution: Distribution::Normal,
                max_iterations: 500,
                tolerance: 1e-6,
                use_numerical_derivatives: false,
            },
        ),
        (
            "GARCH(1,1) with t-dist",
            GarchConfig {
                p: 1,
                q: 1,
                mean_model: MeanModel::Constant,
                distribution: Distribution::StudentT,
                max_iterations: 500,
                tolerance: 1e-6,
                use_numerical_derivatives: false,
            },
        ),
        (
            "GARCH(2,1) with AR mean",
            GarchConfig {
                p: 2,
                q: 1,
                mean_model: MeanModel::AR { order: 1 },
                distribution: Distribution::Normal,
                max_iterations: 500,
                tolerance: 1e-6,
                use_numerical_derivatives: false,
            },
        ),
    ];

    for (name, config) in configs {
        println!("    Fitting {} model...", name);
        let start_time = Instant::now();

        let mut garch_model = GarchModel::new(config);
        match garch_model.fit(_returns) {
            Ok(result) => {
                let duration = start_time.elapsed();
                println!(
                    "      Fitted in {:.2}ms, {} iterations",
                    duration.as_millis(),
                    result.iterations
                );
                println!("      Converged: {}", result.converged);
                println!("      Log-likelihood: {:.2}", result.log_likelihood);
                println!("      AIC: {:.2}, BIC: {:.2}", result.aic, result.bic);

                // Show parameter estimates
                let params = &result.parameters;
                println!(
                    "      Parameters: omega={:.6}, alpha={:.4}, beta={:.4}",
                    params.garch_params.get(0).unwrap_or(&0.0), // omega
                    params.garch_params.get(1).unwrap_or(&0.0), // alpha
                    params.garch_params.get(2).unwrap_or(&0.0)  // beta
                );

                // Volatility forecasting
                match garch_model.forecast_variance(10) {
                    Ok(vol_forecast) => {
                        println!(
                            "      10-step volatility forecast: {:.4} to {:.4}",
                            vol_forecast.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                            vol_forecast
                                .iter()
                                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                        );
                    }
                    Err(e) => println!("      Volatility forecasting failed: {}", e),
                }

                // Model diagnostics
                let standardized_residuals = &result.standardized_residuals;
                let mean_residual = standardized_residuals.mean().unwrap_or(0.0);
                let residual_std = standardized_residuals.variance().sqrt();
                println!(
                    "      Residual diagnostics: mean={:.4}, std={:.4}",
                    mean_residual, residual_std
                );
            }
            Err(e) => {
                println!("      Model fitting failed: {}", e);
            }
        }
    }
}

#[allow(dead_code)]
fn technical_indicators_demo(prices: &Array1<f64>) {
    println!("  Computing technical indicators...");

    let indicators = TechnicalIndicators::new();

    // Moving averages
    if let Ok(sma_20) = indicators.simple_moving_average(_prices, 20) {
        if let Ok(ema_20) = indicators.exponential_moving_average(_prices, 20) {
            println!(
                "    Moving averages (20-period): SMA={:.2}, EMA={:.2}",
                sma_20.last().unwrap_or(&0.0),
                ema_20.last().unwrap_or(&0.0)
            );
        }
    }

    // Bollinger Bands
    match indicators.bollinger_bands(_prices, 20, 2.0) {
        Ok((upper, middle, lower)) => {
            let current_price = prices.last().unwrap_or(&0.0);
            let upper_val = upper.last().unwrap_or(&0.0);
            let lower_val = lower.last().unwrap_or(&0.0);
            let middle_val = middle.last().unwrap_or(&0.0);

            println!(
                "    Bollinger Bands: price={:.2}, upper={:.2}, middle={:.2}, lower={:.2}",
                current_price, upper_val, middle_val, lower_val
            );

            let bb_position = (current_price - lower_val) / (upper_val - lower_val);
            println!(
                "      Band position: {:.1}% (0%=lower, 100%=upper)",
                bb_position * 100.0
            );
        }
        Err(e) => println!("    Bollinger Bands calculation failed: {}", e),
    }

    // RSI (Relative Strength Index)
    match indicators.relative_strength_index(_prices, 14) {
        Ok(rsi) => {
            let current_rsi = rsi.last().unwrap_or(&50.0);
            println!("    RSI (14): {:.1}", current_rsi);

            let signal = if *current_rsi > 70.0 {
                "Overbought"
            } else if *current_rsi < 30.0 {
                "Oversold"
            } else {
                "Neutral"
            };
            println!("      Signal: {}", signal);
        }
        Err(e) => println!("    RSI calculation failed: {}", e),
    }

    // MACD
    match indicators.macd(_prices, 12, 26, 9) {
        Ok((macd_line, signal_line, histogram)) => {
            let macd_val = macd_line.last().unwrap_or(&0.0);
            let signal_val = signal_line.last().unwrap_or(&0.0);
            let hist_val = histogram.last().unwrap_or(&0.0);

            println!(
                "    MACD: line={:.4}, signal={:.4}, histogram={:.4}",
                macd_val, signal_val, hist_val
            );

            let trend_signal = if *hist_val > 0.0 {
                "Bullish"
            } else {
                "Bearish"
            };
            println!("      Trend: {}", trend_signal);
        }
        Err(e) => println!("    MACD calculation failed: {}", e),
    }

    // Average True Range (ATR)
    if let Ok(atr) = indicators.average_true_range(_prices, prices, prices, 14) {
        // Simplified: using same array
        let current_atr = atr.last().unwrap_or(&0.0);
        let current_price = prices.last().unwrap_or(&0.0);
        let atr_percentage = (current_atr / current_price) * 100.0;
        println!(
            "    ATR (14): {:.4} ({:.2}% of price)",
            current_atr, atr_percentage
        );
    }
}

#[allow(dead_code)]
fn risk_metrics_demo(returns: &Array1<f64>) {
    println!("  Computing risk metrics...");

    let risk_calc = RiskMetrics::new();

    // Basic risk metrics
    let mean_return = returns.mean().unwrap_or(0.0);
    let volatility = returns.variance().sqrt();
    let min_return = returns.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_return = returns.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("    Return statistics:");
    println!(
        "      Mean: {:.4} ({:.2}% annualized)",
        mean_return,
        mean_return * 252.0 * 100.0
    );
    println!(
        "      Volatility: {:.4} ({:.2}% annualized)",
        volatility,
        volatility * (252.0_f64).sqrt() * 100.0
    );
    println!("      Min/Max: {:.4} / {:.4}", min_return, max_return);

    // Sharpe ratio (assuming risk-free rate of 2%)
    let risk_free_rate = 0.02 / 252.0; // Daily risk-free rate
    let excess_returns: Array1<f64> = returns.mapv(|r| r - risk_free_rate);
    let sharpe_ratio = excess_returns.mean().unwrap_or(0.0) / excess_returns.variance().sqrt();
    let annualized_sharpe = sharpe_ratio * (252.0_f64).sqrt();
    println!("    Sharpe ratio: {:.2} (annualized)", annualized_sharpe);

    // Downside deviation and Sortino ratio
    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

    if !downside_returns.is_empty() {
        let downside_deviation = (downside_returns.iter().map(|r| r * r).sum::<f64>()
            / downside_returns.len() as f64)
            .sqrt();
        let sortino_ratio = (mean_return - risk_free_rate) / downside_deviation;
        let annualized_sortino = sortino_ratio * (252.0_f64).sqrt();

        println!("    Downside deviation: {:.4}", downside_deviation);
        println!("    Sortino ratio: {:.2} (annualized)", annualized_sortino);
    }

    // Maximum drawdown
    match risk_calc.maximum_drawdown(_returns) {
        Ok(max_dd) => {
            println!("    Maximum drawdown: {:.2}%", max_dd * 100.0);
        }
        Err(e) => println!("    Maximum drawdown calculation failed: {}", e),
    }

    // Skewness and kurtosis
    match risk_calc.skewness(_returns) {
        Ok(skew) => {
            println!(
                "    Skewness: {:.3} ({})",
                skew,
                if skew > 0.0 {
                    "right-tailed"
                } else {
                    "left-tailed"
                }
            );
        }
        Err(e) => println!("    Skewness calculation failed: {}", e),
    }

    match risk_calc.kurtosis(_returns) {
        Ok(kurt) => {
            println!(
                "    Excess kurtosis: {:.3} ({})",
                kurt,
                if kurt > 0.0 {
                    "fat tails"
                } else {
                    "thin tails"
                }
            );
        }
        Err(e) => println!("    Kurtosis calculation failed: {}", e),
    }
}

#[allow(dead_code)]
fn portfolio_analysis_demo() {
    println!("  Portfolio analysis with multiple assets...");

    // Generate correlated asset returns
    let n = 252; // One year of daily data
    let n_assets = 3;
    let mut asset_returns = Array2::zeros((n, n_assets));

    // Correlation structure
    let correlations = vec![
        vec![1.0, 0.6, 0.3],
        vec![0.6, 1.0, 0.4],
        vec![0.3, 0.4, 1.0],
    ];

    // Generate correlated returns using Cholesky decomposition (simplified)
    for i in 0..n {
        let base_factors = vec![
            generate_student_t(4.0, (i * 7 + 1) as u64),
            generate_student_t(4.0, (i * 11 + 3) as u64),
            generate_student_t(4.0, (i * 13 + 7) as u64),
        ];

        // Asset 1: Large cap equity
        asset_returns[[i, 0]] = 0.0008 + 0.015 * base_factors[0];

        // Asset 2: Small cap equity (higher vol, correlated with asset 1)
        asset_returns[[i, 1]] = 0.0006 + 0.020 * (0.6 * base_factors[0] + 0.8 * base_factors[1]);

        // Asset 3: Bonds (lower vol, low correlation)
        asset_returns[[i, 2]] = 0.0003 + 0.005 * (0.3 * base_factors[0] + 0.95 * base_factors[2]);
    }

    println!(
        "    Generated returns for {} assets over {} days",
        n_assets, n
    );

    // Portfolio weights
    let weights = vec![0.6, 0.3, 0.1]; // 60% large cap, 30% small cap, 10% bonds

    let analyzer = PortfolioAnalyzer::new();

    // Calculate portfolio returns
    match analyzer.portfolio_returns(&asset_returns, &weights) {
        Ok(portfolio_returns) => {
            let portfolio_mean = portfolio_returns.mean().unwrap_or(0.0);
            let portfolio_vol = portfolio_returns.variance().sqrt();

            println!("    Portfolio statistics:");
            println!(
                "      Mean return: {:.4} ({:.2}% annualized)",
                portfolio_mean,
                portfolio_mean * 252.0 * 100.0
            );
            println!(
                "      Volatility: {:.4} ({:.2}% annualized)",
                portfolio_vol,
                portfolio_vol * (252.0_f64).sqrt() * 100.0
            );

            // Calculate individual asset statistics
            for i in 0..n_assets {
                let asset_column = asset_returns.column(i);
                let asset_mean = asset_column.mean().unwrap_or(0.0);
                let asset_vol = asset_column.variance().sqrt();
                println!(
                    "      Asset {}: return={:.4}, vol={:.4}, weight={:.1}%",
                    i + 1,
                    asset_mean * 252.0 * 100.0,
                    asset_vol * (252.0_f64).sqrt() * 100.0,
                    weights[i] * 100.0
                );
            }

            // Portfolio efficiency metrics
            let risk_free_rate = 0.02 / 252.0;
            let portfolio_sharpe =
                (portfolio_mean - risk_free_rate) / portfolio_vol * (252.0_f64).sqrt();
            println!("    Portfolio Sharpe ratio: {:.2}", portfolio_sharpe);
        }
        Err(e) => println!("    Portfolio return calculation failed: {}", e),
    }

    // Correlation analysis
    let corr_analyzer = CorrelationAnalyzer::new();
    match corr_analyzer.autocorrelation(&asset_returns.slice(s![.., 0]).to_owned(), 10) {
        Ok(corr_matrix) => {
            println!("    Asset correlation matrix:");
            for i in 0..n_assets {
                print!("      ");
                for j in 0..n_assets {
                    print!("{:6.3} ", corr_matrix[[i, j]]);
                }
                println!();
            }
        }
        Err(e) => println!("    Correlation matrix calculation failed: {}", e),
    }
}

#[allow(dead_code)]
fn var_analysis_demo(returns: &Array1<f64>) {
    println!("  Value at Risk (VaR) analysis...");

    let var_calc = VaRCalculator::new();
    let confidence_levels = vec![0.95, 0.99, 0.995];

    for &confidence in &confidence_levels {
        println!("    {}% confidence level:", confidence * 100.0);

        // Historical VaR
        match var_calc.historical_var(_returns, confidence) {
            Ok(hist_var) => {
                println!(
                    "      Historical VaR: {:.4} ({:.2}%)",
                    hist_var,
                    hist_var * 100.0
                );
            }
            Err(e) => println!("      Historical VaR failed: {}", e),
        }

        // Parametric VaR (assuming normal distribution)
        match var_calc.parametric_var(_returns, confidence) {
            Ok(param_var) => {
                println!(
                    "      Parametric VaR: {:.4} ({:.2}%)",
                    param_var,
                    param_var * 100.0
                );
            }
            Err(e) => println!("      Parametric VaR failed: {}", e),
        }

        // Monte Carlo VaR
        match var_calc.monte_carlo_var(_returns, confidence, 10000) {
            Ok(mc_var) => {
                println!(
                    "      Monte Carlo VaR: {:.4} ({:.2}%)",
                    mc_var,
                    mc_var * 100.0
                );
            }
            Err(e) => println!("      Monte Carlo VaR failed: {}", e),
        }

        // Expected Shortfall (Conditional VaR)
        match var_calc.expected_shortfall(_returns, confidence) {
            Ok(es) => {
                println!("      Expected Shortfall: {:.4} ({:.2}%)", es, es * 100.0);
            }
            Err(e) => println!("      Expected Shortfall failed: {}", e),
        }
    }

    // VaR backtesting
    println!("    VaR backtesting (95% confidence):");
    match var_calc.historical_var(_returns, 0.95) {
        Ok(var_95) => {
            let violations: usize = returns.iter().filter(|&&r| r < var_95).count();
            let violation_rate = violations as f64 / returns.len() as f64;
            let expected_rate = 0.05;

            println!(
                "      Violations: {} out of {} ({:.2}%)",
                violations,
                returns.len(),
                violation_rate * 100.0
            );
            println!("      Expected rate: {:.2}%", expected_rate * 100.0);

            let test_result = if (violation_rate - expected_rate).abs() < 0.02 {
                "PASS"
            } else {
                "FAIL"
            };
            println!("      Backtesting result: {}", test_result);
        }
        Err(e) => println!("      VaR backtesting failed: {}", e),
    }
}

#[allow(dead_code)]
fn backtesting_demo(prices: &Array1<f64>) {
    println!("  Strategy backtesting...");

    let engine = BacktestEngine::new();

    // Simple moving average crossover strategy
    let sma_short = 10;
    let sma_long = 30;

    let strategy = TradingStrategy::MovingAverageCrossover {
        short_window: sma_short,
        long_window: sma_long,
    };

    match engine.backtest(_prices, &strategy) {
        Ok(results) => {
            println!(
                "    Moving Average Crossover ({}/{}) Strategy:",
                sma_short, sma_long
            );
            println!("      Total return: {:.2}%", results.total_return * 100.0);
            println!(
                "      Annualized return: {:.2}%",
                results.annualized_return * 100.0
            );
            println!("      Volatility: {:.2}%", results.volatility * 100.0);
            println!("      Sharpe ratio: {:.2}", results.sharpe_ratio);
            println!(
                "      Maximum drawdown: {:.2}%",
                results.max_drawdown * 100.0
            );
            println!("      Number of trades: {}", results.num_trades);
            println!("      Win rate: {:.1}%", results.win_rate * 100.0);

            if results.num_trades > 0 {
                println!("      Average trade: {:.4}", results.avg_trade_return);
                println!("      Best trade: {:.4}", results.best_trade);
                println!("      Worst trade: {:.4}", results.worst_trade);
            }
        }
        Err(e) => println!("    Strategy backtesting failed: {}", e),
    }

    // Mean reversion strategy
    let mean_reversion_strategy = TradingStrategy::MeanReversion {
        lookback_window: 20,
        threshold: 2.0,
    };

    match engine.backtest(_prices, &mean_reversion_strategy) {
        Ok(results) => {
            println!("    Mean Reversion Strategy:");
            println!("      Total return: {:.2}%", results.total_return * 100.0);
            println!("      Sharpe ratio: {:.2}", results.sharpe_ratio);
            println!("      Number of trades: {}", results.num_trades);
            println!("      Win rate: {:.1}%", results.win_rate * 100.0);
        }
        Err(e) => println!("    Mean reversion backtesting failed: {}", e),
    }
}

#[allow(dead_code)]
fn advanced_features_demo(prices: &Array1<f64>, returns: &Array1<f64>) {
    println!("  Advanced financial feature extraction...");

    let fin_features = FinancialFeatures::new();

    // Price-based features
    match fin_features.price_momentum(_prices, vec![5, 10, 20]) {
        Ok(momentum_features) => {
            println!(
                "    Price momentum features: {} values",
                momentum_features.len()
            );
            for (i, &momentum) in momentum_features.iter().enumerate() {
                let period = [5, 10, 20][i % 3];
                println!("      {}-day momentum: {:.4}", period, momentum);
            }
        }
        Err(e) => println!("    Price momentum calculation failed: {}", e),
    }

    // Volatility features
    match fin_features.rolling_volatility(returns, vec![10, 20, 60]) {
        Ok(vol_features) => {
            println!("    Rolling volatility features:");
            for (i, window) in [10, 20, 60].iter().enumerate() {
                if let Some(&vol) = vol_features.get(i) {
                    let annualized_vol = vol * (252.0_f64).sqrt();
                    println!(
                        "      {}-day volatility: {:.4} ({:.1}% annualized)",
                        window,
                        vol,
                        annualized_vol * 100.0
                    );
                }
            }
        }
        Err(e) => println!("    Rolling volatility calculation failed: {}", e),
    }

    // Market microstructure features
    match fin_features.bid_ask_spread_proxy(_prices) {
        Ok(spread_proxy) => {
            println!("    Bid-ask spread proxy: {:.6}", spread_proxy);
        }
        Err(e) => println!("    Bid-ask spread proxy calculation failed: {}", e),
    }

    // Liquidity measures
    match fin_features.amihud_illiquidity(returns_prices) {
        Ok(illiquidity) => {
            println!("    Amihud illiquidity measure: {:.8}", illiquidity);
        }
        Err(e) => println!("    Amihud illiquidity calculation failed: {}", e),
    }

    // Jump detection
    match fin_features.jump_detection(returns, 0.99) {
        Ok(jumps) => {
            let jump_count = jumps.iter().filter(|&&x| x).count();
            println!("    Jump detection: {} jumps identified", jump_count);
        }
        Err(e) => println!("    Jump detection failed: {}", e),
    }

    // Regime detection using rolling correlations
    let mut rolling_corr = RollingCorrelation::new(30);
    match rolling_corr.compute(returns, returns) {
        Ok(autocorr) => {
            let regime_changes = autocorr
                .windows(2)
                .filter(|window| (window[1] - window[0]).abs() > 0.3)
                .count();
            println!("    Potential regime changes: {}", regime_changes);
        }
        Err(e) => println!("    Rolling correlation calculation failed: {}", e),
    }

    // Feature extraction using general framework
    let feature_config = FeatureConfig::financial();
    let mut extractor = FeatureExtractor::new(feature_config);

    match extractor.extract_features(_prices) {
        Ok(features) => {
            println!(
                "    Comprehensive feature extraction: {} features",
                features.len()
            );
            if features.len() >= 5 {
                println!(
                    "      Sample features: {:?}",
                    features
                        .iter()
                        .take(5)
                        .map(|x| format!("{:.4}", x))
                        .collect::<Vec<_>>()
                );
            }
        }
        Err(e) => println!("    Feature extraction failed: {}", e),
    }
}
*/

// Placeholder main function
fn main() {
    println!("Financial analysis demo disabled due to missing imports");
}
