//! Real-time risk monitoring dashboard facade
//!
//! This module builds a [`RealTimeRiskMonitor`] facade over the
//! already-working [`crate::specialized::finance::risk::realtime`] primitives
//! ([`EventDrivenRisk`], [`IncrementalVaR`], [`PositionLimitChecker`],
//! [`StreamingGreeks`], [`RiskAggregator`]), adding a genuine
//! [`AlertSeverity`] classification and [`RiskAlertType`] discriminant wired
//! into [`RiskDashboard`]/[`RiskSnapshot`] reporting.
//!
//! [`RiskAlert`] itself is re-exported unchanged from the underlying
//! `risk::realtime` module — this facade classifies and aggregates alerts,
//! it does not redefine them.

use crate::specialized::finance::risk::realtime::{
    EventDrivenRisk, IncrementalVaR, MarketTick, PositionLimitChecker, RiskAggregator, RiskMonitor,
    StreamingGreeks,
};

pub use crate::specialized::finance::risk::realtime::RiskAlert;

// ============================================================
// AlertSeverity / RiskAlertType classification
// ============================================================

/// Severity classification for a fired [`RiskAlert`], wired into
/// [`RiskDashboard`]/[`RiskSnapshot`] reporting.
///
/// Ordered `Info < Warning < Critical` so dashboards can compute a "worst
/// severity currently observed" via `Ord`/`max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// No breach, or a breach below the configured limit.
    Info,
    /// The monitored quantity has crossed its configured limit.
    Warning,
    /// The monitored quantity is at least double its configured limit.
    Critical,
}

/// Coarse classification of which underlying [`RiskAlert`] variant fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskAlertType {
    /// A position notional-exposure limit was breached
    /// ([`RiskAlert::LimitBreach`]).
    PositionLimitBreach,
    /// A portfolio VaR limit was exceeded ([`RiskAlert::VarExceeded`]).
    VarLimitBreach,
    /// An option Greek crossed its configured alert threshold
    /// ([`RiskAlert::GreeksThreshold`]).
    GreeksThresholdBreach,
}

impl RiskAlertType {
    /// Classify a [`RiskAlert`] by which monitor kind produced it.
    pub fn classify(alert: &RiskAlert) -> Self {
        match alert {
            RiskAlert::LimitBreach { .. } => RiskAlertType::PositionLimitBreach,
            RiskAlert::VarExceeded { .. } => RiskAlertType::VarLimitBreach,
            RiskAlert::GreeksThreshold { .. } => RiskAlertType::GreeksThresholdBreach,
        }
    }
}

/// Classify the severity of a fired [`RiskAlert`].
///
/// `LimitBreach`/`VarExceeded` alerts carry both the breaching value and the
/// configured limit, so severity is derived from how far past the limit the
/// current value sits (>= 2x limit => `Critical`, >= 1x => `Warning`,
/// otherwise `Info`). `GreeksThreshold` alerts do not carry their configured
/// threshold (only the breaching value), so they are conservatively
/// classified as `Warning`; callers needing finer-grained classification
/// should compare the reported value against their own configured
/// `delta_alert_threshold`.
pub fn classify_severity(alert: &RiskAlert) -> AlertSeverity {
    match alert {
        RiskAlert::LimitBreach { current, limit, .. } => severity_from_ratio(*current, *limit),
        RiskAlert::VarExceeded {
            current_var, limit, ..
        } => severity_from_ratio(*current_var, *limit),
        RiskAlert::GreeksThreshold { .. } => AlertSeverity::Warning,
    }
}

fn severity_from_ratio(current: f64, limit: f64) -> AlertSeverity {
    if limit.abs() < 1e-12 {
        return AlertSeverity::Critical;
    }
    let ratio = (current / limit).abs();
    if ratio >= 2.0 {
        AlertSeverity::Critical
    } else if ratio >= 1.0 {
        AlertSeverity::Warning
    } else {
        AlertSeverity::Info
    }
}

// ============================================================
// Dashboard / snapshot reporting
// ============================================================

/// One classified, sequence-stamped alert as recorded by
/// [`RealTimeRiskMonitor`].
#[derive(Debug, Clone)]
pub struct DashboardAlert {
    /// 1-based sequence number of the tick that produced this alert.
    pub tick_sequence: u64,
    /// Coarse classification of the alert kind.
    pub alert_type: RiskAlertType,
    /// Severity classification.
    pub severity: AlertSeverity,
    /// The original alert payload.
    pub alert: RiskAlert,
}

/// Point-in-time summary of monitored risk state.
#[derive(Debug, Clone)]
pub struct RiskSnapshot {
    /// Number of ticks processed so far.
    pub tick_count: u64,
    /// Total alerts retained in history.
    pub total_alerts: usize,
    /// Count of `Critical`-severity alerts in history.
    pub critical_alerts: usize,
    /// Count of `Warning`-severity alerts in history.
    pub warning_alerts: usize,
    /// Count of `Info`-severity alerts in history.
    pub info_alerts: usize,
    /// Retained alert history (most recent last), capped at the monitor's
    /// configured history size.
    pub recent_alerts: Vec<DashboardAlert>,
}

/// Aggregated dashboard view over a [`RealTimeRiskMonitor`].
#[derive(Debug, Clone)]
pub struct RiskDashboard {
    /// Names of all monitors registered with the underlying dispatcher.
    pub monitor_names: Vec<String>,
    /// Current snapshot of alert counts and history.
    pub snapshot: RiskSnapshot,
}

impl RiskDashboard {
    /// Highest severity currently observed in the retained history, if any
    /// alert has fired.
    pub fn worst_severity(&self) -> Option<AlertSeverity> {
        self.snapshot.recent_alerts.iter().map(|a| a.severity).max()
    }
}

// ============================================================
// RealTimeRiskMonitor
// ============================================================

/// Real-time risk monitor facade.
///
/// Fans out market data to the registered `finance::risk::realtime`
/// monitors via [`EventDrivenRisk`], and layers severity-classified alert
/// history plus [`RiskDashboard`]/[`RiskSnapshot`] reporting on top.
pub struct RealTimeRiskMonitor {
    dispatcher: EventDrivenRisk,
    monitor_names: Vec<String>,
    history: Vec<DashboardAlert>,
    tick_count: u64,
    /// Cap on retained alert history (oldest dropped first).
    max_history: usize,
}

impl RealTimeRiskMonitor {
    /// Create an empty monitor with default history retention (1000 alerts).
    pub fn new() -> Self {
        Self {
            dispatcher: EventDrivenRisk::new(),
            monitor_names: Vec::new(),
            history: Vec::new(),
            tick_count: 0,
            max_history: 1000,
        }
    }

    /// Create an empty monitor with a custom alert-history retention cap.
    pub fn with_history_capacity(max_history: usize) -> Self {
        Self {
            max_history,
            ..Self::new()
        }
    }

    /// Register a position notional-limit monitor.
    pub fn add_position_limit_checker(&mut self, checker: PositionLimitChecker) {
        self.monitor_names.push(checker.name().to_string());
        self.dispatcher.add_monitor(Box::new(checker));
    }

    /// Register an incremental historical-VaR monitor.
    pub fn add_incremental_var(&mut self, ivar: IncrementalVaR) {
        self.monitor_names.push(ivar.name().to_string());
        self.dispatcher.add_monitor(Box::new(ivar));
    }

    /// Register a streaming-Greeks monitor.
    pub fn add_streaming_greeks(&mut self, greeks: StreamingGreeks) {
        self.monitor_names.push(greeks.name().to_string());
        self.dispatcher.add_monitor(Box::new(greeks));
    }

    /// Register a portfolio risk aggregator.
    pub fn add_risk_aggregator(&mut self, aggregator: RiskAggregator) {
        self.monitor_names.push(aggregator.name().to_string());
        self.dispatcher.add_monitor(Box::new(aggregator));
    }

    /// Feed one market tick to all registered monitors, classify every
    /// resulting alert, append it to the retained history, and return the
    /// classified alerts fired by this tick.
    pub fn process_tick(&mut self, tick: &MarketTick) -> Vec<DashboardAlert> {
        self.tick_count += 1;
        let alerts = self.dispatcher.process_tick(tick);
        let classified: Vec<DashboardAlert> = alerts
            .into_iter()
            .map(|alert| DashboardAlert {
                tick_sequence: self.tick_count,
                alert_type: RiskAlertType::classify(&alert),
                severity: classify_severity(&alert),
                alert,
            })
            .collect();

        for entry in &classified {
            self.history.push(entry.clone());
        }
        while self.history.len() > self.max_history {
            self.history.remove(0);
        }

        classified
    }

    /// Point-in-time snapshot of alert counts and recent alert history.
    pub fn snapshot(&self) -> RiskSnapshot {
        let critical_alerts = self
            .history
            .iter()
            .filter(|a| a.severity == AlertSeverity::Critical)
            .count();
        let warning_alerts = self
            .history
            .iter()
            .filter(|a| a.severity == AlertSeverity::Warning)
            .count();
        let info_alerts = self
            .history
            .iter()
            .filter(|a| a.severity == AlertSeverity::Info)
            .count();

        RiskSnapshot {
            tick_count: self.tick_count,
            total_alerts: self.history.len(),
            critical_alerts,
            warning_alerts,
            info_alerts,
            recent_alerts: self.history.clone(),
        }
    }

    /// Aggregated dashboard view (monitor names + current snapshot).
    pub fn dashboard(&self) -> RiskDashboard {
        RiskDashboard {
            monitor_names: self.monitor_names.clone(),
            snapshot: self.snapshot(),
        }
    }
}

impl Default for RealTimeRiskMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tick(symbol: &str, price: f64) -> MarketTick {
        MarketTick {
            symbol: symbol.to_string(),
            price,
            volume: 1000.0,
            timestamp: 1_700_000_000_000,
            bid: price - 0.01,
            ask: price + 0.01,
        }
    }

    #[test]
    fn test_dashboard_matches_direct_position_limit_checker_and_classifies_severity() {
        // Direct primitive: PositionLimitChecker fires with these (current,
        // limit) values.
        let mut direct_checker = PositionLimitChecker::new();
        direct_checker.set_limit("AAPL".to_string(), 5_000.0);
        direct_checker.set_position("AAPL".to_string(), 100.0);
        // notional = 100 * 110 = 11,000 = 2.2x limit -> should classify Critical.
        let tick = make_tick("AAPL", 110.0);
        let direct_alert = direct_checker
            .on_market_data(&tick)
            .expect("direct monitor should succeed")
            .expect("direct monitor should fire an alert");

        // Facade: an equivalently configured checker wrapped by the monitor
        // must fire the same alert (same current/limit values).
        let mut facade_checker = PositionLimitChecker::new();
        facade_checker.set_limit("AAPL".to_string(), 5_000.0);
        facade_checker.set_position("AAPL".to_string(), 100.0);

        let mut monitor = RealTimeRiskMonitor::new();
        monitor.add_position_limit_checker(facade_checker);
        let fired = monitor.process_tick(&tick);

        assert_eq!(fired.len(), 1, "facade should fire exactly one alert");
        match (&fired[0].alert, &direct_alert) {
            (
                RiskAlert::LimitBreach {
                    symbol: s1,
                    current: c1,
                    limit: l1,
                },
                RiskAlert::LimitBreach {
                    symbol: s2,
                    current: c2,
                    limit: l2,
                },
            ) => {
                assert_eq!(s1, s2);
                assert!((c1 - c2).abs() < 1e-9, "current mismatch: {c1} vs {c2}");
                assert!((l1 - l2).abs() < 1e-9, "limit mismatch: {l1} vs {l2}");
            }
            other => panic!("Expected matching LimitBreach alerts, got {other:?}"),
        }
        assert_eq!(fired[0].alert_type, RiskAlertType::PositionLimitBreach);
        assert_eq!(fired[0].severity, AlertSeverity::Critical);

        let snapshot = monitor.snapshot();
        assert_eq!(snapshot.total_alerts, 1);
        assert_eq!(snapshot.critical_alerts, 1);
        assert_eq!(snapshot.warning_alerts, 0);
        assert_eq!(snapshot.tick_count, 1);

        let dashboard = monitor.dashboard();
        assert_eq!(dashboard.monitor_names, vec!["PositionLimitChecker"]);
        assert_eq!(dashboard.worst_severity(), Some(AlertSeverity::Critical));
    }

    #[test]
    fn test_severity_thresholds() {
        let warning_alert = RiskAlert::LimitBreach {
            symbol: "X".to_string(),
            current: 1_200.0,
            limit: 1_000.0,
        }; // 1.2x -> Warning
        let critical_alert = RiskAlert::LimitBreach {
            symbol: "X".to_string(),
            current: 2_500.0,
            limit: 1_000.0,
        }; // 2.5x -> Critical
        assert_eq!(classify_severity(&warning_alert), AlertSeverity::Warning);
        assert_eq!(classify_severity(&critical_alert), AlertSeverity::Critical);
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Critical);
    }

    #[test]
    fn test_history_capacity_is_enforced() {
        let mut monitor = RealTimeRiskMonitor::with_history_capacity(2);
        let mut checker = PositionLimitChecker::new();
        checker.set_limit("X".to_string(), 1.0); // always breaches
        checker.set_position("X".to_string(), 10.0);
        monitor.add_position_limit_checker(checker);

        for _ in 0..5 {
            let tick = make_tick("X", 100.0);
            let _ = monitor.process_tick(&tick);
        }

        let snapshot = monitor.snapshot();
        assert_eq!(snapshot.total_alerts, 2, "history should be capped at 2");
        assert_eq!(snapshot.tick_count, 5);
    }
}
