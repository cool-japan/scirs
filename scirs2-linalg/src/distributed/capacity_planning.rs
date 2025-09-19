//! Capacity planning system for distributed operations

use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::distributed::fault_tolerance::TrendDirection;

/// Capacity planning system
#[derive(Debug)]
pub struct CapacityPlanner {
    /// Demand forecasting models
    demand_models: HashMap<String, DemandForecastModel>,
    /// Capacity scenarios
    capacity_scenarios: Vec<CapacityScenario>,
    /// Planning horizon
    planning_horizon: Duration,
    /// Cost models
    cost_models: HashMap<String, CostModel>,
}

/// Model for forecasting resource demand
#[derive(Debug)]
pub struct DemandForecastModel {
    model_type: ForecastModelType,
    historical_demand: Vec<DemandDataPoint>,
    seasonal_patterns: Vec<SeasonalPattern>,
    trend_analysis: TrendAnalysis,
    forecast_accuracy: f64,
}

/// Types of forecasting models
#[derive(Debug, Clone)]
pub enum ForecastModelType {
    Linear,
    Exponential,
    Seasonal,
    ARIMA,
    NeuralNetwork,
    Ensemble,
}

/// Data point for demand forecasting
#[derive(Debug, Clone)]
pub struct DemandDataPoint {
    timestamp: Instant,
    resource_type: String,
    demand_value: f64,
    context: DemandContext,
}

/// Context for demand data
#[derive(Debug, Clone)]
pub struct DemandContext {
    workload_type: String,
    user_count: usize,
    datasize: usize,
    external_factors: HashMap<String, f64>,
}

/// Seasonal pattern in demand
#[derive(Debug, Clone)]
pub struct SeasonalPattern {
    pattern_type: SeasonalPatternType,
    amplitude: f64,
    period: Duration,
    phase_offset: Duration,
}

/// Types of seasonal patterns
#[derive(Debug, Clone, Copy)]
pub enum SeasonalPatternType {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
    Custom,
}

/// Trend analysis for demand
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    trend_direction: TrendDirection,
    trend_strength: f64,
    change_rate: f64,
    confidence: f64,
}

/// Capacity planning scenario
#[derive(Debug, Clone)]
pub struct CapacityScenario {
    scenario_name: String,
    probability: f64,
    demand_growth_rate: f64,
    resource_requirements: HashMap<String, f64>,
    timeline: Duration,
    investment_required: f64,
}

/// Cost model for capacity planning
#[derive(Debug, Clone)]
pub struct CostModel {
    resource_type: String,
    fixed_costs: f64,
    variable_costs: f64,
    scaling_factor: f64,
    depreciation_rate: f64,
    operational_costs: HashMap<String, f64>,
}

impl CapacityPlanner {
    /// Create a new capacity planner
    pub fn new(planning_horizon: Duration) -> Self {
        Self {
            demand_models: HashMap::new(),
            capacity_scenarios: Vec::new(),
            planning_horizon,
            cost_models: HashMap::new(),
        }
    }

    /// Add a demand forecasting model
    pub fn add_demand_model(&mut self, resource_type: String, model: DemandForecastModel) {
        self.demand_models.insert(resource_type, model);
    }

    /// Add a capacity scenario
    pub fn add_scenario(&mut self, scenario: CapacityScenario) {
        self.capacity_scenarios.push(scenario);
    }

    /// Add a cost model
    pub fn add_cost_model(&mut self, model: CostModel) {
        self.cost_models.insert(model.resource_type.clone(), model);
    }

    /// Forecast demand for a resource type
    pub fn forecast_demand(&self, resource_type: &str, horizon: Duration) -> Result<Vec<f64>, String> {
        let model = self.demand_models.get(resource_type)
            .ok_or_else(|| format!("No demand model found for resource type: {}", resource_type))?;

        let forecast = self.generate_forecast(model, horizon)?;
        Ok(forecast)
    }

    /// Generate forecast using the model
    fn generate_forecast(&self, model: &DemandForecastModel, horizon: Duration) -> Result<Vec<f64>, String> {
        // Simplified forecasting - would use proper algorithms in practice
        let base_demand = model.historical_demand.last()
            .map(|d| d.demand_value)
            .unwrap_or(1.0);

        let trend_factor = match model.trend_analysis.trend_direction {
            TrendDirection::Increasing => 1.0 + model.trend_analysis.change_rate,
            TrendDirection::Decreasing => 1.0 - model.trend_analysis.change_rate,
            _ => 1.0,
        };

        let forecast_points = (horizon.as_secs() / 3600) as usize; // Hourly forecasts
        let mut forecast = Vec::with_capacity(forecast_points);

        for i in 0..forecast_points {
            let time_factor = trend_factor.powi(i as i32);
            let seasonal_factor = self.calculate_seasonal_factor(&model.seasonal_patterns, i);
            let value = base_demand * time_factor * seasonal_factor;
            forecast.push(value);
        }

        Ok(forecast)
    }

    /// Calculate seasonal factor
    fn calculate_seasonal_factor(&self, patterns: &[SeasonalPattern], time_index: usize) -> f64 {
        let mut factor = 1.0;

        for pattern in patterns {
            let period_hours = pattern.period.as_secs() / 3600;
            if period_hours > 0 {
                let phase = (time_index as f64) / (period_hours as f64) * 2.0 * std::f64::consts::PI;
                factor += pattern.amplitude * phase.sin();
            }
        }

        factor.max(0.1) // Ensure positive factor
    }

    /// Calculate capacity requirements for scenarios
    pub fn calculate_capacity_requirements(&self) -> HashMap<String, CapacityRequirement> {
        let mut requirements = HashMap::new();

        for scenario in &self.capacity_scenarios {
            for (resource_type, demand) in &scenario.resource_requirements {
                let requirement = requirements.entry(resource_type.clone())
                    .or_insert_with(|| CapacityRequirement::new(resource_type.clone()));

                requirement.add_scenario_demand(*demand * scenario.probability);
            }
        }

        requirements
    }

    /// Optimize capacity allocation
    pub fn optimize_capacity(&self) -> Result<CapacityAllocation, String> {
        let requirements = self.calculate_capacity_requirements();
        let mut allocation = CapacityAllocation::new();

        for (resource_type, requirement) in requirements {
            if let Some(cost_model) = self.cost_models.get(&resource_type) {
                let optimal_capacity = self.calculate_optimal_capacity(&requirement, cost_model);
                allocation.add_resource_allocation(resource_type, optimal_capacity);
            }
        }

        Ok(allocation)
    }

    /// Calculate optimal capacity for a resource
    fn calculate_optimal_capacity(&self, requirement: &CapacityRequirement, cost_model: &CostModel) -> f64 {
        // Simplified optimization - would use proper optimization algorithms in practice
        let base_requirement = requirement.expected_demand;
        let safety_margin = requirement.peak_demand - requirement.expected_demand;

        // Balance cost vs. capacity
        let optimal_capacity = base_requirement + safety_margin * 0.8;
        optimal_capacity * cost_model.scaling_factor
    }

    /// Generate capacity planning report
    pub fn generate_report(&self) -> CapacityPlanningReport {
        let current_allocation = self.optimize_capacity().unwrap_or_else(|_| CapacityAllocation::new());
        let recommendations = self.generate_recommendations();

        CapacityPlanningReport {
            planning_horizon: self.planning_horizon,
            current_allocation,
            scenarios: self.capacity_scenarios.clone(),
            recommendations,
            total_investment_required: self.calculate_total_investment(),
        }
    }

    /// Generate capacity recommendations
    fn generate_recommendations(&self) -> Vec<CapacityRecommendation> {
        let mut recommendations = Vec::new();

        for scenario in &self.capacity_scenarios {
            let recommendation = CapacityRecommendation {
                scenario_name: scenario.scenario_name.clone(),
                recommended_action: RecommendedAction::Scale,
                rationale: format!("Based on {}% probability scenario", scenario.probability * 100.0),
                priority: if scenario.probability > 0.7 { RecommendationPriority::High } else { RecommendationPriority::Medium },
                estimated_cost: scenario.investment_required,
                timeline: scenario.timeline,
            };
            recommendations.push(recommendation);
        }

        recommendations
    }

    /// Calculate total investment required
    fn calculate_total_investment(&self) -> f64 {
        self.capacity_scenarios.iter()
            .map(|s| s.investment_required * s.probability)
            .sum()
    }
}

/// Capacity requirement for a resource type
#[derive(Debug, Clone)]
pub struct CapacityRequirement {
    resource_type: String,
    expected_demand: f64,
    peak_demand: f64,
    minimum_demand: f64,
    confidence_interval: (f64, f64),
}

impl CapacityRequirement {
    fn new(resource_type: String) -> Self {
        Self {
            resource_type,
            expected_demand: 0.0,
            peak_demand: 0.0,
            minimum_demand: f64::INFINITY,
            confidence_interval: (0.0, 0.0),
        }
    }

    fn add_scenario_demand(&mut self, demand: f64) {
        self.expected_demand += demand;
        self.peak_demand = self.peak_demand.max(demand);
        self.minimum_demand = self.minimum_demand.min(demand);
    }
}

/// Capacity allocation result
#[derive(Debug, Clone)]
pub struct CapacityAllocation {
    resource_allocations: HashMap<String, f64>,
    total_cost: f64,
    utilization_efficiency: f64,
}

impl CapacityAllocation {
    fn new() -> Self {
        Self {
            resource_allocations: HashMap::new(),
            total_cost: 0.0,
            utilization_efficiency: 0.0,
        }
    }

    fn add_resource_allocation(&mut self, resource_type: String, capacity: f64) {
        self.resource_allocations.insert(resource_type, capacity);
    }
}

/// Capacity planning report
#[derive(Debug, Clone)]
pub struct CapacityPlanningReport {
    planning_horizon: Duration,
    current_allocation: CapacityAllocation,
    scenarios: Vec<CapacityScenario>,
    recommendations: Vec<CapacityRecommendation>,
    total_investment_required: f64,
}

/// Capacity recommendation
#[derive(Debug, Clone)]
pub struct CapacityRecommendation {
    scenario_name: String,
    recommended_action: RecommendedAction,
    rationale: String,
    priority: RecommendationPriority,
    estimated_cost: f64,
    timeline: Duration,
}

/// Recommended action for capacity
#[derive(Debug, Clone, Copy)]
pub enum RecommendedAction {
    Scale,
    Optimize,
    Monitor,
    Defer,
}

/// Priority of capacity recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}