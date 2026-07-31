//! Advanced implementations for the stability framework
//!
//! This module contains the implementation details for formal verification,
//! runtime validation, performance modeling, and cryptographic audit trails.

use super::*;
use crate::performance_optimization::PerformanceMetrics;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// A concrete, runnable measurement of the API under verification.
///
/// [`FormalVerificationEngine::verify_contract`] cannot confirm a
/// contract's performance/memory/thread-safety bounds against nothing:
/// those are runtime properties of actually *calling* the API. Without a
/// probe, verification honestly reports [`VerificationStatus::NotVerified`]
/// (via a `verified: false` [`VerificationResult`]) rather than fabricating
/// a pass.
pub struct VerificationProbe {
    /// Invokes the API under test once, returning the wall-clock duration
    /// of the call and, if determinable, the net memory-usage delta caused
    /// by the call (in bytes). Measuring a memory delta is inherently
    /// caller-specific (e.g. via an allocator hook or process RSS sample),
    /// so `None` is an honest "not measured" rather than a fabricated zero.
    pub invoke: Box<dyn Fn() -> (Duration, Option<usize>) + Send + Sync>,
    /// Number of sequential invocations used to build the timing/memory
    /// sample; the *maximum* observed value is compared against the
    /// contract (a conservative choice for a safety bound).
    pub iterations: usize,
    /// Number of concurrent threads used for the thread-safety smoke test
    /// when the contract claims `ThreadSafety::ThreadSafe`. `0` or `1`
    /// disables the concurrent check.
    pub concurrency: usize,
}

impl VerificationProbe {
    /// Convenience constructor for a single-threaded, single-iteration
    /// probe that does not report a memory measurement.
    pub fn from_timing(invoke: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            invoke: Box::new(move || {
                let start = Instant::now();
                invoke();
                (start.elapsed(), None)
            }),
            iterations: 1,
            concurrency: 1,
        }
    }
}

/// Best-effort extraction of a human-readable message from a caught panic
/// payload (as produced by `std::panic::catch_unwind`).
fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

impl Default for FormalVerificationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FormalVerificationEngine {
    /// Create a new formal verification engine
    pub fn new() -> Self {
        Self {
            verification_tasks: Arc::new(Mutex::new(HashMap::new())),
            results_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start formal verification for an API contract.
    ///
    /// `probe` supplies a real, runnable measurement of the API under
    /// test. Without one, none of the contract's performance/memory/
    /// thread-safety bounds can be honestly confirmed, so verification
    /// completes with `VerificationStatus::Failed` and a `verified: false`
    /// result explaining why — never a fabricated `Verified`. With a
    /// probe, the bounds declared in `contract` are checked against real
    /// measurements (see the private `perform_verification` method below).
    pub fn verify_contract(
        &self,
        contract: &ApiContract,
        probe: Option<VerificationProbe>,
    ) -> CoreResult<()> {
        let taskid = format!("{}-{}", contract.module, contract.apiname);

        let properties = self.extract_verification_properties(contract);

        let task = VerificationTask {
            apiname: contract.apiname.clone(),
            module: contract.module.clone(),
            properties: properties.clone(),
            status: VerificationStatus::InProgress,
            started_at: Instant::now(),
        };

        {
            let mut tasks = self.verification_tasks.lock().expect("Operation failed");
            tasks.insert(taskid.clone(), task);
        }

        // Spawn a background thread so a slow probe (many iterations, or a
        // concurrent thread-safety smoke test) never blocks the caller.
        let tasks_clone = Arc::clone(&self.verification_tasks);
        let results_clone = Arc::clone(&self.results_cache);
        let performance = contract.performance.clone();
        let memory = contract.memory.clone();
        let thread_safety = contract.concurrency.thread_safety;

        thread::spawn(move || {
            let result = Self::perform_verification(
                &properties,
                &performance,
                &memory,
                thread_safety,
                probe,
            );
            let verified = result.verified;

            // Store result
            {
                let mut results = results_clone.write().expect("Operation failed");
                results.insert(taskid.clone(), result);
            }

            // Update task status: honestly reflect whether verification
            // actually confirmed the contract, rather than always marking
            // `Verified`.
            {
                let mut tasks = tasks_clone.lock().expect("Operation failed");
                if let Some(task) = tasks.get_mut(&taskid) {
                    task.status = if verified {
                        VerificationStatus::Verified
                    } else {
                        VerificationStatus::Failed
                    };
                }
            }
        });

        Ok(())
    }

    /// Extract verification properties from contract
    fn extract_verification_properties(&self, contract: &ApiContract) -> Vec<VerificationProperty> {
        let mut properties = Vec::new();

        // Performance properties
        properties.push(VerificationProperty {
            name: "performance_bound".to_string(),
            specification: format!(
                "execution_time <= {:?}",
                contract
                    .performance
                    .maxexecution_time
                    .unwrap_or(Duration::from_secs(1))
            ),
            property_type: PropertyType::Safety,
        });

        // Memory properties
        if let Some(max_memory) = contract.memory.max_memory {
            properties.push(VerificationProperty {
                name: "memory_bound".to_string(),
                specification: format!("memory_usage <= {max_memory}"),
                property_type: PropertyType::Safety,
            });
        }

        // Thread safety properties
        if contract.concurrency.thread_safety == ThreadSafety::ThreadSafe {
            properties.push(VerificationProperty {
                name: "thread_safety".to_string(),
                specification: "no_race_conditions AND no_deadlocks".to_string(),
                property_type: PropertyType::Safety,
            });
        }

        properties
    }

    /// Get verification status for an API
    pub fn get_verification_status(&self, apiname: &str, module: &str) -> VerificationStatus {
        let taskid = format!("{module}-{apiname}");

        if let Ok(tasks) = self.verification_tasks.lock() {
            if let Some(task) = tasks.get(&taskid) {
                return task.status;
            }
        }

        VerificationStatus::NotVerified
    }

    /// Get all verification results
    pub fn get_all_results(&self) -> HashMap<String, VerificationResult> {
        if let Ok(results) = self.results_cache.read() {
            results.clone()
        } else {
            HashMap::new()
        }
    }

    /// Check if verification is complete for an API
    pub fn is_verification_complete(&self, apiname: &str, module: &str) -> bool {
        matches!(
            self.get_verification_status(apiname, module),
            VerificationStatus::Verified | VerificationStatus::Failed
        )
    }

    /// Get verification coverage percentage
    pub fn get_verification_coverage(&self) -> f64 {
        if let Ok(tasks) = self.verification_tasks.lock() {
            if tasks.is_empty() {
                return 0.0;
            }

            let verified_count = tasks
                .values()
                .filter(|task| task.status == VerificationStatus::Verified)
                .count();

            (verified_count as f64 / tasks.len() as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Actually check `properties` against real measurements taken from
    /// `probe`.
    ///
    /// Without a probe, nothing was executed, so none of the declared
    /// bounds can be confirmed: this honestly returns `verified: false`
    /// with an explanatory counterexample rather than the previous
    /// unconditional `verified: true`. This crate has no formal-methods
    /// backend (no CBMC/KLEE/model-checker integration, and adding one is
    /// out of scope for a general-purpose core utility crate); what *is*
    /// tractable — and implemented here — is checking the contract's own
    /// numeric bounds against a real invocation of the API.
    fn perform_verification(
        properties: &[VerificationProperty],
        performance: &PerformanceContract,
        memory: &MemoryContract,
        thread_safety: ThreadSafety,
        probe: Option<VerificationProbe>,
    ) -> VerificationResult {
        let start_time = Instant::now();

        let Some(probe) = probe else {
            return VerificationResult {
                verified: false,
                verification_time: start_time.elapsed(),
                checked_properties: vec![],
                counterexample: Some(format!(
                    "no VerificationProbe was supplied: {} declared propert{} could not be \
                     executed against a real workload and so cannot be confirmed",
                    properties.len(),
                    if properties.len() == 1 { "y" } else { "ies" }
                )),
                method: VerificationMethod::StaticAnalysis,
            };
        };

        // Measure the real API under test. A probe that panics is itself a
        // genuine finding (the API under test is not safe to call) and
        // must be caught here: left uncaught, it would unwind straight out
        // of the background thread `verify_contract` spawned to run this
        // function, which would silently abandon the task in `InProgress`
        // forever (nothing left running to ever update its status).
        let iterations = probe.iterations.max(1);
        let mut max_elapsed = Duration::ZERO;
        let mut max_memory_delta: Option<usize> = None;
        let mut probe_panic_message: Option<String> = None;
        for _ in 0..iterations {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (probe.invoke)())) {
                Ok((elapsed, memory_delta)) => {
                    max_elapsed = max_elapsed.max(elapsed);
                    if let Some(delta) = memory_delta {
                        max_memory_delta =
                            Some(max_memory_delta.map_or(delta, |current| current.max(delta)));
                    }
                }
                Err(payload) => {
                    probe_panic_message = Some(panic_payload_to_string(&payload));
                    break;
                }
            }
        }

        if let Some(message) = probe_panic_message {
            return VerificationResult {
                verified: false,
                verification_time: start_time.elapsed(),
                checked_properties: vec![],
                counterexample: Some(format!(
                    "the supplied probe panicked during measurement: {message}"
                )),
                method: VerificationMethod::SymbolicExecution,
            };
        }

        let mut checked_properties = Vec::new();
        let mut counterexamples = Vec::new();

        for property in properties {
            match property.name.as_str() {
                "performance_bound" => {
                    checked_properties.push(property.name.clone());
                    if let Some(bound) = performance.maxexecution_time {
                        if max_elapsed > bound {
                            counterexamples.push(format!(
                                "performance_bound violated: measured max execution time \
                                 {max_elapsed:?} exceeds the contract bound {bound:?} (over \
                                 {iterations} iteration(s))"
                            ));
                        }
                    }
                }
                "memory_bound" => {
                    checked_properties.push(property.name.clone());
                    match (memory.max_memory, max_memory_delta) {
                        (Some(bound), Some(delta)) if delta > bound => {
                            counterexamples.push(format!(
                                "memory_bound violated: measured memory delta {delta} bytes \
                                 exceeds the contract bound {bound} bytes"
                            ));
                        }
                        (Some(_), None) => {
                            counterexamples.push(
                                "memory_bound could not be confirmed: the supplied probe never \
                                 reported a memory measurement"
                                    .to_string(),
                            );
                        }
                        _ => {}
                    }
                }
                "thread_safety" => {
                    checked_properties.push(property.name.clone());
                    if thread_safety == ThreadSafety::ThreadSafe && probe.concurrency > 1 {
                        if let Some(failure) = Self::concurrent_smoke_test(&probe) {
                            counterexamples.push(failure);
                        }
                    }
                }
                _ => {}
            }
        }

        let verified = counterexamples.is_empty();
        VerificationResult {
            verified,
            verification_time: start_time.elapsed(),
            checked_properties,
            counterexample: if counterexamples.is_empty() {
                None
            } else {
                Some(counterexamples.join("; "))
            },
            // `SymbolicExecution` is the closest available label for "the
            // API was actually invoked and measured" (as opposed to the
            // purely-static techniques in this enum); it is concrete
            // rather than symbolic execution, but no better-fitting
            // variant exists.
            method: VerificationMethod::SymbolicExecution,
        }
    }

    /// Runs `probe.invoke` concurrently across `probe.concurrency` threads
    /// (each performing `probe.iterations` calls) and reports a failure
    /// message if any worker panics — a real, if coarse, signal that a
    /// `ThreadSafe` claim does not hold — or if the workers do not all
    /// complete within a generous timeout (a suspected deadlock/hang).
    fn concurrent_smoke_test(probe: &VerificationProbe) -> Option<String> {
        const TIMEOUT: Duration = Duration::from_secs(10);

        let concurrency = probe.concurrency.max(1);
        let iterations_per_thread = probe.iterations.max(1);
        let invoke = &probe.invoke;
        let panic_count = AtomicUsize::new(0);
        let (tx, rx) = mpsc::channel::<()>();

        thread::scope(|scope| {
            for _ in 0..concurrency {
                let tx = tx.clone();
                let panic_count = &panic_count;
                scope.spawn(move || {
                    for _ in 0..iterations_per_thread {
                        let outcome =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                (invoke)();
                            }));
                        if outcome.is_err() {
                            panic_count.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                    let _ = tx.send(());
                });
            }
            drop(tx);

            let deadline = Instant::now() + TIMEOUT;
            let mut completed = 0usize;
            while completed < concurrency {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }
                match rx.recv_timeout(remaining) {
                    Ok(()) => completed += 1,
                    Err(_) => break,
                }
            }

            if completed < concurrency {
                return Some(format!(
                    "thread_safety smoke test timed out: only {completed}/{concurrency} worker \
                     thread(s) completed within {TIMEOUT:?} (suspected deadlock/hang)"
                ));
            }

            let panics = panic_count.load(Ordering::SeqCst);
            if panics > 0 {
                return Some(format!(
                    "thread_safety smoke test failed: {panics} panic(s) observed across \
                     {concurrency} concurrent thread(s) x {iterations_per_thread} iteration(s)"
                ));
            }

            None
        })
    }
}

impl RuntimeContractValidator {
    /// Create a new runtime contract validator
    pub fn new() -> (Self, Receiver<MonitoringEvent>) {
        let (sender, receiver) = mpsc::channel();

        let validator = Self {
            contracts: Arc::new(RwLock::new(HashMap::new())),
            event_sender: sender,
            stats: Arc::new(Mutex::new(ValidationStatistics {
                total_validations: 0,
                violations_detected: 0,
                avg_validation_time: Duration::from_nanos(0),
                success_rate: 1.0,
            })),
            chaos_controller: Arc::new(Mutex::new(ChaosEngineeringController {
                enabled: false,
                faultprobability: 0.01,
                active_faults: Vec::new(),
                fault_history: Vec::new(),
            })),
        };

        (validator, receiver)
    }

    /// Register a contract for runtime validation
    pub fn register_contract(&self, contract: ApiContract) {
        let key = format!("{}-{}", contract.module, contract.apiname);

        if let Ok(mut contracts) = self.contracts.write() {
            contracts.insert(key, contract);
        }
    }

    /// Validate API call against contract in real-time
    pub fn validate_api_call(
        &self,
        apiname: &str,
        module: &str,
        context: &ApiCallContext,
    ) -> CoreResult<()> {
        let start_time = Instant::now();
        let key = format!("{module}-{apiname}");

        // Update statistics
        {
            if let Ok(mut stats) = self.stats.lock() {
                stats.total_validations += 1;
            }
        }

        // Inject chaos if enabled
        self.maybe_inject_fault(apiname, module)?;

        // Get contract
        let contract = {
            if let Ok(contracts) = self.contracts.read() {
                contracts.get(&key).cloned()
            } else {
                return Err(CoreError::ValidationError(ErrorContext::new(
                    "Cannot access contracts for validation".to_string(),
                )));
            }
        };

        let contract = contract.ok_or_else(|| {
            CoreError::ValidationError(ErrorContext::new(format!(
                "No contract found for {module}::{apiname}"
            )))
        })?;

        // Validate performance contract
        if let Some(max_time) = contract.performance.maxexecution_time {
            if context.execution_time > max_time {
                self.report_violation(
                    apiname,
                    module,
                    ContractViolation {
                        violation_type: ViolationType::Performance,
                        expected: format!("{max_time:?}"),
                        actual: format!("{:?}", context.execution_time),
                        severity: ViolationSeverity::High,
                    },
                )?;
            }
        }

        // Validate memory contract
        if let Some(max_memory) = contract.memory.max_memory {
            if context.memory_usage > max_memory {
                self.report_violation(
                    apiname,
                    module,
                    ContractViolation {
                        violation_type: ViolationType::Memory,
                        expected: format!("{max_memory}"),
                        actual: context.memory_usage.to_string(),
                        severity: ViolationSeverity::Medium,
                    },
                )?;
            }
        }

        // Update statistics
        let validation_time = start_time.elapsed();
        {
            if let Ok(mut stats) = self.stats.lock() {
                let total = stats.total_validations as f64;
                let prev_avg = stats.avg_validation_time.as_nanos() as f64;
                let new_avg =
                    (prev_avg * (total - 1.0) + validation_time.as_nanos() as f64) / total;
                stats.avg_validation_time = Duration::from_nanos(new_avg as u64);
                stats.success_rate = (total - stats.violations_detected as f64) / total;
            }
        }

        Ok(())
    }

    /// Enable chaos engineering
    pub fn enable_chaos_engineering(&self, faultprobability: f64) {
        if let Ok(mut controller) = self.chaos_controller.lock() {
            controller.enabled = true;
            controller.faultprobability = faultprobability.clamp(0.0, 1.0);
        }
    }

    /// Maybe inject a chaos fault
    fn maybe_inject_fault(&self, apiname: &str, module: &str) -> CoreResult<()> {
        if let Ok(mut controller) = self.chaos_controller.lock() {
            if !controller.enabled {
                return Ok(());
            }

            // Generate random number for fault probability
            let mut hasher = DefaultHasher::new();
            apiname.hash(&mut hasher);
            module.hash(&mut hasher);
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .hash(&mut hasher);

            let rand_val = (hasher.finish() % 10000) as f64 / 10000.0;

            if rand_val < controller.faultprobability {
                // Inject a random fault
                let fault = match rand_val * 4.0 {
                    x if x < 1.0 => ChaosFault::LatencyInjection(Duration::from_millis(100)),
                    x if x < 2.0 => ChaosFault::MemoryPressure(1024 * 1024), // 1MB
                    x if x < 3.0 => ChaosFault::CpuThrottling(0.5),
                    _ => ChaosFault::RandomFailure(0.1),
                };

                controller.active_faults.push(fault.clone());
                controller
                    .fault_history
                    .push((Instant::now(), fault.clone()));

                // Send monitoring event
                let event = MonitoringEvent {
                    timestamp: Instant::now(),
                    apiname: apiname.to_string(),
                    module: module.to_string(),
                    event_type: MonitoringEventType::ChaosEngineeringFault(fault.clone()),
                    performance_metrics: RuntimePerformanceMetrics {
                        execution_time: Duration::from_nanos(0),
                        memory_usage: 0,
                        cpu_usage: 0.0,
                        cache_hit_rate: 0.0,
                        thread_count: 1,
                    },
                    thread_id: format!("{:?}", thread::current().id()),
                };

                let _ = self.event_sender.send(event);

                // Actually inject the fault
                match fault {
                    ChaosFault::LatencyInjection(delay) => {
                        thread::sleep(delay);
                    }
                    ChaosFault::RandomFailure(prob) if rand_val < prob => {
                        return Err(CoreError::ValidationError(ErrorContext::new(
                            "Chaos engineering: Random failure injected".to_string(),
                        )));
                    }
                    _ => {} // Other faults would require system-level intervention
                }
            }
        }

        Ok(())
    }

    /// Report a contract violation
    fn report_violation(
        &self,
        apiname: &str,
        module: &str,
        violation: ContractViolation,
    ) -> CoreResult<()> {
        // Update statistics
        {
            if let Ok(mut stats) = self.stats.lock() {
                stats.violations_detected += 1;
                let total = stats.total_validations as f64;
                stats.success_rate = (total - stats.violations_detected as f64) / total;
            }
        }

        // Send monitoring event
        let event = MonitoringEvent {
            timestamp: Instant::now(),
            apiname: apiname.to_string(),
            module: module.to_string(),
            event_type: MonitoringEventType::ContractViolation(violation.clone()),
            performance_metrics: RuntimePerformanceMetrics {
                execution_time: Duration::from_nanos(0),
                memory_usage: 0,
                cpu_usage: 0.0,
                cache_hit_rate: 0.0,
                thread_count: 1,
            },
            thread_id: format!("{:?}", thread::current().id()),
        };

        let _ = self.event_sender.send(event);

        // Return error for critical violations
        if violation.severity >= ViolationSeverity::High {
            return Err(CoreError::ValidationError(ErrorContext::new(format!(
                "Critical contract violation in {}::{}: {} (expected: {}, actual: {})",
                module,
                apiname,
                match violation.violation_type {
                    ViolationType::Performance => "Performance",
                    ViolationType::Memory => "Memory",
                    ViolationType::Numerical => "Numerical",
                    ViolationType::Concurrency => "Concurrency",
                    ViolationType::Behavioral => "Behavioral",
                },
                violation.expected,
                violation.actual
            ))));
        }

        Ok(())
    }

    /// Get validation statistics
    pub fn get_statistics(&self) -> Option<ValidationStatistics> {
        self.stats.lock().ok().map(|stats| stats.clone())
    }

    /// Get chaos engineering status
    pub fn get_chaos_status(&self) -> Option<(bool, f64, usize)> {
        if let Ok(controller) = self.chaos_controller.lock() {
            Some((
                controller.enabled,
                controller.faultprobability,
                controller.fault_history.len(),
            ))
        } else {
            None
        }
    }

    /// Disable chaos engineering
    pub fn disable_chaos_engineering(&self) {
        if let Ok(mut controller) = self.chaos_controller.lock() {
            controller.enabled = false;
            controller.active_faults.clear();
        }
    }
}

/// API call context for runtime validation
#[derive(Debug, Clone)]
pub struct ApiCallContext {
    /// Execution time of the call
    pub execution_time: Duration,
    /// Memory usage during the call
    pub memory_usage: usize,
    /// Input parameters hash
    pub input_hash: String,
    /// Output parameters hash
    pub output_hash: String,
    /// Thread ID where call occurred
    pub thread_id: String,
}

impl Default for AdvancedPerformanceModeler {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedPerformanceModeler {
    /// Create a new performance modeler
    pub fn new() -> Self {
        Self {
            performance_history: Arc::new(RwLock::new(Vec::new())),
            prediction_models: Arc::new(RwLock::new(HashMap::new())),
            training_status: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Record a performance measurement
    pub fn record_measurement(
        &self,
        apiname: &str,
        input_characteristics: InputCharacteristics,
        performance: PerformanceMetrics,
        system_state: SystemState,
    ) {
        // Convert PerformanceMetrics to RuntimePerformanceMetrics
        let runtime_performance = RuntimePerformanceMetrics {
            execution_time: Duration::from_secs_f64(
                performance.operation_times.values().sum::<f64>()
                    / performance.operation_times.len().max(1) as f64,
            ),
            memory_usage: 0, // Not available in PerformanceMetrics
            cpu_usage: 0.0,  // Not available in PerformanceMetrics
            cache_hit_rate: performance.cache_hit_rate,
            thread_count: 1, // Default value
        };

        let data_point = PerformanceDataPoint {
            timestamp: Instant::now(),
            apiname: apiname.to_string(),
            input_characteristics,
            performance: runtime_performance,
            system_state,
        };

        if let Ok(mut history) = self.performance_history.write() {
            history.push(data_point);

            // Limit history size to prevent unbounded growth
            if history.len() > 10000 {
                history.remove(0);
            }
        }

        // Trigger model retraining if enough new data
        self.maybe_retrain_model(apiname);
    }

    /// Predict performance for given input characteristics
    pub fn predict_performance(
        &self,
        apiname: &str,
        input_characteristics: InputCharacteristics,
        system_state: &SystemState,
    ) -> Option<RuntimePerformanceMetrics> {
        if let Ok(models) = self.prediction_models.read() {
            if let Some(model) = models.get(apiname) {
                // Simplified prediction based on input size and model parameters
                let base_time = Duration::from_nanos(1000);
                let size_factor = match model.model_type {
                    ModelType::LinearRegression => {
                        // Use linear model: slope * x + intercept
                        if model.parameters.len() >= 2 {
                            model.parameters[0] * input_characteristics.size as f64
                                + model.parameters[1]
                        } else {
                            (input_characteristics.size as f64).sqrt()
                        }
                    }
                    ModelType::PolynomialRegression => (input_characteristics.size as f64).sqrt(),
                    _ => (input_characteristics.size as f64).sqrt(),
                };

                let scaled_time = Duration::from_nanos(
                    (base_time.as_nanos() as f64 * size_factor.max(1.0)) as u64,
                );

                return Some(RuntimePerformanceMetrics {
                    execution_time: scaled_time,
                    memory_usage: input_characteristics.size * 8, // Assume 8 bytes per element
                    cpu_usage: system_state.cpu_utilization * 1.1, // Slightly higher
                    cache_hit_rate: 0.8,                          // Assume good cache performance
                    thread_count: 1,
                });
            }
        }

        None
    }

    /// Maybe retrain model if conditions are met
    fn maybe_retrain_model(&self, apiname: &str) {
        // Check if enough new data points exist
        let should_retrain = {
            if let Ok(history) = self.performance_history.read() {
                let api_data_points = history.iter().filter(|dp| dp.apiname == apiname).count();
                api_data_points > 100 && (api_data_points % 50 == 0)
            } else {
                false
            }
        };

        if should_retrain {
            self.train_model(apiname);
        }
    }

    /// Train a performance prediction model
    fn train_model(&self, apiname: &str) {
        // Set training status
        {
            if let Ok(mut status) = self.training_status.lock() {
                status.insert(apiname.to_string(), TrainingStatus::InProgress);
            }
        }

        let apiname = apiname.to_string();
        let history_clone = Arc::clone(&self.performance_history);
        let models_clone = Arc::clone(&self.prediction_models);
        let status_clone = Arc::clone(&self.training_status);

        // Spawn training thread
        thread::spawn(move || {
            let training_data = {
                if let Ok(history) = history_clone.read() {
                    history
                        .iter()
                        .filter(|dp| dp.apiname == apiname)
                        .cloned()
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                }
            };

            if training_data.len() < 10 {
                // Not enough data
                if let Ok(mut status) = status_clone.lock() {
                    status.insert(apiname.clone(), TrainingStatus::Failed);
                }
                return;
            }

            // Simple linear regression model (simplified)
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;
            let mut sum_x2 = 0.0;
            let n = training_data.len() as f64;

            for dp in &training_data {
                let x = dp.input_characteristics.size as f64;
                let y = dp.performance.execution_time.as_nanos() as f64;

                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
            }

            let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
            let intercept = (sum_y - slope * sum_x) / n;

            // Calculate accuracy (R-squared)
            let y_mean = sum_y / n;
            let mut ss_tot = 0.0;
            let mut ss_res = 0.0;

            for dp in &training_data {
                let x = dp.input_characteristics.size as f64;
                let y = dp.performance.execution_time.as_nanos() as f64;
                let y_pred = slope * x + intercept;

                ss_tot += (y - y_mean).powi(2);
                ss_res += (y - y_pred).powi(2);
            }

            let r_squared = if ss_tot > 0.0 {
                1.0 - (ss_res / ss_tot)
            } else {
                0.0
            };

            let model = PerformancePredictionModel {
                model_type: ModelType::LinearRegression,
                parameters: vec![slope, intercept],
                accuracy: r_squared.clamp(0.0, 1.0),
                training_data_size: training_data.len(),
                last_updated: Instant::now(),
            };

            // Store the trained model
            {
                if let Ok(mut models) = models_clone.write() {
                    models.insert(apiname.clone(), model);
                }
            }

            // Update training status
            {
                if let Ok(mut status) = status_clone.lock() {
                    status.insert(apiname, TrainingStatus::Completed);
                }
            }
        });
    }

    /// Get training status for an API
    pub fn get_training_status(&self, apiname: &str) -> TrainingStatus {
        if let Ok(status) = self.training_status.lock() {
            status
                .get(apiname)
                .copied()
                .unwrap_or(TrainingStatus::NotStarted)
        } else {
            TrainingStatus::NotStarted
        }
    }

    /// Get model accuracy for an API
    pub fn get_model_accuracy(&self, apiname: &str) -> Option<f64> {
        if let Ok(models) = self.prediction_models.read() {
            models.get(apiname).map(|model| model.accuracy)
        } else {
            None
        }
    }

    /// Get number of data points for an API
    pub fn get_data_point_count(&self, apiname: &str) -> usize {
        if let Ok(history) = self.performance_history.read() {
            history.iter().filter(|dp| dp.apiname == apiname).count()
        } else {
            0
        }
    }
}

impl Default for ImmutableAuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

impl ImmutableAuditTrail {
    /// Create a new immutable audit trail
    pub fn new() -> Self {
        Self {
            audit_chain: Arc::new(RwLock::new(Vec::new())),
            current_hash: Arc::new(RwLock::new(0.to_string())),
        }
    }

    /// Add a new audit record
    pub fn add_record(&self, data: AuditData) -> CoreResult<()> {
        let timestamp = SystemTime::now();

        let previous_hash = {
            if let Ok(hash) = self.current_hash.read() {
                hash.clone()
            } else {
                return Err(CoreError::ValidationError(ErrorContext::new(
                    "Cannot access current hash".to_string(),
                )));
            }
        };

        // Create record
        let mut record = AuditRecord {
            timestamp,
            previous_hash: previous_hash.clone(),
            data,
            signature: String::new(), // Would be populated by digital signature
            record_hash: String::new(),
        };

        // Calculate record hash
        record.record_hash = self.calculate_record_hash(&record);

        // Add digital signature (simplified)
        record.signature = record.record_hash.to_string();

        // Add to chain
        {
            if let Ok(mut chain) = self.audit_chain.write() {
                chain.push(record.clone());
            } else {
                return Err(CoreError::ValidationError(ErrorContext::new(
                    "Cannot access audit chain".to_string(),
                )));
            }
        }

        // Update current hash
        {
            if let Ok(mut hash) = self.current_hash.write() {
                *hash = record.record_hash;
            }
        }

        Ok(())
    }

    /// Calculate cryptographic hash of a record
    fn calculate_record_hash(&self, record: &AuditRecord) -> String {
        let mut hasher = DefaultHasher::new();

        record
            .timestamp
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .hash(&mut hasher);
        record.previous_hash.hash(&mut hasher);

        // Hash the data (simplified)
        match &record.data {
            AuditData::ContractRegistration(name) => name.hash(&mut hasher),
            AuditData::ContractValidation {
                apiname,
                module,
                result,
            } => {
                apiname.hash(&mut hasher);
                module.hash(&mut hasher);
                result.hash(&mut hasher);
            }
            AuditData::PerformanceMeasurement {
                apiname,
                module,
                metrics,
            } => {
                apiname.hash(&mut hasher);
                module.hash(&mut hasher);
                metrics.hash(&mut hasher);
            }
            AuditData::ViolationDetection {
                apiname,
                module,
                violation,
            } => {
                apiname.hash(&mut hasher);
                module.hash(&mut hasher);
                violation.hash(&mut hasher);
            }
        }

        format!("{:x}", hasher.finish())
    }

    /// Verify the integrity of the audit trail
    pub fn verify_integrity(&self) -> bool {
        if let Ok(chain) = self.audit_chain.read() {
            if chain.is_empty() {
                return true;
            }

            for (i, record) in chain.iter().enumerate() {
                // Verify hash
                let expected_hash = self.calculate_record_hash(record);
                if record.record_hash != expected_hash {
                    return false;
                }

                // Verify chain linkage
                if i > 0 {
                    let prev_record = &chain[i.saturating_sub(1)];
                    if record.previous_hash != prev_record.record_hash {
                        return false;
                    }
                }
            }

            true
        } else {
            false
        }
    }

    /// Get audit trail length
    pub fn len(&self) -> usize {
        if let Ok(chain) = self.audit_chain.read() {
            chain.len()
        } else {
            0
        }
    }

    /// Check if audit trail is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get recent audit records
    pub fn get_recent_records(&self, count: usize) -> Vec<AuditRecord> {
        if let Ok(chain) = self.audit_chain.read() {
            let start = chain.len().saturating_sub(count);
            chain[start..].to_vec()
        } else {
            Vec::new()
        }
    }

    /// Export audit trail for external verification
    #[cfg(feature = "serialization")]
    pub fn export_trail(&self) -> CoreResult<String> {
        if let Ok(chain) = self.audit_chain.read() {
            serde_json::to_string_pretty(&*chain).map_err(|e| {
                CoreError::ValidationError(ErrorContext::new(format!(
                    "Failed to serialize audit trail: {e}"
                )))
            })
        } else {
            Err(CoreError::ValidationError(ErrorContext::new(
                "Cannot access audit chain for export".to_string(),
            )))
        }
    }

    /// Export audit trail for external verification (fallback without serialization)
    #[cfg(not(feature = "serialization"))]
    pub fn export_trail(&self) -> CoreResult<String> {
        Err(CoreError::ValidationError(ErrorContext::new(
            "Audit trail export requires serialization feature".to_string(),
        )))
    }
}

// Helper implementations for public structs
impl InputCharacteristics {
    /// Create new input characteristics
    pub fn new(size: usize, datatype: String) -> Self {
        Self {
            size,
            datatype,
            memory_layout: "contiguous".to_string(),
            access_pattern: "sequential".to_string(),
        }
    }

    /// Create characteristics for matrix operations
    pub fn matrix(rows: usize, cols: usize) -> Self {
        Self {
            size: rows * cols,
            datatype: "f64".to_string(),
            memory_layout: "row_major".to_string(),
            access_pattern: "matrix".to_string(),
        }
    }

    /// Create characteristics for vector operations
    pub fn vector(length: usize) -> Self {
        Self {
            size: length,
            datatype: "f64".to_string(),
            memory_layout: "contiguous".to_string(),
            access_pattern: "sequential".to_string(),
        }
    }
}

impl SystemState {
    /// Create new system state
    pub fn new() -> Self {
        Self {
            cpu_utilization: 0.5,    // Default 50%
            memory_utilization: 0.6, // Default 60%
            io_load: 0.1,            // Default low
            network_load: 0.05,      // Default very low
            temperature: 65.0,       // Default temperature in Celsius
        }
    }

    /// Create system state from current system metrics (simplified)
    pub fn current() -> Self {
        // In a real implementation, this would query actual system metrics
        Self::new()
    }
}

impl Default for InputCharacteristics {
    fn default() -> Self {
        Self::new(1000, "f64".to_string())
    }
}

impl Default for SystemState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal `ApiContract` for verification tests.
    fn make_contract(
        apiname: &str,
        module: &str,
        max_exec: Option<Duration>,
        max_memory: Option<usize>,
        thread_safe: bool,
    ) -> ApiContract {
        ApiContract {
            apiname: apiname.to_string(),
            module: module.to_string(),
            contract_hash: "test_hash".to_string(),
            created_at: SystemTime::now(),
            verification_status: VerificationStatus::NotVerified,
            stability: StabilityLevel::Stable,
            since_version: Version::new(1, 0, 0),
            performance: PerformanceContract {
                time_complexity: ComplexityBound::Linear,
                space_complexity: ComplexityBound::Constant,
                maxexecution_time: max_exec,
                min_throughput: None,
                memorybandwidth: None,
            },
            numerical: NumericalContract {
                precision: PrecisionGuarantee::MachinePrecision,
                stability: NumericalStability::Stable,
                input_domain: InputDomain {
                    ranges: vec![],
                    exclusions: vec![],
                    special_values: SpecialValueHandling::Propagate,
                },
                output_range: OutputRange {
                    bounds: None,
                    monotonic: None,
                    continuous: true,
                },
            },
            concurrency: ConcurrencyContract {
                thread_safety: if thread_safe {
                    ThreadSafety::ThreadSafe
                } else {
                    ThreadSafety::NotThreadSafe
                },
                atomicity: AtomicityGuarantee::OperationAtomic,
                lock_free: false,
                wait_free: false,
                memory_ordering: MemoryOrdering::AcquireRelease,
            },
            memory: MemoryContract {
                allocation_pattern: AllocationPattern::SingleAllocation,
                max_memory,
                alignment: None,
                locality: LocalityGuarantee::GoodSpatial,
                gc_behavior: GcBehavior::MinimalGc,
            },
            deprecation: None,
        }
    }

    /// Poll until verification for `(module, apiname)` leaves the
    /// `InProgress` state (the background thread resolves near-instantly
    /// for these tests), panicking if it never does.
    fn wait_for_verification(
        engine: &FormalVerificationEngine,
        apiname: &str,
        module: &str,
    ) -> VerificationStatus {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let status = engine.get_verification_status(apiname, module);
            if status != VerificationStatus::InProgress {
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "verification for {module}-{apiname} did not complete within the test timeout"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn test_formal_verification_engine() {
        let engine = FormalVerificationEngine::new();
        assert_eq!(engine.get_verification_coverage(), 0.0);

        let contract = make_contract(
            "test_api",
            "test_module",
            Some(Duration::from_millis(100)),
            Some(1024),
            true,
        );

        // No probe supplied: nothing was actually executed, so this must
        // honestly resolve to Failed rather than a fabricated Verified.
        engine
            .verify_contract(&contract, None)
            .expect("Operation failed");

        let status = wait_for_verification(&engine, "test_api", "test_module");
        assert_eq!(
            status,
            VerificationStatus::Failed,
            "verification with no executable probe must not fabricate Verified"
        );

        let results = engine.get_all_results();
        let result = results
            .get("test_module-test_api")
            .expect("result recorded");
        assert!(!result.verified);
        assert!(result.counterexample.is_some());
    }

    #[test]
    fn test_verify_contract_passes_with_probe_within_bounds() {
        let engine = FormalVerificationEngine::new();
        let contract = make_contract(
            "fast_api",
            "perf_module",
            Some(Duration::from_millis(50)),
            Some(1024),
            false,
        );

        let probe = VerificationProbe {
            invoke: Box::new(|| (Duration::from_millis(1), Some(10))),
            iterations: 5,
            concurrency: 1,
        };

        engine
            .verify_contract(&contract, Some(probe))
            .expect("start verification");
        let status = wait_for_verification(&engine, "fast_api", "perf_module");
        assert_eq!(status, VerificationStatus::Verified);

        let results = engine.get_all_results();
        let result = results
            .get("perf_module-fast_api")
            .expect("result recorded");
        assert!(result.verified);
        assert!(result
            .checked_properties
            .contains(&"performance_bound".to_string()));
        assert!(result
            .checked_properties
            .contains(&"memory_bound".to_string()));
    }

    #[test]
    fn test_verify_contract_detects_real_performance_violation() {
        let engine = FormalVerificationEngine::new();
        let contract = make_contract(
            "slow_api",
            "perf_module",
            Some(Duration::from_millis(10)),
            None,
            false,
        );

        // A real measured duration that genuinely exceeds the contract
        // bound; under the old hardcoded-`true` implementation this would
        // have been reported Verified regardless.
        let probe = VerificationProbe {
            invoke: Box::new(|| (Duration::from_millis(200), None)),
            iterations: 1,
            concurrency: 1,
        };

        engine
            .verify_contract(&contract, Some(probe))
            .expect("start verification");
        let status = wait_for_verification(&engine, "slow_api", "perf_module");
        assert_eq!(status, VerificationStatus::Failed);

        let results = engine.get_all_results();
        let result = results
            .get("perf_module-slow_api")
            .expect("result recorded");
        assert!(!result.verified);
        let counterexample = result
            .counterexample
            .as_ref()
            .expect("counterexample present");
        assert!(
            counterexample.contains("performance_bound"),
            "got: {counterexample}"
        );
    }

    #[test]
    fn test_verify_contract_detects_real_memory_violation() {
        let engine = FormalVerificationEngine::new();
        let contract = make_contract("greedy_api", "mem_module", None, Some(100), false);

        let probe = VerificationProbe {
            invoke: Box::new(|| (Duration::from_micros(1), Some(500))),
            iterations: 1,
            concurrency: 1,
        };

        engine
            .verify_contract(&contract, Some(probe))
            .expect("start verification");
        let status = wait_for_verification(&engine, "greedy_api", "mem_module");
        assert_eq!(status, VerificationStatus::Failed);

        let results = engine.get_all_results();
        let result = results
            .get("mem_module-greedy_api")
            .expect("result recorded");
        let counterexample = result
            .counterexample
            .as_ref()
            .expect("counterexample present");
        assert!(
            counterexample.contains("memory_bound"),
            "got: {counterexample}"
        );
    }

    #[test]
    fn test_verify_contract_concurrent_smoke_test_catches_real_panics() {
        let engine = FormalVerificationEngine::new();
        let contract = make_contract("racy_api", "concurrency_module", None, None, true);

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_probe = Arc::clone(&call_count);
        let probe = VerificationProbe {
            invoke: Box::new(move || {
                let n = call_count_probe.fetch_add(1, Ordering::SeqCst);
                // The sequential performance/memory warm-up consumes calls
                // 0..iterations (4) first; only fail calls from the
                // concurrent phase (n >= 4) so this test exercises the
                // concurrent smoke test specifically; a probe that panics
                // during the *sequential* pass is covered by a separate
                // test (`test_formal_verification_engine`-adjacent
                // behavior lives in `perform_verification`'s own
                // panic-safety, not this concurrency-focused test).
                assert!(n < 4 || n % 5 != 0, "simulated concurrency bug (call #{n})");
                (Duration::from_micros(1), None)
            }),
            iterations: 4,
            concurrency: 4,
        };

        engine
            .verify_contract(&contract, Some(probe))
            .expect("start verification");
        let status = wait_for_verification(&engine, "racy_api", "concurrency_module");
        assert_eq!(status, VerificationStatus::Failed);

        let results = engine.get_all_results();
        let result = results
            .get("concurrency_module-racy_api")
            .expect("result recorded");
        let counterexample = result
            .counterexample
            .as_ref()
            .expect("counterexample present");
        assert!(
            counterexample.contains("thread_safety"),
            "got: {counterexample}"
        );
    }

    #[test]
    fn test_verify_contract_concurrent_smoke_test_passes_when_actually_safe() {
        let engine = FormalVerificationEngine::new();
        let contract = make_contract("safe_api", "concurrency_module2", None, None, true);

        let probe = VerificationProbe {
            invoke: Box::new(|| (Duration::from_micros(1), None)),
            iterations: 4,
            concurrency: 4,
        };

        engine
            .verify_contract(&contract, Some(probe))
            .expect("start verification");
        let status = wait_for_verification(&engine, "safe_api", "concurrency_module2");
        assert_eq!(status, VerificationStatus::Verified);
    }

    /// Regression test: a probe that panics during the sequential
    /// performance/memory measurement pass (before the concurrent
    /// thread-safety smoke test even runs) must resolve to `Failed` with
    /// an explanatory counterexample — not silently hang the background
    /// verification thread forever (an uncaught panic there would abandon
    /// the task in `InProgress` with nothing left running to ever update
    /// it).
    #[test]
    fn test_verify_contract_probe_panic_during_sequential_pass_reports_failure_not_hang() {
        let engine = FormalVerificationEngine::new();
        let contract = make_contract(
            "panicky_api",
            "panic_module",
            Some(Duration::from_millis(50)),
            None,
            false,
        );

        let probe = VerificationProbe {
            invoke: Box::new(|| panic!("probe always panics")),
            iterations: 1,
            concurrency: 1,
        };

        engine
            .verify_contract(&contract, Some(probe))
            .expect("start verification");
        let status = wait_for_verification(&engine, "panicky_api", "panic_module");
        assert_eq!(
            status,
            VerificationStatus::Failed,
            "a probe that panics must resolve to Failed, not hang or fabricate Verified"
        );

        let results = engine.get_all_results();
        let result = results
            .get("panic_module-panicky_api")
            .expect("result recorded");
        assert!(!result.verified);
        let counterexample = result
            .counterexample
            .as_ref()
            .expect("counterexample present");
        assert!(counterexample.contains("panicked"), "got: {counterexample}");
    }

    #[test]
    fn test_runtime_contract_validator() {
        let (validator, receiver) = RuntimeContractValidator::new();

        let stats = validator.get_statistics().expect("Operation failed");
        assert_eq!(stats.total_validations, 0);
        assert_eq!(stats.violations_detected, 0);
        assert_eq!(stats.success_rate, 1.0);
    }

    #[test]
    fn test_performance_modeler() {
        let modeler = AdvancedPerformanceModeler::new();

        let input_chars = InputCharacteristics::new(1000, "f64".to_string());
        let system_state = SystemState::new();
        let performance = PerformanceMetrics {
            operation_times: std::collections::HashMap::new(),
            strategy_success_rates: std::collections::HashMap::new(),
            memorybandwidth_utilization: 0.8,
            cache_hit_rate: 0.8,
            parallel_efficiency: 0.9,
        };

        modeler.record_measurement(
            "test_api",
            input_chars.clone(),
            performance,
            system_state.clone(),
        );

        assert_eq!(modeler.get_data_point_count("test_api"), 1);
        assert_eq!(
            modeler.get_training_status("test_api"),
            TrainingStatus::NotStarted
        );
    }

    #[test]
    fn test_audit_trail() {
        let trail = ImmutableAuditTrail::new();
        assert!(trail.is_empty());
        assert!(trail.verify_integrity());

        let data = AuditData::ContractRegistration("test::api".to_string());
        trail.add_record(data).expect("Operation failed");

        assert_eq!(trail.len(), 1);
        assert!(trail.verify_integrity());
    }

    #[test]
    fn test_input_characteristics() {
        let chars = InputCharacteristics::matrix(10, 10);
        assert_eq!(chars.size, 100);
        assert_eq!(chars.memory_layout, "row_major");

        let vector_chars = InputCharacteristics::vector(50);
        assert_eq!(vector_chars.size, 50);
        assert_eq!(vector_chars.access_pattern, "sequential");
    }

    #[test]
    fn test_system_state() {
        let state = SystemState::current();
        assert!(state.cpu_utilization >= 0.0 && state.cpu_utilization <= 1.0);
        assert!(state.memory_utilization >= 0.0 && state.memory_utilization <= 1.0);
        assert!(state.temperature > 0.0);
    }
}
