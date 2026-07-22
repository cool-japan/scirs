//! Tracy-API-compatible profiler integration.
//!
//! Enable the `tracy` cargo feature to activate a **Pure Rust** trace-event
//! recorder that captures spans, frame marks, and log messages into a
//! process-global, in-memory event log and can export them to the
//! [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU)
//! (also understood by [Perfetto](https://ui.perfetto.dev) and
//! `chrome://tracing`) via [`TracyClient::export_chrome_trace`].
//!
//! This backend deliberately does **not** link the upstream C++ Tracy
//! client: it is built entirely from `std` plus the two dependencies
//! `scirs2-core` already carries unconditionally (`once_cell`,
//! `parking_lot`), so enabling the `tracy` feature never triggers a C/C++
//! compilation step. Without the feature, all types and functions compile
//! to zero-cost no-ops with no external dependencies, exactly as before.
//!
//! # Important characteristics
//!
//! * The process-global event buffer (`TRACE_EVENTS`) grows **unbounded**
//!   for the lifetime of the process while the `tracy` feature is enabled.
//!   This backend is intended for bounded profiling sessions (start the
//!   client, do some work, export, stop the process) — not for
//!   long-running services that never restart. A ring-buffer/size-cap
//!   scheme would be a reasonable follow-up if long-lived usage is needed.
//! * Thread identifiers (`tid` in the exported JSON) are small
//!   process-local monotonic integers assigned the first time a thread
//!   touches the profiler. They are **not** OS thread ids, and will not
//!   match what tools like `ps -T` or an external tracer report.
//!
//! # Usage
//!
//! ```rust,no_run
//! use scirs2_core::profiling::tracy::TracyClient;
//!
//! let client = TracyClient::new();
//! if client.is_active() {
//!     client.message("profiling enabled");
//! }
//! {
//!     let _span = client.span("my_operation");
//!     // work here
//! } // span ends on drop
//!
//! // Export the recorded events (valid even when the `tracy` feature is
//! // disabled -- it just writes an empty-but-valid trace document).
//! let _ = client.export_chrome_trace(std::env::temp_dir().join("trace.json"));
//! ```

#[cfg(feature = "tracy")]
use std::cell::Cell;
#[cfg(feature = "tracy")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "tracy")]
use std::time::Instant;

#[cfg(feature = "tracy")]
use once_cell::sync::Lazy;
#[cfg(feature = "tracy")]
use parking_lot::Mutex;

// ---------------------------------------------------------------------------
// Process-global trace state (only compiled when `tracy` is enabled)
// ---------------------------------------------------------------------------

/// A single Chrome Trace Event Format event.
///
/// See the [Chrome Trace Event Format
/// spec](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU)
/// for the meaning of each field.
#[cfg(feature = "tracy")]
#[derive(Debug, Clone)]
struct TraceEvent {
    /// Event name shown in the timeline.
    name: String,
    /// Event category (e.g. "zone", "frame", "log").
    category: &'static str,
    /// Event phase: `'X'` for complete (duration) events, `'i'` for instants.
    ph: char,
    /// Start timestamp in microseconds, relative to process start.
    ts_micros: f64,
    /// Duration in microseconds, only present for complete (`'X'`) events.
    dur_micros: Option<f64>,
    /// Process-local, process-lifetime-stable thread id (NOT the OS tid).
    tid: u64,
    /// OS process id.
    pid: u32,
    /// Optional free-form text payload (used by [`TracyClient::message`]).
    arg: Option<String>,
}

/// Monotonic epoch captured the first time the profiler is touched. All
/// event timestamps are recorded relative to this instant.
#[cfg(feature = "tracy")]
static PROCESS_START: Lazy<Instant> = Lazy::new(Instant::now);

/// Process-global, mutex-protected event log.
///
/// Deliberately unbounded: see the module-level documentation for the
/// implications of long-running processes.
#[cfg(feature = "tracy")]
static TRACE_EVENTS: Lazy<Mutex<Vec<TraceEvent>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// Counter used to hand out small monotonic per-thread ids.
#[cfg(feature = "tracy")]
static NEXT_TID: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "tracy")]
thread_local! {
    /// Cached process-local thread id for the current OS thread, assigned
    /// lazily on first use.
    static THREAD_TID: Cell<Option<u64>> = const { Cell::new(None) };
}

/// Returns a small, process-local, monotonically-assigned integer id for
/// the calling thread. Stable for the lifetime of the thread, but **not**
/// the OS thread id (`std::thread::ThreadId` has no stable integer
/// accessor, so we roll our own).
#[cfg(feature = "tracy")]
fn current_tid() -> u64 {
    THREAD_TID.with(|cell| {
        if let Some(tid) = cell.get() {
            tid
        } else {
            let tid = NEXT_TID.fetch_add(1, Ordering::Relaxed);
            cell.set(Some(tid));
            tid
        }
    })
}

/// Records a single event into the process-global trace buffer.
#[cfg(feature = "tracy")]
fn record_event(event: TraceEvent) {
    TRACE_EVENTS.lock().push(event);
}

/// Escapes a string for embedding in a JSON string literal.
///
/// Escapes backslash, double quote, and control characters (`< 0x20`),
/// using the standard short escapes for `\n`, `\r`, and `\t`, and
/// `\u00XX` for everything else in that range.
#[cfg(feature = "tracy")]
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Serializes a single [`TraceEvent`] as a JSON object (no trailing comma).
#[cfg(feature = "tracy")]
fn serialize_event(event: &TraceEvent) -> String {
    let mut obj = String::new();
    obj.push('{');
    obj.push_str(&format!("\"name\":\"{}\",", json_escape(&event.name)));
    obj.push_str(&format!("\"cat\":\"{}\",", json_escape(event.category)));
    obj.push_str(&format!("\"ph\":\"{}\",", event.ph));
    obj.push_str(&format!("\"ts\":{},", event.ts_micros));
    if let Some(dur) = event.dur_micros {
        obj.push_str(&format!("\"dur\":{dur},"));
    }
    obj.push_str(&format!("\"pid\":{},", event.pid));
    obj.push_str(&format!("\"tid\":{}", event.tid));
    if event.ph == 'i' {
        obj.push_str(",\"s\":\"g\"");
    }
    if let Some(arg) = &event.arg {
        obj.push_str(&format!(
            ",\"args\":{{\"message\":\"{}\"}}",
            json_escape(arg)
        ));
    }
    obj.push('}');
    obj
}

/// Serializes the full Chrome Trace Event Format document for the given
/// events (a `traceEvents` array plus the `displayTimeUnit` field).
#[cfg(feature = "tracy")]
fn serialize_document(events: &[TraceEvent]) -> String {
    let mut doc = String::from("{\"traceEvents\":[");
    for (i, event) in events.iter().enumerate() {
        if i > 0 {
            doc.push(',');
        }
        doc.push_str(&serialize_event(event));
    }
    doc.push_str("],\"displayTimeUnit\":\"ns\"}");
    doc
}

/// Empty-but-valid Chrome Trace Event Format document, used when the
/// `tracy` feature is disabled (or, trivially, when no events have been
/// recorded yet).
const EMPTY_TRACE_DOCUMENT: &str = "{\"traceEvents\":[],\"displayTimeUnit\":\"ns\"}";

// ---------------------------------------------------------------------------
// Tracy span RAII guard
// ---------------------------------------------------------------------------

/// A profiling span that is emitted to the trace-event log when dropped.
///
/// Obtain one via [`TracyClient::span`].
pub struct TracySpan {
    #[cfg(feature = "tracy")]
    name: String,
    #[cfg(feature = "tracy")]
    start: Instant,
    #[cfg(not(feature = "tracy"))]
    _phantom: (),
}

#[cfg(feature = "tracy")]
impl Drop for TracySpan {
    fn drop(&mut self) {
        let dur = self.start.elapsed();
        let ts_micros = self.start.duration_since(*PROCESS_START).as_secs_f64() * 1_000_000.0;
        record_event(TraceEvent {
            name: std::mem::take(&mut self.name),
            category: "zone",
            ph: 'X',
            ts_micros,
            dur_micros: Some(dur.as_secs_f64() * 1_000_000.0),
            tid: current_tid(),
            pid: std::process::id(),
            arg: None,
        });
    }
}

// ---------------------------------------------------------------------------
// TracyClient
// ---------------------------------------------------------------------------

/// Handle to the (Pure Rust) trace-event profiler client.
///
/// Construct once at application start with [`TracyClient::new`] and keep
/// the handle alive for the duration of profiling.  All methods are safe
/// no-ops when the `tracy` feature is not enabled.
pub struct TracyClient {
    active: bool,
}

impl TracyClient {
    /// Initialise the profiler client.
    ///
    /// When the `tracy` feature is enabled this forces initialisation of
    /// the process-global trace-event buffer and epoch clock. When the
    /// feature is absent this is a pure no-op constructor.
    pub fn new() -> Self {
        #[cfg(feature = "tracy")]
        {
            // Force initialisation of the global statics, matching the old
            // "starts the underlying runtime" semantics.
            Lazy::force(&PROCESS_START);
            Lazy::force(&TRACE_EVENTS);
            TracyClient { active: true }
        }
        #[cfg(not(feature = "tracy"))]
        {
            TracyClient { active: false }
        }
    }

    /// Returns `true` when profiling is active (i.e. the `tracy` feature
    /// is enabled and the client was started successfully).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Begin a named profiling zone.  The returned [`TracySpan`] ends the
    /// zone (and records a complete `'X'` trace event) when dropped.
    ///
    /// When the `tracy` feature is disabled this is a zero-cost no-op.
    #[inline]
    pub fn span(&self, name: &str) -> TracySpan {
        #[cfg(feature = "tracy")]
        {
            TracySpan {
                name: name.to_owned(),
                start: Instant::now(),
            }
        }
        #[cfg(not(feature = "tracy"))]
        {
            let _ = name;
            TracySpan { _phantom: () }
        }
    }

    /// Mark a named frame boundary.
    ///
    /// Recorded as an instant (`'i'`) trace event with global scope. When
    /// the `tracy` feature is disabled this is a no-op.
    #[inline]
    pub fn frame_mark(&self, name: &str) {
        #[cfg(feature = "tracy")]
        {
            let ts_micros =
                Instant::now().duration_since(*PROCESS_START).as_secs_f64() * 1_000_000.0;
            record_event(TraceEvent {
                name: format!("frame: {name}"),
                category: "frame",
                ph: 'i',
                ts_micros,
                dur_micros: None,
                tid: current_tid(),
                pid: std::process::id(),
                arg: None,
            });
        }
        #[cfg(not(feature = "tracy"))]
        let _ = name;
    }

    /// Emit a free-form message to the trace-event log.
    ///
    /// Recorded as an instant (`'i'`) trace event carrying the message as
    /// an argument payload. When the `tracy` feature is disabled this is
    /// a no-op.
    #[inline]
    pub fn message(&self, msg: &str) {
        #[cfg(feature = "tracy")]
        {
            let ts_micros =
                Instant::now().duration_since(*PROCESS_START).as_secs_f64() * 1_000_000.0;
            record_event(TraceEvent {
                name: "message".to_owned(),
                category: "log",
                ph: 'i',
                ts_micros,
                dur_micros: None,
                tid: current_tid(),
                pid: std::process::id(),
                arg: Some(msg.to_owned()),
            });
        }
        #[cfg(not(feature = "tracy"))]
        let _ = msg;
    }

    /// Export all recorded trace events as a [Chrome Trace Event
    /// Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU)
    /// JSON document, viewable at <https://ui.perfetto.dev> or
    /// `chrome://tracing`.
    ///
    /// This method is always available and always safe to call: when the
    /// `tracy` feature is disabled it writes the valid-but-empty document
    /// `{"traceEvents":[],"displayTimeUnit":"ns"}` rather than being
    /// conditionally compiled out, so callers never need feature guards
    /// around the export call itself.
    pub fn export_chrome_trace(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        use std::io::Write;

        #[cfg(feature = "tracy")]
        let document = {
            let events = TRACE_EVENTS.lock().clone();
            serialize_document(&events)
        };
        #[cfg(not(feature = "tracy"))]
        let document = EMPTY_TRACE_DOCUMENT.to_owned();

        let mut file = std::fs::File::create(path)?;
        file.write_all(document.as_bytes())?;
        Ok(())
    }
}

impl Default for TracyClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Macro convenience
// ---------------------------------------------------------------------------

/// Create a Tracy span in the current scope.
///
/// The span ends when the binding goes out of scope.
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_core::profiling::tracy::TracyClient;
/// use scirs2_core::tracy_span;
///
/// let client = TracyClient::new();
/// tracy_span!(client, "my_operation");
/// // work here — span ends at end of block
/// ```
#[macro_export]
macro_rules! tracy_span {
    ($client:expr, $name:expr) => {
        let _tracy_span_guard = $client.span($name);
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracy_client_default_features() {
        // Must succeed regardless of whether the `tracy` feature is enabled.
        let client = TracyClient::new();

        // Without the `tracy` feature (default), the client should be inactive.
        #[cfg(not(feature = "tracy"))]
        assert!(
            !client.is_active(),
            "TracyClient should be inactive without the tracy feature"
        );

        // With the `tracy` feature active, the client should report active.
        #[cfg(feature = "tracy")]
        assert!(
            client.is_active(),
            "TracyClient should be active with the tracy feature"
        );
    }

    #[test]
    fn test_tracy_span_drop() {
        let client = TracyClient::new();
        {
            let _span = client.span("test_span_drop");
            // span is live here
        }
        // span dropped — no panic
    }

    #[test]
    fn test_tracy_frame_mark() {
        let client = TracyClient::new();
        // Must not panic regardless of feature flag.
        client.frame_mark("test_frame");
    }

    #[test]
    fn test_tracy_message() {
        let client = TracyClient::new();
        // Must not panic regardless of feature flag.
        client.message("test message from tracy integration test");
    }

    #[test]
    fn test_tracy_default_impl() {
        let client = TracyClient::default();
        // Default should produce the same result as new().
        #[cfg(not(feature = "tracy"))]
        assert!(!client.is_active());
    }

    #[test]
    fn test_tracy_span_macro() {
        let client = TracyClient::new();
        tracy_span!(client, "macro_test_span");
        // No panic = success
    }

    #[test]
    fn test_export_chrome_trace_produces_valid_json_shape() {
        let client = TracyClient::new();
        client.message("export shape test");
        let path = std::env::temp_dir().join(format!(
            "scirs2_core_tracy_export_shape_{}.json",
            std::process::id()
        ));

        client
            .export_chrome_trace(&path)
            .expect("export_chrome_trace should succeed");

        let contents = std::fs::read_to_string(&path).expect("exported file should be readable");
        let _ = std::fs::remove_file(&path);

        assert!(
            contents.starts_with("{\"traceEvents\":["),
            "exported document should start with the traceEvents array: {contents}"
        );
        assert!(
            contents.ends_with("],\"displayTimeUnit\":\"ns\"}"),
            "exported document should end with the displayTimeUnit field: {contents}"
        );
    }

    #[cfg(feature = "tracy")]
    #[test]
    fn test_export_chrome_trace_records_span_event() {
        // Use a uniquely-named span so this test is robust to other tests
        // (and doctests / concurrent test threads) sharing the same
        // process-global event buffer.
        let client = TracyClient::new();
        let marker = format!(
            "unique_span_marker_{}_{}",
            std::process::id(),
            current_tid()
        );
        {
            let _span = client.span(&marker);
        }

        let path = std::env::temp_dir().join(format!(
            "scirs2_core_tracy_export_span_{}_{}.json",
            std::process::id(),
            current_tid()
        ));
        client
            .export_chrome_trace(&path)
            .expect("export_chrome_trace should succeed");
        let contents = std::fs::read_to_string(&path).expect("exported file should be readable");
        let _ = std::fs::remove_file(&path);

        assert!(
            contents.contains(&marker),
            "exported document should contain the span's name: {contents}"
        );
        assert!(
            contents.contains("\"ph\":\"X\""),
            "exported document should contain a complete-event phase marker: {contents}"
        );
    }
}
