//! Synchronization protocols module
//!
//! This module provides implementations of various clock synchronization protocols
//! including NTP, PTP, GPS, Berkeley algorithm, Cristian's algorithm, and custom protocols.
//! Each protocol has its own configuration, state management, and specific features.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use scirs2_core::error::Result;

use super::gps::GpsConfig;

/// Clock synchronization protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClockSyncProtocol {
    /// Network Time Protocol
    NTP {
        version: u8,
        servers: Vec<String>,
        authentication: bool,
    },
    /// Precision Time Protocol (IEEE 1588)
    PTP {
        version: PtpVersion,
        domain: u8,
        transport: PtpTransport,
        profile: PtpProfile,
    },
    /// Simple Network Time Protocol
    SNTP {
        servers: Vec<String>,
        timeout: Duration,
    },
    /// Berkeley algorithm
    Berkeley {
        fault_tolerance: usize,
        convergence_threshold: Duration,
    },
    /// Cristian's algorithm
    Cristian {
        time_server: String,
        uncertainty_factor: f64,
    },
    /// GPS-based synchronization
    GPS {
        receiver_config: GpsConfig,
        fallback_protocol: Option<Box<ClockSyncProtocol>>,
    },
    /// Custom synchronization protocol
    Custom {
        protocol_name: String,
        parameters: HashMap<String, String>,
    },
}

/// PTP (Precision Time Protocol) version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PtpVersion {
    /// IEEE 1588-2002
    V1,
    /// IEEE 1588-2008
    V2,
    /// IEEE 1588-2019
    V2_1,
}

/// PTP transport layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PtpTransport {
    /// Ethernet transport
    Ethernet,
    /// UDP/IPv4 transport
    UdpIpv4,
    /// UDP/IPv6 transport
    UdpIpv6,
    /// DeviceNet transport
    DeviceNet,
    /// ControlNet transport
    ControlNet,
    /// Profinet transport
    Profinet,
    /// Custom transport
    Custom { transport: String },
}

/// PTP profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PtpProfile {
    /// Default profile
    Default,
    /// Power profile (IEEE C37.238)
    Power,
    /// Telecom profile (ITU-T G.8265.1)
    Telecom,
    /// Enterprise profile
    Enterprise,
    /// Audio-video bridging profile
    AVB,
    /// Automotive profile
    Automotive,
    /// Utility profile (IEEE C37.238)
    Utility,
    /// Custom profile
    Custom { profile: String },
}

/// NTP (Network Time Protocol) implementation
#[derive(Debug, Clone)]
pub struct NtpProtocol {
    /// NTP configuration
    pub config: NtpConfig,
    /// Protocol state
    pub state: NtpState,
    /// Server pool
    pub servers: Vec<NtpServer>,
    /// Authentication manager
    pub auth_manager: Option<NtpAuthManager>,
    /// Statistics
    pub statistics: NtpStatistics,
}

/// NTP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NtpConfig {
    /// NTP version (3 or 4)
    pub version: u8,
    /// Server list
    pub servers: Vec<String>,
    /// Enable authentication
    pub authentication: bool,
    /// Poll interval
    pub poll_interval: Duration,
    /// Maximum poll interval
    pub max_poll_interval: Duration,
    /// Minimum poll interval
    pub min_poll_interval: Duration,
    /// Burst mode
    pub burst_mode: bool,
    /// Iburst mode
    pub iburst_mode: bool,
    /// Prefer this server
    pub prefer: bool,
}

/// NTP protocol state
#[derive(Debug, Clone)]
pub struct NtpState {
    /// Current stratum
    pub stratum: u8,
    /// Reference identifier
    pub reference_id: u32,
    /// Reference timestamp
    pub reference_timestamp: SystemTime,
    /// Root delay
    pub root_delay: Duration,
    /// Root dispersion
    pub root_dispersion: Duration,
    /// Precision
    pub precision: i8,
    /// Poll interval
    pub poll_interval: Duration,
    /// Clock state
    pub clock_state: NtpClockState,
}

/// NTP clock state
#[derive(Debug, Clone)]
pub enum NtpClockState {
    /// Clock not set
    NotSet,
    /// Clock frequency set
    FrequencySet,
    /// Clock synchronized
    Synchronized,
    /// Clock spike detected
    Spike,
    /// Clock step
    Step,
    /// Clock panic
    Panic,
}

/// NTP server information
#[derive(Debug, Clone)]
pub struct NtpServer {
    /// Server address
    pub address: String,
    /// Server port
    pub port: u16,
    /// Server stratum
    pub stratum: u8,
    /// Server poll interval
    pub poll_interval: Duration,
    /// Server reachability
    pub reachability: u8,
    /// Server delay
    pub delay: Duration,
    /// Server dispersion
    pub dispersion: Duration,
    /// Server jitter
    pub jitter: Duration,
    /// Server status
    pub status: NtpServerStatus,
}

/// NTP server status
#[derive(Debug, Clone)]
pub enum NtpServerStatus {
    /// Server reachable
    Reachable,
    /// Server unreachable
    Unreachable,
    /// Server candidate
    Candidate,
    /// Server selected
    Selected,
    /// Server peer
    Peer,
    /// Server rejected
    Rejected,
}

/// NTP authentication manager
#[derive(Debug, Clone)]
pub struct NtpAuthManager {
    /// Authentication keys
    pub keys: HashMap<u32, NtpAuthKey>,
    /// Trusted keys
    pub trusted_keys: Vec<u32>,
    /// Authentication enabled
    pub enabled: bool,
}

/// NTP authentication key
#[derive(Debug, Clone)]
pub struct NtpAuthKey {
    /// Key ID
    pub key_id: u32,
    /// Key type
    pub key_type: NtpAuthKeyType,
    /// Key value
    pub key_value: Vec<u8>,
    /// Key trusted
    pub trusted: bool,
}

/// NTP authentication key types
#[derive(Debug, Clone)]
pub enum NtpAuthKeyType {
    /// MD5 key
    MD5,
    /// SHA1 key
    SHA1,
    /// SHA256 key
    SHA256,
    /// Custom key type
    Custom { key_type: String },
}

/// NTP statistics
#[derive(Debug, Clone)]
pub struct NtpStatistics {
    /// Total packets sent
    pub packets_sent: u64,
    /// Total packets received
    pub packets_received: u64,
    /// Authentication failures
    pub auth_failures: u64,
    /// Bad format packets
    pub bad_format: u64,
    /// Kiss-of-death packets
    pub kiss_of_death: u64,
    /// Clock adjustments
    pub clock_adjustments: u64,
}

/// PTP (Precision Time Protocol) implementation
#[derive(Debug, Clone)]
pub struct PtpProtocol {
    /// PTP configuration
    pub config: PtpConfig,
    /// Protocol state
    pub state: PtpState,
    /// Port state
    pub port_state: PtpPortState,
    /// Clock identity
    pub clock_identity: PtpClockIdentity,
    /// Statistics
    pub statistics: PtpStatistics,
}

/// PTP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtpConfig {
    /// PTP version
    pub version: PtpVersion,
    /// PTP domain
    pub domain: u8,
    /// Transport protocol
    pub transport: PtpTransport,
    /// PTP profile
    pub profile: PtpProfile,
    /// Two-step flag
    pub two_step: bool,
    /// Delay mechanism
    pub delay_mechanism: PtpDelayMechanism,
    /// Announce interval
    pub announce_interval: i8,
    /// Sync interval
    pub sync_interval: i8,
    /// Delay request interval
    pub delay_req_interval: i8,
    /// Path trace enabled
    pub path_trace_enabled: bool,
}

/// PTP delay mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PtpDelayMechanism {
    /// End-to-end delay mechanism
    E2E,
    /// Peer-to-peer delay mechanism
    P2P,
    /// Disabled
    Disabled,
}

/// PTP protocol state
#[derive(Debug, Clone)]
pub struct PtpState {
    /// Clock state
    pub clock_state: PtpClockState,
    /// Number of fault records
    pub fault_record_count: u16,
    /// Mean path delay
    pub mean_path_delay: Duration,
    /// Offset from master
    pub offset_from_master: Duration,
    /// One way delay
    pub one_way_delay: Duration,
    /// Current UTC offset
    pub current_utc_offset: i16,
    /// Current UTC offset valid
    pub current_utc_offset_valid: bool,
}

/// PTP clock state
#[derive(Debug, Clone)]
pub enum PtpClockState {
    /// Initializing
    Initializing,
    /// Faulty
    Faulty,
    /// Disabled
    Disabled,
    /// Listening
    Listening,
    /// Pre-master
    PreMaster,
    /// Master
    Master,
    /// Passive
    Passive,
    /// Uncalibrated
    Uncalibrated,
    /// Slave
    Slave,
}

/// PTP port state
#[derive(Debug, Clone)]
pub struct PtpPortState {
    /// Port state
    pub state: PtpPortStateType,
    /// Log announce interval
    pub log_announce_interval: i8,
    /// Announce receipt timeout
    pub announce_receipt_timeout: u8,
    /// Log sync interval
    pub log_sync_interval: i8,
    /// Delay mechanism
    pub delay_mechanism: PtpDelayMechanism,
    /// Version number
    pub version_number: u8,
}

/// PTP port state types
#[derive(Debug, Clone)]
pub enum PtpPortStateType {
    /// Initializing
    Initializing,
    /// Faulty
    Faulty,
    /// Disabled
    Disabled,
    /// Listening
    Listening,
    /// Pre-master
    PreMaster,
    /// Master
    Master,
    /// Passive
    Passive,
    /// Uncalibrated
    Uncalibrated,
    /// Slave
    Slave,
}

/// PTP clock identity
#[derive(Debug, Clone)]
pub struct PtpClockIdentity {
    /// Clock identity (8 bytes)
    pub identity: [u8; 8],
    /// Priority 1
    pub priority1: u8,
    /// Priority 2
    pub priority2: u8,
    /// Clock class
    pub clock_class: u8,
    /// Clock accuracy
    pub clock_accuracy: u8,
    /// Clock variance
    pub clock_variance: u16,
}

/// PTP statistics
#[derive(Debug, Clone)]
pub struct PtpStatistics {
    /// Announce messages sent
    pub announce_sent: u64,
    /// Announce messages received
    pub announce_received: u64,
    /// Sync messages sent
    pub sync_sent: u64,
    /// Sync messages received
    pub sync_received: u64,
    /// Follow-up messages sent
    pub follow_up_sent: u64,
    /// Follow-up messages received
    pub follow_up_received: u64,
    /// Delay request messages sent
    pub delay_req_sent: u64,
    /// Delay request messages received
    pub delay_req_received: u64,
    /// Delay response messages sent
    pub delay_resp_sent: u64,
    /// Delay response messages received
    pub delay_resp_received: u64,
}

/// SNTP (Simple Network Time Protocol) implementation
#[derive(Debug, Clone)]
pub struct SntpProtocol {
    /// SNTP configuration
    pub config: SntpConfig,
    /// Protocol state
    pub state: SntpState,
    /// Server list
    pub servers: Vec<SntpServer>,
    /// Statistics
    pub statistics: SntpStatistics,
}

/// SNTP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SntpConfig {
    /// Server list
    pub servers: Vec<String>,
    /// Request timeout
    pub timeout: Duration,
    /// Retry count
    pub retry_count: u8,
    /// Port number
    pub port: u16,
    /// Local port
    pub local_port: Option<u16>,
}

/// SNTP protocol state
#[derive(Debug, Clone)]
pub struct SntpState {
    /// Current server index
    pub current_server: usize,
    /// Last successful sync
    pub last_sync: Option<Instant>,
    /// Retry count
    pub retry_count: u8,
    /// Protocol status
    pub status: SntpStatus,
}

/// SNTP status
#[derive(Debug, Clone)]
pub enum SntpStatus {
    /// Idle
    Idle,
    /// Requesting
    Requesting,
    /// Synchronized
    Synchronized,
    /// Failed
    Failed { reason: String },
}

/// SNTP server information
#[derive(Debug, Clone)]
pub struct SntpServer {
    /// Server address
    pub address: String,
    /// Server port
    pub port: u16,
    /// Server stratum
    pub stratum: u8,
    /// Response time
    pub response_time: Duration,
    /// Last response
    pub last_response: Option<Instant>,
    /// Success count
    pub success_count: u64,
    /// Failure count
    pub failure_count: u64,
}

/// SNTP statistics
#[derive(Debug, Clone)]
pub struct SntpStatistics {
    /// Total requests sent
    pub requests_sent: u64,
    /// Total responses received
    pub responses_received: u64,
    /// Timeout count
    pub timeouts: u64,
    /// Error count
    pub errors: u64,
}

/// Berkeley algorithm implementation
#[derive(Debug, Clone)]
pub struct BerkeleyProtocol {
    /// Berkeley configuration
    pub config: BerkeleyConfig,
    /// Protocol state
    pub state: BerkeleyState,
    /// Participant nodes
    pub participants: Vec<BerkeleyParticipant>,
    /// Statistics
    pub statistics: BerkeleyStatistics,
}

/// Berkeley algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BerkeleyConfig {
    /// Fault tolerance level
    pub fault_tolerance: usize,
    /// Convergence threshold
    pub convergence_threshold: Duration,
    /// Sync interval
    pub sync_interval: Duration,
    /// Master selection algorithm
    pub master_selection: MasterSelectionAlgorithm,
    /// Outlier detection
    pub outlier_detection: bool,
    /// Outlier threshold
    pub outlier_threshold: f64,
}

/// Master selection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MasterSelectionAlgorithm {
    /// Select by lowest node ID
    LowestId,
    /// Select by highest priority
    HighestPriority,
    /// Select by best clock quality
    BestClockQuality,
    /// Custom selection algorithm
    Custom { algorithm: String },
}

/// Berkeley algorithm state
#[derive(Debug, Clone)]
pub struct BerkeleyState {
    /// Current role
    pub role: BerkeleyRole,
    /// Master node ID
    pub master_id: Option<String>,
    /// Convergence status
    pub converged: bool,
    /// Last sync time
    pub last_sync: Option<Instant>,
    /// Sync round number
    pub sync_round: u64,
}

/// Berkeley algorithm roles
#[derive(Debug, Clone)]
pub enum BerkeleyRole {
    /// Master node
    Master,
    /// Slave node
    Slave,
    /// Observer node
    Observer,
}

/// Berkeley participant information
#[derive(Debug, Clone)]
pub struct BerkeleyParticipant {
    /// Participant ID
    pub id: String,
    /// Participant address
    pub address: String,
    /// Clock offset
    pub clock_offset: Duration,
    /// Round trip time
    pub round_trip_time: Duration,
    /// Reliability score
    pub reliability: f64,
    /// Last update
    pub last_update: Instant,
    /// Status
    pub status: ParticipantStatus,
}

/// Participant status
#[derive(Debug, Clone)]
pub enum ParticipantStatus {
    /// Active
    Active,
    /// Inactive
    Inactive,
    /// Faulty
    Faulty,
    /// Suspected
    Suspected,
}

/// Berkeley algorithm statistics
#[derive(Debug, Clone)]
pub struct BerkeleyStatistics {
    /// Total sync rounds
    pub sync_rounds: u64,
    /// Successful convergences
    pub successful_convergences: u64,
    /// Failed convergences
    pub failed_convergences: u64,
    /// Average convergence time
    pub avg_convergence_time: Duration,
    /// Participant count
    pub participant_count: usize,
}

/// Cristian's algorithm implementation
#[derive(Debug, Clone)]
pub struct CristianProtocol {
    /// Cristian configuration
    pub config: CristianConfig,
    /// Protocol state
    pub state: CristianState,
    /// Time server
    pub time_server: CristianTimeServer,
    /// Statistics
    pub statistics: CristianStatistics,
}

/// Cristian's algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CristianConfig {
    /// Time server address
    pub time_server: String,
    /// Uncertainty factor
    pub uncertainty_factor: f64,
    /// Request interval
    pub request_interval: Duration,
    /// Timeout
    pub timeout: Duration,
    /// Maximum round trip time
    pub max_round_trip_time: Duration,
}

/// Cristian's algorithm state
#[derive(Debug, Clone)]
pub struct CristianState {
    /// Last sync time
    pub last_sync: Option<Instant>,
    /// Current offset
    pub current_offset: Duration,
    /// Estimated uncertainty
    pub uncertainty: Duration,
    /// Status
    pub status: CristianStatus,
}

/// Cristian's algorithm status
#[derive(Debug, Clone)]
pub enum CristianStatus {
    /// Idle
    Idle,
    /// Requesting
    Requesting,
    /// Synchronized
    Synchronized,
    /// Failed
    Failed { reason: String },
}

/// Cristian time server
#[derive(Debug, Clone)]
pub struct CristianTimeServer {
    /// Server address
    pub address: String,
    /// Server port
    pub port: u16,
    /// Average round trip time
    pub avg_round_trip_time: Duration,
    /// Response time variance
    pub response_variance: Duration,
    /// Reliability score
    pub reliability: f64,
    /// Last response
    pub last_response: Option<Instant>,
}

/// Cristian's algorithm statistics
#[derive(Debug, Clone)]
pub struct CristianStatistics {
    /// Total requests sent
    pub requests_sent: u64,
    /// Total responses received
    pub responses_received: u64,
    /// Successful syncs
    pub successful_syncs: u64,
    /// Failed syncs
    pub failed_syncs: u64,
    /// Average round trip time
    pub avg_round_trip_time: Duration,
}

/// Custom protocol implementation
#[derive(Debug, Clone)]
pub struct CustomProtocol {
    /// Protocol name
    pub name: String,
    /// Protocol configuration
    pub config: HashMap<String, String>,
    /// Protocol state
    pub state: CustomProtocolState,
    /// Statistics
    pub statistics: CustomProtocolStatistics,
}

/// Custom protocol state
#[derive(Debug, Clone)]
pub struct CustomProtocolState {
    /// Protocol status
    pub status: String,
    /// State parameters
    pub parameters: HashMap<String, String>,
    /// Last update
    pub last_update: Instant,
}

/// Custom protocol statistics
#[derive(Debug, Clone)]
pub struct CustomProtocolStatistics {
    /// Protocol-specific metrics
    pub metrics: HashMap<String, f64>,
    /// Event counts
    pub event_counts: HashMap<String, u64>,
    /// Performance metrics
    pub performance: HashMap<String, Duration>,
}

/// Protocol factory for creating protocol instances
pub struct ProtocolFactory;

impl ProtocolFactory {
    /// Create a protocol instance from configuration
    pub fn create_protocol(protocol: &ClockSyncProtocol) -> Result<Box<dyn SyncProtocolTrait>> {
        match protocol {
            ClockSyncProtocol::NTP { .. } => {
                Ok(Box::new(NtpProtocol::new(protocol)?))
            },
            ClockSyncProtocol::PTP { .. } => {
                Ok(Box::new(PtpProtocol::new(protocol)?))
            },
            ClockSyncProtocol::SNTP { .. } => {
                Ok(Box::new(SntpProtocol::new(protocol)?))
            },
            ClockSyncProtocol::Berkeley { .. } => {
                Ok(Box::new(BerkeleyProtocol::new(protocol)?))
            },
            ClockSyncProtocol::Cristian { .. } => {
                Ok(Box::new(CristianProtocol::new(protocol)?))
            },
            ClockSyncProtocol::GPS { .. } => {
                // GPS protocol would be handled by the GPS module
                Err(scirs2_core::error::Error::InvalidInput("GPS protocol handled by GPS module".to_string()))
            },
            ClockSyncProtocol::Custom { .. } => {
                Ok(Box::new(CustomProtocol::new(protocol)?))
            },
        }
    }
}

/// Common trait for all synchronization protocols
pub trait SyncProtocolTrait: std::fmt::Debug + Send + Sync {
    /// Get the protocol name
    fn protocol_name(&self) -> &str;

    /// Initialize the protocol
    fn initialize(&mut self) -> Result<()>;

    /// Perform synchronization
    fn synchronize(&mut self) -> Result<SystemTime>;

    /// Get current offset
    fn get_offset(&self) -> Duration;

    /// Get synchronization quality
    fn get_quality(&self) -> f64;

    /// Check if synchronized
    fn is_synchronized(&self) -> bool;

    /// Get protocol statistics
    fn get_statistics(&self) -> HashMap<String, f64>;

    /// Shutdown the protocol
    fn shutdown(&mut self) -> Result<()>;
}

// Implementation blocks for protocols

impl NtpProtocol {
    /// Create a new NTP protocol instance
    pub fn new(config: &ClockSyncProtocol) -> Result<Self> {
        if let ClockSyncProtocol::NTP { version, servers, authentication } = config {
            Ok(Self {
                config: NtpConfig {
                    version: *version,
                    servers: servers.clone(),
                    authentication: *authentication,
                    poll_interval: Duration::from_secs(64),
                    max_poll_interval: Duration::from_secs(1024),
                    min_poll_interval: Duration::from_secs(16),
                    burst_mode: false,
                    iburst_mode: true,
                    prefer: false,
                },
                state: NtpState::default(),
                servers: Vec::new(),
                auth_manager: if *authentication { Some(NtpAuthManager::default()) } else { None },
                statistics: NtpStatistics::default(),
            })
        } else {
            Err(scirs2_core::error::Error::InvalidInput("Invalid NTP configuration".to_string()))
        }
    }
}

impl SyncProtocolTrait for NtpProtocol {
    fn protocol_name(&self) -> &str { "NTP" }

    fn initialize(&mut self) -> Result<()> {
        // NTP initialization logic
        self.state.clock_state = NtpClockState::NotSet;
        Ok(())
    }

    fn synchronize(&mut self) -> Result<SystemTime> {
        // NTP synchronization logic
        self.statistics.packets_sent += 1;
        Ok(SystemTime::now())
    }

    fn get_offset(&self) -> Duration { Duration::from_millis(0) }
    fn get_quality(&self) -> f64 { 0.95 }
    fn is_synchronized(&self) -> bool { matches!(self.state.clock_state, NtpClockState::Synchronized) }
    fn get_statistics(&self) -> HashMap<String, f64> { HashMap::new() }
    fn shutdown(&mut self) -> Result<()> { Ok(()) }
}

impl PtpProtocol {
    /// Create a new PTP protocol instance
    pub fn new(config: &ClockSyncProtocol) -> Result<Self> {
        if let ClockSyncProtocol::PTP { version, domain, transport, profile } = config {
            Ok(Self {
                config: PtpConfig {
                    version: version.clone(),
                    domain: *domain,
                    transport: transport.clone(),
                    profile: profile.clone(),
                    two_step: false,
                    delay_mechanism: PtpDelayMechanism::E2E,
                    announce_interval: 1,
                    sync_interval: 0,
                    delay_req_interval: 0,
                    path_trace_enabled: false,
                },
                state: PtpState::default(),
                port_state: PtpPortState::default(),
                clock_identity: PtpClockIdentity::default(),
                statistics: PtpStatistics::default(),
            })
        } else {
            Err(scirs2_core::error::Error::InvalidInput("Invalid PTP configuration".to_string()))
        }
    }
}

impl SyncProtocolTrait for PtpProtocol {
    fn protocol_name(&self) -> &str { "PTP" }

    fn initialize(&mut self) -> Result<()> {
        self.state.clock_state = PtpClockState::Initializing;
        self.port_state.state = PtpPortStateType::Initializing;
        Ok(())
    }

    fn synchronize(&mut self) -> Result<SystemTime> {
        self.statistics.sync_sent += 1;
        Ok(SystemTime::now())
    }

    fn get_offset(&self) -> Duration { self.state.offset_from_master }
    fn get_quality(&self) -> f64 { 0.98 }
    fn is_synchronized(&self) -> bool { matches!(self.state.clock_state, PtpClockState::Slave) }
    fn get_statistics(&self) -> HashMap<String, f64> { HashMap::new() }
    fn shutdown(&mut self) -> Result<()> { Ok(()) }
}

impl SntpProtocol {
    /// Create a new SNTP protocol instance
    pub fn new(config: &ClockSyncProtocol) -> Result<Self> {
        if let ClockSyncProtocol::SNTP { servers, timeout } = config {
            Ok(Self {
                config: SntpConfig {
                    servers: servers.clone(),
                    timeout: *timeout,
                    retry_count: 3,
                    port: 123,
                    local_port: None,
                },
                state: SntpState::default(),
                servers: Vec::new(),
                statistics: SntpStatistics::default(),
            })
        } else {
            Err(scirs2_core::error::Error::InvalidInput("Invalid SNTP configuration".to_string()))
        }
    }
}

impl SyncProtocolTrait for SntpProtocol {
    fn protocol_name(&self) -> &str { "SNTP" }

    fn initialize(&mut self) -> Result<()> {
        self.state.status = SntpStatus::Idle;
        Ok(())
    }

    fn synchronize(&mut self) -> Result<SystemTime> {
        self.statistics.requests_sent += 1;
        Ok(SystemTime::now())
    }

    fn get_offset(&self) -> Duration { Duration::from_millis(0) }
    fn get_quality(&self) -> f64 { 0.85 }
    fn is_synchronized(&self) -> bool { matches!(self.state.status, SntpStatus::Synchronized) }
    fn get_statistics(&self) -> HashMap<String, f64> { HashMap::new() }
    fn shutdown(&mut self) -> Result<()> { Ok(()) }
}

impl BerkeleyProtocol {
    /// Create a new Berkeley protocol instance
    pub fn new(config: &ClockSyncProtocol) -> Result<Self> {
        if let ClockSyncProtocol::Berkeley { fault_tolerance, convergence_threshold } = config {
            Ok(Self {
                config: BerkeleyConfig {
                    fault_tolerance: *fault_tolerance,
                    convergence_threshold: *convergence_threshold,
                    sync_interval: Duration::from_secs(30),
                    master_selection: MasterSelectionAlgorithm::LowestId,
                    outlier_detection: true,
                    outlier_threshold: 3.0,
                },
                state: BerkeleyState::default(),
                participants: Vec::new(),
                statistics: BerkeleyStatistics::default(),
            })
        } else {
            Err(scirs2_core::error::Error::InvalidInput("Invalid Berkeley configuration".to_string()))
        }
    }
}

impl SyncProtocolTrait for BerkeleyProtocol {
    fn protocol_name(&self) -> &str { "Berkeley" }

    fn initialize(&mut self) -> Result<()> {
        self.state.role = BerkeleyRole::Slave;
        Ok(())
    }

    fn synchronize(&mut self) -> Result<SystemTime> {
        self.statistics.sync_rounds += 1;
        Ok(SystemTime::now())
    }

    fn get_offset(&self) -> Duration { Duration::from_millis(0) }
    fn get_quality(&self) -> f64 { 0.90 }
    fn is_synchronized(&self) -> bool { self.state.converged }
    fn get_statistics(&self) -> HashMap<String, f64> { HashMap::new() }
    fn shutdown(&mut self) -> Result<()> { Ok(()) }
}

impl CristianProtocol {
    /// Create a new Cristian protocol instance
    pub fn new(config: &ClockSyncProtocol) -> Result<Self> {
        if let ClockSyncProtocol::Cristian { time_server, uncertainty_factor } = config {
            Ok(Self {
                config: CristianConfig {
                    time_server: time_server.clone(),
                    uncertainty_factor: *uncertainty_factor,
                    request_interval: Duration::from_secs(60),
                    timeout: Duration::from_secs(5),
                    max_round_trip_time: Duration::from_millis(100),
                },
                state: CristianState::default(),
                time_server: CristianTimeServer {
                    address: time_server.clone(),
                    port: 123,
                    avg_round_trip_time: Duration::from_millis(10),
                    response_variance: Duration::from_millis(2),
                    reliability: 0.95,
                    last_response: None,
                },
                statistics: CristianStatistics::default(),
            })
        } else {
            Err(scirs2_core::error::Error::InvalidInput("Invalid Cristian configuration".to_string()))
        }
    }
}

impl SyncProtocolTrait for CristianProtocol {
    fn protocol_name(&self) -> &str { "Cristian" }

    fn initialize(&mut self) -> Result<()> {
        self.state.status = CristianStatus::Idle;
        Ok(())
    }

    fn synchronize(&mut self) -> Result<SystemTime> {
        self.statistics.requests_sent += 1;
        Ok(SystemTime::now())
    }

    fn get_offset(&self) -> Duration { self.state.current_offset }
    fn get_quality(&self) -> f64 { 0.88 }
    fn is_synchronized(&self) -> bool { matches!(self.state.status, CristianStatus::Synchronized) }
    fn get_statistics(&self) -> HashMap<String, f64> { HashMap::new() }
    fn shutdown(&mut self) -> Result<()> { Ok(()) }
}

impl CustomProtocol {
    /// Create a new custom protocol instance
    pub fn new(config: &ClockSyncProtocol) -> Result<Self> {
        if let ClockSyncProtocol::Custom { protocol_name, parameters } = config {
            Ok(Self {
                name: protocol_name.clone(),
                config: parameters.clone(),
                state: CustomProtocolState {
                    status: "initialized".to_string(),
                    parameters: HashMap::new(),
                    last_update: Instant::now(),
                },
                statistics: CustomProtocolStatistics {
                    metrics: HashMap::new(),
                    event_counts: HashMap::new(),
                    performance: HashMap::new(),
                },
            })
        } else {
            Err(scirs2_core::error::Error::InvalidInput("Invalid Custom configuration".to_string()))
        }
    }
}

impl SyncProtocolTrait for CustomProtocol {
    fn protocol_name(&self) -> &str { &self.name }

    fn initialize(&mut self) -> Result<()> {
        self.state.status = "initialized".to_string();
        Ok(())
    }

    fn synchronize(&mut self) -> Result<SystemTime> {
        *self.statistics.event_counts.entry("sync_calls".to_string()).or_insert(0) += 1;
        Ok(SystemTime::now())
    }

    fn get_offset(&self) -> Duration { Duration::from_millis(0) }
    fn get_quality(&self) -> f64 { 0.80 }
    fn is_synchronized(&self) -> bool { self.state.status == "synchronized" }
    fn get_statistics(&self) -> HashMap<String, f64> { self.statistics.metrics.clone() }
    fn shutdown(&mut self) -> Result<()> { Ok(()) }
}

// Default implementations

impl Default for ClockSyncProtocol {
    fn default() -> Self {
        ClockSyncProtocol::NTP {
            version: 4,
            servers: vec!["pool.ntp.org".to_string()],
            authentication: false,
        }
    }
}

impl Default for NtpState {
    fn default() -> Self {
        Self {
            stratum: 16, // Unsynchronized
            reference_id: 0,
            reference_timestamp: SystemTime::now(),
            root_delay: Duration::from_millis(0),
            root_dispersion: Duration::from_millis(0),
            precision: -6, // 1/64 second
            poll_interval: Duration::from_secs(64),
            clock_state: NtpClockState::NotSet,
        }
    }
}

impl Default for NtpAuthManager {
    fn default() -> Self {
        Self {
            keys: HashMap::new(),
            trusted_keys: Vec::new(),
            enabled: false,
        }
    }
}

impl Default for NtpStatistics {
    fn default() -> Self {
        Self {
            packets_sent: 0,
            packets_received: 0,
            auth_failures: 0,
            bad_format: 0,
            kiss_of_death: 0,
            clock_adjustments: 0,
        }
    }
}

impl Default for PtpState {
    fn default() -> Self {
        Self {
            clock_state: PtpClockState::Initializing,
            fault_record_count: 0,
            mean_path_delay: Duration::from_millis(0),
            offset_from_master: Duration::from_millis(0),
            one_way_delay: Duration::from_millis(0),
            current_utc_offset: 37, // Current TAI-UTC offset
            current_utc_offset_valid: false,
        }
    }
}

impl Default for PtpPortState {
    fn default() -> Self {
        Self {
            state: PtpPortStateType::Initializing,
            log_announce_interval: 1,
            announce_receipt_timeout: 3,
            log_sync_interval: 0,
            delay_mechanism: PtpDelayMechanism::E2E,
            version_number: 2,
        }
    }
}

impl Default for PtpClockIdentity {
    fn default() -> Self {
        Self {
            identity: [0; 8],
            priority1: 128,
            priority2: 128,
            clock_class: 248, // Default for slave-only clock
            clock_accuracy: 0x31, // Unknown
            clock_variance: 0xFFFF, // Unknown
        }
    }
}

impl Default for PtpStatistics {
    fn default() -> Self {
        Self {
            announce_sent: 0,
            announce_received: 0,
            sync_sent: 0,
            sync_received: 0,
            follow_up_sent: 0,
            follow_up_received: 0,
            delay_req_sent: 0,
            delay_req_received: 0,
            delay_resp_sent: 0,
            delay_resp_received: 0,
        }
    }
}

impl Default for SntpState {
    fn default() -> Self {
        Self {
            current_server: 0,
            last_sync: None,
            retry_count: 0,
            status: SntpStatus::Idle,
        }
    }
}

impl Default for SntpStatistics {
    fn default() -> Self {
        Self {
            requests_sent: 0,
            responses_received: 0,
            timeouts: 0,
            errors: 0,
        }
    }
}

impl Default for BerkeleyState {
    fn default() -> Self {
        Self {
            role: BerkeleyRole::Slave,
            master_id: None,
            converged: false,
            last_sync: None,
            sync_round: 0,
        }
    }
}

impl Default for BerkeleyStatistics {
    fn default() -> Self {
        Self {
            sync_rounds: 0,
            successful_convergences: 0,
            failed_convergences: 0,
            avg_convergence_time: Duration::from_millis(0),
            participant_count: 0,
        }
    }
}

impl Default for CristianState {
    fn default() -> Self {
        Self {
            last_sync: None,
            current_offset: Duration::from_millis(0),
            uncertainty: Duration::from_millis(100),
            status: CristianStatus::Idle,
        }
    }
}

impl Default for CristianStatistics {
    fn default() -> Self {
        Self {
            requests_sent: 0,
            responses_received: 0,
            successful_syncs: 0,
            failed_syncs: 0,
            avg_round_trip_time: Duration::from_millis(0),
        }
    }
}