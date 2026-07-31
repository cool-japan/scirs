//! Utility functions for the audit logging system

/// Get thread ID as a numeric value
#[must_use]
#[allow(dead_code)]
pub fn get_thread_id() -> u64 {
    use std::thread;
    // This is a simplified implementation
    // In production, you'd use proper thread ID detection
    format!("{:?}", thread::current().id())
        .chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| d as u64)
        .fold(0, |acc, d| acc * 10 + d)
}

/// Get hostname from environment variables
#[must_use]
#[allow(dead_code)]
pub fn get_hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Get local IP address
#[must_use]
#[allow(dead_code)]
pub fn get_local_ip() -> Option<String> {
    // Try to get the actual local IP address
    #[cfg(feature = "sysinfo")]
    {
        use std::net::UdpSocket;

        // Determine the local IP via a UDP "connect": for a datagram socket
        // this only asks the OS routing table which local interface/address
        // would be used to reach the given destination -- no handshake and no
        // packet is ever actually sent on the wire, so it resolves instantly
        // regardless of network reachability.
        //
        // NOTE: this previously used `TcpStream::connect("8.8.8.8:80")`,
        // which performs a real SYN and blocks for the platform's full TCP
        // connect timeout (~75s observed on macOS) whenever the destination
        // is unreachable -- e.g. in firewalled/air-gapped/network-restricted
        // deployments. Because `include_system_context` defaults to `true`,
        // that made every default-configured audit-logged event pay a ~75s
        // stall in exactly the kind of environment audit logging is normally
        // deployed into. The UDP-based lookup below cannot block on the
        // network at all.
        if let Ok(socket) = UdpSocket::bind("0.0.0.0:0") {
            if socket.connect("8.8.8.8:80").is_ok() {
                if let Ok(local_addr) = socket.local_addr() {
                    return Some(local_addr.ip().to_string());
                }
            }
        }

        // Fallback: try to get from network interfaces
        // This would require additional network interface detection
        // For now, return a reasonable default
        Some("127.0.0.1".to_string())
    }

    #[cfg(not(feature = "sysinfo"))]
    {
        // Simple fallback without network detection
        use std::env;

        // Check for common environment variables that might contain IP
        if let Ok(ip) = env::var("HOST_IP") {
            return Some(ip);
        }

        if let Ok(ip) = env::var("LOCAL_IP") {
            return Some(ip);
        }

        // Default fallback
        Some("127.0.0.1".to_string())
    }
}

/// Get simplified stack trace
#[must_use]
#[allow(dead_code)]
pub fn get_stack_trace() -> String {
    // Simplified stack trace implementation for compatibility
    let mut result = String::new();
    result.push_str("Stack trace (simplified):\n");

    // Get current thread and function info
    if let Some(name) = std::thread::current().name() {
        result.push_str(&format!("  Thread: {name}\n"));
    } else {
        result.push_str("  Thread: <unnamed>\n");
    }

    // Add caller information (simplified)
    result.push_str(&format!(
        "  Location: {}:{}:{}\n",
        file!(),
        line!(),
        column!()
    ));

    result
}
