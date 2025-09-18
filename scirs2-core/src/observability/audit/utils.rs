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
        use std::net::TcpStream;

        // Try to connect to a remote address to determine local IP
        if let Ok(stream) = TcpStream::connect("8.8.8.8:80") {
            if let Ok(local_addr) = stream.local_addr() {
                return Some(local_addr.ip().to_string());
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
