use std::fmt;

// ---------------------------------------------------------------------------
// String-based identity newtypes
// ---------------------------------------------------------------------------

macro_rules! string_newtype {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

string_newtype!(ClientId);
string_newtype!(BackendId);
string_newtype!(ModelId);
string_newtype!(RequestId);

// ---------------------------------------------------------------------------
// PrefixHash — session prefix hash for cache-aware routing
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrefixHash(u64);

impl PrefixHash {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn value(&self) -> u64 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// LatencyMs — millisecond latency value
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LatencyMs(u64);

impl LatencyMs {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn value(&self) -> u64 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// YearMonth — quota period identifier
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct YearMonth {
    year: u16,
    month: u8,
}

impl YearMonth {
    pub fn new(year: u16, month: u8) -> Self {
        assert!(
            (1..=12).contains(&month),
            "month must be 1..=12, got {month}"
        );
        Self { year, month }
    }

    pub fn year(&self) -> u16 {
        self.year
    }

    pub fn month(&self) -> u8 {
        self.month
    }
}

// ---------------------------------------------------------------------------
// ApiKey — secret value object with redacted Debug and constant-time PartialEq
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ApiKey(String);

impl ApiKey {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for ApiKey {
    fn eq(&self, other: &Self) -> bool {
        let a = self.0.as_bytes();
        let b = other.0.as_bytes();
        let max_len = a.len().max(b.len());
        let mut result = (a.len() != b.len()) as u8;
        for i in 0..max_len {
            let x = if i < a.len() { a[i] } else { 0 };
            let y = if i < b.len() { b[i] } else { 0 };
            result |= x ^ y;
        }
        result == 0
    }
}

impl Eq for ApiKey {}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix: String = self.0.chars().take(6).collect();
        if prefix.chars().count() == 6 {
            write!(f, "ApiKey({prefix}...)")
        } else {
            write!(f, "ApiKey(***)")
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_redacted_debug() {
        let key = ApiKey::new("mb-sk-abcdef1234567890");
        let debug = format!("{key:?}");
        assert_eq!(debug, "ApiKey(mb-sk-...)");
        assert!(!debug.contains("abcdef1234567890"));

        let short_key = ApiKey::new("short");
        let debug_short = format!("{short_key:?}");
        assert_eq!(debug_short, "ApiKey(***)");
    }

    #[test]
    fn test_api_key_constant_time_eq() {
        let key_a = ApiKey::new("mb-sk-abcdef1234567890");
        let key_b = ApiKey::new("mb-sk-abcdef1234567890");
        let key_c = ApiKey::new("mb-sk-different0000000");
        let key_d = ApiKey::new("short");

        assert_eq!(key_a, key_b);
        assert_ne!(key_a, key_c);
        assert_ne!(key_a, key_d);
    }

    #[test]
    fn test_api_key_as_str() {
        let key = ApiKey::new("mb-sk-test123");
        assert_eq!(key.as_str(), "mb-sk-test123");
    }

    #[test]
    fn test_year_month() {
        let ym = YearMonth::new(2025, 6);
        assert_eq!(ym.year(), 2025);
        assert_eq!(ym.month(), 6);
    }

    #[test]
    fn test_api_key_different_length_eq() {
        let short = ApiKey::new("mb-sk-abc");
        let long = ApiKey::new("mb-sk-abcdef1234567890");
        assert_ne!(short, long);
        assert_ne!(long, short);
    }

    #[test]
    fn test_api_key_non_ascii_debug() {
        let key = ApiKey::new("日本語テスト文字列");
        let debug = format!("{key:?}");
        assert!(debug.starts_with("ApiKey("));
    }

    #[test]
    #[should_panic(expected = "month must be 1..=12")]
    fn test_year_month_invalid_month_zero() {
        YearMonth::new(2025, 0);
    }

    #[test]
    #[should_panic(expected = "month must be 1..=12")]
    fn test_year_month_invalid_month_thirteen() {
        YearMonth::new(2025, 13);
    }

    #[test]
    fn test_display_impls() {
        assert_eq!(ClientId::new("team-alpha").to_string(), "team-alpha");
        assert_eq!(BackendId::new("gpu-desktop").to_string(), "gpu-desktop");
        assert_eq!(ModelId::new("llama3-70b").to_string(), "llama3-70b");
        assert_eq!(RequestId::new("req-001").to_string(), "req-001");
    }
}
