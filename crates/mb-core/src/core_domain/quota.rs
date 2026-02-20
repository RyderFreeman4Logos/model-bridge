use std::collections::{HashMap, VecDeque};

use crate::core::{ClientId, QuotaConfig, QuotaInfo, RateLimitInfo, YearMonth};

// ---------------------------------------------------------------------------
// RateLimiter — sliding-window request rate limiter (pure, no system clock)
// ---------------------------------------------------------------------------

pub struct RateLimiter {
    window_ms: u64,
    limit: u32,
    timestamps: VecDeque<u64>,
}

impl RateLimiter {
    pub fn new(window_ms: u64, limit: u32) -> Self {
        Self {
            window_ms,
            limit,
            timestamps: VecDeque::new(),
        }
    }

    /// Check whether a request at `now_ms` is within the rate limit.
    ///
    /// On success, records the timestamp and returns `Ok(())`.
    /// On rejection, returns `Err(RateLimitInfo)` with the time until
    /// the next slot opens.
    pub fn check(&mut self, now_ms: u64) -> Result<(), RateLimitInfo> {
        let window_start = now_ms.saturating_sub(self.window_ms);
        while let Some(&front) = self.timestamps.front() {
            if front < window_start {
                self.timestamps.pop_front();
            } else {
                break;
            }
        }

        if self.timestamps.len() >= self.limit as usize {
            let earliest = self.timestamps.front().copied().unwrap_or(now_ms);
            let retry_after_ms = (earliest + self.window_ms).saturating_sub(now_ms);
            return Err(RateLimitInfo { retry_after_ms });
        }

        self.timestamps.push_back(now_ms);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MonthlyUsage — per-client token consumption for a billing period
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct MonthlyUsage {
    pub period: YearMonth,
    pub tokens_used: u64,
}

// ---------------------------------------------------------------------------
// QuotaTracker — monthly token quota enforcement (pure, no system clock)
// ---------------------------------------------------------------------------

pub struct QuotaTracker {
    usage: HashMap<ClientId, MonthlyUsage>,
}

impl Default for QuotaTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl QuotaTracker {
    pub fn new() -> Self {
        Self {
            usage: HashMap::new(),
        }
    }

    /// Check whether `client` has sufficient quota for `estimated_tokens`.
    ///
    /// Returns `Ok(())` if the client has no monthly limit or is within budget.
    /// Returns `Err(QuotaInfo)` if the estimated usage would exceed the limit.
    pub fn check(
        &self,
        client: &ClientId,
        estimated_tokens: u64,
        config: &QuotaConfig,
        current_period: YearMonth,
    ) -> Result<(), QuotaInfo> {
        let limit = match config.monthly_token_limit {
            Some(l) => l,
            None => return Ok(()),
        };

        let used = self
            .usage
            .get(client)
            .filter(|u| u.period == current_period)
            .map_or(0, |u| u.tokens_used);

        if used + estimated_tokens > limit {
            Err(QuotaInfo { limit, used })
        } else {
            Ok(())
        }
    }

    /// Record actual token consumption for `client` in `current_period`.
    ///
    /// Resets the counter when the period changes (month rollover).
    pub fn record(&mut self, client: &ClientId, actual_tokens: u64, current_period: YearMonth) {
        let entry = self
            .usage
            .entry(client.clone())
            .or_insert_with(|| MonthlyUsage {
                period: current_period,
                tokens_used: 0,
            });

        if entry.period != current_period {
            entry.period = current_period;
            entry.tokens_used = 0;
        }

        entry.tokens_used += actual_tokens;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- RateLimiter --

    #[test]
    fn test_rate_limiter_under_limit() {
        let mut limiter = RateLimiter::new(60_000, 3);
        assert!(limiter.check(1000).is_ok());
        assert!(limiter.check(2000).is_ok());
    }

    #[test]
    fn test_rate_limiter_at_limit() {
        let mut limiter = RateLimiter::new(60_000, 2);
        assert!(limiter.check(1000).is_ok());
        assert!(limiter.check(2000).is_ok());

        let err = limiter.check(3000).unwrap_err();
        // Earliest timestamp (1000) + window (60000) - now (3000) = 58000
        assert_eq!(err.retry_after_ms, 58_000);
    }

    #[test]
    fn test_rate_limiter_window_slides() {
        let mut limiter = RateLimiter::new(10_000, 2);
        assert!(limiter.check(1000).is_ok());
        assert!(limiter.check(2000).is_ok());
        assert!(limiter.check(5000).is_err());

        // At t=12000, the timestamp at t=1000 has expired (12000 - 10000 = 2000 > 1000)
        assert!(limiter.check(12_000).is_ok());
    }

    // -- QuotaTracker --

    #[test]
    fn test_quota_under_limit() {
        let mut tracker = QuotaTracker::new();
        let client = ClientId::new("team-alpha");
        let period = YearMonth::new(2025, 6);
        let config = QuotaConfig {
            monthly_token_limit: Some(100_000),
        };

        tracker.record(&client, 50_000, period);
        assert!(tracker.check(&client, 10_000, &config, period).is_ok());
    }

    #[test]
    fn test_quota_over_limit() {
        let mut tracker = QuotaTracker::new();
        let client = ClientId::new("team-alpha");
        let period = YearMonth::new(2025, 6);
        let config = QuotaConfig {
            monthly_token_limit: Some(100_000),
        };

        tracker.record(&client, 95_000, period);
        let err = tracker.check(&client, 10_000, &config, period).unwrap_err();
        assert_eq!(err.limit, 100_000);
        assert_eq!(err.used, 95_000);
    }

    #[test]
    fn test_quota_month_rollover() {
        let mut tracker = QuotaTracker::new();
        let client = ClientId::new("team-alpha");
        let june = YearMonth::new(2025, 6);
        let july = YearMonth::new(2025, 7);
        let config = QuotaConfig {
            monthly_token_limit: Some(100_000),
        };

        tracker.record(&client, 99_000, june);
        // New month resets usage — should be well within limit
        assert!(tracker.check(&client, 50_000, &config, july).is_ok());
    }

    #[test]
    fn test_quota_unlimited() {
        let tracker = QuotaTracker::new();
        let client = ClientId::new("team-alpha");
        let period = YearMonth::new(2025, 6);
        let config = QuotaConfig {
            monthly_token_limit: None,
        };

        assert!(tracker.check(&client, 999_999_999, &config, period).is_ok());
    }
}
