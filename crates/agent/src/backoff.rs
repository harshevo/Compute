use std::time::Duration;

use rand::Rng;

#[derive(Clone, Copy, Debug)]
pub struct BackoffPolicy {
    pub initial: Duration,
    pub max: Duration,
}

impl Default for BackoffPolicy {
    fn default() -> Self {
        Self {
            initial: Duration::from_secs(1),
            max: Duration::from_secs(60),
        }
    }
}

/// Exponential backoff with jitter, so reconnecting nodes do not all retry at once.
pub struct Backoff {
    policy: BackoffPolicy,
    attempt: u32,
}

impl Backoff {
    pub fn new(policy: BackoffPolicy) -> Self {
        Self { policy, attempt: 0 }
    }

    pub fn reset(&mut self) {
        self.attempt = 0;
    }

    pub fn next_delay(&mut self) -> Duration {
        let ceiling = self.ceiling();
        self.attempt = self.attempt.saturating_add(1);
        ceiling.mul_f64(rand::rng().random_range(0.5..=1.0))
    }

    fn ceiling(&self) -> Duration {
        let doublings = 2u32.saturating_pow(self.attempt);
        self.policy
            .initial
            .saturating_mul(doublings)
            .min(self.policy.max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICY: BackoffPolicy = BackoffPolicy {
        initial: Duration::from_secs(1),
        max: Duration::from_secs(8),
    };

    #[test]
    fn delays_double_up_to_the_maximum() {
        let mut backoff = Backoff::new(POLICY);
        let ceilings: Vec<_> = (0..6)
            .map(|_| {
                let ceiling = backoff.ceiling();
                backoff.next_delay();
                ceiling.as_secs()
            })
            .collect();
        assert_eq!(ceilings, [1, 2, 4, 8, 8, 8]);
    }

    #[test]
    fn jitter_keeps_each_delay_between_half_and_full_ceiling() {
        for _ in 0..200 {
            let mut backoff = Backoff::new(POLICY);
            backoff.next_delay();
            let ceiling = backoff.ceiling();
            let delay = backoff.next_delay();
            assert!(delay >= ceiling / 2 && delay <= ceiling, "{delay:?}");
        }
    }

    #[test]
    fn reset_starts_over() {
        let mut backoff = Backoff::new(POLICY);
        backoff.next_delay();
        backoff.next_delay();
        backoff.reset();
        assert_eq!(backoff.ceiling(), POLICY.initial);
    }

    #[test]
    fn many_failures_do_not_overflow() {
        let mut backoff = Backoff::new(POLICY);
        for _ in 0..100 {
            assert!(backoff.next_delay() <= POLICY.max);
        }
    }
}
