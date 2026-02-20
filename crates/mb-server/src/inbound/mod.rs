pub mod openai_chat;
mod openai_wire;

use mb_core::core::{ApiSpec, InboundAdapter};

/// Registry of all available inbound adapters, keyed by API spec.
///
/// Uses linear scan over a small vec (~3 specs max) rather than a HashMap,
/// since `ApiSpec` does not implement `Hash`.
pub struct InboundAdapterRegistry {
    adapters: Vec<(ApiSpec, Box<dyn InboundAdapter>)>,
}

impl InboundAdapterRegistry {
    pub fn new() -> Self {
        let adapters: Vec<(ApiSpec, Box<dyn InboundAdapter>)> = vec![(
            ApiSpec::OpenAiChat,
            Box::new(openai_chat::OpenAiChatInboundAdapter),
        )];
        Self { adapters }
    }

    pub fn get(&self, spec: &ApiSpec) -> Option<&dyn InboundAdapter> {
        self.adapters
            .iter()
            .find(|(s, _)| s == spec)
            .map(|(_, adapter)| adapter.as_ref())
    }
}

impl Default for InboundAdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_returns_openai_chat() {
        let registry = InboundAdapterRegistry::new();

        let adapter = registry.get(&ApiSpec::OpenAiChat);

        assert!(adapter.is_some());
        assert_eq!(adapter.unwrap().api_spec(), ApiSpec::OpenAiChat);
    }

    #[test]
    fn test_registry_returns_none_for_unregistered() {
        let registry = InboundAdapterRegistry::new();

        let adapter = registry.get(&ApiSpec::OpenAiResponses);

        assert!(adapter.is_none());
    }
}
