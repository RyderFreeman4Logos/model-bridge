pub mod ollama;
pub mod openai_chat;

use mb_core::core::{BackendSpec, OutboundAdapter};

/// Registry of all available outbound adapters, keyed by backend spec.
///
/// Uses linear scan over a small vec (~2 specs max) rather than a HashMap,
/// since `BackendSpec` does not implement `Hash`.
pub struct OutboundAdapterRegistry {
    adapters: Vec<(BackendSpec, Box<dyn OutboundAdapter>)>,
}

impl OutboundAdapterRegistry {
    pub fn new() -> Self {
        let adapters: Vec<(BackendSpec, Box<dyn OutboundAdapter>)> = vec![
            (
                BackendSpec::OpenAiChat,
                Box::new(openai_chat::OpenAiChatOutboundAdapter),
            ),
            (BackendSpec::Ollama, Box::new(ollama::OllamaOutboundAdapter)),
        ];
        Self { adapters }
    }

    pub fn get(&self, spec: &BackendSpec) -> Option<&dyn OutboundAdapter> {
        self.adapters
            .iter()
            .find(|(s, _)| s == spec)
            .map(|(_, adapter)| adapter.as_ref())
    }
}

impl Default for OutboundAdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_returns_openai_chat() {
        let registry = OutboundAdapterRegistry::new();
        let adapter = registry.get(&BackendSpec::OpenAiChat);
        assert!(adapter.is_some());
        assert_eq!(adapter.unwrap().backend_spec(), BackendSpec::OpenAiChat);
    }

    #[test]
    fn test_registry_returns_ollama() {
        let registry = OutboundAdapterRegistry::new();
        let adapter = registry.get(&BackendSpec::Ollama);
        assert!(adapter.is_some());
        assert_eq!(adapter.unwrap().backend_spec(), BackendSpec::Ollama);
    }
}
