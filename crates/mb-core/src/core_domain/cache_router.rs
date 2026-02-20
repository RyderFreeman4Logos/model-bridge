use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::core::{BackendId, ContentPart, Message, MessageContent, ModelId, PrefixHash, Role};

// ---------------------------------------------------------------------------
// CacheAffinityMap — LRU-bounded map of (ModelId, PrefixHash) -> BackendId
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AffinityEntry {
    backend: BackendId,
    last_used: u64,
    hit_count: u64,
}

pub struct CacheAffinityMap {
    entries: HashMap<(ModelId, PrefixHash), AffinityEntry>,
    max_entries: usize,
    counter: u64,
}

impl CacheAffinityMap {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            counter: 0,
        }
    }

    pub fn get(&mut self, model: &ModelId, prefix: PrefixHash) -> Option<&BackendId> {
        let key = (model.clone(), prefix);
        if self.entries.contains_key(&key) {
            self.counter += 1;
            let entry = self.entries.get_mut(&key).expect("checked above");
            entry.last_used = self.counter;
            entry.hit_count += 1;
            Some(&entry.backend)
        } else {
            None
        }
    }

    pub fn record(&mut self, model: &ModelId, prefix: PrefixHash, backend: &BackendId) {
        self.counter += 1;
        let key = (model.clone(), prefix);
        self.entries
            .entry(key)
            .and_modify(|e| {
                e.backend = backend.clone();
                e.last_used = self.counter;
                e.hit_count += 1;
            })
            .or_insert_with(|| AffinityEntry {
                backend: backend.clone(),
                last_used: self.counter,
                hit_count: 1,
            });

        if self.entries.len() > self.max_entries {
            self.evict_lru();
        }
    }

    pub fn evict_backend(&mut self, backend: &BackendId) {
        self.entries.retain(|_, entry| entry.backend != *backend);
    }

    fn evict_lru(&mut self) {
        if let Some(oldest_key) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone())
        {
            self.entries.remove(&oldest_key);
        }
    }
}

// ---------------------------------------------------------------------------
// Prefix hash computation
// ---------------------------------------------------------------------------

pub fn compute_prefix_hash(messages: &[Message], prefix_depth: usize) -> PrefixHash {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut count = 0;

    for msg in messages {
        if count >= prefix_depth {
            break;
        }
        if !matches!(msg.role, Role::System | Role::User) {
            continue;
        }
        hash_message_content(&msg.content, &mut hasher);
        count += 1;
    }

    PrefixHash::new(hasher.finish())
}

fn hash_message_content(content: &MessageContent, hasher: &mut impl Hasher) {
    match content {
        MessageContent::Text(s) => s.hash(hasher),
        MessageContent::Parts(parts) => {
            for part in parts {
                if let ContentPart::Text { text } = part {
                    text.hash(hasher);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: Role, text: &str) -> Message {
        Message {
            role,
            content: MessageContent::Text(text.to_owned()),
            name: None,
            tool_call_id: None,
        }
    }

    fn msg_parts(role: Role, parts: Vec<ContentPart>) -> Message {
        Message {
            role,
            content: MessageContent::Parts(parts),
            name: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn test_cache_hit_returns_correct_backend() {
        let mut map = CacheAffinityMap::new(10);
        let model = ModelId::new("llama3-70b");
        let prefix = PrefixHash::new(42);
        let backend = BackendId::new("gpu-1");

        map.record(&model, prefix, &backend);
        let result = map.get(&model, prefix);

        assert_eq!(result, Some(&BackendId::new("gpu-1")));
    }

    #[test]
    fn test_cache_miss_returns_none() {
        let mut map = CacheAffinityMap::new(10);
        let model = ModelId::new("llama3-70b");
        let prefix = PrefixHash::new(42);

        assert_eq!(map.get(&model, prefix), None);
    }

    #[test]
    fn test_lru_eviction_removes_oldest() {
        let mut map = CacheAffinityMap::new(2);
        let model = ModelId::new("llama3-70b");
        let b1 = BackendId::new("gpu-1");
        let b2 = BackendId::new("gpu-2");
        let b3 = BackendId::new("gpu-3");

        map.record(&model, PrefixHash::new(1), &b1);
        map.record(&model, PrefixHash::new(2), &b2);
        // This should evict the entry with PrefixHash(1) as it has the lowest last_used
        map.record(&model, PrefixHash::new(3), &b3);

        assert_eq!(map.get(&model, PrefixHash::new(1)), None);
        assert_eq!(
            map.get(&model, PrefixHash::new(2)),
            Some(&BackendId::new("gpu-2"))
        );
        assert_eq!(
            map.get(&model, PrefixHash::new(3)),
            Some(&BackendId::new("gpu-3"))
        );
    }

    #[test]
    fn test_backend_eviction_removes_all_entries() {
        let mut map = CacheAffinityMap::new(10);
        let model = ModelId::new("llama3-70b");
        let backend = BackendId::new("gpu-1");
        let other = BackendId::new("gpu-2");

        map.record(&model, PrefixHash::new(1), &backend);
        map.record(&model, PrefixHash::new(2), &backend);
        map.record(&model, PrefixHash::new(3), &other);

        map.evict_backend(&backend);

        assert_eq!(map.get(&model, PrefixHash::new(1)), None);
        assert_eq!(map.get(&model, PrefixHash::new(2)), None);
        assert_eq!(
            map.get(&model, PrefixHash::new(3)),
            Some(&BackendId::new("gpu-2"))
        );
    }

    #[test]
    fn test_prefix_hash_same_input_produces_same_hash() {
        let messages = vec![
            msg(Role::System, "You are a helpful assistant."),
            msg(Role::User, "Hello, world!"),
        ];

        let hash1 = compute_prefix_hash(&messages, 2);
        let hash2 = compute_prefix_hash(&messages, 2);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_prefix_hash_different_input_produces_different_hash() {
        let messages_a = vec![
            msg(Role::System, "You are a helpful assistant."),
            msg(Role::User, "Hello, world!"),
        ];
        let messages_b = vec![
            msg(Role::System, "You are a coding assistant."),
            msg(Role::User, "Write some code."),
        ];

        let hash_a = compute_prefix_hash(&messages_a, 2);
        let hash_b = compute_prefix_hash(&messages_b, 2);

        assert_ne!(hash_a, hash_b);
    }

    #[test]
    fn test_prefix_hash_skips_images() {
        let text_messages = vec![msg_parts(
            Role::User,
            vec![ContentPart::Text {
                text: "Describe this.".to_owned(),
            }],
        )];
        let mixed_messages = vec![msg_parts(
            Role::User,
            vec![
                ContentPart::Text {
                    text: "Describe this.".to_owned(),
                },
                ContentPart::ImageUrl {
                    url: "https://example.com/image.png".to_owned(),
                    detail: None,
                },
            ],
        )];

        let hash_text = compute_prefix_hash(&text_messages, 1);
        let hash_mixed = compute_prefix_hash(&mixed_messages, 1);

        assert_eq!(hash_text, hash_mixed);
    }
}
