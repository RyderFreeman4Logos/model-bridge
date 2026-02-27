use crate::core::{ApiKey, AuthError, ClientId, ModelId};

// ---------------------------------------------------------------------------
// Client permission types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum AllowedModels {
    All,
    Specific(Vec<ModelId>),
}

#[derive(Clone, Debug)]
pub struct RateLimit {
    pub requests_per_minute: u32,
    pub tokens_per_minute: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct QuotaConfig {
    pub monthly_token_limit: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct ClientInfo {
    pub id: ClientId,
    pub allowed_models: AllowedModels,
    pub rate_limit: RateLimit,
    pub quota: QuotaConfig,
}

// ---------------------------------------------------------------------------
// AuthService — authenticates API keys and checks model permissions
// ---------------------------------------------------------------------------

/// Validates client API keys and checks model access permissions.
///
/// Uses `Vec<(ApiKey, ClientInfo)>` instead of `HashMap` because `ApiKey`
/// intentionally does not implement `Hash` (constant-time `PartialEq` only).
/// Linear scan is acceptable: the number of clients is small, and iterating
/// all entries prevents early-exit timing leaks across keys.
pub struct AuthService {
    clients: Vec<(ApiKey, ClientInfo)>,
}

impl AuthService {
    pub fn new(clients: Vec<(ApiKey, ClientInfo)>) -> Self {
        Self { clients }
    }

    /// Authenticate an API key, returning the associated `ClientInfo`.
    ///
    /// Iterates **all** entries regardless of match position to prevent
    /// timing side-channels that would reveal how many keys exist or
    /// where a valid key sits in the list.
    pub fn validate(&self, key: &ApiKey) -> Result<&ClientInfo, AuthError> {
        let mut matched: Option<&ClientInfo> = None;
        for (stored_key, info) in &self.clients {
            if stored_key == key {
                matched = Some(info);
            }
        }
        matched.ok_or(AuthError::InvalidApiKey)
    }

    /// Check whether `client` is permitted to access `model`.
    pub fn check_model_permission(client: &ClientInfo, model: &ModelId) -> Result<(), AuthError> {
        match &client.allowed_models {
            AllowedModels::All => Ok(()),
            AllowedModels::Specific(models) => {
                if models.contains(model) {
                    Ok(())
                } else {
                    Err(AuthError::ModelNotPermitted {
                        model: model.clone(),
                        client: client.id.clone(),
                    })
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

    fn make_client(id: &str, allowed: AllowedModels) -> ClientInfo {
        ClientInfo {
            id: ClientId::new(id),
            allowed_models: allowed,
            rate_limit: RateLimit {
                requests_per_minute: 60,
                tokens_per_minute: None,
            },
            quota: QuotaConfig {
                monthly_token_limit: None,
            },
        }
    }

    #[test]
    fn test_valid_key() {
        let key = ApiKey::new("mb-sk-valid000000000000000000000000");
        let client = make_client("team-alpha", AllowedModels::All);
        let svc = AuthService::new(vec![(key.clone(), client)]);

        let result = svc.validate(&ApiKey::new("mb-sk-valid000000000000000000000000"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().id.as_str(), "team-alpha");
    }

    #[test]
    fn test_invalid_key() {
        let key = ApiKey::new("mb-sk-valid000000000000000000000000");
        let client = make_client("team-alpha", AllowedModels::All);
        let svc = AuthService::new(vec![(key, client)]);

        let result = svc.validate(&ApiKey::new("mb-sk-wrong000000000000000000000000"));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AuthError::InvalidApiKey));
    }

    #[test]
    fn test_model_permitted_specific() {
        let client = make_client(
            "team-alpha",
            AllowedModels::Specific(vec![ModelId::new("llama3-70b"), ModelId::new("gpt-4")]),
        );

        let result = AuthService::check_model_permission(&client, &ModelId::new("llama3-70b"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_not_permitted() {
        let client = make_client(
            "team-alpha",
            AllowedModels::Specific(vec![ModelId::new("llama3-70b")]),
        );

        let result = AuthService::check_model_permission(&client, &ModelId::new("gpt-4"));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AuthError::ModelNotPermitted { .. }
        ));
    }

    #[test]
    fn test_wildcard_allows_all() {
        let client = make_client("team-alpha", AllowedModels::All);

        let result =
            AuthService::check_model_permission(&client, &ModelId::new("any-model-at-all"));
        assert!(result.is_ok());
    }
}
