//! Names of the providers the built-in catalog ships.
//!
//! An application that refers to a provider directly, to run a login flow or
//! seed an install, names it here rather than spelling the id inline. The
//! constants are the catalog ids; the functions build the [`ProviderId`] for
//! them. Every constant names a row in the built-in catalog, and a test keeps
//! the two lists the same.

use super::ProviderId;

/// Built-in provider identifiers.
pub mod ids {
    pub const ANTHROPIC: &str = "anthropic";
    pub const BEDROCK: &str = "bedrock";
    pub const BEDROCK_OPENAI: &str = "bedrock-openai";
    pub const DEEPSEEK: &str = "deepseek";
    pub const FIREWORKS: &str = "fireworks";
    pub const GEMINI: &str = "gemini";
    pub const INCEPTION: &str = "inception";
    pub const LITELLM: &str = "litellm";
    pub const MINIMAX: &str = "minimax";
    pub const MODAL: &str = "modal";
    pub const MOONSHOT: &str = "moonshot";
    pub const OLLAMA: &str = "ollama";
    pub const OPENAI: &str = "openai";
    /// The ChatGPT-subscription deployment that stands in for
    /// [`OPENAI`] when a Codex OAuth credential is present.
    pub const OPENAI_CODEX: &str = "openai-codex";
    pub const OPENROUTER: &str = "openrouter";
    pub const POOLSIDE: &str = "poolside";
    pub const VENICE: &str = "venice";
    pub const ZAI: &str = "zai";

    /// Every built-in provider id, in catalog order.
    pub const ALL: [&str; 18] = [
        ANTHROPIC,
        BEDROCK,
        BEDROCK_OPENAI,
        DEEPSEEK,
        FIREWORKS,
        GEMINI,
        INCEPTION,
        LITELLM,
        MINIMAX,
        MODAL,
        MOONSHOT,
        OLLAMA,
        OPENAI,
        OPENAI_CODEX,
        OPENROUTER,
        POOLSIDE,
        VENICE,
        ZAI,
    ];
}

/// The `anthropic` provider id.
pub fn anthropic() -> ProviderId {
    ProviderId::new(ids::ANTHROPIC)
}

/// The `openai` provider id.
pub fn openai() -> ProviderId {
    ProviderId::new(ids::OPENAI)
}

/// The `openai-codex` provider id.
pub fn openai_codex() -> ProviderId {
    ProviderId::new(ids::OPENAI_CODEX)
}

/// The `gemini` provider id.
pub fn gemini() -> ProviderId {
    ProviderId::new(ids::GEMINI)
}

#[cfg(all(test, feature = "builtin-catalog"))]
mod tests {
    use std::collections::BTreeSet;
    use std::error::Error as StdError;

    use super::ids;
    use crate::catalog::Catalog;

    #[test]
    fn the_constants_name_exactly_the_builtin_providers() -> Result<(), Box<dyn StdError>> {
        let catalog = Catalog::builder().with_builtin().build()?;
        let shipped: BTreeSet<&str> = catalog.providers().map(|p| p.id().as_str()).collect();
        let named: BTreeSet<&str> = ids::ALL.into_iter().collect();
        assert_eq!(named, shipped);
        assert_eq!(super::openai_codex().as_str(), ids::OPENAI_CODEX);
        Ok(())
    }
}
