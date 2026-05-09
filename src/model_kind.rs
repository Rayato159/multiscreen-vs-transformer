#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelKind {
    Multiscreen,
    Transformer,
}

impl ModelKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "multiscreen" | "multi-screen" | "screening" | "ms" => Some(Self::Multiscreen),
            "transformer" | "tfm" | "attention" => Some(Self::Transformer),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Multiscreen => "multiscreen",
            Self::Transformer => "transformer",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Multiscreen => "Multiscreen",
            Self::Transformer => "Transformer",
        }
    }

    pub fn default_param_path(self) -> &'static str {
        match self {
            Self::Multiscreen => "models/sat_multiscreen.params",
            Self::Transformer => "models/sat_transformer.params",
        }
    }
}
