use std::{
    path::Path,
    sync::{Arc, OnceLock},
};

use crate::EngineError;

use super::super::{
    bullet::{build_model_from_bullet_quantized, build_model_from_native_bullet_quantized},
    io::invalid_eval_file,
    types::*,
};

impl NnueModel {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, EngineError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|error| EngineError::InvalidEvalFile {
            path: path.display().to_string(),
            message: error.to_string(),
        })?;
        Self::from_bytes(path, &bytes)
    }

    fn from_bytes(path: &Path, bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.len() < 8 {
            return Err(invalid_eval_file(path, "file too short for eval header"));
        }
        if &bytes[..8] == BULLET_QUANT_MAGIC {
            return build_model_from_bullet_quantized(path, bytes);
        }
        build_model_from_native_bullet_quantized(path, bytes)
    }

    pub(crate) fn shared_embedded_default() -> Option<Result<Arc<Self>, EngineError>> {
        if option_env!("SABLE_ENGINE_HAS_EMBEDDED_EVAL").unwrap_or("0") != "1" {
            return None;
        }
        static SHARED_EMBEDDED_DEFAULT: OnceLock<Result<Arc<NnueModel>, EngineError>> =
            OnceLock::new();
        Some(
            SHARED_EMBEDDED_DEFAULT
                .get_or_init(|| {
                    let label = NnueModel::embedded_default_label().unwrap_or("<embedded>");
                    let bytes = include_bytes!(env!("SABLE_ENGINE_EMBEDDED_EVAL_PATH"));
                    NnueModel::from_bytes(Path::new(label), bytes).map(Arc::new)
                })
                .clone(),
        )
    }

    pub fn has_embedded_default() -> bool {
        option_env!("SABLE_ENGINE_HAS_EMBEDDED_EVAL").unwrap_or("0") == "1"
    }

    pub fn embedded_default_label() -> Option<&'static str> {
        if !Self::has_embedded_default() {
            return None;
        }
        let label = option_env!("SABLE_ENGINE_EMBEDDED_EVAL_LABEL").unwrap_or("embedded");
        if label == "none" { None } else { Some(label) }
    }
}
