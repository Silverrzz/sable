use std::path::Path;

use crate::EngineError;
pub(super) fn invalid_eval_file(path: &Path, message: &str) -> EngineError {
    EngineError::InvalidEvalFile {
        path: path.display().to_string(),
        message: message.to_owned(),
    }
}

pub(super) fn read_i16(bytes: &[u8], offset: usize) -> Result<i16, EngineError> {
    if offset + 2 > bytes.len() {
        return Err(EngineError::InvalidEvalFile {
            path: "<buffer>".to_owned(),
            message: "buffer out of bounds for i16".to_owned(),
        });
    }
    Ok(i16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

pub(super) fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, EngineError> {
    if offset + 4 > bytes.len() {
        return Err(EngineError::InvalidEvalFile {
            path: "<buffer>".to_owned(),
            message: "buffer out of bounds for u32".to_owned(),
        });
    }
    Ok(u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

pub(super) fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, EngineError> {
    if offset + 4 > bytes.len() {
        return Err(EngineError::InvalidEvalFile {
            path: "<buffer>".to_owned(),
            message: "buffer out of bounds for i32".to_owned(),
        });
    }
    Ok(i32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}
