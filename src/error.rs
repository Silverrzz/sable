#[derive(Clone, Debug)]
pub enum EngineError {
    InvalidFen(String),
    InvalidMove(String),
    InvalidSearchMove(String),
    InvalidOption(String),
    InvalidOptionValue { option: String, value: String },
    InvalidEvalFile { path: String, message: String },
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFen(fen) => write!(f, "invalid FEN: {fen}"),
            Self::InvalidMove(mv) => write!(f, "invalid move: {mv}"),
            Self::InvalidSearchMove(mv) => write!(f, "invalid search move: {mv}"),
            Self::InvalidOption(option) => write!(f, "invalid option: {option}"),
            Self::InvalidOptionValue { option, value } => {
                write!(f, "invalid option value for {option}: {value}")
            }
            Self::InvalidEvalFile { path, message } => {
                write!(f, "invalid eval file {path}: {message}")
            }
        }
    }
}

impl std::error::Error for EngineError {}
