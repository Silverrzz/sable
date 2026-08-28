mod chess;
mod engine;
mod error;
mod evaluation;
mod options;
mod perft;
mod pieces;
mod protocol;
mod search;
mod simd;

pub const SPSA_UCI_OPTIONS_ENABLED: bool = true;

pub(crate) use chess::Square;
pub use chess::{Board, Color, GameStatus, Move, Piece};
pub use engine::{Engine, PvLeafOutput, VerboseEval, VerboseEvalSquare};
pub use error::EngineError;
pub use evaluation::{
    NnueArchitectureId, NnueOutput, PieceContribution, embedded_eval_hash, embedded_eval_label,
    has_embedded_eval,
};
pub(crate) use options::EngineOptions;
pub use search::{
    SearchBudget, SearchInfo, SearchLimits, SearchRequest, SearchResult, SpsaParameter,
    StaticEval, StaticEvalSource, TimeControl, spsa_parameters,
};
pub use simd::runtime_backend_name as runtime_simd_backend;
