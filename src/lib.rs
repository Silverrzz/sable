mod engine;
mod error;
mod evaluation;
mod options;
mod perft;
mod pieces;
mod protocol;
mod search;
mod simd;

pub(crate) use cozy_chess::Square;
pub use cozy_chess::{Board, Color, GameStatus, Move, Piece};
pub use engine::{Engine, VerboseEval, VerboseEvalSquare};
pub use error::EngineError;
pub use evaluation::{
    EvalMode, NnueArchitectureId, PieceContribution, embedded_eval_label, has_embedded_eval,
};
pub(crate) use options::EngineOptions;
pub use search::{
    SearchBudget, SearchInfo, SearchLimits, SearchRequest, SearchResult, StaticEval,
    StaticEvalSource, TimeControl,
};
