use crate::Move;

#[derive(Clone, Debug, Default)]
pub struct TimeControl {
    pub white_time_ms: Option<u64>,
    pub black_time_ms: Option<u64>,
    pub white_increment_ms: Option<u64>,
    pub black_increment_ms: Option<u64>,
    pub moves_to_go: Option<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct SearchLimits {
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub soft_nodes: Option<u64>,
    pub hard_nodes: Option<u64>,
    pub mate: Option<u32>,
    pub move_time_ms: Option<u64>,
    pub infinite: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SearchRequest {
    pub ponder: bool,
    pub search_moves: Vec<String>,
    pub time_control: Option<TimeControl>,
    pub limits: SearchLimits,
}

#[derive(Clone, Debug, Default)]
pub struct SearchBudget {
    pub soft_time_ms: Option<u64>,
    pub hard_time_ms: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct SearchInfo {
    pub budget: SearchBudget,
    pub depth: u32,
    pub seldepth: u32,
    pub nodes: u64,
    pub time_ms: u64,
    pub nps: u64,
    pub score_cp: i32,
    pub score_mate: Option<i32>,
    pub multi_pv: Option<u32>,
    pub hashfull: u16,
    pub pv: Vec<Move>,
}

#[derive(Clone, Debug, Default)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub ponder_move: Option<Move>,
    pub info: SearchInfo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticEvalSource {
    Nnue,
    Terminal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticEval {
    pub score_cp: i32,
    pub score_mate: Option<i32>,
    pub source: StaticEvalSource,
}
