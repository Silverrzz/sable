use crate::{
    protocol::uci::mate_score_to_uci,
    search::{StaticEval, StaticEvalSource},
};

pub(super) fn terminal_static_eval(score_cp: i32) -> StaticEval {
    StaticEval {
        score_cp,
        score_mate: mate_score_to_uci(score_cp),
        source: StaticEvalSource::Terminal,
    }
}
