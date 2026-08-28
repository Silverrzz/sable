mod api;
mod evaluator;
mod features;
mod material;
mod model;
mod types;

pub use api::{embedded_eval_hash, embedded_eval_label, has_embedded_eval};
pub use types::{
    NnueAccumulators, NnueArchitectureId, NnueModel, NnueOutput, PieceContribution,
};

pub(crate) use evaluator::Evaluator;
pub(crate) use material::{is_board_drawn, material_score_for_white};
pub(crate) use types::{DRAW_SCORE, LOSS_SCORE, NnueFinnyTable};

pub(crate) fn evaluate_position(board: &crate::Board, evaluator: &Evaluator) -> i32 {
    if material::is_board_drawn(board) {
        DRAW_SCORE
    } else {
        scale_rule50_score(board, evaluator.evaluate_for_side_to_move(board))
    }
}

pub(crate) fn scale_rule50_score(board: &crate::Board, score: i32) -> i32 {
    let scale = 200_i32.saturating_sub(i32::from(crate::chess::halfmove_clock(board)));
    score.saturating_mul(scale) / 200
}
