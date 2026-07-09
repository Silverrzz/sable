mod api;
mod evaluator;
mod features;
mod material;
mod model;
mod types;

pub use api::{embedded_eval_hash, embedded_eval_label, has_embedded_eval};
pub use types::{
    NnueAccumulators, NnueArchitectureId, NnueModel, PieceContribution,
};

pub(crate) use evaluator::Evaluator;
pub(crate) use material::{is_board_drawn, material_score_for_white};
pub(crate) use types::{DRAW_SCORE, LOSS_SCORE, NnueFinnyTable};

pub(crate) fn evaluate_position(board: &crate::Board, evaluator: &Evaluator) -> i32 {
    if material::is_board_drawn(board) {
        DRAW_SCORE
    } else {
        evaluator.evaluate_for_side_to_move(board)
    }
}
