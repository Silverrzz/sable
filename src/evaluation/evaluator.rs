
use crate::Board;

use super::{api::SharedNnueModel, hce::hce_score_for_side_to_move, types::*};

#[derive(Clone, Debug)]
pub(crate) struct Evaluator {
    mode: EvalMode,
    nnue: Option<SharedNnueModel>,
}

impl Evaluator {
    pub(crate) fn new(mode: EvalMode, nnue: Option<SharedNnueModel>) -> Self {
        Self { mode, nnue }
    }

    pub(crate) fn set_mode(&mut self, mode: EvalMode) {
        self.mode = mode;
    }

    pub(crate) fn set_nnue_model(&mut self, nnue: SharedNnueModel) {
        self.nnue = Some(nnue);
    }

    pub(crate) fn has_nnue_model(&self) -> bool {
        self.nnue.is_some()
    }

    pub(crate) fn active_nnue_model(&self) -> Option<&NnueModel> {
        if self.mode == EvalMode::Nnue {
            self.nnue.as_deref()
        } else {
            None
        }
    }

    pub(crate) fn evaluate_for_side_to_move(&self, board: &Board) -> i32 {
        if let Some(model) = self.active_nnue_model() {
            return model.evaluate_for_side_to_move(board);
        }
        hce_score_for_side_to_move(board)
    }
}
