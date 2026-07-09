
use crate::Board;

use super::{api::SharedNnueModel, types::*};

#[derive(Clone, Debug)]
pub(crate) struct Evaluator {
    nnue: Option<SharedNnueModel>,
}

impl Evaluator {
    pub(crate) fn new(nnue: Option<SharedNnueModel>) -> Self {
        Self { nnue }
    }

    pub(crate) fn set_nnue_model(&mut self, nnue: SharedNnueModel) {
        self.nnue = Some(nnue);
    }

    pub(crate) fn has_nnue_model(&self) -> bool {
        self.nnue.is_some()
    }

    pub(crate) fn active_nnue_model(&self) -> Option<&NnueModel> {
        self.nnue.as_deref()
    }

    pub(crate) fn loaded_nnue_model(&self) -> Option<&NnueModel> {
        self.nnue.as_deref()
    }

    pub(crate) fn evaluate_for_side_to_move(&self, board: &Board) -> i32 {
        self.active_nnue_model()
            .expect("evaluation requires a loaded NNUE model")
            .evaluate_for_side_to_move(board)
    }
}
