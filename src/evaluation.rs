mod material;

use crate::{Color, Piece, Square};

pub(crate) use material::{is_board_drawn, material_score_for_white};

pub(crate) const DRAW_SCORE: i32 = 0;
pub(crate) const LOSS_SCORE: i32 = -30_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NnueArchitectureId {
    Goliath,
}

impl NnueArchitectureId {
    pub fn as_str(self) -> &'static str {
        "goliath"
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PieceContribution {
    pub square: Square,
    pub piece: Piece,
    pub color: Color,
    pub score_white_cp: i32,
}

pub fn has_embedded_eval() -> bool {
    true
}

pub fn embedded_eval_label() -> Option<&'static str> {
    Some(env!("SABLE_ENGINE_EMBEDDED_EVAL_LABEL"))
}

pub fn embedded_eval_hash() -> Option<&'static str> {
    Some(env!("SABLE_ENGINE_EMBEDDED_EVAL_HASH"))
}
