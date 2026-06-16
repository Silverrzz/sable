
use crate::{
    Board, Piece,
};

use super::constants::*;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SearchProfile {
    reduce_late_quiet_checks: bool,
}

impl SearchProfile {
    pub(super) fn for_board(board: &Board) -> Self {
        Self {
            reduce_late_quiet_checks: is_sparse_pawnless_endgame(board),
        }
    }

    pub(super) fn reduce_late_quiet_checks(self) -> bool {
        self.reduce_late_quiet_checks
    }
}

pub(super) fn is_sparse_pawnless_endgame(board: &Board) -> bool {
    board.pieces(Piece::Pawn).is_empty()
        && non_king_piece_count(board) <= SPARSE_ENDGAME_MAX_NON_KING_PIECES
}

pub(super) fn non_king_piece_count(board: &Board) -> u32 {
    [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
    ]
    .into_iter()
    .map(|piece| board.pieces(piece).len() as u32)
    .sum()
}
