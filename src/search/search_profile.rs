
use crate::{
    Board, Piece,
};

use super::constants::*;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SearchProfile {
    sparse_pawnless_endgame: bool,
}

impl SearchProfile {
    pub(super) fn for_board(board: &Board) -> Self {
        Self {
            sparse_pawnless_endgame: is_sparse_pawnless_endgame(board),
        }
    }

    pub(super) fn sparse_pawnless_endgame(self) -> bool {
        self.sparse_pawnless_endgame
    }
}

pub(super) fn is_sparse_pawnless_endgame(board: &Board) -> bool {
    crate::chess::pieces(board, Piece::Pawn).is_empty()
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
    .map(|piece| crate::chess::pieces(board, piece).len() as u32)
    .sum()
}
