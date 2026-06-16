
use crate::{
    Board, Color, Move, Piece, Square,
};
use cozy_chess::Rank;
pub(in crate::search) fn captured_piece(
    board: &Board,
    moving_piece: Piece,
    mv: Move,
    ep_target: Option<Square>,
) -> Option<Piece> {
    if is_en_passant(moving_piece, mv, ep_target) {
        Some(Piece::Pawn)
    } else {
        board.piece_on(mv.to)
    }
}

pub(in crate::search) fn is_en_passant(moving_piece: Piece, mv: Move, ep_target: Option<Square>) -> bool {
    moving_piece == Piece::Pawn && ep_target == Some(mv.to)
}

pub(in crate::search) fn en_passant_target(board: &Board, side: Color) -> Option<Square> {
    board
        .en_passant()
        .map(|file| Square::new(file, Rank::Sixth.relative_to(side)))
}
