use super::{BitBoard, Color, Square};

#[inline]
pub(crate) fn get_pawn_attacks(square: Square, color: Color) -> BitBoard {
    BitBoard::from_cozy(cozy_chess::get_pawn_attacks(square.to_cozy(), color.to_cozy()))
}

#[inline]
pub(crate) fn get_knight_moves(square: Square) -> BitBoard {
    BitBoard::from_cozy(cozy_chess::get_knight_moves(square.to_cozy()))
}

#[inline]
pub(crate) fn get_bishop_moves(square: Square, occupied: BitBoard) -> BitBoard {
    BitBoard::from_cozy(cozy_chess::get_bishop_moves(
        square.to_cozy(),
        occupied.to_cozy(),
    ))
}

#[inline]
pub(crate) fn get_rook_moves(square: Square, occupied: BitBoard) -> BitBoard {
    BitBoard::from_cozy(cozy_chess::get_rook_moves(
        square.to_cozy(),
        occupied.to_cozy(),
    ))
}

#[inline]
pub(crate) fn get_king_moves(square: Square) -> BitBoard {
    BitBoard::from_cozy(cozy_chess::get_king_moves(square.to_cozy()))
}
