
use crate::{Board, Color, Piece};

use super::types::*;

pub(crate) fn material_score_for_white(board: &Board) -> i32 {
    material_balance(board)
}

pub(super) fn material_balance(board: &Board) -> i32 {
    let white = board.colors(Color::White);
    let black = board.colors(Color::Black);
    [
        (Piece::Pawn, PAWN_VALUE),
        (Piece::Knight, KNIGHT_VALUE),
        (Piece::Bishop, BISHOP_VALUE),
        (Piece::Rook, ROOK_VALUE),
        (Piece::Queen, QUEEN_VALUE),
    ]
    .into_iter()
    .map(|(piece, value)| {
        let pieces = board.pieces(piece);
        ((pieces & white).len() as i32 - (pieces & black).len() as i32) * value
    })
    .sum()
}

pub(crate) fn is_board_drawn(board: &Board) -> bool {
    has_insufficient_material(board)
}

pub(super) fn has_insufficient_material(board: &Board) -> bool {
    if !board.pieces(Piece::Pawn).is_empty()
        || !board.pieces(Piece::Rook).is_empty()
        || !board.pieces(Piece::Queen).is_empty()
    {
        return false;
    }

    let knights = board.pieces(Piece::Knight).len();
    let bishops = board.pieces(Piece::Bishop);
    let bishop_count = bishops.len();
    let minor_count = knights + bishop_count;

    if minor_count <= 1 {
        return true;
    }
    if knights > 0 {
        return false;
    }

    let mut bishop_square_color = None;
    for square in bishops {
        let square_color = ((square.file() as u8) ^ (square.rank() as u8)) & 1;
        match bishop_square_color {
            Some(color) if color != square_color => return false,
            Some(_) => {}
            None => bishop_square_color = Some(square_color),
        }
    }
    true
}
