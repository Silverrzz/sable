use crate::{Color, Square};
use cozy_chess::BitBoard;

use super::{FILE_A_MASK, FILE_H_MASK, PASSED_PAWN_MASKS};

#[inline(always)]
pub(super) fn pst_square_index(square: Square, color: Color) -> usize {
    let index = square as usize;
    if color == Color::White { index ^ 56 } else { index }
}

#[inline(always)]
pub(super) fn collect_pawn_attacks(pawns: BitBoard, color: Color) -> BitBoard {
    let bits = pawns.0;
    let attacks = if color == Color::White {
        ((bits & !FILE_A_MASK) << 7) | ((bits & !FILE_H_MASK) << 9)
    } else {
        ((bits & !FILE_H_MASK) >> 7) | ((bits & !FILE_A_MASK) >> 9)
    };
    BitBoard(attacks)
}

#[inline(always)]
pub(super) fn build_pawn_files(pawns: BitBoard) -> [BitBoard; 8] {
    let mut files = [BitBoard::EMPTY; 8];
    for square in pawns {
        files[square_file(square)] |= square.bitboard();
    }
    files
}

#[inline(always)]
pub(super) fn is_outpost(
    square: Square,
    side: Color,
    support: BitBoard,
    enemy_pawn_attacks: BitBoard,
) -> bool {
    let rank = relative_rank(square, side);
    (3..=5).contains(&rank) && support.has(square) && !enemy_pawn_attacks.has(square)
}

#[inline(always)]
pub(super) fn is_passed_pawn(square: Square, side: Color, enemy_pawns: BitBoard) -> bool {
    PASSED_PAWN_MASKS[side as usize][square as usize].is_disjoint(enemy_pawns)
}

#[inline(always)]
pub(super) fn square_mask(file: usize, rank: usize) -> BitBoard {
    BitBoard(1u64 << (rank * 8 + file))
}

#[inline(always)]
pub(super) fn square_file(square: Square) -> usize {
    square as usize & 7
}

#[inline(always)]
pub(super) fn square_rank(square: Square) -> usize {
    square as usize >> 3
}

#[inline(always)]
pub(super) fn relative_rank(square: Square, side: Color) -> usize {
    let rank = square_rank(square);
    if side == Color::White { rank } else { 7 - rank }
}
