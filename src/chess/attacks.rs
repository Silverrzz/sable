use super::{BitBoard, Color, Square};

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_B: u64 = 0x0202_0202_0202_0202;
const FILE_G: u64 = 0x4040_4040_4040_4040;
const FILE_H: u64 = 0x8080_8080_8080_8080;
const NOT_FILE_A: u64 = !FILE_A;
const NOT_FILE_H: u64 = !FILE_H;
const NOT_FILE_AB: u64 = !(FILE_A | FILE_B);
const NOT_FILE_GH: u64 = !(FILE_G | FILE_H);

#[inline]
pub(crate) fn get_pawn_attacks(square: Square, color: Color) -> BitBoard {
    let bit = square.bitboard().0;
    match color {
        Color::White => BitBoard(((bit & NOT_FILE_A) << 7) | ((bit & NOT_FILE_H) << 9)),
        Color::Black => BitBoard(((bit & NOT_FILE_A) >> 9) | ((bit & NOT_FILE_H) >> 7)),
    }
}

#[inline]
pub(crate) fn get_knight_moves(square: Square) -> BitBoard {
    let bit = square.bitboard().0;
    BitBoard(
        ((bit & NOT_FILE_A) << 15)
            | ((bit & NOT_FILE_H) << 17)
            | ((bit & NOT_FILE_AB) << 6)
            | ((bit & NOT_FILE_GH) << 10)
            | ((bit & NOT_FILE_A) >> 17)
            | ((bit & NOT_FILE_H) >> 15)
            | ((bit & NOT_FILE_AB) >> 10)
            | ((bit & NOT_FILE_GH) >> 6),
    )
}

#[inline]
pub(crate) fn get_bishop_moves(square: Square, occupied: BitBoard) -> BitBoard {
    sliding_moves(square, occupied, &[(1, 1), (-1, 1), (1, -1), (-1, -1)])
}

#[inline]
pub(crate) fn get_rook_moves(square: Square, occupied: BitBoard) -> BitBoard {
    sliding_moves(square, occupied, &[(1, 0), (-1, 0), (0, 1), (0, -1)])
}

#[inline]
pub(crate) fn get_king_moves(square: Square) -> BitBoard {
    let bit = square.bitboard().0;
    BitBoard(
        (bit << 8)
            | (bit >> 8)
            | ((bit & NOT_FILE_H) << 1)
            | ((bit & NOT_FILE_A) >> 1)
            | ((bit & NOT_FILE_H) << 9)
            | ((bit & NOT_FILE_A) << 7)
            | ((bit & NOT_FILE_H) >> 7)
            | ((bit & NOT_FILE_A) >> 9),
    )
}

#[inline]
fn sliding_moves(square: Square, occupied: BitBoard, directions: &[(i8, i8)]) -> BitBoard {
    let mut moves = 0u64;
    let from = square as i8;
    let file = from & 7;
    let rank = from >> 3;

    for &(df, dr) in directions {
        let mut next_file = file + df;
        let mut next_rank = rank + dr;
        while (0..8).contains(&next_file) && (0..8).contains(&next_rank) {
            let next = (next_rank * 8 + next_file) as usize;
            let bit = 1u64 << next;
            moves |= bit;
            if occupied.0 & bit != 0 {
                break;
            }
            next_file += df;
            next_rank += dr;
        }
    }

    BitBoard(moves)
}
