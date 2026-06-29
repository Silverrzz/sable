use super::{BitBoard, Color, Square};

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_B: u64 = 0x0202_0202_0202_0202;
const FILE_G: u64 = 0x4040_4040_4040_4040;
const FILE_H: u64 = 0x8080_8080_8080_8080;
const NOT_FILE_A: u64 = !FILE_A;
const NOT_FILE_H: u64 = !FILE_H;
const NOT_FILE_AB: u64 = !(FILE_A | FILE_B);
const NOT_FILE_GH: u64 = !(FILE_G | FILE_H);

pub(crate) const NORTH: usize = 0;
pub(crate) const SOUTH: usize = 1;
pub(crate) const EAST: usize = 2;
pub(crate) const WEST: usize = 3;
pub(crate) const NORTH_EAST: usize = 4;
pub(crate) const NORTH_WEST: usize = 5;
pub(crate) const SOUTH_EAST: usize = 6;
pub(crate) const SOUTH_WEST: usize = 7;

const RAYS: [[u64; 64]; 8] = [
    build_rays(0, 1),
    build_rays(0, -1),
    build_rays(1, 0),
    build_rays(-1, 0),
    build_rays(1, 1),
    build_rays(-1, 1),
    build_rays(1, -1),
    build_rays(-1, -1),
];

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
    BitBoard(
        ray_attacks(square, occupied, NORTH_EAST, true)
            | ray_attacks(square, occupied, NORTH_WEST, true)
            | ray_attacks(square, occupied, SOUTH_EAST, false)
            | ray_attacks(square, occupied, SOUTH_WEST, false),
    )
}

#[inline]
pub(crate) fn get_rook_moves(square: Square, occupied: BitBoard) -> BitBoard {
    BitBoard(
        ray_attacks(square, occupied, NORTH, true)
            | ray_attacks(square, occupied, SOUTH, false)
            | ray_attacks(square, occupied, EAST, true)
            | ray_attacks(square, occupied, WEST, false),
    )
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
pub(crate) fn ray_mask(square: Square, direction: usize) -> BitBoard {
    BitBoard(RAYS[direction][square as usize])
}

#[inline]
fn ray_attacks(square: Square, occupied: BitBoard, direction: usize, increasing: bool) -> u64 {
    let ray = RAYS[direction][square as usize];
    let blockers = ray & occupied.0;
    if blockers == 0 {
        return ray;
    }
    let blocker = if increasing {
        blockers.trailing_zeros() as usize
    } else {
        63 - blockers.leading_zeros() as usize
    };
    ray ^ RAYS[direction][blocker]
}

const fn build_rays(df: i8, dr: i8) -> [u64; 64] {
    let mut rays = [0u64; 64];
    let mut square = 0;
    while square < 64 {
        rays[square] = build_ray(square, df, dr);
        square += 1;
    }
    rays
}

const fn build_ray(square: usize, df: i8, dr: i8) -> u64 {
    let mut ray = 0u64;
    let mut file = (square as i8 & 7) + df;
    let mut rank = (square as i8 >> 3) + dr;
    while file >= 0 && file < 8 && rank >= 0 && rank < 8 {
        ray |= 1u64 << (rank as usize * 8 + file as usize);
        file += df;
        rank += dr;
    }
    ray
}
