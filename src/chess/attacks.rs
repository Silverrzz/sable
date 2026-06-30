use super::{BitBoard, Color, Square};

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_B: u64 = 0x0202_0202_0202_0202;
const FILE_G: u64 = 0x4040_4040_4040_4040;
const FILE_H: u64 = 0x8080_8080_8080_8080;
const NOT_FILE_A: u64 = !FILE_A;
const NOT_FILE_H: u64 = !FILE_H;
const NOT_FILE_AB: u64 = !(FILE_A | FILE_B);
const NOT_FILE_GH: u64 = !(FILE_G | FILE_H);

const ROOK_TABLE_SIZE: usize = 4096;
const BISHOP_TABLE_SIZE: usize = 512;
const BETWEEN: [[u64; 64]; 64] = build_between();

include!(concat!(env!("OUT_DIR"), "/attack_tables.rs"));

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
    let square = square as usize;
    let mask = BISHOP_RELEVANT_MASKS[square];
    BitBoard(BISHOP_ATTACKS[square][slider_index(occupied.0, mask)])
}

#[inline]
pub(crate) fn get_rook_moves(square: Square, occupied: BitBoard) -> BitBoard {
    let square = square as usize;
    let mask = ROOK_RELEVANT_MASKS[square];
    BitBoard(ROOK_ATTACKS[square][slider_index(occupied.0, mask)])
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
pub(crate) fn between_squares(a: Square, b: Square) -> BitBoard {
    BitBoard(BETWEEN[a as usize][b as usize])
}

#[inline]
pub(crate) fn line_squares(a: Square, b: Square) -> BitBoard {
    let between = BETWEEN[a as usize][b as usize];
    if between != 0 || a == b || same_line(a as usize, b as usize) {
        BitBoard(between | a.bitboard().0 | b.bitboard().0)
    } else {
        BitBoard::EMPTY
    }
}

#[inline]
fn slider_index(occupied: u64, mask: u64) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("bmi2") {
            return unsafe { pext_index(occupied, mask) };
        }
    }
    #[cfg(target_arch = "x86")]
    {
        if std::arch::is_x86_feature_detected!("bmi2") {
            return unsafe { pext_index(occupied, mask) };
        }
    }
    compact_index(occupied, mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn pext_index(occupied: u64, mask: u64) -> usize {
    core::arch::x86_64::_pext_u64(occupied, mask) as usize
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "bmi2")]
unsafe fn pext_index(occupied: u64, mask: u64) -> usize {
    core::arch::x86::_pext_u64(occupied, mask) as usize
}

#[inline]
fn compact_index(occupied: u64, mut mask: u64) -> usize {
    let occupied = occupied & mask;
    let mut index = 0usize;
    let mut offset = 0usize;
    while mask != 0 {
        let bit = mask & mask.wrapping_neg();
        if occupied & bit != 0 {
            index |= 1usize << offset;
        }
        mask ^= bit;
        offset += 1;
    }
    index
}

const fn build_between() -> [[u64; 64]; 64] {
    let mut between = [[0u64; 64]; 64];
    let mut from = 0;
    while from < 64 {
        let mut to = 0;
        while to < 64 {
            between[from][to] = build_between_pair(from, to);
            to += 1;
        }
        from += 1;
    }
    between
}

const fn build_between_pair(from: usize, to: usize) -> u64 {
    if from == to || !same_line(from, to) {
        return 0;
    }
    let from_file = (from & 7) as i8;
    let from_rank = (from >> 3) as i8;
    let to_file = (to & 7) as i8;
    let to_rank = (to >> 3) as i8;
    let df = signum(to_file - from_file);
    let dr = signum(to_rank - from_rank);
    let mut file = from_file + df;
    let mut rank = from_rank + dr;
    let mut mask = 0u64;
    while file != to_file || rank != to_rank {
        mask |= 1u64 << ((rank as usize) * 8 + file as usize);
        file += df;
        rank += dr;
    }
    mask
}

const fn same_line(a: usize, b: usize) -> bool {
    let af = (a & 7) as i8;
    let ar = (a >> 3) as i8;
    let bf = (b & 7) as i8;
    let br = (b >> 3) as i8;
    af == bf || ar == br || abs(af - bf) == abs(ar - br)
}

const fn signum(value: i8) -> i8 {
    if value > 0 {
        1
    } else if value < 0 {
        -1
    } else {
        0
    }
}

const fn abs(value: i8) -> i8 {
    if value < 0 {
        -value
    } else {
        value
    }
}
