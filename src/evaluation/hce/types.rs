use crate::{Board, Color, Piece, Square};
use cozy_chess::{BitBoard, get_king_moves};

use super::geometry::{build_pawn_files, collect_pawn_attacks};

#[derive(Clone, Copy, Default)]
pub(super) struct HceScore {
    pub(super) mg: i32,
    pub(super) eg: i32,
}

impl HceScore {
    #[inline(always)]
    pub(super) fn add(&mut self, mg: i32, eg: i32) {
        self.mg += mg;
        self.eg += eg;
    }

    #[inline(always)]
    pub(super) fn add_score(&mut self, other: HceScore) {
        self.mg += other.mg;
        self.eg += other.eg;
    }
}

#[derive(Clone, Copy)]
pub(super) struct HceInfo {
    colors: [BitBoard; 2],
    piece_colors: [[BitBoard; 6]; 2],
    pub(super) occupied: BitBoard,
    pub(super) pawn_attacks: [BitBoard; 2],
    pub(super) pawn_files: [[BitBoard; 8]; 2],
    king_squares: [Option<Square>; 2],
    pub(super) king_zones: [BitBoard; 2],
}

impl HceInfo {
    #[inline]
    pub(super) fn new(board: &Board) -> Self {
        let colors = [board.colors(Color::White), board.colors(Color::Black)];
        let occupied = colors[Color::White as usize] | colors[Color::Black as usize];
        let mut piece_colors = [[BitBoard::EMPTY; 6]; 2];
        for color in [Color::White, Color::Black] {
            let color_idx = color as usize;
            for piece in [
                Piece::Pawn,
                Piece::Knight,
                Piece::Bishop,
                Piece::Rook,
                Piece::Queen,
                Piece::King,
            ] {
                piece_colors[color_idx][piece as usize] = board.colored_pieces(color, piece);
            }
        }

        let mut pawn_files = [[BitBoard::EMPTY; 8]; 2];
        for color in [Color::White, Color::Black] {
            let color_idx = color as usize;
            pawn_files[color_idx] = build_pawn_files(piece_colors[color_idx][Piece::Pawn as usize]);
        }

        let pawn_attacks = [
            collect_pawn_attacks(
                piece_colors[Color::White as usize][Piece::Pawn as usize],
                Color::White,
            ),
            collect_pawn_attacks(
                piece_colors[Color::Black as usize][Piece::Pawn as usize],
                Color::Black,
            ),
        ];

        let king_squares = [
            piece_colors[Color::White as usize][Piece::King as usize].next_square(),
            piece_colors[Color::Black as usize][Piece::King as usize].next_square(),
        ];
        let king_zones = [
            king_squares[Color::White as usize]
                .map(|king| get_king_moves(king) | king.bitboard())
                .unwrap_or(BitBoard::EMPTY),
            king_squares[Color::Black as usize]
                .map(|king| get_king_moves(king) | king.bitboard())
                .unwrap_or(BitBoard::EMPTY),
        ];

        Self {
            colors,
            piece_colors,
            occupied,
            pawn_attacks,
            pawn_files,
            king_squares,
            king_zones,
        }
    }

    #[inline(always)]
    pub(super) fn color(&self, color: Color) -> BitBoard {
        self.colors[color as usize]
    }

    #[inline(always)]
    pub(super) fn piece_color(&self, color: Color, piece: Piece) -> BitBoard {
        self.piece_colors[color as usize][piece as usize]
    }

    #[inline(always)]
    pub(super) fn king_square(&self, color: Color) -> Option<Square> {
        self.king_squares[color as usize]
    }
}
