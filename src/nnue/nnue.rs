use crate::{
    Board, Color, Move, Piece, Square,
    chess::{self, File, Rank},
    search::MAX_ORDERING_PLY,
};

use super::{Accumulator, FeatureUpdates, PieceFeature, should_mirror};

pub struct Nnue {
    stack: Vec<Accumulator>,
    cursor: usize,
}

impl Nnue {
    #[inline]
    pub fn new(board: &Board) -> Self {
        let mut stack = vec![Accumulator::default(); MAX_ORDERING_PLY + 1];
        stack[0] = Accumulator::from_board(board);
        Self { stack, cursor: 0 }
    }

    #[inline]
    pub fn eval(&self, board: &Board) -> i32 {
        self.stack[self.cursor].eval(board)
    }

    #[inline]
    fn reset(&mut self, board: &Board, perspective: Color) {
        self.stack[self.cursor].reset(board, perspective);
    }

    #[inline]
    pub fn update(&mut self, board: &Board) {
        for perspective in [Color::White, Color::Black] {
            if self.stack[self.cursor].needs_refresh[perspective as usize] {
                self.reset(board, perspective);
            } else if self.stack[self.cursor].dirty[perspective as usize] {
                self.update_dirty(board, perspective);
            }
        }
    }

    #[inline]
    fn update_dirty(&mut self, board: &Board, perspective: Color) {
        let mut clean_index = None;
        for i in (0..self.cursor).rev() {
            if self.stack[i].needs_refresh[perspective as usize] {
                break;
            }

            if !self.stack[i].dirty[perspective as usize] {
                clean_index = Some(i);
                break;
            }
        }

        let king = chess::king(board, perspective);

        let Some(clean_index) = clean_index else {
            self.reset(board, perspective);
            return;
        };

        for i in clean_index..self.cursor {
            let [clean, dirty] = self.stack.get_disjoint_mut([i, i + 1]).unwrap();
            dirty.update(clean, king, perspective);
        }
    }

    #[inline]
    pub fn make_move(
        &mut self,
        board: &Board,
        mv: Move,
        piece: Piece,
        captured_piece: Option<Piece>,
    ) {
        let mut updates = FeatureUpdates::default();
        let (src, mut dest) = (mv.from, mv.to);
        let stm = chess::side_to_move(board);

        if piece == Piece::King && chess::colors(board, stm).has(dest) {
            let rank = Rank::First.relative_to(stm);
            let (king_file, rook_file) = if dest.file() as u8 > src.file() as u8 {
                (File::G, File::F)
            } else {
                (File::C, File::D)
            };
            let king_dest = Square::new(king_file, rank);
            let rook_dest = Square::new(rook_file, rank);

            updates.add = Some(PieceFeature::new(Piece::King, stm, king_dest));
            updates.add2 = Some(PieceFeature::new(Piece::Rook, stm, rook_dest));
            updates.sub = Some(PieceFeature::new(Piece::King, stm, src));
            updates.sub2 = Some(PieceFeature::new(Piece::Rook, stm, dest));

            dest = king_dest;
        } else {
            updates.add = Some(PieceFeature::new(mv.promotion.unwrap_or(piece), stm, dest));
            updates.sub = Some(PieceFeature::new(piece, stm, src));
            if let Some(captured) = captured_piece {
                let victim = if piece == Piece::Pawn && !chess::colors(board, !stm).has(dest) {
                    Square::new(dest.file(), src.rank())
                } else {
                    dest
                };
                updates.sub2 = Some(PieceFeature::new(captured, !stm, victim));
            }
        }

        self.stack[self.cursor].updates = updates;
        self.cursor += 1;
        if self.cursor == self.stack.len() {
            self.stack.push(Accumulator::default());
        }
        self.stack[self.cursor].dirty = [true; 2];
        self.stack[self.cursor].needs_refresh = self.stack[self.cursor - 1].needs_refresh;

        if piece == Piece::King && should_mirror(src) != should_mirror(dest) {
            self.stack[self.cursor].needs_refresh[stm as usize] = true;
        }
    }

    #[inline]
    pub fn unmake_move(&mut self) {
        self.cursor -= 1;
    }
}
