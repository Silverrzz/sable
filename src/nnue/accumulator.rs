use arrayvec::ArrayVec;

use crate::{
    Board, Color, Piece, Square,
    chess,
    evaluation::PieceContribution,
    pieces::ALL_PIECES,
};

use super::{FeatureUpdates, INPUT, L1, NET, PieceFeature, feed_forward};

#[derive(Debug, Clone)]
pub struct Accumulator {
    values: [[i16; L1]; 2],
    pub needs_refresh: [bool; 2],
    pub dirty: [bool; 2],
    pub updates: FeatureUpdates,
}

impl Accumulator {
    pub fn from_board(board: &Board) -> Self {
        let mut accumulator = Self::default();
        accumulator.reset(board, Color::White);
        accumulator.reset(board, Color::Black);
        accumulator
    }

    #[inline]
    pub fn eval(&self, board: &Board) -> i32 {
        let hmc = usize::from(chess::halfmove_clock(board)).min(100);
        let bucket = if hmc <= 5 { 0 } else { hmc.div_ceil(10) };
        let weights = &NET.l0w[768 + bucket];
        let stm = chess::side_to_move(board);
        let score = feed_forward(&self.values[stm as usize], &self.values[!stm as usize], weights);
        const LIMIT: i32 = -crate::evaluation::LOSS_SCORE - crate::search::MAX_ORDERING_PLY as i32 - 1;
        score.clamp(-LIMIT, LIMIT)
    }

    pub(crate) fn piece_contributions(board: &Board, base_score: i32) -> Vec<PieceContribution> {
        let accumulator = Self::from_board(board);
        let sign = match chess::side_to_move(board) {
            Color::White => 1,
            Color::Black => -1,
        };
        let mut contributions = Vec::new();
        for color in [Color::White, Color::Black] {
            for piece in ALL_PIECES {
                if piece == Piece::King {
                    continue;
                }
                for square in chess::colored_pieces(board, color, piece) {
                    let mut removed = accumulator.clone();
                    let feature = PieceFeature::new(piece, color, square);
                    for perspective in [Color::White, Color::Black] {
                        let index = feature.to_index(chess::king(board, perspective), perspective);
                        let values = &mut removed.values[perspective as usize];
                        for (value, &weight) in values.iter_mut().zip(&NET.l0w[index]) {
                            *value = value.wrapping_sub(weight);
                        }
                    }
                    contributions.push(PieceContribution {
                        square,
                        piece,
                        color,
                        score_white_cp: (base_score - removed.eval(board)) * sign,
                    });
                }
            }
        }
        contributions
    }

    #[inline]
    pub fn update(&mut self, prev: &Accumulator, king: Square, perspective: Color) {
        let values = &mut self.values[perspective as usize];
        let weights = &NET.l0w;
        let previous = &prev.values[perspective as usize];
        let updates = prev.updates;
        let index = |feature: PieceFeature| feature.to_index(king, perspective);

        match (updates.add, updates.add2, updates.sub, updates.sub2) {
            (Some(add), None, Some(sub), None) => {
                acc_add_sub(previous, values, weights, index(add), index(sub));
            }
            (Some(add), None, Some(sub), Some(sub2)) => {
                acc_add_sub2(
                    previous,
                    values,
                    weights,
                    index(add),
                    index(sub),
                    index(sub2),
                );
            }
            (Some(add), Some(add2), Some(sub), Some(sub2)) => acc_add2_sub2(
                previous,
                values,
                weights,
                index(add),
                index(add2),
                index(sub),
                index(sub2),
            ),
            _ => unreachable!("Invalid accumulator updates"),
        }

        self.dirty[perspective as usize] = false;
    }

    #[inline]
    pub fn reset(&mut self, board: &Board, perspective: Color) {
        let king = chess::king(board, perspective);
        let mut adds: ArrayVec<usize, 64> = ArrayVec::new();

        for sq in chess::colors(board, Color::White) | chess::colors(board, Color::Black) {
            let piece = chess::piece_on(board, sq).unwrap();
            let color = chess::color_on(board, sq).unwrap();
            adds.push(PieceFeature::new(piece, color, sq).to_index(king, perspective));
        }

        let weights = &NET.l0w;
        let (chunks, rem) = adds.as_chunks();
        let values = &mut self.values[perspective as usize];
        *values = NET.l0b;

        for &[add1, add2, add3, add4] in chunks {
            acc_add4(values, weights, add1, add2, add3, add4);
        }

        for &add in rem {
            acc_add(values, weights, add);
        }

        self.dirty[perspective as usize] = false;
        self.needs_refresh[perspective as usize] = false;
    }
}

impl Default for Accumulator {
    #[inline]
    fn default() -> Self {
        Self {
            values: [NET.l0b; 2],
            dirty: [false; 2],
            needs_refresh: [false; 2],
            updates: FeatureUpdates::default(),
        }
    }
}

#[inline]
fn acc_add(values: &mut [i16; L1], weights: &[[i16; L1]; INPUT], add: usize) {
    let add = &weights[add];
    for i in 0..L1 {
        values[i] = values[i].wrapping_add(add[i]);
    }
}

pub fn acc_add_sub(
    src: &[i16; L1],
    dst: &mut [i16; L1],
    weights: &[[i16; L1]; INPUT],
    add: usize,
    sub: usize,
) {
    let add: &[i16; L1] = &weights[add];
    let sub: &[i16; L1] = &weights[sub];

    for i in 0..L1 {
        dst[i] = src[i].wrapping_add(add[i]).wrapping_sub(sub[i]);
    }
}

pub fn acc_add_sub2(
    src: &[i16; L1],
    dst: &mut [i16; L1],
    weights: &[[i16; L1]; INPUT],
    add: usize,
    sub1: usize,
    sub2: usize,
) {
    let add: &[i16; L1] = &weights[add];
    let sub1: &[i16; L1] = &weights[sub1];
    let sub2: &[i16; L1] = &weights[sub2];

    for i in 0..L1 {
        dst[i] = src[i].wrapping_add(add[i]).wrapping_sub(sub1[i]).wrapping_sub(sub2[i]);
    }
}

pub fn acc_add2_sub2(
    src: &[i16; L1],
    dst: &mut [i16; L1],
    weights: &[[i16; L1]; INPUT],
    add1: usize,
    add2: usize,
    sub1: usize,
    sub2: usize,
) {
    let add1: &[i16; L1] = &weights[add1];
    let add2: &[i16; L1] = &weights[add2];
    let sub1: &[i16; L1] = &weights[sub1];
    let sub2: &[i16; L1] = &weights[sub2];

    for i in 0..L1 {
        dst[i] = src[i]
            .wrapping_add(add1[i])
            .wrapping_add(add2[i])
            .wrapping_sub(sub1[i])
            .wrapping_sub(sub2[i]);
    }
}

#[inline]
fn acc_add4(
    values: &mut [i16; L1],
    weights: &[[i16; L1]; INPUT],
    add1: usize,
    add2: usize,
    add3: usize,
    add4: usize,
) {
    let add1 = &weights[add1];
    let add2 = &weights[add2];
    let add3 = &weights[add3];
    let add4 = &weights[add4];

    for i in 0..L1 {
        values[i] = values[i]
            .wrapping_add(add1[i])
            .wrapping_add(add2[i])
            .wrapping_add(add3[i])
            .wrapping_add(add4[i]);
    }
}
