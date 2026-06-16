
use crate::{
    Board, Color, Move, Piece,
};

use super::{
    board_moves::{en_passant_target, is_en_passant},
    constants::*,
    transposition::{Bound, is_mate_score},
};

#[derive(Clone, Debug)]
pub(in crate::search) struct CorrectionHistory {
    pawn: Vec<i32>,
    minor: Vec<i32>,
    non_pawn: Vec<i32>,
    continuation: Vec<i32>,
}

impl Default for CorrectionHistory {
    fn default() -> Self {
        Self {
            pawn: vec![0; 2 * CORRECTION_HISTORY_BUCKETS],
            minor: vec![0; 2 * CORRECTION_HISTORY_BUCKETS],
            non_pawn: vec![0; 2 * CORRECTION_HISTORY_BUCKETS],
            continuation: vec![0; 2 * 64 * 64],
        }
    }
}

impl CorrectionHistory {
    pub(in crate::search) fn decay(&mut self) {
        decay_correction_table(&mut self.pawn);
        decay_correction_table(&mut self.minor);
        decay_correction_table(&mut self.non_pawn);
        decay_correction_table(&mut self.continuation);
    }

    pub(in crate::search) fn corrected_eval(&self, board: &Board, raw_eval: i32, previous_move: Option<Move>) -> i32 {
        raw_eval.saturating_add(self.correction(board, previous_move))
    }

    pub(in crate::search) fn update(
        &mut self,
        board: &Board,
        previous_move: Option<Move>,
        raw_eval: i32,
        score: i32,
        depth: u32,
    ) {
        let target = score
            .saturating_sub(raw_eval)
            .clamp(-MAX_CORRECTION_HISTORY_SCORE, MAX_CORRECTION_HISTORY_SCORE);
        let weight = correction_history_weight(depth);
        let side = board.side_to_move() as usize;
        update_correction_value(&mut self.pawn[pawn_correction_index(board, side)], target, weight);
        update_correction_value(
            &mut self.minor[minor_correction_index(board, side)],
            target,
            weight,
        );
        update_correction_value(
            &mut self.non_pawn[non_pawn_correction_index(board, side)],
            target,
            weight,
        );
        if let Some(previous_move) = previous_move {
            update_correction_value(
                &mut self.continuation[continuation_correction_index(side, previous_move)],
                target,
                weight,
            );
        }
    }

    pub(in crate::search) fn correction(&self, board: &Board, previous_move: Option<Move>) -> i32 {
        let side = board.side_to_move() as usize;
        let mut sum = self.pawn[pawn_correction_index(board, side)]
            .saturating_add(self.minor[minor_correction_index(board, side)])
            .saturating_add(self.non_pawn[non_pawn_correction_index(board, side)]);
        let mut count = 3;
        if let Some(previous_move) = previous_move {
            sum = sum.saturating_add(
                self.continuation[continuation_correction_index(side, previous_move)],
            );
            count += 1;
        }
        sum / count
    }
}

pub(in crate::search) fn should_update_correction_history(
    board: &Board,
    best_move: Option<Move>,
    bound: Bound,
    raw_eval: i32,
    score: i32,
) -> bool {
    let quiet_best_move = best_move.is_some_and(|mv| is_quiet_correction_move(board, mv));
    !is_mate_score(score)
        && quiet_best_move
        && match bound {
            Bound::Exact => true,
            Bound::Lower => score > raw_eval,
            Bound::Upper => score < raw_eval,
        }
}

pub(in crate::search) fn is_quiet_correction_move(board: &Board, mv: Move) -> bool {
    let moving_piece = board.piece_on(mv.from).unwrap_or(Piece::Pawn);
    mv.promotion.is_none()
        && board.piece_on(mv.to).is_none()
        && !is_en_passant(
            moving_piece,
            mv,
            en_passant_target(board, board.side_to_move()),
        )
}

pub(in crate::search) fn correction_history_weight(depth: u32) -> i32 {
    (depth.min(16) as i32).saturating_add(1)
}

pub(in crate::search) fn update_correction_value(value: &mut i32, target: i32, weight: i32) {
    let delta = target.saturating_sub(*value);
    *value = (*value)
        .saturating_add(delta.saturating_mul(weight) / CORRECTION_HISTORY_UPDATE_DIVISOR)
        .clamp(-MAX_CORRECTION_HISTORY_SCORE, MAX_CORRECTION_HISTORY_SCORE);
}

pub(in crate::search) fn decay_correction_table(values: &mut [i32]) {
    for value in values {
        *value /= 2;
    }
}

pub(in crate::search) fn pawn_correction_index(board: &Board, side: usize) -> usize {
    correction_index(side, pawn_correction_key(board))
}

pub(in crate::search) fn minor_correction_index(board: &Board, side: usize) -> usize {
    correction_index(side, minor_correction_key(board))
}

pub(in crate::search) fn non_pawn_correction_index(board: &Board, side: usize) -> usize {
    correction_index(side, non_pawn_correction_key(board))
}

pub(in crate::search) fn continuation_correction_index(side: usize, previous_move: Move) -> usize {
    ((side * 64) + previous_move.from as usize) * 64 + previous_move.to as usize
}

pub(in crate::search) fn correction_index(side: usize, key: u64) -> usize {
    side * CORRECTION_HISTORY_BUCKETS + (key as usize & (CORRECTION_HISTORY_BUCKETS - 1))
}

pub(in crate::search) fn pawn_correction_key(board: &Board) -> u64 {
    let white = colored_piece_bits(board, Color::White, Piece::Pawn);
    let black = colored_piece_bits(board, Color::Black, Piece::Pawn);
    mix_correction_key(white ^ black.rotate_left(1))
}

pub(in crate::search) fn minor_correction_key(board: &Board) -> u64 {
    let white = colored_piece_bits(board, Color::White, Piece::Knight)
        ^ colored_piece_bits(board, Color::White, Piece::Bishop).rotate_left(11);
    let black = colored_piece_bits(board, Color::Black, Piece::Knight).rotate_left(23)
        ^ colored_piece_bits(board, Color::Black, Piece::Bishop).rotate_left(37);
    mix_correction_key(white ^ black)
}

pub(in crate::search) fn non_pawn_correction_key(board: &Board) -> u64 {
    let white = colored_piece_bits(board, Color::White, Piece::Rook)
        ^ colored_piece_bits(board, Color::White, Piece::Queen).rotate_left(13)
        ^ colored_piece_bits(board, Color::White, Piece::King).rotate_left(29);
    let black = colored_piece_bits(board, Color::Black, Piece::Rook).rotate_left(41)
        ^ colored_piece_bits(board, Color::Black, Piece::Queen).rotate_left(47)
        ^ colored_piece_bits(board, Color::Black, Piece::King).rotate_left(53);
    mix_correction_key(white ^ black)
}

pub(in crate::search) fn colored_piece_bits(board: &Board, color: Color, piece: Piece) -> u64 {
    (board.pieces(piece) & board.colors(color)).0
}

pub(in crate::search) fn mix_correction_key(mut key: u64) -> u64 {
    key ^= key >> 33;
    key = key.wrapping_mul(0xff51afd7ed558ccd);
    key ^= key >> 33;
    key = key.wrapping_mul(0xc4ceb9fe1a85ec53);
    key ^ (key >> 33)
}
