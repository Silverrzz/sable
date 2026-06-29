use crate::{
    Board, Color, Move, Piece,
    evaluation::LOSS_SCORE,
};

use super::{
    board_moves::{en_passant_target, is_en_passant},
    constants::*,
    transposition::{Bound, is_mate_score},
};

const CORRECTION_CONTINUATION_SIZE: usize = 2 * 6 * 64;
const CORRECTION_UPDATE_SCALE_DIVISOR: i32 = 128;

#[derive(Clone, Copy, Debug)]
pub(in crate::search) struct CorrectionMove {
    mv: Move,
    piece: Piece,
}

impl CorrectionMove {
    pub(in crate::search) fn new(mv: Move, piece: Piece) -> Self {
        Self { mv, piece }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::search) struct CorrectionContext {
    previous: Option<CorrectionMove>,
    previous_same_side: Option<CorrectionMove>,
}

impl CorrectionContext {
    pub(in crate::search) fn after_move(self, mv: Move, piece: Piece) -> Self {
        Self {
            previous: Some(CorrectionMove::new(mv, piece)),
            previous_same_side: self.previous,
        }
    }

    pub(in crate::search) fn without_move_context(self) -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug)]
pub(in crate::search) struct CorrectionHistory {
    pawn: Vec<i32>,
    minor: Vec<i32>,
    non_pawn_white: Vec<i32>,
    non_pawn_black: Vec<i32>,
    continuation_previous: Vec<i32>,
    continuation_same_side: Vec<i32>,
}

impl Default for CorrectionHistory {
    fn default() -> Self {
        Self {
            pawn: vec![0; 2 * CORRECTION_HISTORY_BUCKETS],
            minor: vec![0; 2 * CORRECTION_HISTORY_BUCKETS],
            non_pawn_white: vec![0; 2 * CORRECTION_HISTORY_BUCKETS],
            non_pawn_black: vec![0; 2 * CORRECTION_HISTORY_BUCKETS],
            continuation_previous: vec![0; CORRECTION_CONTINUATION_SIZE],
            continuation_same_side: vec![0; CORRECTION_CONTINUATION_SIZE],
        }
    }
}

impl CorrectionHistory {
    pub(in crate::search) fn decay(&mut self) {
        decay_correction_table(&mut self.pawn);
        decay_correction_table(&mut self.minor);
        decay_correction_table(&mut self.non_pawn_white);
        decay_correction_table(&mut self.non_pawn_black);
        decay_correction_table(&mut self.continuation_previous);
        decay_correction_table(&mut self.continuation_same_side);
    }

    pub(in crate::search) fn corrected_eval(
        &self,
        board: &Board,
        raw_eval: i32,
        correction_context: CorrectionContext,
    ) -> i32 {
        let corrected = raw_eval.saturating_add(self.correction(board, correction_context));
        corrected.clamp(
            LOSS_SCORE + MATE_PRUNING_GUARD,
            -LOSS_SCORE - MATE_PRUNING_GUARD,
        )
    }

    pub(in crate::search) fn update(
        &mut self,
        board: &Board,
        correction_context: CorrectionContext,
        raw_eval: i32,
        score: i32,
        depth: u32,
    ) {
        let target = score
            .saturating_sub(raw_eval)
            .clamp(-MAX_CORRECTION_HISTORY_SCORE, MAX_CORRECTION_HISTORY_SCORE);
        let weight = correction_history_weight(depth);
        let side = crate::chess::side_to_move(board) as usize;

        update_correction_value(
            &mut self.pawn[pawn_correction_index(board, side)],
            target,
            scaled_update_weight(weight, CORRECTION_HISTORY_PAWN_UPDATE_SCALE),
        );
        update_correction_value(
            &mut self.minor[minor_correction_index(board, side)],
            target,
            scaled_update_weight(weight, CORRECTION_HISTORY_MINOR_UPDATE_SCALE),
        );
        update_correction_value(
            &mut self.non_pawn_white[non_pawn_correction_index(board, side, Color::White)],
            target,
            scaled_update_weight(weight, CORRECTION_HISTORY_NON_PAWN_UPDATE_SCALE),
        );
        update_correction_value(
            &mut self.non_pawn_black[non_pawn_correction_index(board, side, Color::Black)],
            target,
            scaled_update_weight(weight, CORRECTION_HISTORY_NON_PAWN_UPDATE_SCALE),
        );
        if let Some(previous) = correction_context.previous {
            update_correction_value(
                &mut self.continuation_previous[continuation_correction_index(side, previous)],
                target,
                scaled_update_weight(weight, CORRECTION_HISTORY_PREVIOUS_UPDATE_SCALE),
            );
        }
        if let Some(previous_same_side) = correction_context.previous_same_side {
            update_correction_value(
                &mut self.continuation_same_side
                    [continuation_correction_index(side, previous_same_side)],
                target,
                scaled_update_weight(weight, CORRECTION_HISTORY_SAME_SIDE_UPDATE_SCALE),
            );
        }
    }

    pub(in crate::search) fn correction(
        &self,
        board: &Board,
        correction_context: CorrectionContext,
    ) -> i32 {
        let side = crate::chess::side_to_move(board) as usize;
        let mut sum = 0_i32;
        let mut weight_sum = 0_i32;
        add_weighted_correction(
            &mut sum,
            &mut weight_sum,
            self.pawn[pawn_correction_index(board, side)],
            CORRECTION_HISTORY_PAWN_WEIGHT,
        );
        add_weighted_correction(
            &mut sum,
            &mut weight_sum,
            self.minor[minor_correction_index(board, side)],
            CORRECTION_HISTORY_MINOR_WEIGHT,
        );
        add_weighted_correction(
            &mut sum,
            &mut weight_sum,
            self.non_pawn_white[non_pawn_correction_index(board, side, Color::White)],
            CORRECTION_HISTORY_NON_PAWN_WEIGHT,
        );
        add_weighted_correction(
            &mut sum,
            &mut weight_sum,
            self.non_pawn_black[non_pawn_correction_index(board, side, Color::Black)],
            CORRECTION_HISTORY_NON_PAWN_WEIGHT,
        );
        if let Some(previous) = correction_context.previous {
            add_weighted_correction(
                &mut sum,
                &mut weight_sum,
                self.continuation_previous[continuation_correction_index(side, previous)],
                CORRECTION_HISTORY_PREVIOUS_WEIGHT,
            );
        }
        if let Some(previous_same_side) = correction_context.previous_same_side {
            add_weighted_correction(
                &mut sum,
                &mut weight_sum,
                self.continuation_same_side[continuation_correction_index(
                    side,
                    previous_same_side,
                )],
                CORRECTION_HISTORY_SAME_SIDE_WEIGHT,
            );
        }
        if weight_sum == 0 {
            0
        } else {
            sum / weight_sum
        }
    }
}

pub(in crate::search) fn should_update_correction_history(
    board: &Board,
    best_move: Option<Move>,
    bound: Bound,
    corrected_eval: i32,
    score: i32,
) -> bool {
    let quiet_best_move = best_move.is_some_and(|mv| is_quiet_correction_move(board, mv));
    !is_mate_score(score)
        && quiet_best_move
        && match bound {
            Bound::Exact => true,
            Bound::Lower => score > corrected_eval,
            Bound::Upper => score < corrected_eval,
        }
}

pub(in crate::search) fn is_quiet_correction_move(board: &Board, mv: Move) -> bool {
    let moving_piece = crate::chess::piece_on(board, mv.from).unwrap_or(Piece::Pawn);
    mv.promotion.is_none()
        && crate::chess::piece_on(board, mv.to).is_none()
        && !is_en_passant(
            moving_piece,
            mv,
            en_passant_target(board, crate::chess::side_to_move(board)),
        )
}

pub(in crate::search) fn correction_history_weight(depth: u32) -> i32 {
    let depth = depth.min(16) as i32 + 1;
    depth.saturating_mul(depth)
}

pub(in crate::search) fn scaled_update_weight(weight: i32, scale: i32) -> i32 {
    weight.saturating_mul(scale) / CORRECTION_UPDATE_SCALE_DIVISOR
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

pub(in crate::search) fn add_weighted_correction(
    sum: &mut i32,
    weight_sum: &mut i32,
    value: i32,
    weight: i32,
) {
    *sum = sum.saturating_add(value.saturating_mul(weight));
    *weight_sum = weight_sum.saturating_add(weight);
}

pub(in crate::search) fn pawn_correction_index(board: &Board, side: usize) -> usize {
    correction_index(side, pawn_correction_key(board))
}

pub(in crate::search) fn minor_correction_index(board: &Board, side: usize) -> usize {
    correction_index(side, minor_correction_key(board))
}

pub(in crate::search) fn non_pawn_correction_index(
    board: &Board,
    side: usize,
    color: Color,
) -> usize {
    correction_index(side, non_pawn_correction_key(board, color))
}

pub(in crate::search) fn continuation_correction_index(
    side: usize,
    previous: CorrectionMove,
) -> usize {
    ((side * 6) + previous.piece as usize) * 64 + previous.mv.to as usize
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

pub(in crate::search) fn non_pawn_correction_key(board: &Board, color: Color) -> u64 {
    let pieces = colored_piece_bits(board, color, Piece::Rook)
        ^ colored_piece_bits(board, color, Piece::Queen).rotate_left(13)
        ^ colored_piece_bits(board, color, Piece::King).rotate_left(29);
    mix_correction_key(pieces)
}

pub(in crate::search) fn colored_piece_bits(board: &Board, color: Color, piece: Piece) -> u64 {
    (crate::chess::pieces(board, piece) & crate::chess::colors(board, color)).0
}

pub(in crate::search) fn mix_correction_key(mut key: u64) -> u64 {
    key ^= key >> 33;
    key = key.wrapping_mul(0xff51afd7ed558ccd);
    key ^= key >> 33;
    key = key.wrapping_mul(0xc4ceb9fe1a85ec53);
    key ^ (key >> 33)
}
