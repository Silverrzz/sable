use crate::Board;

use super::{
    constants::*,
    correction_history::{correction_history_weight, pawn_correction_key},
    transposition::{Bound, is_mate_score},
};

const VOLATILITY_HISTORY_BUCKETS: usize = 16_384;
const VOLATILITY_UPDATE_SCALE_DIVISOR: i32 = 128;

#[derive(Clone, Debug)]
pub(in crate::search) struct VolatilityHistory {
    pawn: Vec<i32>,
}

impl Default for VolatilityHistory {
    fn default() -> Self {
        Self {
            pawn: vec![volatility_baseline(); 2 * VOLATILITY_HISTORY_BUCKETS],
        }
    }
}

impl VolatilityHistory {
    pub(in crate::search) fn decay(&mut self) {
        let maximum = MAX_VOLATILITY_HISTORY_SCORE();
        let baseline = volatility_baseline();
        for value in &mut self.pawn {
            let current = (*value).clamp(0, maximum);
            *value = baseline
                .saturating_add(current.saturating_sub(baseline) / 2)
                .clamp(0, maximum);
        }
    }

    pub(in crate::search) fn volatility(&self, board: &Board) -> i32 {
        self.pawn[volatility_index(board)].clamp(0, MAX_VOLATILITY_HISTORY_SCORE())
    }

    pub(in crate::search) fn update(
        &mut self,
        board: &Board,
        raw_eval: i32,
        score: i32,
        depth: u32,
    ) {
        let maximum = MAX_VOLATILITY_HISTORY_SCORE();
        let target = score.abs_diff(raw_eval).min(maximum as u32) as i32;
        let weight = correction_history_weight(depth)
            .saturating_mul(VOLATILITY_HISTORY_UPDATE_SCALE())
            / VOLATILITY_UPDATE_SCALE_DIVISOR;
        let value = &mut self.pawn[volatility_index(board)];
        let current = (*value).clamp(0, maximum);
        let delta = target.saturating_sub(current);
        *value = current
            .saturating_add(
                delta.saturating_mul(weight) / CORRECTION_HISTORY_UPDATE_DIVISOR,
            )
            .clamp(0, maximum);
    }
}

pub(in crate::search) fn should_update_volatility_history(
    bound: Bound,
    raw_eval: i32,
    score: i32,
) -> bool {
    !is_mate_score(score)
        && match bound {
            Bound::Exact => true,
            Bound::Lower => score > raw_eval,
            Bound::Upper => score < raw_eval,
        }
}

fn volatility_index(board: &Board) -> usize {
    let side = crate::chess::side_to_move(board) as usize;
    side * VOLATILITY_HISTORY_BUCKETS
        + (pawn_correction_key(board) as usize & (VOLATILITY_HISTORY_BUCKETS - 1))
}

fn volatility_baseline() -> i32 {
    VOLATILITY_HISTORY_BASELINE().clamp(0, MAX_VOLATILITY_HISTORY_SCORE())
}
