
use crate::Move;

use super::super::{
    constants::*,
    state::context::SearchContext,
    types::SearchBudget,
};

#[derive(Clone, Debug)]
pub(in crate::search) struct IterativeTimeManager {
    base_soft_ms: Option<u64>,
    hard_ms: Option<u64>,
    previous_total_nodes: u64,
    previous_elapsed_ms: u64,
    previous_iteration_nodes: Option<u64>,
    last_iteration_nodes: Option<u64>,
    last_iteration_ms: Option<u64>,
    last_best_move: Option<Move>,
    best_move_stability: u32,
    last_score: Option<i32>,
    last_score_delta: Option<i32>,
    fail_low_multiplier_per_mille: u64,
    score_stability: u32,
}

impl IterativeTimeManager {
    pub(in crate::search) fn new(budget: &SearchBudget) -> Self {
        Self {
            base_soft_ms: budget.soft_time_ms,
            hard_ms: budget.hard_time_ms,
            previous_total_nodes: 0,
            previous_elapsed_ms: 0,
            previous_iteration_nodes: None,
            last_iteration_nodes: None,
            last_iteration_ms: None,
            last_best_move: None,
            best_move_stability: 0,
            last_score: None,
            last_score_delta: None,
            fail_low_multiplier_per_mille: 1_000,
            score_stability: 0,
        }
    }

    pub(in crate::search) fn should_stop_before_iteration(
        &self,
        completed_depth: u32,
        context: &mut SearchContext<'_>,
    ) -> bool {
        let pondering = context.is_pondering();
        let elapsed_ms = context.clock_elapsed_ms();
        self.should_stop_at_iteration_start(completed_depth, elapsed_ms, pondering)
    }

    pub(in crate::search) fn should_stop_at_iteration_start(
        &self,
        completed_depth: u32,
        elapsed_ms: u64,
        pondering: bool,
    ) -> bool {
        if pondering || completed_depth == 0 {
            return false;
        }
        let Some(limit_ms) = self.effective_soft_time_ms() else {
            return false;
        };
        if elapsed_ms >= limit_ms {
            return true;
        }
        if completed_depth < TIME_MANAGER_MIN_PREDICTION_DEPTH || !self.dynamic_soft_enabled() {
            return false;
        }
        let Some(predicted_ms) = self.predicted_next_iteration_ms() else {
            return false;
        };
        elapsed_ms.saturating_add(predicted_ms) >= limit_ms
    }

    pub(in crate::search) fn record_completed_iteration(
        &mut self,
        elapsed_ms: u64,
        total_nodes: u64,
        best_move: Move,
        score: i32,
    ) {
        let iteration_nodes = total_nodes.saturating_sub(self.previous_total_nodes);
        let iteration_ms = elapsed_ms.saturating_sub(self.previous_elapsed_ms);
        self.previous_total_nodes = total_nodes;
        self.previous_elapsed_ms = elapsed_ms;
        self.previous_iteration_nodes = self.last_iteration_nodes;
        self.last_iteration_nodes = Some(iteration_nodes);
        self.last_iteration_ms = Some(iteration_ms);

        if self.last_best_move == Some(best_move) {
            self.best_move_stability = self.best_move_stability.saturating_add(1);
        } else {
            self.best_move_stability = 0;
            self.last_best_move = Some(best_move);
        }

        if let Some(last_score) = self.last_score {
            let delta = score.abs_diff(last_score).min(i32::MAX as u32) as i32;
            let score_drop = if score < last_score {
                last_score.saturating_sub(score)
            } else {
                0
            };
            self.last_score_delta = Some(delta);
            self.fail_low_multiplier_per_mille = fail_low_multiplier_per_mille(score_drop);
            if delta <= TIME_MANAGER_STABLE_SCORE_CP {
                self.score_stability = self.score_stability.saturating_add(1);
            } else {
                self.score_stability = 0;
            }
        } else {
            self.fail_low_multiplier_per_mille = 1_000;
        }
        self.last_score = Some(score);
    }

    pub(in crate::search) fn effective_soft_time_ms(&self) -> Option<u64> {
        let soft_ms = self.base_soft_ms?;
        let scaled = if self.dynamic_soft_enabled() {
            scale_time_ms(soft_ms, self.soft_multiplier_per_mille())
        } else {
            soft_ms
        };
        Some(match self.hard_ms {
            Some(hard_ms) => scaled.min(hard_ms).max(1),
            None => scaled.max(1),
        })
    }

    pub(in crate::search) fn dynamic_soft_enabled(&self) -> bool {
        match (self.base_soft_ms, self.hard_ms) {
            (Some(soft_ms), Some(hard_ms)) => hard_ms > soft_ms,
            (Some(_), None) => true,
            _ => false,
        }
    }

    pub(in crate::search) fn soft_multiplier_per_mille(&self) -> u64 {
        let move_factor: u64 = match self.best_move_stability {
            0 => 1_250,
            1 => 1_100,
            2 => 1_000,
            3 => 900,
            _ => 800,
        };
        let score_factor: u64 = match (self.last_score_delta, self.score_stability) {
            (Some(delta), _) if delta >= 250 => 1_350,
            (Some(delta), _) if delta >= 120 => 1_200,
            (Some(delta), _) if delta >= 60 => 1_100,
            (Some(delta), stability) if delta <= 12 && stability >= 3 => 850,
            (Some(delta), stability) if delta <= TIME_MANAGER_STABLE_SCORE_CP && stability >= 2 => {
                925
            }
            _ => 1_000,
        };
        move_factor
            .saturating_mul(score_factor)
            .saturating_div(1_000)
            .saturating_mul(self.fail_low_multiplier_per_mille)
            .saturating_div(1_000)
            .clamp(
                TIME_MANAGER_MIN_SOFT_MULTIPLIER,
                TIME_MANAGER_MAX_SOFT_MULTIPLIER,
            )
    }

    pub(in crate::search) fn predicted_next_iteration_ms(&self) -> Option<u64> {
        let last_iteration_ms = self.last_iteration_ms?;
        let growth = self.node_growth_per_mille();
        Some(scale_time_ms(last_iteration_ms.max(1), growth).max(1))
    }

    pub(in crate::search) fn node_growth_per_mille(&self) -> u64 {
        let Some(last_nodes) = self.last_iteration_nodes else {
            return TIME_MANAGER_DEFAULT_NODE_GROWTH_PERMILLE;
        };
        let Some(previous_nodes) = self.previous_iteration_nodes else {
            return TIME_MANAGER_DEFAULT_NODE_GROWTH_PERMILLE;
        };
        if previous_nodes == 0 {
            return TIME_MANAGER_DEFAULT_NODE_GROWTH_PERMILLE;
        }
        last_nodes
            .saturating_mul(1_000)
            .saturating_div(previous_nodes)
            .clamp(
                TIME_MANAGER_MIN_NODE_GROWTH_PERMILLE,
                TIME_MANAGER_MAX_NODE_GROWTH_PERMILLE,
            )
    }
}

pub(in crate::search) fn fail_low_multiplier_per_mille(score_drop: i32) -> u64 {
    if score_drop >= TIME_MANAGER_FAIL_LOW_BIG_DROP_CP {
        TIME_MANAGER_FAIL_LOW_BIG_MULTIPLIER
    } else if score_drop >= TIME_MANAGER_FAIL_LOW_MEDIUM_DROP_CP {
        TIME_MANAGER_FAIL_LOW_MEDIUM_MULTIPLIER
    } else if score_drop >= TIME_MANAGER_FAIL_LOW_SMALL_DROP_CP {
        TIME_MANAGER_FAIL_LOW_SMALL_MULTIPLIER
    } else {
        1_000
    }
}

pub(in crate::search) fn scale_time_ms(time_ms: u64, multiplier_per_mille: u64) -> u64 {
    let scaled = u128::from(time_ms)
        .saturating_mul(u128::from(multiplier_per_mille))
        .saturating_div(1_000);
    scaled.min(u128::from(u64::MAX)) as u64
}
