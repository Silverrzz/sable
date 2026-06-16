use std::{
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::{Duration, Instant},
};

use crate::{
    Board,
    evaluation::{Evaluator, NnueAccumulators, NnueEvalScratch, NnueFinnyTable},
};

use super::{EvalPending, SearchContext};
use crate::search::{
    constants::MAX_ORDERING_PLY,
    constants::STOP_CHECK_NODE_INTERVAL,
    moves::move_ordering::MoveOrdering,
    state::{
        correction_history::CorrectionHistory,
        position_key::PositionKey,
        stop_reason::StopReason,
        transposition::TranspositionTable,
    },
};

pub(super) struct SearchClock {
    pub(super) timed_started: Instant,
}

pub(super) struct NodeCounters<'a> {
    pub(super) nodes: u64,
    pub(super) seldepth: u32,
    pub(super) next_stop_check_node: u64,
    pub(super) shared_nodes: Option<&'a AtomicU64>,
    pub(super) published_nodes: u64,
}

pub(super) struct SearchControls<'a> {
    pub(super) hard_time_ms: Option<u64>,
    pub(super) node_limit: Option<u64>,
    pub(super) soft_node_limit: Option<u64>,
    pub(super) stop_flag: Option<&'a AtomicBool>,
    pub(super) lazy_stop_flag: Option<&'a AtomicBool>,
    pub(super) ponder_flag: Option<&'a AtomicBool>,
    pub(super) was_pondering: bool,
}

pub(super) struct EvalStackState {
    pub(super) evaluator: Evaluator,
    pub(super) chess960: bool,
    pub(super) stack: Vec<NnueAccumulators>,
    pub(super) ply: usize,
    pub(super) boards: Vec<Board>,
    pub(super) pending: Vec<EvalPending>,
    pub(super) materialized: usize,
    pub(super) scratch: Option<NnueEvalScratch>,
    pub(super) finny: Option<NnueFinnyTable>,
}

pub(super) struct RepetitionState {
    pub(super) game_history: Vec<PositionKey>,
    pub(super) path_keys: Vec<PositionKey>,
}

pub(super) struct SearchHeuristics {
    pub(super) static_eval_stack: [Option<i32>; MAX_ORDERING_PLY],
    pub(super) ordering: MoveOrdering,
    pub(super) correction_history: CorrectionHistory,
    pub(super) transposition_table: TranspositionTable,
}

impl<'a> SearchContext<'a> {
    pub(in crate::search) fn set_lazy_smp_state(
        &mut self,
        lazy_stop_flag: Option<&'a AtomicBool>,
        shared_nodes: Option<&'a AtomicU64>,
    ) {
        self.controls.lazy_stop_flag = lazy_stop_flag;
        self.counters.shared_nodes = shared_nodes;
    }

    pub(in crate::search) fn clock_elapsed(&mut self) -> Duration {
        self.update_ponder_state();
        self.clock.timed_started.elapsed()
    }

    pub(in crate::search) fn clock_elapsed_ms(&mut self) -> u64 {
        self.clock_elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64
    }

    pub(in crate::search) fn is_pondering(&mut self) -> bool {
        self.update_ponder_state();
        self.controls.was_pondering
    }

    pub(in crate::search) fn update_ponder_state(&mut self) {
        let Some(flag) = self.controls.ponder_flag else {
            return;
        };
        let pondering = flag.load(Ordering::Relaxed);
        if self.controls.was_pondering && !pondering {
            self.clock.timed_started = Instant::now();
        }
        self.controls.was_pondering = pondering;
    }

    pub(in crate::search) fn note_searched_move(&mut self, ply: u16) {
        self.counters.nodes = self.counters.nodes.saturating_add(1);
        self.counters.seldepth = self.counters.seldepth.max(ply as u32);
    }

    pub(in crate::search) fn flush_shared_node_counts(&mut self) {
        if let Some(shared_nodes) = self.counters.shared_nodes {
            let delta = self.counters.nodes.saturating_sub(self.counters.published_nodes);
            if delta > 0 {
                shared_nodes.fetch_add(delta, Ordering::Relaxed);
                self.counters.published_nodes = self.counters.nodes;
            }
        }
    }

    pub(in crate::search) fn total_nodes(&self) -> u64 {
        self.counters.shared_nodes.map_or(self.counters.nodes, |nodes| {
            nodes
                .load(Ordering::Relaxed)
                .saturating_add(self.counters.nodes.saturating_sub(self.counters.published_nodes))
        })
    }

    pub(in crate::search) fn local_nodes(&self) -> u64 {
        self.counters.nodes
    }

    pub(in crate::search) fn should_stop(&mut self) -> Option<StopReason> {
        if let Some(node_limit) = self.controls.node_limit {
            let nodes = if self.counters.shared_nodes.is_some() {
                self.total_nodes()
            } else {
                self.counters.nodes
            };
            if nodes >= node_limit {
                return Some(StopReason::NodeLimit);
            }
        }
        if self
            .controls
            .stop_flag
            .map(|flag| flag.load(Ordering::Relaxed))
            .unwrap_or(false)
        {
            return Some(StopReason::ExternalStop);
        }
        if self.is_pondering() {
            return None;
        }

        let Some(hard_time_ms) = self.controls.hard_time_ms else {
            if self.controls.lazy_stop_flag.is_none() {
                return None;
            }
            if self.counters.nodes < self.counters.next_stop_check_node {
                return None;
            }
            self.flush_shared_node_counts();
            self.counters.next_stop_check_node = self
                .counters
                .nodes
                .saturating_add(STOP_CHECK_NODE_INTERVAL);
            return self
                .controls
                .lazy_stop_flag
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
                .then_some(StopReason::ExternalStop);
        };

        if self.counters.nodes < self.counters.next_stop_check_node {
            return None;
        }
        self.flush_shared_node_counts();
        self.counters.next_stop_check_node = self
            .counters
            .nodes
            .saturating_add(STOP_CHECK_NODE_INTERVAL);
        if self
            .controls
            .lazy_stop_flag
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return Some(StopReason::ExternalStop);
        }
        if self.clock_elapsed_ms() >= hard_time_ms {
            return Some(StopReason::TimeHard);
        }
        None
    }

    pub(in crate::search) fn should_stop_before_iteration_for_nodes(
        &self,
        completed_depth: u32,
    ) -> bool {
        completed_depth > 0
            && self
                .controls
                .soft_node_limit
                .is_some_and(|soft_node_limit| self.total_nodes() >= soft_node_limit)
    }

    pub(in crate::search) fn seldepth(&self) -> u32 {
        self.counters.seldepth
    }

    pub(in crate::search) fn chess960(&self) -> bool {
        self.eval.chess960
    }

    pub(in crate::search) fn game_history(&self) -> &[PositionKey] {
        &self.repetition.game_history
    }

    pub(in crate::search) fn ordering(&self) -> &MoveOrdering {
        &self.heuristics.ordering
    }

    pub(in crate::search) fn ordering_mut(&mut self) -> &mut MoveOrdering {
        &mut self.heuristics.ordering
    }

    pub(in crate::search) fn transposition_table(&self) -> &TranspositionTable {
        &self.heuristics.transposition_table
    }
}
