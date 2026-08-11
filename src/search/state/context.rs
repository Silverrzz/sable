use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use crate::{
    Board, Move,
    evaluation::{Evaluator, evaluate_position},
};

use super::{
    correction_history::{CorrectionContext, CorrectionHistory},
    position_key::{PositionKey, actual_game_repetition_count, is_repetition, position_key},
    transposition::TranspositionTable,
};
use crate::search::{constants::*, moves::move_ordering::MoveOrdering};

mod parts;

use parts::{
    EvalStackState, NodeCounters, RepetitionState, SearchClock, SearchControls, SearchHeuristics,
};

pub(in crate::search) struct SearchContext<'a> {
    clock: SearchClock,
    counters: NodeCounters<'a>,
    controls: SearchControls<'a>,
    eval: EvalStackState,
    repetition: RepetitionState,
    heuristics: SearchHeuristics,
}

pub(in crate::search) struct SearchContextConfig<'a> {
    pub(in crate::search) root_board: &'a Board,
    pub(in crate::search) started: Instant,
    pub(in crate::search) hard_time_ms: Option<u64>,
    pub(in crate::search) node_limit: Option<u64>,
    pub(in crate::search) soft_node_limit: Option<u64>,
    pub(in crate::search) evaluator: Evaluator,
    pub(in crate::search) stop_flag: Option<&'a AtomicBool>,
    pub(in crate::search) ponder_flag: Option<&'a AtomicBool>,
    pub(in crate::search) game_history: &'a [PositionKey],
    pub(in crate::search) transposition_table: TranspositionTable,
    pub(in crate::search) search_state: PersistentSearchState,
}

#[derive(Clone, Copy, Debug)]
enum EvalPending {
    Root,
    Move {
        mv: Move,
        moving_piece: crate::Piece,
        captured_piece: Option<crate::Piece>,
    },
    NullMove,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PersistentSearchState {
    pub(in crate::search) ordering: MoveOrdering,
    pub(in crate::search) correction_history: CorrectionHistory,
}

impl PersistentSearchState {
    pub(in crate::search) fn decay(&mut self) {
        self.ordering.decay_persistent();
        self.correction_history.decay();
    }
}

impl<'a> SearchContext<'a> {
    pub(in crate::search) fn new(mut config: SearchContextConfig<'a>) -> Self {
        config.search_state.decay();
        let SearchContextConfig {
            root_board,
            started,
            hard_time_ms,
            node_limit,
            soft_node_limit,
            evaluator,
            stop_flag,
            ponder_flag,
            game_history,
            transposition_table,
            search_state,
        } = config;

        let mut game_history = game_history.to_vec();
        let root_key = position_key(root_board);
        if game_history.last() != Some(&root_key) {
            game_history.push(root_key);
        }
        let history_without_current = game_history.len().saturating_sub(1);
        let mut repetition_table = Vec::with_capacity(game_history.len() + MAX_ORDERING_PLY);
        repetition_table.extend_from_slice(&game_history[..history_without_current]);
        let eval_state = evaluator
            .active_nnue_model()
            .and_then(|nnue| nnue.initial_accumulators(root_board));
        let mut eval_stack = Vec::with_capacity(MAX_ORDERING_PLY + 1);
        if let Some(accumulators) = eval_state {
            eval_stack.push(accumulators);
            while eval_stack.len() < MAX_ORDERING_PLY + 1 {
                let accumulator = crate::evaluation::NnueAccumulators::empty_like(&eval_stack[0]);
                eval_stack.push(accumulator);
            }
        }
        let mut eval_boards = Vec::with_capacity(MAX_ORDERING_PLY + 1);
        eval_boards.push(root_board.clone());
        let mut eval_pending = Vec::with_capacity(MAX_ORDERING_PLY + 1);
        eval_pending.push(EvalPending::Root);
        if !eval_stack.is_empty() {
            while eval_boards.len() < MAX_ORDERING_PLY + 1 {
                eval_boards.push(root_board.clone());
                eval_pending.push(EvalPending::Root);
            }
        }
        let mut eval_finny = evaluator
            .active_nnue_model()
            .and_then(|nnue| nnue.new_finny_table());
        let seeded_finny = if let (Some(model), Some(accumulators), Some(table)) = (
            evaluator.active_nnue_model(),
            eval_stack.first(),
            eval_finny.as_mut(),
        ) {
            model.seed_finny_table(table, root_board, accumulators)
        } else {
            eval_finny.is_none()
        };
        if !seeded_finny {
            eval_finny = None;
        }
        let was_pondering = ponder_flag
            .map(|flag| flag.load(Ordering::Relaxed))
            .unwrap_or(false);
        let PersistentSearchState {
            mut ordering,
            correction_history,
        } = search_state;
        ordering.clear_search_local();
        let mut context = Self {
            clock: SearchClock {
                timed_started: started,
            },
            counters: NodeCounters {
                nodes: 0,
                seldepth: 0,
                next_stop_check_node: 0,
                shared_nodes: None,
                published_nodes: 0,
            },
            controls: SearchControls {
                hard_time_ms,
                node_limit,
                soft_node_limit,
                stop_flag,
                lazy_stop_flag: None,
                ponder_flag,
                was_pondering,
            },
            eval: EvalStackState {
                evaluator,
                stack: eval_stack,
                ply: 0,
                boards: eval_boards,
                pending: eval_pending,
                materialized: 0,
                finny: eval_finny,
            },
            repetition: RepetitionState {
                game_history,
                path_keys: repetition_table,
            },
            heuristics: SearchHeuristics {
                static_eval_stack: [None; MAX_ORDERING_PLY],
                ordering,
                correction_history,
                transposition_table,
            },
        };
        context.refresh_static_eval_at_ply(root_board, CorrectionContext::default(), 0);
        context
    }

    pub(in crate::search) fn repetition_keys(&self) -> &[PositionKey] {
        &self.repetition.path_keys
    }

    /// actual game repetition count at root, current visit included
    pub(in crate::search) fn actual_game_repetition_count(&self, board: &Board) -> u8 {
        actual_game_repetition_count(board, &self.repetition.game_history)
    }

    #[inline]
    pub(in crate::search) fn push_repetition_key(&mut self, key: PositionKey) {
        self.repetition.path_keys.push(key);
    }

    pub(in crate::search) fn push_position(&mut self, board: &Board, key: PositionKey) -> bool {
        let repeated = is_repetition(
            key,
            crate::chess::halfmove_clock(board),
            &self.repetition.path_keys,
        );
        self.repetition.path_keys.push(key);
        repeated
    }

    pub(in crate::search) fn pop_position(&mut self, key: PositionKey) {
        let popped = self.repetition.path_keys.pop();
        debug_assert_eq!(popped, Some(key));
    }

    fn push_eval_frame(&mut self, after: &Board, pending: EvalPending) {
        if self.eval.stack.is_empty() {
            return;
        }
        let next_ply = self.eval.ply + 1;
        debug_assert!(next_ply <= self.eval.boards.len());
        if next_ply == self.eval.boards.len() {
            self.eval.boards.push(after.clone());
            self.eval.pending.push(pending);
        } else {
            self.eval.boards[next_ply].clone_from(after);
            self.eval.pending[next_ply] = pending;
        }
        self.eval.ply = next_ply;
    }

    pub(in crate::search) fn push_eval_state(
        &mut self,
        after: &Board,
        mv: Move,
        moving_piece: crate::Piece,
        captured_piece: Option<crate::Piece>,
    ) {
        self.push_eval_frame(
            after,
            EvalPending::Move {
                mv,
                moving_piece,
                captured_piece,
            },
        );
    }

    pub(in crate::search) fn pop_eval_state(&mut self) {
        if self.eval.stack.is_empty() {
            return;
        };
        debug_assert!(self.eval.ply > 0);
        self.eval.ply = self.eval.ply.saturating_sub(1);
        self.eval.materialized = self.eval.materialized.min(self.eval.ply);
    }

    pub(in crate::search) fn push_null_eval_state(&mut self, after: &Board) {
        self.push_eval_frame(after, EvalPending::NullMove);
    }

    pub(in crate::search) fn pop_null_eval_state(&mut self) {
        if self.eval.stack.is_empty() {
            return;
        };
        debug_assert!(self.eval.ply > 0);
        self.eval.ply = self.eval.ply.saturating_sub(1);
        self.eval.materialized = self.eval.materialized.min(self.eval.ply);
    }

    /// brings current eval ply up to date
    fn materialize_eval_stack(&mut self) {
        while self.eval.materialized < self.eval.ply {
            let ply = self.eval.materialized + 1;
            if ply == self.eval.stack.len() {
                let empty =
                    crate::evaluation::NnueAccumulators::empty_like(&self.eval.stack[ply - 1]);
                self.eval.stack.push(empty);
            }
            let Some(model) = self.eval.evaluator.active_nnue_model() else {
                return;
            };
            let before = &self.eval.boards[ply - 1];
            let after = &self.eval.boards[ply];
            let (previous, next) = self.eval.stack.split_at_mut(ply);
            let source = &previous[ply - 1];
            let target = &mut next[0];
            let updated = match self.eval.pending[ply] {
                EvalPending::Move {
                    mv,
                    moving_piece,
                    captured_piece,
                } => model.update_accumulators_after_move(
                    source,
                    target,
                    before,
                    after,
                    mv,
                    moving_piece,
                    captured_piece,
                    self.eval.finny.as_mut(),
                ),
                EvalPending::NullMove => {
                    target.clone_from(source);
                    model.apply_null_move_delta(target, before)
                }
                EvalPending::Root => {
                    target.clone_from(source);
                    true
                }
            };
            if !updated {
                model.refresh_accumulators_into_with_finny(target, after, self.eval.finny.as_mut());
            }
            self.eval.materialized = ply;
        }
    }

    pub(in crate::search) fn evaluate(&mut self, board: &Board) -> i32 {
        if !self.eval.stack.is_empty() {
            self.materialize_eval_stack();
        }
        if let (Some(model), Some(accumulators)) = (
            self.eval.evaluator.active_nnue_model(),
            self.eval.stack.get(self.eval.ply),
        ) {
            return model.evaluate_for_side_to_move_with_accumulators(board, accumulators);
        }
        evaluate_position(board, &self.eval.evaluator)
    }

    pub(in crate::search) fn refresh_static_eval_at_ply(
        &mut self,
        board: &Board,
        correction_context: CorrectionContext,
        ply: u16,
    ) -> i32 {
        let raw_eval = self.evaluate(board);
        let static_eval = self.corrected_static_eval(board, raw_eval, correction_context);
        self.record_static_eval_at_ply(ply, static_eval);
        static_eval
    }

    pub(in crate::search) fn clear_static_eval_at_ply(&mut self, ply: u16) {
        if let Some(slot) = self.heuristics.static_eval_stack.get_mut(ply as usize) {
            *slot = None;
        }
    }

    pub(in crate::search) fn record_static_eval_at_ply(&mut self, ply: u16, static_eval: i32) {
        if let Some(slot) = self.heuristics.static_eval_stack.get_mut(ply as usize) {
            *slot = Some(static_eval);
        }
    }

    pub(in crate::search) fn is_static_eval_improving(&self, ply: u16, static_eval: i32) -> bool {
        for distance in [2_u16, 4] {
            let Some(parent_ply) = ply.checked_sub(distance) else {
                continue;
            };
            if let Some(previous_eval) = self
                .heuristics
                .static_eval_stack
                .get(parent_ply as usize)
                .and_then(|eval| *eval)
            {
                return static_eval > previous_eval;
            }
        }
        true
    }

    pub(in crate::search) fn corrected_static_eval(
        &self,
        board: &Board,
        raw_eval: i32,
        correction_context: CorrectionContext,
    ) -> i32 {
        self.heuristics
            .correction_history
            .corrected_eval(board, raw_eval, correction_context)
    }

    pub(in crate::search) fn update_correction_history(
        &mut self,
        board: &Board,
        correction_context: CorrectionContext,
        raw_eval: i32,
        score: i32,
        depth: u32,
    ) {
        self.heuristics.correction_history.update(
            board,
            correction_context,
            raw_eval,
            score,
            depth,
        );
    }

    /// moves persistent tables out without cloning
    /// only call this when search is done with the context
    pub(in crate::search) fn take_persistent_state(&mut self) -> PersistentSearchState {
        PersistentSearchState {
            ordering: std::mem::take(&mut self.heuristics.ordering),
            correction_history: std::mem::take(&mut self.heuristics.correction_history),
        }
    }
}
