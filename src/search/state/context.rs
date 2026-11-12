use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use crate::{Board, Move, Piece, nnue::Nnue};

use super::{
    correction_history::CorrectionHistory,
    move_context::MoveContext,
    position_key::{PositionKey, actual_game_repetition_count, is_repetition, position_key},
    transposition::TranspositionTable,
};
use crate::search::{constants::*, moves::move_ordering::MoveOrdering};

mod parts;

use parts::{
    NodeCounters, RepetitionState, SearchClock, SearchControls, SearchHeuristics,
};

pub(in crate::search) struct SearchContext<'a> {
    clock: SearchClock,
    counters: NodeCounters<'a>,
    controls: SearchControls<'a>,
    nnue: Nnue,
    repetition: RepetitionState,
    heuristics: SearchHeuristics,
}

pub(in crate::search) struct SearchContextConfig<'a> {
    pub(in crate::search) root_board: &'a Board,
    pub(in crate::search) started: Instant,
    pub(in crate::search) hard_time_ms: Option<u64>,
    pub(in crate::search) node_limit: Option<u64>,
    pub(in crate::search) soft_node_limit: Option<u64>,
    pub(in crate::search) stop_flag: Option<&'a AtomicBool>,
    pub(in crate::search) ponder_flag: Option<&'a AtomicBool>,
    pub(in crate::search) game_history: &'a [PositionKey],
    pub(in crate::search) transposition_table: TranspositionTable,
    pub(in crate::search) search_state: PersistentSearchState,
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
            nnue: Nnue::new(root_board),
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
        context.refresh_static_eval_at_ply(root_board, MoveContext::default(), 0);
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

    pub(in crate::search) fn push_eval_state(
        &mut self,
        board: &Board,
        mv: Move,
        moving_piece: Piece,
        captured_piece: Option<Piece>,
    ) {
        self.nnue.make_move(board, mv, moving_piece, captured_piece);
    }

    pub(in crate::search) fn pop_eval_state(&mut self) {
        self.nnue.unmake_move();
    }

    pub(in crate::search) fn evaluate(&mut self, board: &Board) -> i32 {
        self.nnue.update(board);
        self.nnue.eval(board)
    }

    pub(in crate::search) fn refresh_static_eval_at_ply(
        &mut self,
        board: &Board,
        move_context: MoveContext,
        ply: u16,
    ) -> i32 {
        let raw_eval = self.evaluate(board);
        let static_eval = self.corrected_static_eval(board, raw_eval, move_context);
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
        move_context: MoveContext,
    ) -> i32 {
        self.heuristics
            .correction_history
            .corrected_eval(board, raw_eval, move_context)
    }

    pub(in crate::search) fn update_correction_history(
        &mut self,
        board: &Board,
        move_context: MoveContext,
        raw_eval: i32,
        score: i32,
        depth: u32,
    ) {
        self.heuristics
            .correction_history
            .update(board, move_context, raw_eval, score, depth);
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
