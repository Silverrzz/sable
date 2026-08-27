use crate::{Board, Move};

use super::super::{
    moves::move_generation::ordered_root_moves,
    state::{context::SearchContext, move_context::MoveContext},
};
use super::{
    depth::search_root_child,
    outcome::{SearchOutcome, debug_validate_pv, is_better_root_outcome, parent_outcome},
};

#[derive(Clone, Debug)]
pub(in crate::search) struct RootMoveResult {
    pub(in crate::search) mv: Move,
    pub(in crate::search) score: i32,
    pub(in crate::search) repetition_draw: bool,
    pub(in crate::search) pv: Vec<Move>,
}

pub(in crate::search) fn search_root_multi_pv_iteration(
    board: &Board,
    candidate_moves: &[Move],
    depth: u32,
    previous_results: &[RootMoveResult],
    requested_multi_pv: usize,
    context: &mut SearchContext<'_>,
) -> Option<Vec<RootMoveResult>> {
    context.refresh_static_eval_at_ply(board, MoveContext::default(), 0);
    let root_repetitions = context.actual_game_repetition_count(board);
    let previous_best = previous_results.first().map(|result| result.mv);
    let moves = ordered_root_moves(board, candidate_moves, previous_best, context.ordering());
    let mut results = Vec::with_capacity(moves.len());

    for (move_index, ordered) in moves.into_iter().enumerate() {
        if context.should_stop() {
            return None;
        }
        let previous_pv = previous_results
            .iter()
            .find(|result| result.mv == ordered.mv)
            .map(|result| result.pv.as_slice())
            .unwrap_or(&[]);
        let mv = ordered.mv;
        let child = search_root_child(
            board,
            ordered,
            depth,
            previous_pv,
            Some(ordered.mv),
            i32::MIN + 1,
            i32::MAX,
            move_index as u32,
            false,
            context,
        )?;
        let outcome = parent_outcome(mv, child);
        debug_validate_pv(board, &outcome.pv, "MULTIPV");
        results.push(RootMoveResult {
            mv,
            score: outcome.score,
            repetition_draw: outcome.repetition_draw,
            pv: outcome.pv,
        });
    }

    results.sort_by(|a, b| {
        let a_outcome = SearchOutcome {
            score: a.score,
            repetition_draw: a.repetition_draw,
            pv: Vec::new(),
        };
        let b_outcome = SearchOutcome {
            score: b.score,
            repetition_draw: b.repetition_draw,
            pv: Vec::new(),
        };
        if is_better_root_outcome(&a_outcome, &b_outcome, root_repetitions) {
            std::cmp::Ordering::Less
        } else if is_better_root_outcome(&b_outcome, &a_outcome, root_repetitions) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    results.truncate(requested_multi_pv.max(1));
    Some(results)
}
