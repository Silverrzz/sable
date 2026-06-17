use crate::Board;

use super::{
    PruneResult, negamax,
    super::{
        context::SearchContext,
        pruning::{null_move_reduction, should_try_null_move},
        root::terminal_outcome,
    },
};

pub(super) struct NullMoveParams<'a> {
    pub(super) board: &'a Board,
    pub(super) repetition: bool,
    pub(super) depth: u32,
    pub(super) root_depth: u32,
    pub(super) beta: i32,
    pub(super) is_pv_node: bool,
    pub(super) in_check: bool,
    pub(super) needs_full_mate_search: bool,
    pub(super) static_eval: Option<i32>,
    pub(super) allow_null_move: bool,
    pub(super) ply: u16,
}

pub(super) fn try_null_move_pruning(
    params: NullMoveParams<'_>,
    context: &mut SearchContext<'_>,
) -> PruneResult {
    if params.needs_full_mate_search
        || !should_try_null_move(
            params.board,
            params.depth,
            params.is_pv_node,
            params.in_check,
            params.allow_null_move,
        )
        || !static_eval_supports_null_move(params.static_eval, params.beta)
    {
        return PruneResult::Continue;
    }

    let Some(null_board) = params.board.null_move() else {
        return PruneResult::Continue;
    };
    let reduction = null_move_reduction(params.depth);
    let null_depth = params.depth.saturating_sub(1 + reduction);
    let null_alpha = params.beta.saturating_neg();
    let null_beta = null_alpha.saturating_add(1);
    context.push_null_eval_state(params.board, &null_board);
    let null_result = negamax(
        &null_board,
        params.repetition,
        null_depth,
        params.root_depth,
        null_alpha,
        null_beta,
        &[],
        None,
        context,
        params.ply + 1,
        false,
    );
    context.pop_null_eval_state(params.board);
    let Some(null_result) = null_result else {
        return PruneResult::Interrupted;
    };
    if -null_result.score >= params.beta {
        PruneResult::Done(terminal_outcome(params.beta, false))
    } else {
        PruneResult::Continue
    }
}

#[inline]
fn static_eval_supports_null_move(static_eval: Option<i32>, beta: i32) -> bool {
    static_eval.is_some_and(|eval| eval >= beta)
}
