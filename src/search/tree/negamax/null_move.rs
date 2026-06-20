use crate::Board;

use super::{
    PruneResult, negamax,
    super::{
        context::SearchContext,
        correction_history::CorrectionContext,
        pruning::{null_move_reduction, should_try_null_move, should_verify_null_move},
        root::terminal_outcome,
        search_profile::SearchProfile,
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
    pub(super) correction_context: CorrectionContext,
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
    {
        return PruneResult::Continue;
    }
    let Some(static_eval) = static_eval_for_null_move(params.static_eval, params.beta) else {
        return PruneResult::Continue;
    };

    let Some(null_board) = params.board.null_move() else {
        return PruneResult::Continue;
    };
    let search_profile = SearchProfile::for_board(params.board);
    let reduction = null_move_reduction(params.depth, static_eval, params.beta, search_profile);
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
        params.correction_context.without_move_context(),
        context,
        params.ply + 1,
        false,
        None,
    );
    context.pop_null_eval_state(params.board);
    let Some(null_result) = null_result else {
        return PruneResult::Interrupted;
    };
    if -null_result.score < params.beta {
        return PruneResult::Continue;
    }

    if should_verify_null_move(params.depth, search_profile) {
        let verification_depth = params.depth.saturating_sub(reduction);
        let verification = negamax(
            params.board,
            params.repetition,
            verification_depth,
            params.root_depth,
            params.beta.saturating_sub(1),
            params.beta,
            &[],
            None,
            params.correction_context,
            context,
            params.ply,
            false,
            None,
        );
        let Some(verification) = verification else {
            return PruneResult::Interrupted;
        };
        if verification.score < params.beta {
            return PruneResult::Continue;
        }
    }

    PruneResult::Done(terminal_outcome(params.beta, false))
}

#[inline]
fn static_eval_for_null_move(static_eval: Option<i32>, beta: i32) -> Option<i32> {
    static_eval.filter(|eval| *eval >= beta)
}
