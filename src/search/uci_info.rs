
use crate::{
    Board, GameStatus, Move,
    protocol::uci::{format_uci_move_for_board, mate_score_to_uci},
};

use super::{root::PvMove, state::context::SearchContext, types::*};

pub(super) fn build_search_info(
    board: &Board,
    budget: &SearchBudget,
    depth: u32,
    context: &mut SearchContext<'_>,
    chess960: bool,
    score_cp: i32,
    pv: &[PvMove],
) -> SearchInfo {
    let elapsed = context.clock_elapsed();
    let elapsed_ns = elapsed.as_nanos();
    let elapsed_ms = elapsed.as_millis().min(u128::from(u64::MAX)) as u64;
    let nodes = context.total_nodes();
    let nps = nodes_per_second(nodes, elapsed_ns);
    let (pv, pv_uci) = playable_pv(board, pv, chess960);
    SearchInfo {
        budget: budget.clone(),
        depth: Some(depth),
        seldepth: Some(context.seldepth().max(depth)),
        nodes: Some(nodes),
        time_ms: Some(elapsed_ms),
        nps,
        score_cp: Some(score_cp),
        score_mate: mate_score_to_uci(score_cp),
        multi_pv: None,
        hashfull: Some(context.transposition_table().hashfull()),
        pv,
        pv_uci,
    }
}

fn playable_pv(board: &Board, pv: &[PvMove], chess960: bool) -> (Vec<Move>, Vec<String>) {
    let mut board = board.clone();
    let mut moves = Vec::with_capacity(pv.len());
    let mut pv_uci = Vec::with_capacity(pv.len());
    for pv_move in pv.iter().rev() {
        if crate::chess::status(&board) != GameStatus::Ongoing {
            break;
        }
        if !crate::chess::is_legal(&board, pv_move.mv) {
            break;
        }
        pv_uci.push(format_uci_move_for_board(&board, pv_move.mv, chess960));
        moves.push(pv_move.mv);
        crate::chess::play_unchecked(&mut board, pv_move.mv);
    }
    (moves, pv_uci)
}

pub(super) fn nodes_per_second(nodes: u64, elapsed_ns: u128) -> Option<u64> {
    let nps = u128::from(nodes)
        .saturating_mul(1_000_000_000)
        .checked_div(elapsed_ns)?;
    Some(nps.min(u128::from(u64::MAX)) as u64)
}
