use crate::{Board, GameStatus, Move, protocol::uci::mate_score_to_uci};

use super::{state::context::SearchContext, types::*};

pub(super) fn build_search_info(
    board: &Board,
    budget: &SearchBudget,
    depth: u32,
    context: &mut SearchContext<'_>,
    score_cp: i32,
    pv: &[Move],
) -> SearchInfo {
    let elapsed = context.clock_elapsed();
    let elapsed_ns = elapsed.as_nanos();
    let elapsed_ms = elapsed.as_millis().min(u128::from(u64::MAX)) as u64;
    let nodes = context.total_nodes();
    let nps = nodes_per_second(nodes, elapsed_ns);
    let pv = playable_pv(board, pv);
    SearchInfo {
        budget: budget.clone(),
        depth,
        seldepth: context.seldepth(),
        nodes,
        time_ms: elapsed_ms,
        nps,
        score_cp,
        score_mate: mate_score_to_uci(score_cp),
        multi_pv: None,
        hashfull: context.transposition_table().hashfull(),
        pv,
    }
}

fn playable_pv(board: &Board, pv: &[Move]) -> Vec<Move> {
    let mut board = board.clone();
    let mut moves = Vec::with_capacity(pv.len());
    for &mv in pv.iter().rev() {
        if crate::chess::status(&board) != GameStatus::Ongoing {
            break;
        }
        if !crate::chess::is_legal(&board, mv) {
            break;
        }
        moves.push(mv);
        crate::chess::play_unchecked(&mut board, mv);
    }
    moves
}

pub(super) fn nodes_per_second(nodes: u64, elapsed_ns: u128) -> u64 {
    u128::from(nodes)
        .saturating_mul(1_000_000_000)
        .checked_div(elapsed_ns)
        .unwrap_or(0)
        .min(u128::from(u64::MAX)) as u64
}
