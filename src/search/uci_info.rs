
use crate::protocol::uci::mate_score_to_uci;

use super::{root::PvMove, state::context::SearchContext, types::*};

pub(super) fn build_search_info(
    budget: &SearchBudget,
    depth: u32,
    context: &mut SearchContext<'_>,
    score_cp: i32,
    pv: &[PvMove],
) -> SearchInfo {
    let elapsed = context.clock_elapsed();
    let elapsed_ns = elapsed.as_nanos();
    let elapsed_ms = elapsed.as_millis().min(u128::from(u64::MAX)) as u64;
    let nodes = context.total_nodes();
    let nps = nodes_per_second(nodes, elapsed_ns);
    let pv_uci = pv.iter().map(|mv| mv.to_uci()).collect::<Vec<_>>();
    let pv = pv.iter().map(|mv| mv.mv).collect::<Vec<_>>();
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

pub(super) fn nodes_per_second(nodes: u64, elapsed_ns: u128) -> Option<u64> {
    let nps = u128::from(nodes)
        .saturating_mul(1_000_000_000)
        .checked_div(elapsed_ns)?;
    Some(nps.min(u128::from(u64::MAX)) as u64)
}
