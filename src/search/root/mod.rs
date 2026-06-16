mod candidates;
mod depth;
mod driver;
mod lazy_smp;
mod limits;
mod multi_pv;
mod outcome;
mod time_manager;

pub(crate) use self::candidates::select_candidate_moves;
pub(crate) use self::driver::run_search;
pub(crate) use self::limits::max_depth_from_limits;
pub(in crate::search) use self::depth::{
    search_child_with_pvs, search_root_iteration, search_root_ordered_move,
};
pub(in crate::search) use self::driver::{
    RootSearchControls, RootSearchInput, RootSearchJob, RootSearchRuntime, run_search_single,
};
pub(in crate::search) use self::outcome::{
    PvMove, SearchOutcome, is_better_score, parent_outcome, terminal_outcome,
};
