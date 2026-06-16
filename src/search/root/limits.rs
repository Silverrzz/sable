use super::super::{constants::DEFAULT_MAX_DEPTH, types::SearchRequest};

pub(crate) fn max_depth_from_limits(request: &SearchRequest) -> u32 {
    let default_depth = if request.limits.move_time_ms.is_some()
        || request.time_control.is_some()
        || request.limits.infinite
        || request.limits.nodes.is_some()
        || request.limits.soft_nodes.is_some()
        || request.limits.hard_nodes.is_some()
        || request.limits.mate.is_some()
    {
        DEFAULT_MAX_DEPTH
    } else {
        1
    };
    let mut depth_cap = request.limits.depth.unwrap_or(default_depth).max(1);
    if let Some(mate_limit) = request.limits.mate {
        depth_cap = depth_cap.min(mate_limit.saturating_mul(2).max(1));
    }
    depth_cap
}
