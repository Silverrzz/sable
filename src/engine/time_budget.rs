use crate::{
    Board, Color, EngineOptions,
    search::{SearchBudget, SearchRequest},
};

const TIME_SAFETY_MARGIN_MS: u64 = 25;
const CLOCK_RESERVE_DIVISOR: u64 = 40;
const CLOCK_RESERVE_MAX_MS: u64 = 500;
const DEFAULT_TIME_ALLOCATION_DIVISOR: u64 = 22;
const INCREMENT_TIME_NUMERATOR: u64 = 7;
const INCREMENT_TIME_DENOMINATOR: u64 = 10;
const HARD_TIME_SOFT_MULTIPLIER: u64 = 4;
const HARD_TIME_CLOCK_NUMERATOR: u64 = 4;
const HARD_TIME_CLOCK_DENOMINATOR: u64 = 5;

pub(super) fn compute_search_budget(
    board: &Board,
    options: &EngineOptions,
    request: &SearchRequest,
) -> SearchBudget {
    if request.limits.infinite {
        return SearchBudget::default();
    }

    if let Some(move_time_ms) = request.limits.move_time_ms {
        let allocated =
            move_time_ms.saturating_sub(move_time_reserve_ms(move_time_ms, options.move_overhead_ms));
        return SearchBudget {
            soft_time_ms: Some(allocated),
            hard_time_ms: Some(allocated),
        };
    }

    let Some(time_control) = &request.time_control else {
        return SearchBudget::default();
    };
    let Some((base_ms, increment_ms)) = side_clock(board, time_control) else {
        return SearchBudget::default();
    };

    let available_ms = base_ms.saturating_sub(clock_reserve_ms(base_ms, options.move_overhead_ms));
    let divisor = time_control
        .moves_to_go
        .map(u64::from)
        .unwrap_or(DEFAULT_TIME_ALLOCATION_DIVISOR)
        .max(1);
    let increment_share =
        increment_ms.saturating_mul(INCREMENT_TIME_NUMERATOR) / INCREMENT_TIME_DENOMINATOR;
    let target_soft = (available_ms / divisor)
        .saturating_add(increment_share)
        .min(available_ms);
    let hard_by_soft = target_soft.saturating_mul(HARD_TIME_SOFT_MULTIPLIER);
    let hard_by_clock =
        available_ms.saturating_mul(HARD_TIME_CLOCK_NUMERATOR) / HARD_TIME_CLOCK_DENOMINATOR;
    let hard = hard_by_soft.min(hard_by_clock).min(available_ms);
    let soft = target_soft.min(hard);

    SearchBudget {
        soft_time_ms: Some(soft),
        hard_time_ms: Some(hard),
    }
}

fn side_clock(
    board: &Board,
    time_control: &crate::search::TimeControl,
) -> Option<(u64, u64)> {
    match board.side_to_move() {
        Color::White => time_control
            .white_time_ms
            .map(|base_ms| (base_ms, time_control.white_increment_ms.unwrap_or(0))),
        Color::Black => time_control
            .black_time_ms
            .map(|base_ms| (base_ms, time_control.black_increment_ms.unwrap_or(0))),
    }
}

fn move_time_reserve_ms(move_time_ms: u64, overhead_ms: u64) -> u64 {
    overhead_ms.saturating_add(TIME_SAFETY_MARGIN_MS.min(move_time_ms / 2))
}

fn clock_reserve_ms(base_ms: u64, overhead_ms: u64) -> u64 {
    overhead_ms.saturating_add(CLOCK_RESERVE_MAX_MS.min(base_ms / CLOCK_RESERVE_DIVISOR))
}
