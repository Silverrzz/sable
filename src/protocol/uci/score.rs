use crate::evaluation::LOSS_SCORE;

const MATE_SCORE_PLY_WINDOW: i32 = 128;

pub(crate) fn mate_score_to_uci(score: i32) -> Option<i32> {
    if score >= -LOSS_SCORE - MATE_SCORE_PLY_WINDOW {
        let plies = -LOSS_SCORE - score;
        Some((plies + 1) / 2)
    } else if score <= LOSS_SCORE + MATE_SCORE_PLY_WINDOW {
        let plies = score - LOSS_SCORE;
        Some(-((plies + 1) / 2))
    } else {
        None
    }
}
