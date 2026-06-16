
use crate::{
    Board, Move, Square,
};
use cozy_chess::{
    File, Rank, get_pawn_attacks,
};

pub(crate) type PositionKey = u64;

pub(crate) fn position_key(board: &Board) -> PositionKey {
    if effective_ep_file(board).is_some() {
        board.hash()
    } else {
        board.hash_without_ep()
    }
}

#[inline]
pub(in crate::search) fn is_repetition(
    key: PositionKey,
    halfmove_clock: u8,
    previous_keys: &[PositionKey],
) -> bool {
    let end = previous_keys.len();
    for offset in 1..=end.min(usize::from(halfmove_clock)) {
        if previous_keys[end - offset] == key {
            return true;
        }
    }
    false
}

fn previous_game_repetitions(board: &Board, state_keys: &[PositionKey]) -> u8 {
    let key = position_key(board);
    let size = state_keys.len();
    if size < 2 {
        return 0;
    }

    let mut prior = 0_u8;
    for &existing in state_keys[..size - 1]
        .iter()
        .rev()
        .take(usize::from(board.halfmove_clock()))
    {
        if existing == key {
            prior += 1;
            if prior >= 2 {
                break;
            }
        }
    }
    prior
}

#[inline]
pub(in crate::search) fn actual_game_repetition_count(board: &Board, state_keys: &[PositionKey]) -> u8 {
    1 + previous_game_repetitions(board, state_keys)
}

#[inline]
pub(crate) fn is_claimable_repetition_draw(board: &Board, state_keys: &[PositionKey]) -> bool {
    previous_game_repetitions(board, state_keys) >= 2
}

pub(in crate::search) fn effective_ep_file(board: &Board) -> Option<File> {
    let ep_file = board.en_passant()?;
    let color = board.side_to_move();
    let ep_square = Square::new(ep_file, Rank::Sixth.relative_to(color));
    let attackers = get_pawn_attacks(ep_square, !color);
    for attacker in attackers {
        let mv = Move {
            from: attacker,
            to: ep_square,
            promotion: None,
        };
        if board.is_legal(mv) {
            return Some(ep_file);
        }
    }
    None
}
