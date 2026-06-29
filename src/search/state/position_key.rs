
use crate::{
    Board, Move, Square,
    chess::{File, Rank, get_pawn_attacks},
};

pub(crate) type PositionKey = u64;

pub(crate) fn position_key(board: &Board) -> PositionKey {
    if effective_ep_file(board).is_some() {
        crate::chess::hash(board)
    } else {
        crate::chess::hash_without_ep(board)
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
        .take(usize::from(crate::chess::halfmove_clock(board)))
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
    let ep_file = crate::chess::en_passant(board)?;
    let color = crate::chess::side_to_move(board);
    let ep_square = Square::new(ep_file, Rank::Sixth.relative_to(color));
    let attackers = get_pawn_attacks(ep_square, !color);
    for attacker in attackers {
        let mv = Move {
            from: attacker,
            to: ep_square,
            promotion: None,
        };
        if crate::chess::is_legal(board, mv) {
            return Some(ep_file);
        }
    }
    None
}
