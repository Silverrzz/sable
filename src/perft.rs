use crate::{Board, chess::{generate_moves}};

pub(crate) fn perft(board: &Board, depth: u32) -> u64 {
    perft_impl(board.clone(), depth)
}

fn perft_impl(board: Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if depth == 1 {
        let mut total = 0_u64;
        generate_moves(&board, |piece_moves| {
            for _ in piece_moves {
                total = total.saturating_add(1);
            }
            false
        });
        return total;
    }

    let mut total = 0_u64;
    generate_moves(&board, |piece_moves| {
        for mv in piece_moves {
            let mut next = board.clone();
            crate::chess::play(&mut next, mv);
            total = total.saturating_add(perft_impl(next, depth - 1));
        }
        false
    });
    total
}
