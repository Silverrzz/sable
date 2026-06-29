use crate::{
    Board,
    chess::{generate_moves, play_unchecked_with_piece},
};

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
            total += piece_moves.len() as u64;
            false
        });
        return total;
    }

    let mut total = 0_u64;
    generate_moves(&board, |piece_moves| {
        let piece = piece_moves.piece;
        for mv in piece_moves {
            let mut next = board.clone();
            play_unchecked_with_piece(&mut next, mv, piece);
            total += perft_impl(next, depth - 1);
        }
        false
    });
    total
}
