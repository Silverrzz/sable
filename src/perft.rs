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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::board_from_fen;

    #[test]
    fn start_position_move_generation_matches_reference_counts() {
        let board = Board::default();
        for (depth, nodes) in [(1, 20), (2, 400), (3, 8_902), (4, 197_281)] {
            assert_eq!(perft(&board, depth), nodes, "depth {depth}");
        }
    }

    #[test]
    fn kiwipete_move_generation_matches_reference_counts() {
        let board = board_from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            false,
        )
        .expect("reference FEN is valid");
        for (depth, nodes) in [(1, 48), (2, 2_039), (3, 97_862)] {
            assert_eq!(perft(&board, depth), nodes, "depth {depth}");
        }
    }
}
