use crate::{
    Board, Color, Piece,
    evaluation::{Evaluator, PieceContribution, material_score_for_white},
    pieces::ALL_PIECES,
    search::{StaticEval, StaticEvalSource},
};

#[derive(Clone, Copy, Debug)]
pub struct VerboseEvalSquare {
    pub piece: Piece,
    pub color: Color,
}

#[derive(Clone, Debug)]
pub struct VerboseEval {
    pub squares: [Option<VerboseEvalSquare>; 64],
    pub white_king_square: u8,
    pub black_king_square: u8,
    pub material_score_white_cp: i32,
    pub nnue_score_white_cp: Option<i32>,
    pub final_score_stm_cp: i32,
    pub side_to_move: Color,
    pub source: StaticEvalSource,
    pub piece_contributions: Vec<PieceContribution>,
}

pub(super) fn build_verbose_eval(
    board: &Board,
    evaluator: &Evaluator,
    static_eval: StaticEval,
) -> VerboseEval {
    let squares = piece_map(board);
    let white_king_square = king_square(board, Color::White, 4);
    let black_king_square = king_square(board, Color::Black, 60);
    let material_score_white_cp = material_score_for_white(board);
    let nnue_score_white_cp = nnue_score_for_white(board, static_eval);
    let piece_contributions = nnue_piece_contributions(board, evaluator, static_eval.source);

    VerboseEval {
        squares,
        white_king_square,
        black_king_square,
        material_score_white_cp,
        nnue_score_white_cp,
        final_score_stm_cp: static_eval.score_cp,
        side_to_move: board.side_to_move(),
        source: static_eval.source,
        piece_contributions,
    }
}

fn piece_map(board: &Board) -> [Option<VerboseEvalSquare>; 64] {
    let mut squares = [None; 64];
    for piece in ALL_PIECES {
        for color in [Color::White, Color::Black] {
            for sq in board.pieces(piece) & board.colors(color) {
                squares[sq as usize] = Some(VerboseEvalSquare { piece, color });
            }
        }
    }
    squares
}

fn king_square(board: &Board, color: Color, fallback: u8) -> u8 {
    (board.pieces(Piece::King) & board.colors(color))
        .into_iter()
        .next()
        .map(|sq| sq as u8)
        .unwrap_or(fallback)
}

fn nnue_score_for_white(board: &Board, static_eval: StaticEval) -> Option<i32> {
    if static_eval.source != StaticEvalSource::Nnue {
        return None;
    }
    Some(match board.side_to_move() {
        Color::White => static_eval.score_cp,
        Color::Black => -static_eval.score_cp,
    })
}

fn nnue_piece_contributions(
    board: &Board,
    evaluator: &Evaluator,
    source: StaticEvalSource,
) -> Vec<PieceContribution> {
    evaluator
        .active_nnue_model()
        .filter(|_| source == StaticEvalSource::Nnue)
        .map(|ev| ev.piece_contributions_white(board))
        .unwrap_or_default()
}
