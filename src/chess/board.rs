use super::{BitBoard, Color, File, GameStatus, Move, Piece, Rank, Square};

pub type Board = cozy_chess::Board;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct CastleRights {
    pub short: Option<File>,
    pub long: Option<File>,
}

impl CastleRights {
    #[inline]
    fn from_cozy(rights: &cozy_chess::CastleRights) -> Self {
        Self {
            short: rights.short.map(File::from_cozy),
            long: rights.long.map(File::from_cozy),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PieceMoves {
    pub piece: Piece,
    pub from: Square,
    pub to: BitBoard,
}

impl PieceMoves {
    #[inline]
    fn from_cozy(moves: cozy_chess::PieceMoves) -> Self {
        Self {
            piece: Piece::from_cozy(moves.piece),
            from: Square::from_cozy(moves.from),
            to: BitBoard::from_cozy(moves.to),
        }
    }

    pub fn len(&self) -> usize {
        const PROMOTION_MASK: BitBoard = BitBoard(
            Rank::First.bitboard().0 | Rank::Eighth.bitboard().0,
        );
        if self.piece == Piece::Pawn {
            ((self.to & !PROMOTION_MASK).len() + (self.to & PROMOTION_MASK).len() * 4) as usize
        } else {
            self.to.len() as usize
        }
    }

}

impl IntoIterator for PieceMoves {
    type Item = Move;
    type IntoIter = PieceMovesIter;

    fn into_iter(self) -> Self::IntoIter {
        PieceMovesIter {
            moves: self,
            promotion: 0,
        }
    }
}

pub struct PieceMovesIter {
    moves: PieceMoves,
    promotion: u8,
}

impl Iterator for PieceMovesIter {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        let from = self.moves.from;
        let to = self.moves.to.next_square()?;
        let is_promotion = self.moves.piece == Piece::Pawn
            && matches!(to.rank(), Rank::First | Rank::Eighth);
        let promotion = if is_promotion {
            let promotion = match self.promotion {
                0 => Piece::Knight,
                1 => Piece::Bishop,
                2 => Piece::Rook,
                3 => Piece::Queen,
                _ => unreachable!(),
            };
            if self.promotion < 3 {
                self.promotion += 1;
            } else {
                self.promotion = 0;
                self.moves.to ^= to.bitboard();
            }
            Some(promotion)
        } else {
            self.moves.to ^= to.bitboard();
            None
        };
        Some(Move {
            from,
            to,
            promotion,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for PieceMovesIter {
    fn len(&self) -> usize {
        self.moves.len() - self.promotion as usize
    }
}

#[inline]
pub(crate) fn side_to_move(board: &Board) -> Color {
    Color::from_cozy(board.side_to_move())
}

#[inline]
pub(crate) fn status(board: &Board) -> GameStatus {
    GameStatus::from_cozy(board.status())
}

#[inline]
pub(crate) fn piece_on(board: &Board, square: Square) -> Option<Piece> {
    board.piece_on(square.to_cozy()).map(Piece::from_cozy)
}

#[inline]
pub(crate) fn color_on(board: &Board, square: Square) -> Option<Color> {
    board.color_on(square.to_cozy()).map(Color::from_cozy)
}

#[inline]
pub(crate) fn pieces(board: &Board, piece: Piece) -> BitBoard {
    BitBoard::from_cozy(board.pieces(piece.to_cozy()))
}

#[inline]
pub(crate) fn colors(board: &Board, color: Color) -> BitBoard {
    BitBoard::from_cozy(board.colors(color.to_cozy()))
}

#[inline]
pub(crate) fn colored_pieces(board: &Board, color: Color, piece: Piece) -> BitBoard {
    BitBoard::from_cozy(board.colored_pieces(color.to_cozy(), piece.to_cozy()))
}

#[inline]
pub(crate) fn occupied(board: &Board) -> BitBoard {
    BitBoard::from_cozy(board.occupied())
}

#[inline]
pub(crate) fn en_passant(board: &Board) -> Option<File> {
    board.en_passant().map(File::from_cozy)
}

#[inline]
pub(crate) fn castle_rights(board: &Board, color: Color) -> CastleRights {
    CastleRights::from_cozy(board.castle_rights(color.to_cozy()))
}

#[inline]
pub(crate) fn king(board: &Board, color: Color) -> Square {
    Square::from_cozy(board.king(color.to_cozy()))
}

#[inline]
pub(crate) fn checkers(board: &Board) -> BitBoard {
    BitBoard::from_cozy(board.checkers())
}

#[inline]
pub(crate) fn halfmove_clock(board: &Board) -> u8 {
    board.halfmove_clock()
}

#[inline]
pub(crate) fn hash(board: &Board) -> u64 {
    board.hash()
}

#[inline]
pub(crate) fn hash_without_ep(board: &Board) -> u64 {
    board.hash_without_ep()
}

#[inline]
pub(crate) fn is_legal(board: &Board, mv: Move) -> bool {
    board.is_legal(mv.to_cozy())
}

#[inline]
pub(crate) fn play(board: &mut Board, mv: Move) {
    board.play(mv.to_cozy());
}

#[inline]
pub(crate) fn play_unchecked(board: &mut Board, mv: Move) {
    board.play_unchecked(mv.to_cozy());
}

#[inline]
pub(crate) fn null_move(board: &Board) -> Option<Board> {
    board.null_move()
}

#[inline]
pub(crate) fn board_from_fen(fen: &str, chess960: bool) -> Result<Board, cozy_chess::FenParseError> {
    Board::from_fen(fen, chess960)
}

#[inline]
pub(crate) fn generate_moves<F>(board: &Board, mut listener: F)
where
    F: FnMut(PieceMoves) -> bool,
{
    board.generate_moves(|moves| listener(PieceMoves::from_cozy(moves)));
}
