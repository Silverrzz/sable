use crate::Piece;

pub(crate) const ALL_PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

pub(crate) const NON_PAWN_MATERIAL: [Piece; 4] = [
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
];
