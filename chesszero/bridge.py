"""pgx chess <-> python-chess conversion. The ONLY module that knows both encodings.

Verified pgx chess v2 facts (see plan Global Constraints):
- label = from_square * 73 + plane; squares FILE-major (a1=0, a2=1, ..., h8=63)
- black-to-move: rank-flip squares with (sq // 8) * 8 + (7 - sq % 8)
- planes 0-8 underpromotions (plane//3: 0=R 1=B 2=N; plane%3: 0=straight
  1=right-capture 2=left-capture); planes 9-72 via pgx TO_PLANE/FROM_PLANE
"""
from __future__ import annotations

import chess
import numpy as np
import pgx
import pgx._src.games.chess as _cg
import pgx.experimental.chess as _pxc

ENV = pgx.make("chess")

TO_PLANE = np.asarray(_cg.TO_PLANE)      # (64, 64) -> plane
FROM_PLANE = np.asarray(_cg.FROM_PLANE)  # (64, 73) -> to-square
_UNDERPROMO = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}
_UNDERPROMO_INV = {v: k for k, v in _UNDERPROMO.items()}
_DIR_IDX = {0: 0, 1: 1, -1: 2}           # file delta -> direction index


def pc_sq_to_pgx(sq: int) -> int:
    """python-chess (rank-major) <-> pgx (file-major); involution."""
    return (sq % 8) * 8 + (sq // 8)


def flip_sq(sq: int) -> int:
    """Mirror ranks (a1<->a8) in pgx numbering."""
    return (sq // 8) * 8 + (7 - (sq % 8))


def move_to_action(move: chess.Move, turn: chess.Color) -> int:
    frm, to = pc_sq_to_pgx(move.from_square), pc_sq_to_pgx(move.to_square)
    if turn == chess.BLACK:
        frm, to = flip_sq(frm), flip_sq(to)
    if move.promotion in _UNDERPROMO:
        plane = _UNDERPROMO[move.promotion] * 3 + _DIR_IDX[(to // 8) - (frm // 8)]
    else:
        plane = int(TO_PLANE[frm, to])
    return int(frm * 73 + plane)


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    frm, plane = action // 73, action % 73
    to = int(FROM_PLANE[frm, plane])
    if board.turn == chess.BLACK:
        frm, to = flip_sq(frm), flip_sq(to)
    frm_pc, to_pc = pc_sq_to_pgx(frm), pc_sq_to_pgx(to)
    promotion = None
    if plane < 9:
        promotion = _UNDERPROMO_INV[plane // 3]
    elif (board.piece_type_at(frm_pc) == chess.PAWN
          and chess.square_rank(to_pc) in (0, 7)):
        promotion = chess.QUEEN
    return chess.Move(frm_pc, to_pc, promotion=promotion)


def state_from_fen(fen: str):
    """pgx State from FEN. History planes are empty — for play, prefer
    stepping the env move-by-move (engine does this)."""
    return _pxc.from_fen(fen)


def fen_from_state(state) -> str:
    return _pxc.to_fen(state)
