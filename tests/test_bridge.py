import random

import chess
import numpy as np

from chesszero import bridge


PROMO_FENS = [
    "8/P6k/8/8/8/8/p6K/8 w - - 0 1",
    "8/P6k/8/8/8/8/p6K/8 b - - 0 1",
    "1n1n3k/2P5/8/8/8/8/2p5/1N1N3K w - - 0 1",
    "1n1n3k/2P5/8/8/8/8/2p5/1N1N3K b - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "8/2p5/8/KP5r/5pPk/8/4P3/6R1 b - g3 0 1",
]


def _assert_position_matches(board: chess.Board):
    state = bridge.state_from_fen(board.fen())
    pgx_legal = set(np.where(np.asarray(state.legal_action_mask))[0].tolist())
    legal = list(board.legal_moves)
    encoded = {bridge.move_to_action(m, board.turn) for m in legal}
    assert encoded == pgx_legal, board.fen()
    decoded = {bridge.action_to_move(a, board) for a in pgx_legal}
    assert decoded == set(legal), board.fen()


def test_fen_roundtrip():
    state = bridge.state_from_fen(chess.STARTING_FEN)
    assert bridge.fen_from_state(state) == chess.STARTING_FEN


def test_fixed_positions_exact():
    for fen in PROMO_FENS:
        _assert_position_matches(chess.Board(fen))


def test_random_games_exact():
    rng = random.Random(42)
    for _ in range(2):
        board = chess.Board()
        while not board.is_game_over() and board.ply() < 60:
            _assert_position_matches(board)
            board.push(rng.choice(list(board.legal_moves)))
