# tests/test_engine.py
import chess
import jax
import jax.numpy as jnp
import pytest

from chesszero.config import Config, NetConfig
from chesszero.engine import Engine
from chesszero.net import ChessNet


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    import orbax.checkpoint as ocp
    cfg = Config(net=NetConfig(channels=16, blocks=1, precision="fp32"))
    net = ChessNet(cfg.net)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 119)))
    best = tmp_path_factory.mktemp("ck") / "best"
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(best.absolute(), params, force=True)
    return Engine(best, cfg)


def test_plays_legal_from_startpos(engine):
    engine.reset()
    mv = engine.best_move(0.2)
    assert mv in chess.Board().legal_moves


def test_follows_game_and_plays_black(engine):
    engine.reset()
    for uci in ["e2e4", "e7e5", "g1f3"]:
        engine.push_uci(uci)
    mv = engine.best_move(0.2)          # black to move
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3"]:
        board.push_uci(uci)
    assert mv in board.legal_moves


def test_many_random_plies_stay_legal(engine):
    engine.reset()
    board = chess.Board()
    for _ in range(30):
        if board.is_game_over():
            break
        mv = engine.best_move(0.05)
        assert mv in board.legal_moves
        board.push(mv)
        engine.push_uci(mv.uci())
