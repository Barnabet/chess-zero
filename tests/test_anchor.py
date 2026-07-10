import sys
import time

from chesszero.anchor import AnchorRunner, parse_anchor_output


def test_parse_anchor_output():
    text = ("[Negamax2] game 6/6 as Black: 1-0 -> loss | 30 moves\n"
            "[Negamax2] score 4.5/6 (75%)\n"
            "[Negamax3] score 1/6 (17%)\n")
    assert parse_anchor_output(text) == {"negamax2": 0.75, "negamax3": 1 / 6}
    assert parse_anchor_output("garbage\n") == {}


def test_anchor_runner_roundtrip():
    fake = [sys.executable, "-c",
            "print('[Negamax2] score 3/6 (50%)'); print('[Negamax3] score 0/6 (0%)')"]
    r = AnchorRunner("unused", "unused", cmd=fake)
    r.start()
    assert r.running
    for _ in range(100):
        res = r.poll()
        if res is not None:
            break
        time.sleep(0.05)
    assert res == {"negamax2": 0.5, "negamax3": 0.0}
    assert not r.running


def test_anchor_runner_failure_returns_empty():
    r = AnchorRunner("unused", "unused",
                     cmd=[sys.executable, "-c", "import sys; sys.exit(3)"])
    r.start()
    for _ in range(100):
        res = r.poll()
        if res is not None:
            break
        time.sleep(0.05)
    assert res == {}
