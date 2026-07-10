#!/usr/bin/env python
"""Spar the current best checkpoint against Stockfish and/or simple baselines.

Safe to run next to training (takes the 0.15 GPU memory slice). Loads
runs/<run>/best once at startup — restart to pick up a newer promotion.

Opponents (--vs, any mix):
  <number>   Stockfish at that UCI_Elo (1350-2850)
  random     uniform random legal moves
  greedy     grabs the biggest immediate capture/promotion, else random
  negamax2   2-ply alpha-beta on material (negamax3 for 3-ply, slower)

Examples:
  python scripts/versus_stockfish.py --vs random greedy --games 4
  python scripts/versus_stockfish.py --vs greedy 1350 --movetime 0.5
  python scripts/versus_stockfish.py --vs 1350 --watch   # Enter steps moves

In --watch mode press Enter to advance one ply, or q+Enter to stop watching
(the game finishes at full speed and the match continues).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

import argparse
import math
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine


class _Tee:
    """Mirror stdout to a log file so anchor runs keep their per-game detail."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text):
        for s in self._streams:
            s.write(text)

    def flush(self):
        for s in self._streams:
            s.flush()

SF_ELO_MIN, SF_ELO_MAX = 1350, 2850   # Stockfish 14.1 UCI_Elo range

PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


# -- opponents ----------------------------------------------------------------
class RandomPlayer:
    name = "Random"

    def __init__(self, rng):
        self.rng = rng

    def play(self, board, movetime):
        return self.rng.choice(list(board.legal_moves))


class GreedyPlayer:
    """Takes the most valuable immediate capture/promotion, else random."""
    name = "Greedy"

    def __init__(self, rng):
        self.rng = rng

    def play(self, board, movetime):
        def gain(m):
            g = 0
            if board.is_capture(m):
                victim = board.piece_at(m.to_square)
                g += PIECE_VALUES[victim.piece_type] if victim \
                    else PIECE_VALUES[chess.PAWN]      # en passant
            if m.promotion:
                g += PIECE_VALUES[m.promotion] - PIECE_VALUES[chess.PAWN]
            return g
        moves = list(board.legal_moves)
        best = max(gain(m) for m in moves)
        return self.rng.choice([m for m in moves if gain(m) == best])


class NegamaxPlayer:
    """Alpha-beta on pure material, captures searched first."""

    def __init__(self, rng, depth):
        self.rng = rng
        self.depth = depth
        self.name = f"Negamax{depth}"

    def _eval(self, board):     # side-to-move perspective, centipawns
        score = 0
        for piece_type, value in PIECE_VALUES.items():
            score += value * (len(board.pieces(piece_type, board.turn))
                              - len(board.pieces(piece_type, not board.turn)))
        return score + self.rng.uniform(0, 10)   # jitter for variety

    def _negamax(self, board, depth, alpha, beta):
        if board.is_checkmate():
            return -100_000 - depth               # prefer faster mates
        if board.is_game_over(claim_draw=True):
            return 0
        if depth == 0:
            return self._eval(board)
        moves = sorted(board.legal_moves,
                       key=board.is_capture, reverse=True)
        for m in moves:
            board.push(m)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return alpha

    def play(self, board, movetime):
        best_score, best_moves = -float("inf"), []
        for m in board.legal_moves:
            board.push(m)
            score = -self._negamax(board, self.depth - 1,
                                   -float("inf"), float("inf"))
            board.pop()
            if score > best_score:
                best_score, best_moves = score, [m]
            elif score == best_score:
                best_moves.append(m)
        return self.rng.choice(best_moves)


class ChessZeroPlayer:
    """Another ChessZero checkpoint as the opponent (net vs net).

    Keeps its own Engine in sync by replaying board.move_stack deltas, so it
    works with random openings and multi-game matches (a shrinking stack
    means a new game started)."""

    def __init__(self, best_dir, cfg, fixed_sims: int):
        from pathlib import Path
        from chesszero.engine import Engine
        self.eng = Engine(best_dir, cfg, fixed_sims=fixed_sims)
        p = Path(best_dir)
        self.name = "Zero-" + (p.parent.name if p.name == "best" else p.name)
        self._pushed = 0

    def play(self, board, movetime):
        if len(board.move_stack) < self._pushed:      # new game
            self.eng.reset()
            self._pushed = 0
        for mv in board.move_stack[self._pushed:]:
            self.eng.push_uci(mv.uci())
        self._pushed = len(board.move_stack)
        return self.eng.best_move(movetime)


class StockfishPlayer:
    def __init__(self, sf, elo):
        self.sf = sf
        self.elo = elo
        self.name = f"Stockfish-{elo}"
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

    def play(self, board, movetime):
        return self.sf.play(board, chess.engine.Limit(time=movetime)).move


# -- display ------------------------------------------------------------------
def render_board(board: chess.Board, last_move: "chess.Move | None" = None) -> str:
    marked = {last_move.from_square, last_move.to_square} if last_move else set()
    lines = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            sym = piece.unicode_symbol() if piece else "·"
            cells.append(f"[{sym}]" if sq in marked else f" {sym} ")
        lines.append(f" {rank + 1} " + "".join(cells))
    lines.append("    " + "".join(f" {f} " for f in "abcdefgh"))
    return "\n".join(lines)


def print_position(board: chess.Board, ply: int, mover: str, san: str,
                   last_move: "chess.Move | None" = None):
    print(f"\nply {ply}: {mover} played {san}")
    print(render_board(board, last_move))


# -- match loop ---------------------------------------------------------------
def play_game(eng, opponent, movetime: float, we_are_white: bool,
              opening_plies: int, max_plies: int, watch: bool,
              rng: random.Random) -> tuple[float, str, str, int, bool]:
    """Returns (score for our net, result, termination, plies, watch still on)."""
    board = chess.Board()
    eng.reset()
    for _ in range(opening_plies):           # shared random opening for variety
        if board.is_game_over():
            break
        mv = rng.choice(list(board.legal_moves))
        eng.push_uci(mv.uci())
        board.push(mv)
    if watch and opening_plies:
        print(f"\n[opening: {opening_plies} random plies]")
        print(render_board(board, board.peek() if board.move_stack else None))

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        our_turn = board.turn == (chess.WHITE if we_are_white else chess.BLACK)
        if our_turn:
            move = eng.best_move(movetime)
        else:
            move = opponent.play(board, movetime)
        san = board.san(move)
        mover = ("ChessZero" if our_turn else opponent.name) \
            + (" (W)" if board.turn == chess.WHITE else " (B)")
        eng.push_uci(move.uci())
        board.push(move)
        if watch:
            print_position(board, board.ply(), mover, san, move)
            try:
                if input("Enter = next move, q = stop watching > ").strip() \
                        .lower() == "q":
                    watch = False
            except EOFError:
                watch = False

    if board.ply() >= max_plies and not board.is_game_over(claim_draw=True):
        result, score = "1/2-1/2", 0.5
        termination = f"adjudicated at {max_plies} plies"
    else:
        outcome = board.outcome(claim_draw=True)
        result = outcome.result()
        termination = outcome.termination.name.lower().replace("_", " ")
        if result == "1/2-1/2":
            score = 0.5
        else:
            white_won = result == "1-0"
            score = 1.0 if white_won == we_are_white else 0.0
    return score, result, termination, board.ply(), watch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vs", nargs="+", default=["1350"], metavar="OPPONENT",
                    help="mix of: Stockfish Elo numbers, random, greedy, "
                         "negamax2, negamax3, zero:<best_dir> (another "
                         "ChessZero checkpoint) (default: 1350)")
    ap.add_argument("--games", type=int, default=2,
                    help="games per opponent, colors alternate (default 2)")
    ap.add_argument("--movetime", type=float, default=0.3,
                    help="seconds per move for both engines (default 0.3)")
    ap.add_argument("--watch", action="store_true",
                    help="print the board and wait for Enter between moves")
    ap.add_argument("--best-dir", default="runs/v1/best")
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--sf", default="/usr/games/stockfish")
    ap.add_argument("--opening-plies", type=int, default=4,
                    help="random opening plies for game variety (default 4)")
    ap.add_argument("--clean-first", action="store_true",
                    help="first game of the session starts from the standard "
                         "position (no random opening plies)")
    ap.add_argument("--max-plies", type=int, default=400,
                    help="adjudicate draw beyond this many plies (default 400)")
    ap.add_argument("--sims", type=int, default=0,
                    help="pin the search simulation count "
                         "(0 = auto by movetime)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from chesszero.config import Config
    from chesszero.engine import Engine

    run_dir = Path(args.best_dir).resolve().parent
    if run_dir.is_dir():
        log_dir = run_dir / "anchors"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / time.strftime("versus-%Y%m%d-%H%M%S.log")
        sys.stdout = _Tee(sys.stdout, open(log_path, "w"))
        print(f"(full output logged to {log_path})")

    cfg = Config.from_yaml(args.config)
    print(f"loading best checkpoint from {args.best_dir} "
          f"(net {cfg.net.blocks}x{cfg.net.channels}) ...", flush=True)
    eng = Engine(args.best_dir, cfg, fixed_sims=args.sims)
    rng = random.Random(args.seed)

    baselines = {"random": lambda: RandomPlayer(rng),
                 "greedy": lambda: GreedyPlayer(rng),
                 "negamax2": lambda: NegamaxPlayer(rng, 2),
                 "negamax3": lambda: NegamaxPlayer(rng, 3)}
    sf = None
    first_done = False
    try:
        for token in args.vs:
            if token.lower() in baselines:
                opponent = baselines[token.lower()]()
                sf_elo = None
            elif token.lower().startswith("zero:"):
                opponent = ChessZeroPlayer(token[5:], cfg, args.sims)
                sf_elo = None
            else:
                try:
                    elo = int(token)
                except ValueError:
                    print(f"unknown opponent {token!r} — use an Elo number or "
                          f"one of: {', '.join(baselines)}")
                    continue
                sf_elo = max(SF_ELO_MIN, min(SF_ELO_MAX, elo))
                if sf_elo != elo:
                    print(f"note: {elo} outside Stockfish range, using {sf_elo}")
                if sf is None:
                    sf = chess.engine.SimpleEngine.popen_uci(args.sf)
                opponent = StockfishPlayer(sf, sf_elo)

            total, watch = 0.0, args.watch
            plies_sum, terminations = 0, {}
            for g in range(args.games):
                we_are_white = g % 2 == 0
                opening = 0 if args.clean_first and not first_done \
                    else args.opening_plies
                first_done = True
                score, result, termination, plies, watch = play_game(
                    eng, opponent, args.movetime, we_are_white,
                    opening, args.max_plies, watch, rng)
                total += score
                plies_sum += plies
                terminations[termination] = terminations.get(termination, 0) + 1
                tag = {1.0: "WIN", 0.5: "draw", 0.0: "loss"}[score]
                print(f"[{opponent.name}] game {g + 1}/{args.games} "
                      f"as {'White' if we_are_white else 'Black'}: "
                      f"{result} -> {tag} | {(plies + 1) // 2} moves, "
                      f"{termination} | total {total:g}/{g + 1}", flush=True)
            args.watch = watch
            pct = total / args.games
            terms = ", ".join(f"{t} x{n}" for t, n in
                              sorted(terminations.items(), key=lambda kv: -kv[1]))
            line = (f"[{opponent.name}] score {total:g}/{args.games} ({pct:.0%})"
                    f" | avg {plies_sum / args.games / 2:.0f} moves | {terms}")
            if sf_elo is not None and 0.0 < pct < 1.0:
                line += (f" | implied Elo "
                         f"~{sf_elo + 400 * math.log10(pct / (1 - pct)):.0f}")
            print(line, flush=True)
    finally:
        if sf is not None:
            sf.quit()


if __name__ == "__main__":
    main()
