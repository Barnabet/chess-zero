#!/usr/bin/env python
"""Spar the current best checkpoint against Stockfish at chosen Elo levels.

Safe to run next to training (takes the 0.15 GPU memory slice). Loads
runs/<run>/best once at startup — restart to pick up a newer promotion.

Examples:
  python scripts/versus_stockfish.py --elo 1350 --games 4
  python scripts/versus_stockfish.py --elo 1350 1500 --movetime 0.5
  python scripts/versus_stockfish.py --elo 1350 --watch     # Enter steps moves

In --watch mode press Enter to advance one ply, or q+Enter to stop watching
(the game finishes at full speed and the match continues).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

import argparse
import math
import random

import chess
import chess.engine

SF_ELO_MIN, SF_ELO_MAX = 1350, 2850   # Stockfish 14.1 UCI_Elo range


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


def play_game(eng, sf, elo: int, movetime: float, we_are_white: bool,
              opening_plies: int, max_plies: int, watch: bool,
              rng: random.Random) -> tuple[float, str, bool]:
    """Returns (score for our net, result string, watch still on)."""
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
            move = sf.play(board, chess.engine.Limit(time=movetime)).move
        san = board.san(move)
        mover = ("ChessZero" if our_turn else f"Stockfish-{elo}") \
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
        result = "1/2-1/2 (adjudicated at max plies)"
        score = 0.5
    else:
        result = board.result(claim_draw=True)
        if result == "1/2-1/2":
            score = 0.5
        else:
            white_won = result == "1-0"
            score = 1.0 if white_won == we_are_white else 0.0
    return score, result, watch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elo", type=int, nargs="+", default=[1350])
    ap.add_argument("--games", type=int, default=2,
                    help="games per Elo level, colors alternate (default 2)")
    ap.add_argument("--movetime", type=float, default=0.3,
                    help="seconds per move for both engines (default 0.3)")
    ap.add_argument("--watch", action="store_true",
                    help="print the board and wait for Enter between moves")
    ap.add_argument("--best-dir", default="runs/v1/best")
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--sf", default="/usr/games/stockfish")
    ap.add_argument("--opening-plies", type=int, default=4,
                    help="random opening plies for game variety (default 4)")
    ap.add_argument("--max-plies", type=int, default=400,
                    help="adjudicate draw beyond this many plies (default 400)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from chesszero.config import Config
    from chesszero.engine import Engine

    cfg = Config.from_yaml(args.config)
    print(f"loading best checkpoint from {args.best_dir} "
          f"(net {cfg.net.blocks}x{cfg.net.channels}) ...", flush=True)
    eng = Engine(args.best_dir, cfg)
    rng = random.Random(args.seed)

    sf = chess.engine.SimpleEngine.popen_uci(args.sf)
    try:
        for elo in args.elo:
            level = max(SF_ELO_MIN, min(SF_ELO_MAX, elo))
            if level != elo:
                print(f"note: {elo} outside Stockfish range, using {level}")
            sf.configure({"UCI_LimitStrength": True, "UCI_Elo": level})
            total, results = 0.0, []
            for g in range(args.games):
                we_are_white = g % 2 == 0
                score, result, args.watch = play_game(
                    eng, sf, level, args.movetime, we_are_white,
                    args.opening_plies, args.max_plies, args.watch, rng)
                total += score
                results.append(score)
                tag = {1.0: "WIN", 0.5: "draw", 0.0: "loss"}[score]
                print(f"[elo {level}] game {g + 1}/{args.games} "
                      f"as {'White' if we_are_white else 'Black'}: "
                      f"{result} -> {tag}", flush=True)
            pct = total / args.games
            line = f"[elo {level}] score {total:g}/{args.games} ({pct:.0%})"
            if 0.0 < pct < 1.0:
                line += f" | implied Elo ~{level + 400 * math.log10(pct / (1 - pct)):.0f}"
            print(line, flush=True)
    finally:
        sf.quit()


if __name__ == "__main__":
    main()
