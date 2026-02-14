"""
Hub Server -- central equity calculator for distributed poker stream.

Receives card data from player clients via HTTP, aggregates community cards
and hole cards, recalculates equity on every update, and prints results.

Usage
-----
    python hub.py                     # default port 8000
    python hub.py --port 9000         # custom port
    python hub.py --host 0.0.0.0      # listen on all interfaces (for internet)
    python hub.py --log-file hub.log  # also write logs to a file
"""

from __future__ import annotations

import argparse
import logging
import threading
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from equity_calculator import calculate_equity, format_equity_table

# ── Logging setup ─────────────────────────────────────────────────────
logger = logging.getLogger("hub")


def _setup_logging(log_file: Optional[str] = None) -> None:
    """Configure logging to console (and optionally a file)."""
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )


# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(title="Poker Equity Hub")


# ── Request / response models ────────────────────────────────────────
class HandReport(BaseModel):
    player_name: str
    hole_cards: list[str]          # e.g. ["Ah", "Kd"]
    community_cards: list[str]     # e.g. ["Jc", "Ts", "9h"]
    hand_id: Optional[int] = None  # optional -- hub auto-detects hands if omitted


class StatusResponse(BaseModel):
    hand_id: int
    community_cards: list[str]
    players: dict[str, list[str]]  # player_name -> hole_cards
    equity: dict[str, float]       # player_name -> equity %
    players_connected: int
    last_update: Optional[str]


# ── In-memory hand state ─────────────────────────────────────────────
class HandState:
    def __init__(self) -> None:
        self.hand_id: int = 0
        self.community_cards: list[str] = []
        self.players: dict[str, list[str]] = {}  # name -> [card1, card2]
        self.equity: dict[str, float] = {}        # name -> equity (0-1)
        self.last_update: Optional[str] = None
        self._lock = threading.Lock()

    def reset(self, hand_id: int) -> None:
        with self._lock:
            self.hand_id = hand_id
            self.community_cards = []
            self.players = {}
            self.equity = {}
            self.last_update = None
        logger.info("=" * 45)
        logger.info("NEW HAND #%d", hand_id)
        logger.info("=" * 45)

    def _detect_new_hand(self, report: HandReport, client_ip: str) -> bool:
        """Check if this report indicates a new hand. Must be called under self._lock.

        A new hand is detected when:
        - Client explicitly sends a hand_id greater than the current one
        - The board shrinks (e.g. river/turn -> preflop/flop = different hand)
        - Board was non-empty and now the report has an empty board (preflop)
        - The reported board cards are completely different from the current board
          (e.g. current flop is Kd Jc 9h, report says Ah 5c 3d)
        """
        # Explicit hand_id advance from client
        if report.hand_id is not None and report.hand_id > self.hand_id:
            logger.info("=" * 45)
            logger.info("NEW HAND #%d (explicit from %s)", report.hand_id, report.player_name)
            logger.info("=" * 45)
            self.hand_id = report.hand_id
            return True

        # Board shrinks: e.g. we had 4 community cards, now report has 3 or 0
        if self.community_cards and len(report.community_cards) < len(self.community_cards):
            self.hand_id += 1
            logger.info("=" * 45)
            logger.info(
                "NEW HAND #%d (board shrank: %d -> %d cards, detected via %s)",
                self.hand_id, len(self.community_cards),
                len(report.community_cards), report.player_name,
            )
            logger.info("=" * 45)
            return True

        # Board was non-empty and report has empty board (preflop of new hand)
        if self.community_cards and not report.community_cards:
            self.hand_id += 1
            logger.info("=" * 45)
            logger.info(
                "NEW HAND #%d (board cleared to preflop, detected via %s)",
                self.hand_id, report.player_name,
            )
            logger.info("=" * 45)
            return True

        # Same number of board cards but completely different cards = new hand
        # (e.g. flop Kd Jc 9h -> flop Ah 5c 3d)
        if (self.community_cards
                and report.community_cards
                and len(report.community_cards) == len(self.community_cards)
                and set(report.community_cards) != set(self.community_cards)):
            # Check if the new board shares NO cards with the old board
            overlap = set(report.community_cards) & set(self.community_cards)
            if not overlap:
                self.hand_id += 1
                logger.info("=" * 45)
                logger.info(
                    "NEW HAND #%d (board changed completely: %s -> %s, detected via %s)",
                    self.hand_id,
                    " ".join(self.community_cards),
                    " ".join(report.community_cards),
                    report.player_name,
                )
                logger.info("=" * 45)
                return True

        return False

    def update(self, report: HandReport, client_ip: str) -> None:
        with self._lock:
            # Stale hand_id check (only if client sends one)
            if report.hand_id is not None and report.hand_id < self.hand_id:
                logger.warning(
                    "Stale report from %s [%s] -- hand %d < current %d, ignoring",
                    report.player_name, client_ip, report.hand_id, self.hand_id,
                )
                return

            # Auto-detect new hand from board state
            is_new_hand = self._detect_new_hand(report, client_ip)
            if is_new_hand:
                self.community_cards = []
                self.players = {}
                self.equity = {}

            # First hand auto-start (hub just started, hand_id is 0)
            if self.hand_id == 0:
                self.hand_id = 1
                logger.info("=" * 45)
                logger.info("NEW HAND #1 (first report)")
                logger.info("=" * 45)

            # Update community cards
            if report.community_cards:
                if not self.community_cards:
                    self.community_cards = report.community_cards
                    logger.info(
                        "Board set by %s: %s",
                        report.player_name, " ".join(self.community_cards),
                    )
                elif len(report.community_cards) > len(self.community_cards):
                    old_board = " ".join(self.community_cards)
                    self.community_cards = report.community_cards
                    logger.info(
                        "Board updated by %s: %s -> %s (new street)",
                        report.player_name, old_board, " ".join(self.community_cards),
                    )
                elif report.community_cards != self.community_cards:
                    if len(report.community_cards) == len(self.community_cards):
                        logger.warning(
                            "Board MISMATCH from %s [%s]: %s vs current %s -- keeping current",
                            report.player_name, client_ip,
                            " ".join(report.community_cards),
                            " ".join(self.community_cards),
                        )

            # Update player hole cards
            is_new_player = report.player_name not in self.players
            old_cards = self.players.get(report.player_name)
            self.players[report.player_name] = report.hole_cards
            self.last_update = datetime.now().strftime("%H:%M:%S")

        # Log the report
        cards_str = " ".join(report.hole_cards)
        if is_new_player:
            logger.info(
                "REPORT from %s [%s]: %s  (new player, %d total)",
                report.player_name, client_ip, cards_str, len(self.players),
            )
        elif old_cards != report.hole_cards:
            logger.info(
                "REPORT from %s [%s]: %s  (updated from %s)",
                report.player_name, client_ip, cards_str,
                " ".join(old_cards) if old_cards else "?",
            )
        else:
            logger.info(
                "REPORT from %s [%s]: %s  (unchanged)",
                report.player_name, client_ip, cards_str,
            )

        # Recalculate equity (outside the lock -- calculate_equity is read-only)
        self._recalculate()

        # Print full hand state after every report
        self._print_state()

    def _recalculate(self) -> None:
        with self._lock:
            community = list(self.community_cards)
            players_snapshot = dict(self.players)

        if len(players_snapshot) < 2:
            # Not enough players for equity -- _print_state will still show what we have
            return

        # Build input for equity calculator
        player_names = list(players_snapshot.keys())
        player_hands = [
            (i + 1, players_snapshot[name])
            for i, name in enumerate(player_names)
        ]

        try:
            equities = calculate_equity(
                community_cards=community,
                player_hands=player_hands,
            )
        except Exception as e:
            logger.error("Equity calculation failed: %s", e)
            return

        # Store equity by player name
        with self._lock:
            self.equity = {}
            for eq, name in zip(equities, player_names):
                self.equity[name] = eq.equity

    def _print_state(self) -> None:
        """Print a full snapshot of the current hand to the console."""
        with self._lock:
            hand_id = self.hand_id
            community = list(self.community_cards)
            players = dict(self.players)
            equity = dict(self.equity)
            n_players = len(players)

        board_str = " ".join(community) if community else "(preflop)"
        n_board = len(community)
        street = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}.get(n_board, f"{n_board} cards")

        logger.info("")
        logger.info("+" + "=" * 50 + "+")
        logger.info("|  HAND #%-5d  |  %-8s  |  Players: %d", hand_id, street, n_players)
        logger.info("+" + "-" * 50 + "+")
        logger.info("|  Board: %-41s|", board_str)
        logger.info("+" + "-" * 50 + "+")

        if not players:
            logger.info("|  (no players have reported yet)%19s|", "")
        else:
            for name, cards in players.items():
                cards_str = " ".join(cards)
                eq = equity.get(name)
                if eq is not None:
                    logger.info(
                        "|  %-14s  %-6s  |  Equity: %5.1f%%  %6s|",
                        name, cards_str, eq * 100, "",
                    )
                else:
                    logger.info(
                        "|  %-14s  %-6s  |  (waiting for more)%3s|",
                        name, cards_str, "",
                    )

        logger.info("+" + "=" * 50 + "+")
        logger.info("")

    def get_status(self) -> StatusResponse:
        with self._lock:
            return StatusResponse(
                hand_id=self.hand_id,
                community_cards=list(self.community_cards),
                players={k: list(v) for k, v in self.players.items()},
                equity={k: round(v * 100, 1) for k, v in self.equity.items()},
                players_connected=len(self.players),
                last_update=self.last_update,
            )


# ── Global state ──────────────────────────────────────────────────────
state = HandState()


# ── Endpoints ─────────────────────────────────────────────────────────
@app.post("/hand")
def post_hand(report: HandReport, request: Request):
    """Receive a card report from a player client."""
    client_ip = request.client.host if request.client else "unknown"

    if len(report.hole_cards) != 2:
        logger.warning("Bad request from %s [%s]: hole_cards has %d cards",
                        report.player_name, client_ip, len(report.hole_cards))
        raise HTTPException(status_code=400, detail="hole_cards must have exactly 2 cards")
    if report.hand_id is not None and report.hand_id < 1:
        logger.warning("Bad request from %s [%s]: hand_id=%d",
                        report.player_name, client_ip, report.hand_id)
        raise HTTPException(status_code=400, detail="hand_id must be >= 1")

    state.update(report, client_ip)
    return {"status": "ok", "hand_id": state.hand_id}


@app.post("/new_hand")
def new_hand(hand_id: Optional[int] = None, request: Request = None):
    """Reset state for a new hand."""
    client_ip = request.client.host if request and request.client else "unknown"
    new_id = hand_id if hand_id is not None else state.hand_id + 1
    logger.info("New hand requested by [%s] -> hand #%d", client_ip, new_id)
    state.reset(new_id)
    return {"status": "ok", "hand_id": new_id}


@app.get("/status")
def get_status(request: Request):
    """Get the current hand state and equity."""
    client_ip = request.client.host if request.client else "unknown"
    logger.debug("Status check from [%s]", client_ip)
    return state.get_status()


# ── CLI entry point ───────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Poker Equity Hub Server")
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to listen on (default: 8000).",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0 for all interfaces).",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Also write logs to this file (e.g. hub.log).",
    )
    args = parser.parse_args()

    _setup_logging(log_file=args.log_file)

    logger.info("=" * 50)
    logger.info("  Poker Equity Hub")
    logger.info("  Listening on http://%s:%d", args.host, args.port)
    logger.info("  Endpoints:")
    logger.info("    POST /hand       - player reports cards")
    logger.info("    POST /new_hand   - reset for new hand")
    logger.info("    GET  /status     - current state + equity")
    logger.info("=" * 50)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
