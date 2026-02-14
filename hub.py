"""
Hub Server -- central equity calculator for distributed poker stream.

Receives card data from player clients via HTTP, aggregates community cards
and hole cards, recalculates equity on every update, and prints results.

Usage
-----
    python hub.py                     # default port 8000
    python hub.py --port 9000         # custom port
    python hub.py --host 0.0.0.0      # listen on all interfaces (for internet)
"""

from __future__ import annotations

import argparse
import threading
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from equity_calculator import calculate_equity, format_equity_table

# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(title="Poker Equity Hub")


# ── Request / response models ────────────────────────────────────────
class HandReport(BaseModel):
    player_name: str
    hole_cards: list[str]          # e.g. ["Ah", "Kd"]
    community_cards: list[str]     # e.g. ["Jc", "Ts", "9h"]
    hand_id: int


class StatusResponse(BaseModel):
    hand_id: int
    community_cards: list[str]
    players: dict[str, list[str]]  # player_name -> hole_cards
    equity: dict[str, float]       # player_name -> equity %
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
        print(f"\n[Hub] === New hand #{hand_id} ===\n")

    def update(self, report: HandReport) -> None:
        with self._lock:
            # Auto-advance hand if client sends a newer hand_id
            if report.hand_id > self.hand_id:
                self.hand_id = report.hand_id
                self.community_cards = []
                self.players = {}
                self.equity = {}
                print(f"\n[Hub] === New hand #{self.hand_id} (auto-advanced) ===\n")

            if report.hand_id < self.hand_id:
                print(f"[Hub] Ignoring stale report from {report.player_name} "
                      f"(hand {report.hand_id} < current {self.hand_id})")
                return

            # Update community cards (use longest valid set, or first reporter)
            if report.community_cards:
                if not self.community_cards:
                    self.community_cards = report.community_cards
                    print(f"[Hub] Board set by {report.player_name}: "
                          f"{' '.join(self.community_cards)}")
                elif len(report.community_cards) > len(self.community_cards):
                    # New street detected -- more community cards now
                    self.community_cards = report.community_cards
                    print(f"[Hub] Board updated by {report.player_name}: "
                          f"{' '.join(self.community_cards)}")
                elif report.community_cards != self.community_cards:
                    if len(report.community_cards) == len(self.community_cards):
                        print(f"[Hub] WARNING: {report.player_name} reports different board: "
                              f"{report.community_cards} vs current {self.community_cards}. "
                              f"Keeping current.")

            # Update player hole cards
            self.players[report.player_name] = report.hole_cards
            self.last_update = datetime.now().strftime("%H:%M:%S")
            print(f"[Hub] {report.player_name} reported: {' '.join(report.hole_cards)}")

        # Recalculate equity (outside the lock -- calculate_equity is read-only)
        self._recalculate()

    def _recalculate(self) -> None:
        with self._lock:
            community = list(self.community_cards)
            players_snapshot = dict(self.players)

        if len(players_snapshot) < 2:
            print(f"[Hub] {len(players_snapshot)} player(s) reported -- "
                  f"need at least 2 for equity.\n")
            return

        # Build input for equity calculator
        # Assign seats by insertion order (1-based)
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
            print(f"[Hub] Equity calculation failed: {e}")
            return

        # Store equity by player name
        with self._lock:
            self.equity = {}
            for eq, name in zip(equities, player_names):
                self.equity[name] = eq.equity

        # Print formatted table
        lines = []
        lines.append(f"=== Hand #{self.hand_id} ===")
        board_str = " ".join(community) if community else "(preflop)"
        lines.append(f"Board: {board_str}")
        lines.append("-" * 45)
        for eq, name in zip(equities, player_names):
            cards_str = " ".join(eq.cards)
            lines.append(f"  {name:<12s}  {cards_str:<6s}  |  Equity: {eq.equity * 100:5.1f}%")
        lines.append("-" * 45)
        print("\n" + "\n".join(lines) + "\n")

    def get_status(self) -> StatusResponse:
        with self._lock:
            return StatusResponse(
                hand_id=self.hand_id,
                community_cards=list(self.community_cards),
                players={k: list(v) for k, v in self.players.items()},
                equity={k: round(v * 100, 1) for k, v in self.equity.items()},
                last_update=self.last_update,
            )


# ── Global state ──────────────────────────────────────────────────────
state = HandState()


# ── Endpoints ─────────────────────────────────────────────────────────
@app.post("/hand")
def post_hand(report: HandReport):
    """Receive a card report from a player client."""
    if len(report.hole_cards) != 2:
        raise HTTPException(status_code=400, detail="hole_cards must have exactly 2 cards")
    if report.hand_id < 1:
        raise HTTPException(status_code=400, detail="hand_id must be >= 1")
    state.update(report)
    return {"status": "ok", "hand_id": state.hand_id}


@app.post("/new_hand")
def new_hand(hand_id: Optional[int] = None):
    """Reset state for a new hand."""
    new_id = hand_id if hand_id is not None else state.hand_id + 1
    state.reset(new_id)
    return {"status": "ok", "hand_id": new_id}


@app.get("/status")
def get_status():
    """Get the current hand state and equity."""
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
    args = parser.parse_args()

    print("=" * 50)
    print("  Poker Equity Hub")
    print(f"  Listening on http://{args.host}:{args.port}")
    print("  Endpoints:")
    print(f"    POST /hand       - player reports cards")
    print(f"    POST /new_hand   - reset for new hand")
    print(f"    GET  /status     - current state + equity")
    print("=" * 50)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
