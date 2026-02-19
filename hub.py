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

from equity_calculator import calculate_equity, calculate_equity_with_unknowns, format_equity_table
from strategy import recommend as strategy_recommend, Recommendation

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
class TablePlayerReport(BaseModel):
    name: str
    status: str                            # "active" or "folded"
    stack: Optional[float] = None


class HandReport(BaseModel):
    player_name: str
    hole_cards: list[str]                  # e.g. ["Ah", "Kd"]
    community_cards: list[str]             # e.g. ["Jc", "Ts", "9h"]
    hand_id: Optional[int] = None          # optional -- hub auto-detects hands if omitted
    stack: Optional[float] = None          # player's chip stack
    pot_size: Optional[float] = None       # current pot size
    total_players: Optional[int] = None    # total players with cards (from AI vision)
    table_players: list[TablePlayerReport] = []  # all players with names + status


class RecommendationResponse(BaseModel):
    action: str
    sizing: Optional[float] = None
    reasoning: str
    confidence: str


class StatusResponse(BaseModel):
    hand_id: int
    community_cards: list[str]
    players: dict[str, list[str]]          # player_name -> hole_cards
    stacks: dict[str, Optional[float]]     # player_name -> chip stack
    pot_size: Optional[float]
    equity: dict[str, float]               # player_name -> equity %
    total_players: Optional[int]           # agreed total (from unanimous vote)
    unknown_players: int                   # total - known
    recommendations: dict[str, RecommendationResponse]  # player_name -> recommendation
    folded_players: list[str]              # list of player names who folded
    active_players: list[str]              # list of active player names
    players_connected: int
    last_update: Optional[str]


# ── In-memory hand state ─────────────────────────────────────────────
class HandState:
    def __init__(self) -> None:
        self.hand_id: int = 0
        self.community_cards: list[str] = []
        self.players: dict[str, list[str]] = {}    # name -> [card1, card2]
        self.stacks: dict[str, Optional[float]] = {}  # name -> stack
        self.pot_size: Optional[float] = None
        self.equity: dict[str, float] = {}          # name -> equity (0-1)
        self.total_players_votes: dict[str, int] = {}  # name -> reported total
        self.total_players: Optional[int] = None       # agreed-upon count
        self.recommendations: dict[str, Recommendation] = {}  # name -> rec
        self.folded_players: set[str] = set()          # player names confirmed folded
        self.active_players: set[str] = set()          # player names confirmed active
        # Per-reporter votes: player_name -> {reporter_name -> "active"/"folded"}
        self.player_status_votes: dict[str, dict[str, str]] = {}
        self.last_update: Optional[str] = None
        self._lock = threading.Lock()

    def reset(self, hand_id: int) -> None:
        with self._lock:
            self.hand_id = hand_id
            self.community_cards = []
            self.players = {}
            self.stacks = {}
            self.pot_size = None
            self.equity = {}
            self.total_players_votes = {}
            self.total_players = None
            self.recommendations = {}
            self.folded_players = set()
            self.active_players = set()
            self.player_status_votes = {}
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
                self.stacks = {}
                self.pot_size = None
                self.equity = {}
                self.total_players_votes = {}
                self.total_players = None
                self.recommendations = {}
                self.folded_players = set()
                self.active_players = set()
                self.player_status_votes = {}

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

            # Update player hole cards and stack
            is_new_player = report.player_name not in self.players
            old_cards = self.players.get(report.player_name)
            self.players[report.player_name] = report.hole_cards
            if report.stack is not None:
                self.stacks[report.player_name] = report.stack
            # Update pot size (use latest reported value)
            if report.pot_size is not None:
                self.pot_size = report.pot_size

            # Update total-players vote (unanimous consensus)
            if report.total_players is not None:
                self.total_players_votes[report.player_name] = report.total_players
                votes = set(self.total_players_votes.values())
                if len(votes) == 1:
                    agreed = votes.pop()
                    if agreed != self.total_players:
                        self.total_players = agreed
                        logger.info(
                            "TOTAL PLAYERS agreed: %d (unanimous from %d client(s))",
                            agreed, len(self.total_players_votes),
                        )
                else:
                    logger.warning(
                        "TOTAL PLAYERS disagreement: votes=%s -- keeping %s",
                        dict(self.total_players_votes),
                        self.total_players if self.total_players else "unknown",
                    )

            # Update player status from table_players reports
            if report.table_players:
                for tp in report.table_players:
                    name = tp.name.strip()
                    if not name:
                        continue
                    status = tp.status.strip().lower()
                    if status not in ("active", "folded"):
                        status = "active"
                    # Record this reporter's vote for this player
                    if name not in self.player_status_votes:
                        self.player_status_votes[name] = {}
                    self.player_status_votes[name][report.player_name] = status
                    # Also update stack from table_players if available
                    if tp.stack is not None and name not in self.stacks:
                        self.stacks[name] = tp.stack

                # Recalculate folded/active sets from votes.
                # Two modes:
                #   - Player has their own client (in self.players): only
                #     trust their OWN report (their screen shows greyed cards)
                #   - Player has NO client: trust other players' reports
                #     (on their screens, a folded player has no cards)
                new_folded: set[str] = set()
                new_active: set[str] = set()
                for pname, votes in self.player_status_votes.items():
                    has_own_client = pname in self.players
                    if has_own_client:
                        self_vote = votes.get(pname)
                        if self_vote == "folded":
                            new_folded.add(pname)
                        else:
                            new_active.add(pname)
                    else:
                        # No own client -- trust majority of other reporters
                        fold_votes = sum(1 for v in votes.values() if v == "folded")
                        if fold_votes > len(votes) / 2:
                            new_folded.add(pname)
                        else:
                            new_active.add(pname)

                # Log newly folded players
                for pname in new_folded - self.folded_players:
                    has_own = pname in self.players
                    logger.info(
                        "FOLD detected: %s (%s)",
                        pname,
                        "self-reported" if has_own else
                        f"reported by other clients: {[r for r, v in self.player_status_votes[pname].items() if v == 'folded']}",
                    )

                self.folded_players = new_folded
                self.active_players = new_active

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
            stacks_snapshot = dict(self.stacks)
            pot = self.pot_size
            total_pl = self.total_players
            folded = set(self.folded_players)

        # Filter out folded players from equity calculation
        active_players_snapshot = {
            name: cards for name, cards in players_snapshot.items()
            if name not in folded
        }
        n_active_known = len(active_players_snapshot)
        n_known_total = len(players_snapshot)
        n_unknown = max(0, total_pl - n_known_total) if total_pl is not None else 0

        # Need at least 1 active known player + at least 1 other (known or unknown)
        if n_active_known == 0:
            return
        if n_active_known < 2 and n_unknown == 0:
            return

        # Build input for equity calculator (only active players)
        active_names = list(active_players_snapshot.keys())
        player_hands = [
            (i + 1, active_players_snapshot[name])
            for i, name in enumerate(active_names)
        ]

        try:
            if n_unknown > 0:
                equities = calculate_equity_with_unknowns(
                    community_cards=community,
                    player_hands=player_hands,
                    num_unknown=n_unknown,
                )
            else:
                equities = calculate_equity(
                    community_cards=community,
                    player_hands=player_hands,
                )
        except Exception as e:
            logger.error("Equity calculation failed: %s", e)
            return

        # Determine street for strategy
        n_board = len(community)
        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(n_board, "unknown")

        # Store equity by player name and generate recommendations
        with self._lock:
            self.equity = {}
            self.recommendations = {}

            # Folded players get 0% equity, no recommendation
            for name in folded:
                self.equity[name] = 0.0

            for eq in equities:
                if eq.seat == -1:
                    # Aggregate unknown-player equity
                    self.equity["Unknown"] = eq.equity
                    continue
                # Map seat index back to active player name
                idx = eq.seat - 1
                if 0 <= idx < len(active_names):
                    name = active_names[idx]
                    self.equity[name] = eq.equity

                    # Strategy recommendation (only for active players)
                    try:
                        rec = strategy_recommend(
                            player_name=name,
                            equity=eq.equity,
                            pot_size=pot,
                            stack=stacks_snapshot.get(name),
                            street=street,
                            num_unknown=n_unknown,
                        )
                        self.recommendations[name] = rec
                    except Exception as e:
                        logger.error("Strategy recommendation failed for %s: %s", name, e)

    def _print_state(self) -> None:
        """Print a full snapshot of the current hand to the console."""
        with self._lock:
            hand_id = self.hand_id
            community = list(self.community_cards)
            players = dict(self.players)
            stacks = dict(self.stacks)
            pot = self.pot_size
            equity = dict(self.equity)
            total_pl = self.total_players
            recs = dict(self.recommendations)
            n_known = len(players)
            folded = set(self.folded_players)
            active = set(self.active_players)

        n_unknown = max(0, total_pl - n_known) if total_pl is not None else 0
        n_total = total_pl if total_pl is not None else n_known
        n_folded = len(folded)

        board_str = " ".join(community) if community else "(preflop)"
        n_board = len(community)
        street = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}.get(n_board, f"{n_board} cards")
        pot_str = f"Pot: {pot:,.0f}" if pot else "Pot: --"
        players_str = f"{n_known}/{n_total}" if n_unknown > 0 else str(n_known)

        logger.info("")
        logger.info("+" + "=" * 78 + "+")
        logger.info("|  HAND #%-5d  |  %-8s  |  Players: %-5s |  %-10s|%14s|",
                     hand_id, street, players_str, pot_str,
                     f"{n_unknown} unknown" if n_unknown > 0 else "")
        if n_folded > 0:
            logger.info("|  Folded: %-68s|",
                         ", ".join(sorted(folded)))
        logger.info("+" + "-" * 78 + "+")
        logger.info("|  Board: %-69s|", board_str)
        logger.info("+" + "-" * 78 + "+")

        if not players and n_unknown == 0:
            logger.info("|  (no players have reported yet)%-47s|", "")
        else:
            # Show active (known) players first
            for name, cards in players.items():
                is_folded = name in folded
                cards_str = " ".join(cards)
                stack = stacks.get(name)
                stack_str = f"{stack:,.0f}" if stack is not None else "--"
                eq = equity.get(name)
                rec = recs.get(name)
                rec_str = rec.action if rec else ""
                if rec and rec.sizing is not None:
                    rec_str += f" {rec.sizing:,.0f}"

                if is_folded:
                    logger.info(
                        "|  %-12s  %-6s  |  Stack: %8s  |  FOLDED    0.0%%  |  %-12s|",
                        name, cards_str, stack_str, "",
                    )
                elif eq is not None:
                    logger.info(
                        "|  %-12s  %-6s  |  Stack: %8s  |  Equity: %5.1f%%  |  %-12s|",
                        name, cards_str, stack_str, eq * 100, rec_str,
                    )
                else:
                    logger.info(
                        "|  %-12s  %-6s  |  Stack: %8s  |  (waiting)       |  %-12s|",
                        name, cards_str, stack_str, "",
                    )

            # Show folded players who are NOT in self.players (no cards reported by a client)
            for name in sorted(folded):
                if name not in players:
                    stack = stacks.get(name)
                    stack_str = f"{stack:,.0f}" if stack is not None else "--"
                    logger.info(
                        "|  %-12s  %-6s  |  Stack: %8s  |  FOLDED    0.0%%  |  %-12s|",
                        name, "--", stack_str, "",
                    )

            # Show unknown player aggregate
            if n_unknown > 0:
                unknown_eq = equity.get("Unknown")
                if unknown_eq is not None:
                    logger.info(
                        "|  %-12s  %-6s  |  Stack: %8s  |  Equity: %5.1f%%  |  %-12s|",
                        f"Unknown({n_unknown})", "?? ??", "--", unknown_eq * 100, "--",
                    )
                else:
                    logger.info(
                        "|  %-12s  %-6s  |  Stack: %8s  |  (waiting)       |  %-12s|",
                        f"Unknown({n_unknown})", "?? ??", "--", "",
                    )

        logger.info("+" + "=" * 78 + "+")

        # Print strategy reasoning below the table
        if recs:
            logger.info("  Strategy:")
            for name, rec in recs.items():
                logger.info("    %-12s  %s  [%s]", name, rec.reasoning, rec.confidence)

        logger.info("")

    def get_status(self) -> StatusResponse:
        with self._lock:
            n_known = len(self.players)
            n_unknown = max(0, self.total_players - n_known) if self.total_players else 0
            recs_resp: dict[str, RecommendationResponse] = {}
            for name, rec in self.recommendations.items():
                recs_resp[name] = RecommendationResponse(
                    action=rec.action,
                    sizing=rec.sizing,
                    reasoning=rec.reasoning,
                    confidence=rec.confidence,
                )
            return StatusResponse(
                hand_id=self.hand_id,
                community_cards=list(self.community_cards),
                players={k: list(v) for k, v in self.players.items()},
                stacks=dict(self.stacks),
                pot_size=self.pot_size,
                equity={k: round(v * 100, 1) for k, v in self.equity.items()},
                total_players=self.total_players,
                unknown_players=n_unknown,
                recommendations=recs_resp,
                folded_players=sorted(self.folded_players),
                active_players=sorted(self.active_players),
                players_connected=n_known,
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

    # Return recommendation for this player (if available)
    rec_data = None
    with state._lock:
        rec = state.recommendations.get(report.player_name)
        if rec:
            rec_data = {
                "action": rec.action,
                "sizing": rec.sizing,
                "reasoning": rec.reasoning,
                "confidence": rec.confidence,
            }
        # Also report if this player is folded
        is_folded = report.player_name in state.folded_players

    return {
        "status": "ok",
        "hand_id": state.hand_id,
        "recommendation": rec_data,
        "is_folded": is_folded,
    }


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
