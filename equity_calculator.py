"""
Equity Calculator – compute win/tie equity for each player using the treys library.

Supports exhaustive enumeration (fast enough on the turn/river) and
Monte Carlo sampling (for flop or preflop with many unknowns).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Optional

from treys import Card, Deck, Evaluator

_evaluator = Evaluator()


@dataclass
class PlayerEquity:
    seat: int
    cards: list[str]       # original string notation, e.g. ["Ah", "Kd"]
    equity: float          # 0.0 – 1.0
    win_count: int = 0
    tie_count: int = 0
    total_simulations: int = 0


def _to_treys(cards: list[str]) -> list[int]:
    """Convert a list of card strings (e.g. ['Ah', 'Kd']) to treys int format."""
    return [Card.new(c) for c in cards]


def calculate_equity(
    community_cards: list[str],
    player_hands: list[tuple[int, list[str]]],  # [(seat, [card1, card2]), ...]
    num_simulations: int = 10_000,
    exhaustive_threshold: int = 50_000,
) -> list[PlayerEquity]:
    """Calculate equity for each player given known board and hole cards.

    Parameters
    ----------
    community_cards : list[str]
        Board cards in treys notation (0–5 cards).
    player_hands : list[tuple[int, list[str]]]
        Each entry is (seat_number, [card1, card2]).
    num_simulations : int
        Number of Monte Carlo iterations when not doing exhaustive enumeration.
    exhaustive_threshold : int
        If the total number of possible remaining-board combinations is at or below
        this value, do exhaustive enumeration instead of Monte Carlo.

    Returns
    -------
    list[PlayerEquity]
        One entry per player, sorted by seat number.
    """
    if not player_hands:
        return []

    # ── Convert cards to treys format ─────────────────────────────────
    board_treys = _to_treys(community_cards)
    hands_treys = [_to_treys(h) for _, h in player_hands]

    # ── Build remaining deck ──────────────────────────────────────────
    known_cards = set(board_treys)
    for h in hands_treys:
        known_cards.update(h)
    full_deck = Deck.GetFullDeck()
    remaining = [c for c in full_deck if c not in known_cards]

    cards_to_deal = 5 - len(board_treys)  # how many community cards left to deal

    # ── Decide exhaustive vs. Monte Carlo ─────────────────────────────
    if cards_to_deal == 0:
        # River – just evaluate once
        return _evaluate_single_board(board_treys, hands_treys, player_hands)

    total_combos = _n_choose_k(len(remaining), cards_to_deal)
    use_exhaustive = total_combos <= exhaustive_threshold

    if use_exhaustive:
        return _exhaustive(
            board_treys, hands_treys, remaining, cards_to_deal, player_hands
        )
    else:
        return _monte_carlo(
            board_treys, hands_treys, remaining, cards_to_deal, player_hands,
            num_simulations,
        )


# ── Internal helpers ──────────────────────────────────────────────────

def _n_choose_k(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    result = 1
    for i in range(min(k, n - k)):
        result = result * (n - i) // (i + 1)
    return result


def _evaluate_single_board(
    board: list[int],
    hands: list[list[int]],
    player_info: list[tuple[int, list[str]]],
) -> list[PlayerEquity]:
    """Evaluate a complete 5-card board for all players."""
    scores = [_evaluator.evaluate(board, h) for h in hands]
    best = min(scores)  # lower is better in treys
    winners = [i for i, s in enumerate(scores) if s == best]

    results = []
    for idx, (seat, cards) in enumerate(player_info):
        if len(winners) == 1 and idx == winners[0]:
            eq = 1.0
            w, t = 1, 0
        elif idx in winners:
            eq = 1.0 / len(winners)
            w, t = 0, 1
        else:
            eq = 0.0
            w, t = 0, 0
        results.append(PlayerEquity(
            seat=seat, cards=cards, equity=eq,
            win_count=w, tie_count=t, total_simulations=1,
        ))
    return sorted(results, key=lambda r: r.seat)


def _run_simulation(
    board: list[int],
    hands: list[list[int]],
    extra_cards: tuple[int, ...],
    wins: list[int],
    ties: list[int],
) -> None:
    """Score one board runout and update win/tie counters in-place."""
    full_board = board + list(extra_cards)
    scores = [_evaluator.evaluate(full_board, h) for h in hands]
    best = min(scores)
    winners = [i for i, s in enumerate(scores) if s == best]
    if len(winners) == 1:
        wins[winners[0]] += 1
    else:
        for w in winners:
            ties[w] += 1


def _build_results(
    wins: list[int],
    ties: list[int],
    total: int,
    player_info: list[tuple[int, list[str]]],
) -> list[PlayerEquity]:
    results = []
    num_players = len(player_info)
    for idx, (seat, cards) in enumerate(player_info):
        # Equity = (wins + ties / num_tying) / total
        # Simplified: wins get full credit, ties split
        eq = (wins[idx] + ties[idx] / max(num_players, 1)) / max(total, 1)
        results.append(PlayerEquity(
            seat=seat, cards=cards, equity=eq,
            win_count=wins[idx], tie_count=ties[idx],
            total_simulations=total,
        ))
    return sorted(results, key=lambda r: r.seat)


def _exhaustive(
    board: list[int],
    hands: list[list[int]],
    remaining: list[int],
    cards_to_deal: int,
    player_info: list[tuple[int, list[str]]],
) -> list[PlayerEquity]:
    """Enumerate all possible remaining boards."""
    n = len(hands)
    wins = [0] * n
    ties = [0] * n
    total = 0

    for combo in itertools.combinations(remaining, cards_to_deal):
        _run_simulation(board, hands, combo, wins, ties)
        total += 1

    print(f"[EquityCalc] Exhaustive: {total} boards evaluated")
    return _build_results(wins, ties, total, player_info)


def _monte_carlo(
    board: list[int],
    hands: list[list[int]],
    remaining: list[int],
    cards_to_deal: int,
    player_info: list[tuple[int, list[str]]],
    num_simulations: int,
) -> list[PlayerEquity]:
    """Run Monte Carlo sampling over random remaining boards."""
    n = len(hands)
    wins = [0] * n
    ties = [0] * n

    for _ in range(num_simulations):
        sampled = tuple(random.sample(remaining, cards_to_deal))
        _run_simulation(board, hands, sampled, wins, ties)

    print(f"[EquityCalc] Monte Carlo: {num_simulations} simulations")
    return _build_results(wins, ties, num_simulations, player_info)


def calculate_equity_with_unknowns(
    community_cards: list[str],
    player_hands: list[tuple[int, list[str]]],  # known players: [(seat, [c1, c2]), ...]
    num_unknown: int = 0,
    num_simulations: int = 10_000,
) -> list[PlayerEquity]:
    """Calculate equity when some opponents have unknown (random) hands.

    Parameters
    ----------
    community_cards : list[str]
        Board cards in treys notation (0-5 cards).
    player_hands : list[tuple[int, list[str]]]
        Known players -- each entry is (seat_number, [card1, card2]).
    num_unknown : int
        Number of opponents whose cards are unknown.  Their hands are
        sampled uniformly from the remaining deck each iteration.
    num_simulations : int
        Monte Carlo iterations.

    Returns
    -------
    list[PlayerEquity]
        One entry per known player (sorted by seat), plus one aggregate
        entry for all unknown players combined (seat=-1, cards=["??","??"]).
    """
    # If no unknowns, delegate to the standard calculator
    if num_unknown <= 0:
        return calculate_equity(community_cards, player_hands, num_simulations)

    # ── Convert known cards ───────────────────────────────────────────
    board_treys = _to_treys(community_cards)
    known_hands_treys = [_to_treys(h) for _, h in player_hands]

    known_cards = set(board_treys)
    for h in known_hands_treys:
        known_cards.update(h)
    full_deck = Deck.GetFullDeck()
    remaining = [c for c in full_deck if c not in known_cards]

    cards_to_deal = 5 - len(board_treys)  # community cards still to come
    n_known = len(player_hands)
    # Each unknown player needs 2 cards; plus remaining community cards
    cards_needed_per_sim = num_unknown * 2 + cards_to_deal

    if cards_needed_per_sim > len(remaining):
        raise ValueError(
            f"Not enough cards in deck: need {cards_needed_per_sim} "
            f"but only {len(remaining)} remain"
        )

    # ── Monte Carlo with unknown hands ────────────────────────────────
    wins_known = [0] * n_known
    ties_known = [0] * n_known
    wins_unknown = 0
    ties_unknown = 0

    for _ in range(num_simulations):
        sampled = random.sample(remaining, cards_needed_per_sim)

        # Deal unknown hands (2 cards each)
        unknown_hands = []
        idx = 0
        for _ in range(num_unknown):
            unknown_hands.append(sampled[idx:idx + 2])
            idx += 2

        # Remaining community cards
        extra_board = sampled[idx:idx + cards_to_deal]
        full_board = board_treys + extra_board

        # Evaluate all hands together
        all_hands = known_hands_treys + unknown_hands
        scores = [_evaluator.evaluate(full_board, h) for h in all_hands]
        best = min(scores)
        winners = [i for i, s in enumerate(scores) if s == best]

        if len(winners) == 1:
            w = winners[0]
            if w < n_known:
                wins_known[w] += 1
            else:
                wins_unknown += 1
        else:
            for w in winners:
                if w < n_known:
                    ties_known[w] += 1
                else:
                    ties_unknown += 1

    # ── Build results ─────────────────────────────────────────────────
    total = num_simulations
    n_all = n_known + num_unknown
    results: list[PlayerEquity] = []

    for i, (seat, cards) in enumerate(player_hands):
        eq = (wins_known[i] + ties_known[i] / max(n_all, 1)) / max(total, 1)
        results.append(PlayerEquity(
            seat=seat, cards=cards, equity=eq,
            win_count=wins_known[i], tie_count=ties_known[i],
            total_simulations=total,
        ))

    # Aggregate unknown-player equity
    unknown_eq = (wins_unknown + ties_unknown / max(n_all, 1)) / max(total, 1)
    results.append(PlayerEquity(
        seat=-1, cards=["??", "??"], equity=unknown_eq,
        win_count=wins_unknown, tie_count=ties_unknown,
        total_simulations=total,
    ))

    print(f"[EquityCalc] Monte Carlo with {num_unknown} unknown(s): "
          f"{num_simulations} simulations")
    return sorted(results, key=lambda r: r.seat)


def format_equity_table(
    community_cards: list[str],
    equities: list[PlayerEquity],
    hand_number: Optional[int] = None,
) -> str:
    """Return a formatted string for console display."""
    lines = []
    if hand_number is not None:
        lines.append(f"=== Hand #{hand_number} ===")

    board_str = " ".join(community_cards) if community_cards else "(preflop)"
    lines.append(f"Board: {board_str}")
    lines.append("-" * 40)

    for pe in equities:
        cards_str = " ".join(pe.cards)
        lines.append(f"  Seat {pe.seat}: {cards_str:<6s}  |  Equity: {pe.equity * 100:5.1f}%")

    lines.append("-" * 40)
    return "\n".join(lines)
