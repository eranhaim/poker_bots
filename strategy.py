"""
Strategy Module -- EV-based optimal play recommendations.

Provides simple action recommendations (Fold / Check / Call / Bet / Raise / All-in)
based on equity, pot size, stack sizes, and street.  Works in both perfect-info
(all cards known) and partial-info (some opponents unknown) scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Recommendation:
    action: str             # "Fold", "Check", "Call", "Bet", "Raise", "All-in"
    sizing: Optional[float] # chip amount for bet/raise, None for fold/check/call
    reasoning: str          # brief human-readable explanation
    confidence: str         # "high" (all info known) or "estimated" (partial info)


# ── Tunable thresholds ────────────────────────────────────────────────
# These govern action selection based on raw equity.
# In partial-info mode the thresholds shift to be more conservative.

_THRESHOLDS_PERFECT = {
    "fold_below": 0.25,     # equity < 25% -> fold (or check if free)
    "call_below": 0.50,     # 25-50% -> call / check
    "raise_below": 0.70,    # 50-70% -> bet / raise 2/3 pot
    # >= 70% -> bet pot / all-in
}

_THRESHOLDS_PARTIAL = {
    "fold_below": 0.20,     # more conservative: fold only when very weak
    "call_below": 0.45,     # widen call range slightly
    "raise_below": 0.65,
}


def _pick_thresholds(num_unknown: int) -> dict[str, float]:
    return _THRESHOLDS_PARTIAL if num_unknown > 0 else _THRESHOLDS_PERFECT


def _bet_sizing(equity: float, pot: float, stack: float) -> tuple[str, Optional[float]]:
    """Choose an action + sizing based on equity vs pot geometry.

    Returns (action_label, chip_amount | None).
    """
    if pot <= 0:
        # No pot information -- fall back to qualitative advice only
        if equity >= 0.70:
            return "Raise", None
        elif equity >= 0.50:
            return "Bet", None
        else:
            return "Check", None

    # Stack-to-pot ratio
    spr = stack / pot if pot > 0 else float("inf")

    if equity >= 0.70:
        if spr <= 1.5:
            # Short-stacked relative to pot -- just shove
            return "All-in", round(stack, 0)
        else:
            # Pot-sized bet
            sizing = round(pot, 0)
            return "Raise", min(sizing, stack)
    elif equity >= 0.50:
        # 2/3 pot bet
        sizing = round(pot * 2 / 3, 0)
        return "Bet", min(sizing, stack)
    else:
        return "Check", None


def recommend(
    player_name: str,
    equity: float,
    pot_size: Optional[float],
    stack: Optional[float],
    street: str,
    num_unknown: int = 0,
) -> Recommendation:
    """Produce an action recommendation for a single player.

    Parameters
    ----------
    player_name : str
        For logging / display.
    equity : float
        Player's equity as a fraction (0.0 - 1.0).
    pot_size : float or None
        Current pot.  If None or 0, sizing advice is omitted.
    stack : float or None
        Player's remaining chip stack.
    street : str
        One of "preflop", "flop", "turn", "river".
    num_unknown : int
        How many opponents are unknown (0 = perfect info).

    Returns
    -------
    Recommendation
    """
    pot = pot_size if pot_size and pot_size > 0 else 0.0
    stk = stack if stack and stack > 0 else 0.0
    thresholds = _pick_thresholds(num_unknown)
    confidence = "estimated" if num_unknown > 0 else "high"

    # ── Decide action ─────────────────────────────────────────────────
    if equity < thresholds["fold_below"]:
        # Weak hand -- fold unless checking is free
        action = "Check/Fold"
        sizing = None
        reasoning = (
            f"Equity {equity*100:.0f}% is below the {thresholds['fold_below']*100:.0f}% "
            f"threshold -- fold to any bet, check if free"
        )

    elif equity < thresholds["call_below"]:
        # Mediocre hand -- call if getting correct odds, otherwise check
        if pot > 0 and stk > 0:
            # Assume a half-pot bet to face: implied call = pot * 0.5
            implied_bet = pot * 0.5
            pot_odds = implied_bet / (pot + implied_bet) if (pot + implied_bet) > 0 else 1.0
            if equity >= pot_odds:
                action = "Call"
                sizing = None
                reasoning = (
                    f"Equity {equity*100:.0f}% beats pot-odds "
                    f"({pot_odds*100:.0f}%) -- call is +EV"
                )
            else:
                action = "Check/Fold"
                sizing = None
                reasoning = (
                    f"Equity {equity*100:.0f}% is below pot-odds "
                    f"({pot_odds*100:.0f}%) -- fold to a bet, check if free"
                )
        else:
            action = "Call"
            sizing = None
            reasoning = f"Equity {equity*100:.0f}% -- call or check"

    elif equity < thresholds["raise_below"]:
        # Strong hand -- bet or raise
        action, sizing = _bet_sizing(equity, pot, stk)
        if sizing is not None:
            reasoning = (
                f"Equity {equity*100:.0f}% -- value bet {sizing:,.0f} "
                f"(~{sizing/pot*100:.0f}% pot)" if pot > 0
                else f"Equity {equity*100:.0f}% -- value bet"
            )
        else:
            reasoning = f"Equity {equity*100:.0f}% -- bet for value"

    else:
        # Very strong hand -- big bet or all-in
        action, sizing = _bet_sizing(equity, pot, stk)
        if sizing is not None and pot > 0:
            reasoning = (
                f"Equity {equity*100:.0f}% -- strong hand, "
                f"{'all-in' if action == 'All-in' else f'raise {sizing:,.0f}'} "
                f"(~{sizing/pot*100:.0f}% pot)"
            )
        else:
            reasoning = f"Equity {equity*100:.0f}% -- strong hand, bet big"

    # Append partial-info caveat
    if num_unknown > 0:
        reasoning += f" [estimated vs {num_unknown} unknown opponent(s)]"

    return Recommendation(
        action=action,
        sizing=sizing,
        reasoning=reasoning,
        confidence=confidence,
    )
