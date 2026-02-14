"""
Card Detector – uses OpenAI GPT-4o Vision to identify poker cards from a screenshot.

Returns structured data with community cards and each player's hole cards,
using treys notation: rank (A K Q J T 9 8 7 6 5 4 3 2) + suit (h d c s).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

# ── Valid card tokens (for validation) ────────────────────────────────
VALID_RANKS = set("A K Q J T 9 8 7 6 5 4 3 2".split())
VALID_SUITS = set("h d c s")
VALID_CARDS = {f"{r}{s}" for r in VALID_RANKS for s in VALID_SUITS}


# ── Data classes ──────────────────────────────────────────────────────
@dataclass
class PlayerHand:
    seat: int
    cards: list[str]  # e.g. ["Ah", "Kd"]


@dataclass
class DetectedCards:
    community_cards: list[str] = field(default_factory=list)  # e.g. ["Ah", "Kd", "Jc"]
    players: list[PlayerHand] = field(default_factory=list)

    @property
    def street(self) -> str:
        n = len(self.community_cards)
        if n == 0:
            return "preflop"
        elif n == 3:
            return "flop"
        elif n == 4:
            return "turn"
        elif n == 5:
            return "river"
        return f"unknown ({n} cards)"


# ── Rank/suit mapping: plain English → treys code ─────────────────────
_RANK_MAP = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "7", "8": "8", "9": "9", "10": "T", "ten": "T",
    "jack": "J", "queen": "Q", "king": "K", "ace": "A",
    # also accept shorthand the model might use
    "t": "T", "j": "J", "q": "Q", "k": "K", "a": "A",
}
_SUIT_MAP = {
    "hearts": "h", "diamonds": "d", "clubs": "c", "spades": "s",
    # shorthand
    "h": "h", "d": "d", "c": "c", "s": "s",
    "heart": "h", "diamond": "d", "club": "c", "spade": "s",
}


def _convert_to_treys(rank: str, suit: str) -> str:
    """Convert plain-English rank and suit to treys notation (e.g. 'Ah')."""
    r = _RANK_MAP.get(rank.strip().lower())
    s = _SUIT_MAP.get(suit.strip().lower())
    if r is None:
        raise ValueError(f"Unknown rank: {rank!r}")
    if s is None:
        raise ValueError(f"Unknown suit: {suit!r}")
    return f"{r}{s}"


# ── OpenAI function-calling schema ────────────────────────────────────
# The model reports cards in plain English (rank + suit as separate fields).
# We convert to treys notation in Python -- no room for notation confusion.
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "report_cards",
            "description": (
                "Report the poker cards visible on the screen. "
                "For each card, provide the rank and suit as separate plain-English fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "community_cards": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "rank": {
                                    "type": "string",
                                    "enum": ["2", "3", "4", "5", "6", "7", "8", "9", "10",
                                             "Jack", "Queen", "King", "Ace"],
                                    "description": "The rank of the card.",
                                },
                                "suit": {
                                    "type": "string",
                                    "enum": ["Hearts", "Diamonds", "Clubs", "Spades"],
                                    "description": "The suit of the card.",
                                },
                            },
                            "required": ["rank", "suit"],
                        },
                        "description": (
                            "The community / board cards in the center of the table. "
                            "Empty array if preflop (no cards dealt yet)."
                        ),
                    },
                    "players": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "seat": {
                                    "type": "integer",
                                    "description": "Seat number (1-based, clockwise from bottom).",
                                },
                                "card1_rank": {
                                    "type": "string",
                                    "enum": ["2", "3", "4", "5", "6", "7", "8", "9", "10",
                                             "Jack", "Queen", "King", "Ace"],
                                },
                                "card1_suit": {
                                    "type": "string",
                                    "enum": ["Hearts", "Diamonds", "Clubs", "Spades"],
                                },
                                "card2_rank": {
                                    "type": "string",
                                    "enum": ["2", "3", "4", "5", "6", "7", "8", "9", "10",
                                             "Jack", "Queen", "King", "Ace"],
                                },
                                "card2_suit": {
                                    "type": "string",
                                    "enum": ["Hearts", "Diamonds", "Clubs", "Spades"],
                                },
                            },
                            "required": ["seat", "card1_rank", "card1_suit",
                                         "card2_rank", "card2_suit"],
                        },
                        "description": "Players whose hole cards are face-up and visible.",
                    },
                },
                "required": ["community_cards", "players"],
            },
        },
    }
]

_SYSTEM_PROMPT = """\
You are an expert poker card reader with perfect attention to detail.
You will be shown a screenshot of a poker table.

YOUR TASK:
1. Look at the community cards (board) in the center of the table.
2. Look at each player's hole cards (the two cards in front of each player).
3. For each card, identify its RANK and SUIT separately, then report via the function.

HOW TO IDENTIFY CARDS:
- RANK: Read the number or letter printed in the top-left corner of the card.
  Possible values: 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace
- SUIT: Look at the symbol printed on the card.
  * Hearts = red heart shape
  * Diamonds = red diamond/rhombus shape
  * Clubs = black clover/trefoil shape
  * Spades = black pointed/spade shape

WATCH OUT FOR:
- 6 vs 9: check which way the number is oriented
- Clubs vs Spades: clubs have rounded lobes (like a clover), spades have a single point at the top
- Hearts vs Diamonds: hearts are rounded at the top, diamonds are angular
- 3 vs 8: 8 has two enclosed loops, 3 is open on one side

RULES:
- Only report cards you can CLEARLY see. Never guess.
- If no community cards are dealt (preflop), return an empty community_cards array.
- If a player's cards are face-down, do NOT include that player.
- Number seats clockwise starting from 1.
"""


# ── Helpers ───────────────────────────────────────────────────────────
def _encode_image(image_path: str) -> str:
    """Read an image file and return its base64-encoded content."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _validate_card(card: str) -> str:
    """Normalise and validate a single card string. Raises ValueError on bad input."""
    card = card.strip()
    # Handle common OCR mistakes: '10' → 'T'
    if card.startswith("10") and len(card) == 3:
        card = "T" + card[2]
    if len(card) != 2:
        raise ValueError(f"Invalid card string: {card!r}")
    card = card[0].upper() + card[1].lower()
    if card not in VALID_CARDS:
        raise ValueError(f"Unknown card: {card!r}")
    return card


def _validate_no_duplicates(detected: DetectedCards) -> None:
    """Raise ValueError if any card appears more than once."""
    all_cards: list[str] = list(detected.community_cards)
    for p in detected.players:
        all_cards.extend(p.cards)
    seen: set[str] = set()
    for c in all_cards:
        if c in seen:
            raise ValueError(f"Duplicate card detected: {c}")
        seen.add(c)


# ── Main detection function ───────────────────────────────────────────
def detect_cards(
    image_path: str,
    model: str = "gpt-4.1",
    api_key: Optional[str] = None,
) -> DetectedCards:
    """Send a screenshot to OpenAI Vision and return the detected cards.

    Parameters
    ----------
    image_path : str
        Path to the screenshot PNG file.
    model : str
        OpenAI model to use (must support vision + function calling).
        Default is 'gpt-4.1' which has the best vision accuracy.
    api_key : str, optional
        OpenAI API key. Falls back to the OPENAI_API_KEY env var.

    Returns
    -------
    DetectedCards
        Parsed community cards and player hole cards.

    Raises
    ------
    RuntimeError
        If the model doesn't call the expected function or returns bad data.
    ValueError
        If a card string is invalid or duplicated.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Screenshot not found: {image_path}")

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    b64_image = _encode_image(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is a screenshot of a poker table. "
                            "Please identify all visible cards and call the report_cards function."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        tools=_TOOLS,
        tool_choice={"type": "function", "function": {"name": "report_cards"}},
        temperature=0,
        max_tokens=1024,
    )

    # ── Parse the function call ───────────────────────────────────────
    message = response.choices[0].message
    if not message.tool_calls:
        raise RuntimeError(
            "Model did not call the report_cards function. "
            f"Response: {message.content}"
        )

    tool_call = message.tool_calls[0]
    if tool_call.function.name != "report_cards":
        raise RuntimeError(f"Unexpected function call: {tool_call.function.name}")

    data = json.loads(tool_call.function.arguments)

    # ── Build result with validation ──────────────────────────────────
    community = []
    for c in data.get("community_cards", []):
        if isinstance(c, dict):
            # New structured format: {"rank": "King", "suit": "Spades"}
            community.append(_convert_to_treys(c["rank"], c["suit"]))
        else:
            # Fallback: plain string like "Ks"
            community.append(_validate_card(c))

    players = []
    for p in data.get("players", []):
        if "card1_rank" in p:
            # New structured format with separate rank/suit fields
            card1 = _convert_to_treys(p["card1_rank"], p["card1_suit"])
            card2 = _convert_to_treys(p["card2_rank"], p["card2_suit"])
            cards = [card1, card2]
        elif "cards" in p:
            # Fallback: old format with card strings
            cards = [_validate_card(c) for c in p["cards"]]
            if len(cards) != 2:
                print(f"[CardDetector] WARNING: seat {p['seat']} has {len(cards)} cards, skipping")
                continue
        else:
            print(f"[CardDetector] WARNING: seat {p.get('seat', '?')} missing card data, skipping")
            continue
        players.append(PlayerHand(seat=p["seat"], cards=cards))

    result = DetectedCards(community_cards=community, players=players)
    _validate_no_duplicates(result)

    print(f"[CardDetector] Detected: board={result.community_cards}  "
          f"players={[(p.seat, p.cards) for p in result.players]}  "
          f"street={result.street}")

    return result
