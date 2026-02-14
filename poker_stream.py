"""
Poker Stream Equity Tool – main entry point.

Captures the poker table screen, sends it to GPT-4o Vision to detect cards,
calculates equity for each visible player, and prints results to the console.

Usage
-----
# Manual mode (press Enter to capture, 'q' to quit):
    python poker_stream.py

# Auto mode (capture every N seconds):
    python poker_stream.py --auto --interval 10

# Use a specific monitor:
    python poker_stream.py --monitor 1

# Use an existing screenshot instead of capturing:
    python poker_stream.py --image path/to/screenshot.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()  # load .env file (OPENAI_API_KEY, etc.)

from screen_recorder import ScreenRecorder
from card_detector import detect_cards, DetectedCards
from equity_calculator import calculate_equity, format_equity_table


def process_screenshot(image_path: str, hand_number: int, model: str = "gpt-4o") -> None:
    """Run the full pipeline on a single screenshot: detect cards -> compute equity -> print."""
    print(f"\n[Stream] Analysing screenshot: {image_path}  (model: {model})")

    # ── Step 1: Detect cards via Vision API ───────────────────────────
    try:
        detected: DetectedCards = detect_cards(image_path, model=model)
    except Exception as e:
        print(f"[Stream] Card detection failed: {e}")
        return

    if not detected.players:
        print("[Stream] No player cards detected - skipping equity calculation.")
        return

    if len(detected.players) < 2:
        print("[Stream] Only one player's cards visible - need at least 2 for equity.")
        return

    # ── Step 2: Calculate equity ──────────────────────────────────────
    player_hands = [(p.seat, p.cards) for p in detected.players]

    try:
        equities = calculate_equity(
            community_cards=detected.community_cards,
            player_hands=player_hands,
        )
    except Exception as e:
        print(f"[Stream] Equity calculation failed: {e}")
        return

    # ── Step 3: Display results ───────────────────────────────────────
    table = format_equity_table(
        community_cards=detected.community_cards,
        equities=equities,
        hand_number=hand_number,
    )
    print(f"\n{table}\n")


def run_manual(recorder: ScreenRecorder, model: str = "gpt-4o") -> None:
    """Manual mode: press Enter to capture, 'q' + Enter to quit."""
    hand_number = 0
    print("\n" + "=" * 50)
    print("  Poker Stream Equity Tool - Manual Mode")
    print(f"  Model: {model}")
    print("  Press Enter to capture, 'q' to quit.")
    print("=" * 50 + "\n")

    while True:
        user_input = input("> ").strip().lower()
        if user_input in ("q", "quit", "exit"):
            print("Goodbye!")
            break

        hand_number += 1
        image_path = recorder.capture(f"hand{hand_number}")
        process_screenshot(image_path, hand_number, model=model)


def run_auto(recorder: ScreenRecorder, interval: float, model: str = "gpt-4o") -> None:
    """Auto mode: capture every `interval` seconds."""
    hand_number = 0
    print("\n" + "=" * 50)
    print(f"  Poker Stream Equity Tool - Auto Mode ({interval}s)")
    print(f"  Model: {model}")
    print("  Ctrl+C to stop.")
    print("=" * 50 + "\n")

    try:
        while True:
            hand_number += 1
            image_path = recorder.capture(f"hand{hand_number}")
            process_screenshot(image_path, hand_number, model=model)
            print(f"[Stream] Next capture in {interval}s ...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


def run_single_image(image_path: str, model: str = "gpt-4o") -> None:
    """Process a single existing screenshot file."""
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)
    process_screenshot(image_path, hand_number=1, model=model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poker stream equity calculator: capture -> detect -> calculate."
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-capture mode (periodic screenshots).",
    )
    parser.add_argument(
        "--interval", type=float, default=10.0,
        help="Seconds between auto-captures (default: 10).",
    )
    parser.add_argument(
        "--monitor", type=int, default=0,
        help="Monitor index: 0 = all, 1 = primary, 2 = secondary, etc. (default: 0).",
    )
    parser.add_argument(
        "--dir", type=str, default="screenshots",
        help="Directory to save screenshots (default: screenshots).",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to an existing screenshot to process (skips capture).",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="OpenAI vision model to use (default: gpt-4o).",
    )
    args = parser.parse_args()

    # ── Verify API key is available ───────────────────────────────────
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.")
        print("Either add it to a .env file:  OPENAI_API_KEY=sk-...")
        print("Or set it in your terminal:    set OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # ── Single image mode ─────────────────────────────────────────────
    if args.image:
        run_single_image(args.image, model=args.model)
        return

    # ── Live capture modes ────────────────────────────────────────────
    recorder = ScreenRecorder(save_dir=args.dir, monitor=args.monitor)

    if args.auto:
        run_auto(recorder, interval=args.interval, model=args.model)
    else:
        run_manual(recorder, model=args.model)


if __name__ == "__main__":
    main()
