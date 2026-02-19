"""
Player Client -- runs on each player's computer.

Captures the screen every N seconds, detects hole cards + community cards
via GPT-4o Vision, and sends the data to the central Hub server.
The hub handles hand detection automatically.

Optionally, with --auto-play, the client will automatically execute the
hub's recommended action (Fold, Check, Call, Raise, etc.) on the ClubGG
window using visual button matching.

Usage
-----
# Default: auto-capture every 5 seconds (just run and forget):
    python client.py --name Alice --hub-url http://host:8000

# With auto-play enabled (executes recommendations automatically):
    python client.py --name Alice --hub-url http://host:8000 --auto-play

# Custom interval and action delay:
    python client.py --name Alice --hub-url http://host:8000 --auto-play --action-delay 2.0

# Manual mode (press Enter to capture):
    python client.py --name Alice --hub-url http://host:8000 --manual

# Process an existing screenshot:
    python client.py --name Alice --hub-url http://host:8000 --image screenshot.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

from screen_recorder import ScreenRecorder
from card_detector import detect_cards, DetectedCards


def send_to_hub(
    hub_url: str,
    player_name: str,
    detected: DetectedCards,
) -> dict | None:
    """POST the detected cards to the hub. Returns the response dict, or None on failure."""
    # The client always sends the first player's hole cards it finds.
    # In a single-player screen, there should be exactly 1 player (the hero).
    if not detected.players:
        print("[Client] No hole cards detected in screenshot.")
        return None

    hero = detected.players[0]
    payload = {
        "player_name": player_name,
        "hole_cards": hero.cards,
        "community_cards": detected.community_cards,
        "stack": hero.stack,
        "pot_size": detected.pot_size,
        "total_players": detected.total_players,
        "table_players": [
            {"name": tp.name, "status": tp.status, "stack": tp.stack}
            for tp in detected.table_players
        ],
    }

    url = hub_url.rstrip("/") + "/hand"
    headers = {"ngrok-skip-browser-warning": "true"}
    print(f"[Client] Sending to hub: {payload}")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"[Client] Hub response: {data}")
        return data
    except requests.ConnectionError:
        print(f"[Client] ERROR: Cannot reach hub at {url}")
        return None
    except requests.HTTPError as e:
        print(f"[Client] ERROR: Hub returned {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        print(f"[Client] ERROR: {e}")
        return None


def process_and_send(
    image_path: str,
    hub_url: str,
    player_name: str,
    model: str,
) -> dict | None:
    """Capture -> detect -> send pipeline. Returns hub response dict or None."""
    print(f"\n[Client] Analysing: {image_path}  (model: {model})")

    try:
        detected = detect_cards(image_path, model=model)
    except Exception as e:
        print(f"[Client] Card detection failed: {e}")
        return None

    return send_to_hub(hub_url, player_name, detected)


def _maybe_auto_play(hub_response: dict | None, auto_play: bool, action_delay: float) -> None:
    """If auto-play is enabled and the hub returned a recommendation, execute it."""
    if not auto_play or hub_response is None:
        return

    rec = hub_response.get("recommendation")
    if rec is None:
        print("[Client] No recommendation from hub yet (need more players)")
        return

    is_folded = hub_response.get("is_folded", False)
    if is_folded:
        print("[Client] Player is folded -- skipping auto-play")
        return

    action = rec.get("action", "")
    sizing = rec.get("sizing")
    confidence = rec.get("confidence", "?")

    # Color-coded action display
    _action_colors = {
        "Fold": "\033[91m", "Check": "\033[92m", "Call": "\033[93m",
        "Bet": "\033[96m", "Raise": "\033[95m", "All-in": "\033[91;1m",
    }
    _rst = "\033[0m"
    base_action = action.split("/")[0].strip().title() if "/" in action else action.strip().title()
    color = _action_colors.get(base_action, "")
    sizing_str = f" {sizing}" if sizing is not None else ""
    print(f"[Client] AUTO-PLAY: \033[1m{color}{action}{sizing_str}{_rst}  [{confidence}]")

    try:
        from automator import execute_action
        execute_action(action, sizing=sizing, delay=action_delay)
    except ImportError:
        print("[Client] ERROR: automator module not found. "
              "Make sure automator.py is in the same directory.")
    except Exception as e:
        print(f"[Client] ERROR executing action: {e}")


def run_auto(
    recorder: ScreenRecorder,
    hub_url: str,
    player_name: str,
    model: str,
    interval: float,
    auto_play: bool = False,
    action_delay: float = 1.0,
) -> None:
    """Auto mode: capture every `interval` seconds. Default behavior."""
    capture_num = 0
    play_mode = "AUTO-PLAY" if auto_play else "observe-only"
    print("\n" + "=" * 50)
    print(f"  Player Client - {player_name}")
    print(f"  Hub: {hub_url}")
    print(f"  Model: {model}")
    print(f"  Mode: {play_mode}")
    print(f"  Capturing every {interval}s (Ctrl+C to stop)")
    if auto_play:
        print(f"  Action delay: {action_delay}s")
    print("=" * 50 + "\n")

    try:
        while True:
            capture_num += 1
            image_path = recorder.capture(f"cap{capture_num}")
            hub_response = process_and_send(image_path, hub_url, player_name, model)
            _maybe_auto_play(hub_response, auto_play, action_delay)
            print(f"[Client] Next capture in {interval}s ...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


def run_manual(
    recorder: ScreenRecorder,
    hub_url: str,
    player_name: str,
    model: str,
) -> None:
    """Manual mode: press Enter to capture, 'q' to quit."""
    capture_num = 0
    print("\n" + "=" * 50)
    print(f"  Player Client - {player_name} (Manual)")
    print(f"  Hub: {hub_url}")
    print(f"  Model: {model}")
    print("  Press Enter to capture, 'q' to quit.")
    print("=" * 50 + "\n")

    while True:
        user_input = input("> ").strip().lower()

        if user_input in ("q", "quit", "exit"):
            print("Goodbye!")
            break

        capture_num += 1
        image_path = recorder.capture(f"cap{capture_num}")
        process_and_send(image_path, hub_url, player_name, model)


def run_single_image(
    image_path: str,
    hub_url: str,
    player_name: str,
    model: str,
) -> None:
    """Process a single existing screenshot."""
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)
    process_and_send(image_path, hub_url, player_name, model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poker player client: auto-captures screen and sends cards to the hub."
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="Player name (e.g. Alice, Bob).",
    )
    parser.add_argument(
        "--hub-url", type=str, required=True,
        help="Hub server URL (e.g. http://host:8000 or https://xyz.ngrok-free.app).",
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Manual mode: press Enter to capture instead of auto-capturing.",
    )
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="Seconds between auto-captures (default: 5).",
    )
    parser.add_argument(
        "--monitor", type=int, default=0,
        help="Monitor index: 0 = all, 1 = primary, etc. (default: 0).",
    )
    parser.add_argument(
        "--dir", type=str, default="screenshots",
        help="Directory to save screenshots (default: screenshots).",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to an existing screenshot (skips capture).",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="OpenAI vision model (default: gpt-4o).",
    )
    parser.add_argument(
        "--auto-play", action="store_true",
        help="Automatically execute the hub's recommended action on ClubGG.",
    )
    parser.add_argument(
        "--action-delay", type=float, default=1.0,
        help="Seconds to wait before clicking a button in auto-play mode (default: 1.0).",
    )
    args = parser.parse_args()

    # Verify API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.")
        print("Either add it to a .env file:  OPENAI_API_KEY=sk-...")
        print("Or set it in your terminal:    set OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Single image mode
    if args.image:
        run_single_image(args.image, args.hub_url, args.name, args.model)
        return

    # Live capture modes
    recorder = ScreenRecorder(save_dir=args.dir, monitor=args.monitor)

    if args.manual:
        run_manual(recorder, args.hub_url, args.name, args.model)
    else:
        run_auto(
            recorder, args.hub_url, args.name, args.model, args.interval,
            auto_play=args.auto_play,
            action_delay=args.action_delay,
        )


if __name__ == "__main__":
    main()
