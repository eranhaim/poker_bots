"""
Player Client -- runs on each player's computer.

Captures the screen every N seconds, detects hole cards + community cards
via GPT-4.1 Vision, and sends the data to the central Hub server.
The hub handles hand detection automatically.

Usage
-----
# Default: auto-capture every 5 seconds (just run and forget):
    python client.py --name Alice --hub-url http://host:8000

# Custom interval:
    python client.py --name Alice --hub-url http://host:8000 --interval 3

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
) -> bool:
    """POST the detected cards to the hub. Returns True on success."""
    # The client always sends the first player's hole cards it finds.
    # In a single-player screen, there should be exactly 1 player (the hero).
    if not detected.players:
        print("[Client] No hole cards detected in screenshot.")
        return False

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
        return True
    except requests.ConnectionError:
        print(f"[Client] ERROR: Cannot reach hub at {url}")
        return False
    except requests.HTTPError as e:
        print(f"[Client] ERROR: Hub returned {e.response.status_code}: {e.response.text}")
        return False
    except Exception as e:
        print(f"[Client] ERROR: {e}")
        return False


def process_and_send(
    image_path: str,
    hub_url: str,
    player_name: str,
    model: str,
) -> bool:
    """Capture -> detect -> send pipeline. Returns True on success."""
    print(f"\n[Client] Analysing: {image_path}  (model: {model})")

    try:
        detected = detect_cards(image_path, model=model)
    except Exception as e:
        print(f"[Client] Card detection failed: {e}")
        return False

    return send_to_hub(hub_url, player_name, detected)


def run_auto(
    recorder: ScreenRecorder,
    hub_url: str,
    player_name: str,
    model: str,
    interval: float,
) -> None:
    """Auto mode: capture every `interval` seconds. Default behavior."""
    capture_num = 0
    print("\n" + "=" * 50)
    print(f"  Player Client - {player_name}")
    print(f"  Hub: {hub_url}")
    print(f"  Model: {model}")
    print(f"  Capturing every {interval}s (Ctrl+C to stop)")
    print("=" * 50 + "\n")

    try:
        while True:
            capture_num += 1
            image_path = recorder.capture(f"cap{capture_num}")
            process_and_send(image_path, hub_url, player_name, model)
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
        run_auto(recorder, args.hub_url, args.name, args.model, args.interval)


if __name__ == "__main__":
    main()
