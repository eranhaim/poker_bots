"""
Player Client -- runs on each player's computer.

Captures the screen, detects hole cards + community cards via GPT-4o Vision,
and sends the data to the central Hub server.

Usage
-----
# Manual mode (press Enter to capture):
    python client.py --name Alice --hub-url http://host:8000

# Auto mode (capture every N seconds):
    python client.py --name Alice --hub-url http://host:8000 --auto --interval 10

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
    hand_id: int,
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
        "hand_id": hand_id,
    }

    url = hub_url.rstrip("/") + "/hand"
    print(f"[Client] Sending to hub: {payload}")

    try:
        resp = requests.post(url, json=payload, timeout=10)
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
    hand_id: int,
    model: str,
) -> bool:
    """Capture -> detect -> send pipeline. Returns True on success."""
    print(f"\n[Client] Analysing: {image_path}  (model: {model})")

    try:
        detected = detect_cards(image_path, model=model)
    except Exception as e:
        print(f"[Client] Card detection failed: {e}")
        return False

    return send_to_hub(hub_url, player_name, hand_id, detected)


def run_manual(
    recorder: ScreenRecorder,
    hub_url: str,
    player_name: str,
    model: str,
) -> None:
    """Manual mode: press Enter to capture, 'n' for new hand, 'q' to quit."""
    hand_id = 1
    print("\n" + "=" * 50)
    print(f"  Player Client - {player_name}")
    print(f"  Hub: {hub_url}")
    print(f"  Model: {model}")
    print("  Commands:")
    print("    Enter  = capture & send")
    print("    n      = new hand (increment hand_id)")
    print("    q      = quit")
    print("=" * 50 + "\n")

    while True:
        user_input = input(f"[hand #{hand_id}] > ").strip().lower()

        if user_input in ("q", "quit", "exit"):
            print("Goodbye!")
            break

        if user_input in ("n", "new", "new_hand"):
            hand_id += 1
            print(f"[Client] New hand #{hand_id}")
            # Notify hub of new hand
            try:
                resp = requests.post(
                    hub_url.rstrip("/") + "/new_hand",
                    params={"hand_id": hand_id},
                    timeout=10,
                )
                print(f"[Client] Hub new_hand response: {resp.json()}")
            except Exception as e:
                print(f"[Client] Could not notify hub: {e}")
            continue

        image_path = recorder.capture(f"hand{hand_id}")
        process_and_send(image_path, hub_url, player_name, hand_id, model)


def run_auto(
    recorder: ScreenRecorder,
    hub_url: str,
    player_name: str,
    model: str,
    interval: float,
    hand_id: int = 1,
) -> None:
    """Auto mode: capture every `interval` seconds."""
    print("\n" + "=" * 50)
    print(f"  Player Client - {player_name} (Auto, {interval}s)")
    print(f"  Hub: {hub_url}")
    print(f"  Model: {model}")
    print("  Ctrl+C to stop.")
    print("=" * 50 + "\n")

    try:
        while True:
            image_path = recorder.capture(f"hand{hand_id}")
            process_and_send(image_path, hub_url, player_name, hand_id, model)
            print(f"[Client] Next capture in {interval}s ...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


def run_single_image(
    image_path: str,
    hub_url: str,
    player_name: str,
    model: str,
    hand_id: int = 1,
) -> None:
    """Process a single existing screenshot."""
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)
    process_and_send(image_path, hub_url, player_name, hand_id, model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poker player client: capture -> detect -> send to hub."
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="Player name (e.g. Alice, Bob).",
    )
    parser.add_argument(
        "--hub-url", type=str, required=True,
        help="Hub server URL (e.g. http://123.45.67.89:8000).",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-capture mode (periodic).",
    )
    parser.add_argument(
        "--interval", type=float, default=10.0,
        help="Seconds between auto-captures (default: 10).",
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
        "--hand-id", type=int, default=1,
        help="Starting hand ID (default: 1).",
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
        run_single_image(args.image, args.hub_url, args.name, args.model, args.hand_id)
        return

    # Live capture modes
    recorder = ScreenRecorder(save_dir=args.dir, monitor=args.monitor)

    if args.auto:
        run_auto(recorder, args.hub_url, args.name, args.model, args.interval, args.hand_id)
    else:
        run_manual(recorder, args.hub_url, args.name, args.model)


if __name__ == "__main__":
    main()
