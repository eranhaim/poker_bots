"""
Automator -- uses PyAutoGUI + OpenCV to click ClubGG poker buttons.

Takes a recommended action (Fold, Check, Call, Bet, Raise, All-in) and
executes it by finding the corresponding button on screen via template
matching, then clicking it.

Usage
-----
Execute an action programmatically:
    from automator import execute_action
    execute_action("Call")
    execute_action("Raise", sizing=6.50)

Capture a button template interactively:
    python automator.py --capture fold
    python automator.py --capture call
    python automator.py --capture raise
    python automator.py --capture check
    python automator.py --capture betinput
    python automator.py --capture allin

Test by executing an action from CLI:
    python automator.py --execute Call
    python automator.py --execute Raise --sizing 6.50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyautogui

# Disable PyAutoGUI's fail-safe pause (we handle delays ourselves)
pyautogui.PAUSE = 0.1

# ── Configuration ─────────────────────────────────────────────────────
TEMPLATES_DIR = Path(__file__).parent / "templates"
CONFIDENCE_THRESHOLD = 0.8
DEFAULT_ACTION_DELAY = 1.0  # seconds to wait before clicking

# Map action names to template filenames (without extension)
# These are the INITIAL buttons in the main action bar:
_ACTION_TEMPLATES: dict[str, list[str]] = {
    "Fold":   ["fold"],
    "Check":  ["check"],
    "Call":   ["call"],
    "Bet":    ["betstart", "raise"],  # "betstart" = BET in main bar (opens panel)
    "Raise":  ["raise", "betstart"],  # "raise" = RAISE in main bar (opens panel)
    "All-in": ["allin", "raise", "betstart"],
}

# Template for the CONFIRM button inside the bet panel (green BET button)
_BET_CONFIRM_TEMPLATE = "bet"
# Template for the bet/raise input field inside the panel
_BET_INPUT_TEMPLATE = "betinput"


# ── Template matching ─────────────────────────────────────────────────
def _load_template(name: str) -> Optional[np.ndarray]:
    """Load a template image from the templates directory."""
    path = TEMPLATES_DIR / f"{name}.png"
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def _screenshot_cv2() -> np.ndarray:
    """Take a screenshot and return it as an OpenCV BGR image."""
    pil_img = pyautogui.screenshot()
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def find_button(template_name: str, confidence: float = CONFIDENCE_THRESHOLD) -> Optional[tuple[int, int]]:
    """Find a button on screen using template matching.

    Returns the center (x, y) coordinates of the best match, or None
    if no match exceeds the confidence threshold.
    """
    template = _load_template(template_name)
    if template is None:
        print(f"[Automator] WARNING: Template '{template_name}.png' not found in {TEMPLATES_DIR}")
        return None

    screen = _screenshot_cv2()
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < confidence:
        print(f"[Automator] Template '{template_name}' best match: {max_val:.3f} "
              f"(below threshold {confidence:.2f})")
        return None

    # max_loc is the top-left corner; compute center
    h, w = template.shape[:2]
    cx = max_loc[0] + w // 2
    cy = max_loc[1] + h // 2

    print(f"[Automator] Found '{template_name}' at ({cx}, {cy}) "
          f"confidence={max_val:.3f}")
    return (cx, cy)


def _find_first_button(template_names: list[str]) -> Optional[tuple[int, int, str]]:
    """Try multiple template names in order, return the first match.

    Returns (x, y, matched_name) or None.
    """
    for name in template_names:
        pos = find_button(name)
        if pos is not None:
            return (pos[0], pos[1], name)
    return None


# ── Action execution ──────────────────────────────────────────────────
def execute_action(
    action: str,
    sizing: Optional[float] = None,
    delay: float = DEFAULT_ACTION_DELAY,
) -> bool:
    """Execute a poker action by finding and clicking the appropriate ClubGG button.

    Parameters
    ----------
    action : str
        One of: "Fold", "Check", "Call", "Bet", "Raise", "All-in"
    sizing : float, optional
        Chip amount for Bet/Raise actions. Ignored for Fold/Check/Call.
    delay : float
        Seconds to wait before clicking (default 1.0).

    Returns
    -------
    bool
        True if the action was executed successfully, False otherwise.
    """
    action_clean = action.strip()

    # Handle compound actions like "Check/Fold", "Call/Fold", "Bet/Raise"
    # These mean "try the first action, if not available try the second"
    if "/" in action_clean:
        parts = [a.strip() for a in action_clean.split("/")]
        print(f"[Automator] Compound action: {parts} -- trying in order")
        for part in parts:
            result = execute_action(part, sizing=sizing, delay=delay)
            if result:
                return True
        print(f"[Automator] SKIP: None of {parts} found on screen")
        return False

    action_title = action_clean.title()
    if action_title not in _ACTION_TEMPLATES:
        # Try matching common variations
        action_map = {
            "All-In": "All-in", "Allin": "All-in", "All In": "All-in",
            "Shove": "All-in",
        }
        action_title = action_map.get(action_title, action_title)

    if action_title not in _ACTION_TEMPLATES:
        print(f"[Automator] ERROR: Unknown action '{action}'. "
              f"Valid actions: {list(_ACTION_TEMPLATES.keys())}")
        return False

    template_names = _ACTION_TEMPLATES[action_title]

    print(f"[Automator] Executing: {action_title}"
          f"{f' (sizing={sizing})' if sizing is not None else ''}")

    # For Bet/Raise with sizing: first type the amount, then click confirm
    if action_title in ("Bet", "Raise") and sizing is not None:
        return _execute_bet_raise(action_title, sizing, template_names, delay)

    # For All-in: try the allin button first, fall back to raise
    if action_title == "All-in":
        return _execute_allin(template_names, delay)

    # Simple click actions: Fold, Check, Call
    return _click_button(template_names, action_title, delay)


def _click_button(
    template_names: list[str],
    action_label: str,
    delay: float,
) -> bool:
    """Find and click a button."""
    match = _find_first_button(template_names)
    if match is None:
        print(f"[Automator] SKIP: Could not find {action_label} button on screen")
        return False

    x, y, matched = match
    print(f"[Automator] Clicking {action_label} ({matched}) at ({x}, {y}) "
          f"in {delay:.1f}s ...")
    time.sleep(delay)
    pyautogui.click(x, y)
    print(f"[Automator] DONE: Clicked {action_label}")
    return True


def _execute_bet_raise(
    action_label: str,
    sizing: float,
    button_templates: list[str],
    delay: float,
) -> bool:
    """Click Raise/Bet to open the panel, type the amount, then click BET to confirm.

    ClubGG flow:
      1. Click RAISE/BET button -> opens the bet panel with input focused
      2. Type the desired amount (input already has focus)
      3. Click the BET confirm button to submit
    """
    # Step 1: Click the Raise/Bet button to open the bet panel
    match = _find_first_button(button_templates)
    if match is None:
        print(f"[Automator] SKIP: Could not find {action_label} button on screen")
        return False

    x, y, matched = match
    print(f"[Automator] Clicking {action_label} ({matched}) at ({x}, {y}) "
          f"in {delay:.1f}s to open bet panel ...")
    time.sleep(delay)
    pyautogui.click(x, y)
    print(f"[Automator] Bet panel should be open now, waiting for it ...")
    time.sleep(1.0)

    # Step 2: Type the sizing amount
    # The input field should already have focus after clicking Raise/Bet.
    # Select all existing text and replace with our amount.
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.1)

    # Format sizing: use integer if whole number, else 2 decimal places
    if sizing == int(sizing):
        sizing_str = str(int(sizing))
    else:
        sizing_str = f"{sizing:.2f}"

    pyautogui.typewrite(sizing_str, interval=0.03)
    print(f"[Automator] Typed sizing: {sizing_str}")
    time.sleep(0.5)

    # Step 3: Click the BET confirm button inside the panel
    bet_pos = find_button(_BET_CONFIRM_TEMPLATE)
    if bet_pos is not None:
        bx, by = bet_pos
        print(f"[Automator] Clicking BET confirm at ({bx}, {by}) ...")
        time.sleep(0.3)
        pyautogui.click(bx, by)
        print(f"[Automator] DONE: {action_label} {sizing_str}")
        return True

    print(f"[Automator] WARNING: Could not find BET confirm button after typing amount")
    return False


def _execute_allin(template_names: list[str], delay: float) -> bool:
    """Execute an all-in action."""
    # Try dedicated all-in button first
    allin_pos = find_button("allin")
    if allin_pos is not None:
        x, y = allin_pos
        print(f"[Automator] Clicking All-in at ({x}, {y}) in {delay:.1f}s ...")
        time.sleep(delay)
        pyautogui.click(x, y)
        print("[Automator] DONE: Clicked All-in")
        return True

    # Fall back to raise button (ClubGG sometimes shows "All-in" as the raise button)
    print("[Automator] No dedicated All-in button, trying Raise button ...")
    return _click_button(["raise", "bet"], "All-in", delay)


# ── Template capture utility ─────────────────────────────────────────
def capture_template(name: str) -> None:
    """Interactive utility: capture a screen region and save as a template.

    Pauses for the user to position their mouse, takes a screenshot,
    then lets the user select the region via an OpenCV window.
    """
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TEMPLATES_DIR / f"{name}.png"

    print(f"\n[Template Capture] Capturing template: '{name}'")
    print("  1. Make sure the ClubGG table is visible on screen")
    print(f"  2. You have 3 seconds to position the window ...")
    time.sleep(3)

    print("  3. Taking screenshot ...")
    screen = _screenshot_cv2()

    # Show the screenshot in an OpenCV window for region selection
    print("  4. A window will open. Draw a rectangle around the button:")
    print("     - Click and drag to select the region")
    print("     - Press ENTER or SPACE to confirm")
    print("     - Press 'c' to cancel and retry")

    win_name = f"Select {name} button - ENTER to confirm - ESC to cancel"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
    roi = cv2.selectROI(win_name, screen, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("[Template Capture] Cancelled (no region selected)")
        return

    # Crop and save
    template = screen[y:y + h, x:x + w]
    cv2.imwrite(str(output_path), template)
    print(f"[Template Capture] Saved: {output_path} ({w}x{h} pixels)")


def list_templates() -> None:
    """Print all available template images."""
    if not TEMPLATES_DIR.exists():
        print(f"[Automator] No templates directory at {TEMPLATES_DIR}")
        return

    templates = sorted(TEMPLATES_DIR.glob("*.png"))
    if not templates:
        print(f"[Automator] No templates found in {TEMPLATES_DIR}")
        return

    print(f"\n[Automator] Available templates in {TEMPLATES_DIR}:")
    for t in templates:
        img = cv2.imread(str(t))
        if img is not None:
            h, w = img.shape[:2]
            print(f"  {t.stem:15s}  {w}x{h} px")
        else:
            print(f"  {t.stem:15s}  (unreadable)")


# ── CLI ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ClubGG Button Automator -- template capture and action execution."
    )
    sub = parser.add_subparsers(dest="command")

    # Capture subcommand
    cap = sub.add_parser("capture", help="Capture a button template from screen")
    cap.add_argument(
        "name", type=str,
        help="Template name (e.g. fold, call, check, raise, betinput, allin)",
    )

    # Execute subcommand
    exe = sub.add_parser("execute", help="Execute a poker action")
    exe.add_argument(
        "action", type=str,
        help="Action to execute (Fold, Check, Call, Bet, Raise, All-in)",
    )
    exe.add_argument(
        "--sizing", type=float, default=None,
        help="Bet/raise amount",
    )
    exe.add_argument(
        "--delay", type=float, default=DEFAULT_ACTION_DELAY,
        help=f"Seconds to wait before clicking (default: {DEFAULT_ACTION_DELAY})",
    )

    # List subcommand
    sub.add_parser("list", help="List available templates")

    args = parser.parse_args()

    if args.command == "capture":
        capture_template(args.name)
    elif args.command == "execute":
        success = execute_action(args.action, sizing=args.sizing, delay=args.delay)
        sys.exit(0 if success else 1)
    elif args.command == "list":
        list_templates()
    else:
        parser.print_help()
        print("\n--- Quick start ---")
        print("1. Capture button templates:")
        print("     python automator.py capture fold")
        print("     python automator.py capture check")
        print("     python automator.py capture call")
        print("     python automator.py capture raise")
        print("     python automator.py capture betinput")
        print("     python automator.py capture allin")
        print("2. List captured templates:")
        print("     python automator.py list")
        print("3. Test an action:")
        print("     python automator.py execute Call")
        print("     python automator.py execute Raise --sizing 6.50")


if __name__ == "__main__":
    main()
