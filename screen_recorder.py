"""
Screen Recorder – capture screenshots on demand or periodically.

Usage examples
--------------
# On-demand capture from anywhere in your code:
    from screen_recorder import recorder
    recorder.capture()                       # saves a timestamped PNG
    recorder.capture("my_label")             # saves with a custom label

# Periodic capture (runs in a background thread):
    recorder.start_periodic(interval=5)      # every 5 seconds
    ...                                      # do other work
    recorder.stop_periodic()                 # stop the background thread

# Standalone – run this file directly for periodic capture:
    python screen_recorder.py --interval 5
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import mss
import mss.tools


class ScreenRecorder:
    """Captures and saves screenshots to a folder."""

    def __init__(self, save_dir: str = "screenshots", monitor: int = 0):
        """
        Parameters
        ----------
        save_dir : str
            Directory where screenshots are saved (created automatically).
        monitor : int
            Which monitor to capture.
            0 = all monitors combined, 1 = primary, 2 = secondary, etc.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor

        self._periodic_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._counter = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # On-demand capture
    # ------------------------------------------------------------------
    def capture(self, label: str = "") -> str:
        """Take a single screenshot and save it.

        Parameters
        ----------
        label : str, optional
            A short label inserted into the filename for easy identification.

        Returns
        -------
        str
            The full path of the saved screenshot.
        """
        with self._lock:
            self._counter += 1
            count = self._counter

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
        tag = f"_{label}" if label else ""
        filename = f"screenshot_{count:04d}{tag}_{timestamp}.png"
        filepath = self.save_dir / filename

        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[self.monitor])
            mss.tools.to_png(shot.rgb, shot.size, output=str(filepath))

        print(f"[ScreenRecorder] saved: {filepath}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Periodic capture (background thread)
    # ------------------------------------------------------------------
    def start_periodic(self, interval: float = 5.0) -> None:
        """Start capturing screenshots at a fixed interval (seconds).

        Runs in a daemon thread so it won't block your program from exiting.
        Call :meth:`stop_periodic` to stop early.
        """
        if self._periodic_thread and self._periodic_thread.is_alive():
            print("[ScreenRecorder] periodic capture already running")
            return

        self._stop_event.clear()
        self._periodic_thread = threading.Thread(
            target=self._periodic_loop,
            args=(interval,),
            daemon=True,
        )
        self._periodic_thread.start()
        print(f"[ScreenRecorder] periodic capture started (every {interval}s)")

    def stop_periodic(self) -> None:
        """Stop the periodic capture thread."""
        if not self._periodic_thread or not self._periodic_thread.is_alive():
            print("[ScreenRecorder] periodic capture is not running")
            return

        self._stop_event.set()
        self._periodic_thread.join(timeout=10)
        print("[ScreenRecorder] periodic capture stopped")

    @property
    def is_running(self) -> bool:
        """True if the periodic capture thread is active."""
        return bool(self._periodic_thread and self._periodic_thread.is_alive())

    def _periodic_loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            self.capture("periodic")
            self._stop_event.wait(timeout=interval)


# ------------------------------------------------------------------
# Module-level convenience instance
# ------------------------------------------------------------------
recorder = ScreenRecorder()


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Capture screenshots periodically.")
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=5.0,
        help="Seconds between captures (default: 5)",
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default="screenshots",
        help='Save directory (default: "screenshots")',
    )
    parser.add_argument(
        "--monitor", "-m",
        type=int,
        default=0,
        help="Monitor index: 0 = all, 1 = primary, 2 = secondary, etc. (default: 0)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Take a single screenshot and exit.",
    )
    args = parser.parse_args()

    rec = ScreenRecorder(save_dir=args.dir, monitor=args.monitor)

    if args.once:
        rec.capture("manual")
        return

    print(f"Capturing every {args.interval}s -> {args.dir}/  (Ctrl+C to stop)")
    rec.start_periodic(interval=args.interval)

    try:
        while rec.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        rec.stop_periodic()


if __name__ == "__main__":
    main()
