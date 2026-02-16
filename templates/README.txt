BUTTON TEMPLATE IMAGES FOR ClubGG AUTOMATOR
=============================================

This folder stores small PNG screenshots of ClubGG poker buttons.
The automator uses these to find buttons on screen via image matching.

REQUIRED TEMPLATES:
  fold.png       - The "Fold" button
  check.png      - The "Check" button
  call.png       - The "Call" button
  raise.png      - The "Raise" / "Bet" confirm button
  betinput.png   - The text input field where you type the raise amount
  allin.png      - The "All-in" button (optional if no distinct button)

HOW TO CAPTURE:
  1. Open a ClubGG table so the buttons are visible
  2. Run:  python automator.py capture fold
  3. Wait 3 seconds (position your screen)
  4. An OpenCV window opens -- drag a rectangle around the button
  5. Press ENTER to save, ESC to cancel
  6. Repeat for each button: check, call, raise, betinput, allin

TIPS:
  - Capture ONLY the button, not surrounding area
  - Make sure the button is in its normal (clickable) state
  - If buttons change appearance (e.g. highlighted), capture both states
  - Re-capture if ClubGG updates its UI or you change screen resolution
