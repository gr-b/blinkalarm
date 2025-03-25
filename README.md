# BlinkAlarm 👁️❤️‍🩹🔕 - Prevent eyestrain, red eyes, and dry eyes with real-time computer vision!

![Annotated Frame Example](images/2-annotated.png)
*BlinkAlarm in action - watching you not watching it*

Ever caught yourself staring at your screen like a zombie, eyes drier than a desert? This is a common problem for software engineers and other knowledge workers.

Here's a wild fact: When you're zoned in on your screen, your blink rate drops from a healthy 15-20 blinks per minute to a measly 3-4 (or in some cases, even fewer). Unfortunately, the more focused you are, the more likely you are to completely forget to blink altogether. Depending on total per-day screentime over a career or lifetime, this can really add up to poor eye health and declining vision. BlinkAlarm is here to help retrain your natural blinking patterns! (although it may drive you nuts)

![BlinkAlarm Logo](images/logo1.png)

# Blinks per minute

A normal human blink rate is 15-20 per minute (or 3-4 seconds between blinks). However, when focusing or working, the blink rate naturally declines. For example, when playing a a sport like tennis, the blink rate also declines, especially during intense periods. It's completely healthy for short, intense periods. It becomes problematic only when focused periods happen for 20+ minutes at a time. Many ultra-productive knowledge workers, especially those who work from home, may have many hours per-day of sustained focus.

If you find yourself in this category, I'd suggest trying out BlinkAlarm at the very least for a single session of 60+ minutes to get statistics on how often you do actually blink while working.


## What can it do

- 👀 Real-time blink detection (runs >50x faster than real-time on an M2 macbook Pro)
- 🎯 Head posture compensation (because this isn't a phone selfie camera -- you're sitting naturally, with dynamic positions)
- 🔊 Gentle (or strong) reminders when you've accidentally entered into a staring contest
- 📊 Session stats on your blink rate
- 🪶 Lightweight: runs on Google's FaceMesh, which is heavily optimized for cheap mobile devices
- ⚡ Adjustable processing rate (`--frames 5`) if you need it to be even lighter in the background

## Install

```bash
git clone github.com/gr-b/blinkalarm

pip install poetry # If you don't already have it
poetry install
```

## How to run

```bash
poetry run python blinkalarm.py
```

# Blinks per minute

- A normal human blink rate is 15-20 per minute (or 3-4 seconds between blinks).
- However, after trying it set to 15 per minute, I found it incredibly annoying
- I set the default to 6 per minute (reminder every 10s without blinking) and this is much more manageable. I plan to gradually work my way up from here. This way, only egregious cases cause the alarm at first.

To run with more blinks per minute required:
```bash
poetry run python blinkalarm.py --required-bpm 12
```

### Your Training Program 🏋️‍♂️
1. Rookie Level: Start at 6 blinks/minute (default setting)
2. Intermediate: Level up to 8-10 blinks/minute
3. Pro Status: Hit that healthy 12-15 blinks/minute

Fair warning, this can get pretty annoying. Let me know if you have ideas for how I could make it less annoying, while retaining the utility.

# Run it faster / consume fewer resources

It's already really lightweight, but if you need it to be even more, increase the `--frames N` parameter (default processes 1 frame per `1` frame captured).

`poetry run python blinkalarm.py --frames 5` Process a frame for every `5` frames captured

## Coming Soon™ (let me know if you want any of these and I may actually implement it)

- [ ] Run on startup
- [ ] Volume control
- [ ] Fancy GUI
- [ ] 20-20-20 rule integration (look 20 feet away every 20 minutes for 20 seconds)
- [ ] Option for a visual pop-up rather than audio alarm.

## Dependencies

- Python 3.8+ (because we're not savages)
- A webcam (your laptop's built-in one is perfect)
- Some dependencies (let poetry handle the heavy lifting):
  - OpenCV (for the computer vision magic)
  - MediaPipe (for the fancy face stuff)
  - NumPy (because math)
  - SimpleAudio (for those gentle "hey, blink!" reminders)

## Where It Works

- Definitely works on macOS (tested on M2 MacBook Pro)
- Should work on Windows and Linux (but we need brave volunteers to confirm)

---

You only get one pair of eyes - treat them right! 👀✨