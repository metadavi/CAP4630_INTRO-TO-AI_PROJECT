"""
Multi-Frame Voter
=================
Accumulates OCR readings across a sliding window of frames and returns the
most likely plate string along with an aggregated confidence score.

Why multi-frame?
    A single frame may be blurry, partially occluded, or poorly lit.
    By collecting readings across several frames and taking a weighted
    majority vote, we dramatically reduce the character error rate.

Algorithm
---------
1. Each call to `update()` adds a (plate_text, confidence) pair to a deque
   of length VOTE_WINDOW.
2. When the window is full, `decide()` returns the candidate that:
     a. appears at least VOTE_MIN_HITS times, AND
     b. has the highest mean confidence among qualifying candidates.
3. The window then slides (oldest reading drops out).
"""

from collections import deque, defaultdict

import config


class FrameVoter:
    """
    Sliding-window majority voter for plate readings.

    Usage
    -----
        voter = FrameVoter()
        for frame in video_stream:
            text, conf = ocr.read(plate_crop)
            result = voter.update(text, conf)
            if result:
                plate, agg_conf = result
                # process the stable reading
    """

    def __init__(self,
                 window:    int   = config.VOTE_WINDOW,
                 min_hits:  int   = config.VOTE_MIN_HITS):
        self.window   = window
        self.min_hits = min_hits
        self._buffer: deque[tuple[str, float]] = deque(maxlen=window)

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self,
               plate_text: str,
               confidence: float) -> tuple[str, float] | None:
        """
        Add a new reading and attempt to reach a decision.

        Returns
        -------
        (plate_text, aggregated_confidence)  if a stable plate is found,
        None                                 if the window is not yet confident.
        """
        if plate_text:                          # ignore empty readings
            self._buffer.append((plate_text, confidence))

        if len(self._buffer) < self.window:
            return None                         # window not full yet

        return self._decide()

    def reset(self) -> None:
        """Clear the buffer (call after a gate decision is made)."""
        self._buffer.clear()

    # ── Private helpers ────────────────────────────────────────────────────

    def _decide(self) -> tuple[str, float] | None:
        """
        Evaluate the current window and return the winning candidate
        (or None if no candidate meets the threshold).
        """
        hits:    defaultdict[str, int]   = defaultdict(int)
        conf_sum: defaultdict[str, float] = defaultdict(float)

        for text, conf in self._buffer:
            hits[text]     += 1
            conf_sum[text] += conf

        # Filter to candidates that appear at least min_hits times
        # and meet minimum plate length
        qualifying = {
            text: conf_sum[text] / hits[text]
            for text, count in hits.items()
            if count >= self.min_hits
            and len(text) >= config.MIN_PLATE_CHARS
        }

        if not qualifying:
            return None

        # Pick the candidate with the highest mean confidence
        best_text = max(qualifying, key=qualifying.__getitem__)
        best_conf = qualifying[best_text]
        return best_text, best_conf

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.window
