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
2. When the window is full, `decide()` finds clusters of similar readings
   using edit distance (Levenshtein) <= VOTE_FUZZY_DIST. This handles the
   Florida orange-graphic issue where the same plate reads slightly differently
   each frame (e.g. QJZH62 vs QJZM62 -> same cluster).
3. The largest cluster that appears >= VOTE_MIN_HITS times wins.
   Within the cluster, the most-frequent exact string is returned.
"""

from collections import deque, defaultdict

import config


def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            )
        prev = curr
    return prev[len(b)]


class FrameVoter:
    """
    Sliding-window fuzzy voter for plate readings.

    Groups readings that differ by <= VOTE_FUZZY_DIST characters (edit distance)
    so that plates partially obscured by graphics (Florida orange) still lock in.

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
                 window:     int = config.VOTE_WINDOW,
                 min_hits:   int = config.VOTE_MIN_HITS,
                 fuzzy_dist: int = config.VOTE_FUZZY_DIST):
        self.window     = window
        self.min_hits   = min_hits
        self.fuzzy_dist = fuzzy_dist
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
        Cluster the buffered readings by fuzzy edit distance and return
        the best candidate (or None if no cluster meets the threshold).
        """
        readings = list(self._buffer)

        # Build clusters: each reading joins the first existing cluster
        # whose representative is within fuzzy_dist of it.
        clusters: list[list[tuple[str, float]]] = []
        for text, conf in readings:
            if len(text) < config.MIN_PLATE_CHARS:
                continue
            placed = False
            for cluster in clusters:
                rep = cluster[0][0]         # representative = first member
                if _edit_distance(text, rep) <= self.fuzzy_dist:
                    cluster.append((text, conf))
                    placed = True
                    break
            if not placed:
                clusters.append([(text, conf)])

        # Find the best qualifying cluster
        best_text = None
        best_conf = 0.0

        for cluster in clusters:
            if len(cluster) < self.min_hits:
                continue

            # Within the cluster pick the most-frequent exact string
            freq: defaultdict[str, int]   = defaultdict(int)
            csum: defaultdict[str, float] = defaultdict(float)
            for t, c in cluster:
                freq[t] += 1
                csum[t] += c

            top = max(freq, key=freq.__getitem__)
            mean_conf = csum[top] / freq[top]

            if mean_conf > best_conf:
                best_text = top
                best_conf = mean_conf

        if best_text is None:
            return None
        return best_text, best_conf

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.window
