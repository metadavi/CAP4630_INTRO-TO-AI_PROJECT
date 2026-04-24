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
3. The largest qualifying cluster wins. Its final string is built by
   character-level majority voting across all readings in the cluster —
   for each position, the most common character wins. This produces a
   stable, consistent output even when individual frames vary by 1-2 chars.
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


def _char_majority_vote(strings: list[str]) -> str:
    """
    Reconstruct the most likely plate string by voting on each character
    position independently across all strings in the cluster.

    Strings of different lengths are handled by picking the most common
    length first, then only using strings of that length for the vote.
    This avoids misaligned positions from insertion/deletion errors.
    """
    if not strings:
        return ""

    # Find the most common length
    length_votes: defaultdict[int, int] = defaultdict(int)
    for s in strings:
        length_votes[len(s)] += 1
    best_len = max(length_votes, key=length_votes.__getitem__)

    # Keep only strings matching the majority length
    candidates = [s for s in strings if len(s) == best_len]
    if not candidates:
        candidates = strings[:1]
        best_len = len(candidates[0])

    # Vote on each position
    result = []
    for i in range(best_len):
        char_votes: defaultdict[str, int] = defaultdict(int)
        for s in candidates:
            if i < len(s):
                char_votes[s[i]] += 1
        if char_votes:
            result.append(max(char_votes, key=char_votes.__getitem__))

    return "".join(result)


class FrameVoter:
    """
    Sliding-window fuzzy voter for plate readings.

    Groups readings that differ by <= VOTE_FUZZY_DIST characters (edit distance)
    so that plates partially obscured by graphics (Florida orange) still lock in.
    Uses character-level majority voting within the winning cluster to produce
    a stable, consistent plate string even when individual frames vary.

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
        if plate_text:
            self._buffer.append((plate_text, confidence))

        if len(self._buffer) < self.window:
            return None

        return self._decide()

    def reset(self) -> None:
        """Clear the buffer (call after a gate decision is made)."""
        self._buffer.clear()

    # ── Private helpers ────────────────────────────────────────────────────

    def _decide(self) -> tuple[str, float] | None:
        """
        Cluster buffered readings by edit distance, then reconstruct the
        best plate string using character-level majority voting.
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
                rep = cluster[0][0]
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

            texts      = [t for t, _ in cluster]
            confs      = [c for _, c in cluster]
            mean_conf  = sum(confs) / len(confs)

            if mean_conf > best_conf:
                # Reconstruct string via per-position majority vote
                best_text = _char_majority_vote(texts)
                best_conf = mean_conf

        if not best_text or len(best_text) < config.MIN_PLATE_CHARS:
            return None
        return best_text, best_conf

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.window
