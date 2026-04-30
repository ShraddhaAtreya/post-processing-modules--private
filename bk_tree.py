# =============================================================
# bktree.py
# Kannada OCR Post-Processing System
# BK-Tree index for fast Levenshtein candidate search
#
# Integrates with: new_validator.py
# Replace: linear scan in _find_candidates_levenshtein()
# Build cost: once at validator startup (~8-12 seconds)
# Search cost: O(log n) vs previous O(n)
# =============================================================

from typing import List, Optional, Dict, Tuple


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Standard Levenshtein distance.
    Duplicated here so bktree.py is fully standalone
    and does not create a circular import with new_validator.py.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


class BKTree:
    """
    Burkhard-Keller Tree for approximate string matching
    using Levenshtein distance as the metric.

    Triangle inequality property of Levenshtein distance allows
    pruning of large portions of the search space per query.

    Usage:
        tree = BKTree()
        tree.build(word_list)           # once at startup
        results = tree.search(word, 2)  # fast at query time
    """

    def __init__(self):
        # Each node: { distance_from_parent: (word, children_dict) }
        # Root stored as (word, children)
        self._root: Optional[Tuple[str, Dict]] = None
        self._size: int = 0
        self.is_ready: bool = False

    # ----------------------------------------------------------
    # BUILD
    # ----------------------------------------------------------

    def build(self, word_list: List[str]) -> None:
        """
        Build the BK-Tree from a list of words.
        Called once at validator startup after dictionaries load.

        Args:
            word_list: Full dictionary word list (1,095,946 words).
        """
        if not word_list:
            print("[BKTree] ⚠️  Empty word list — tree not built.")
            return

        print(f"[BKTree] Building from {len(word_list):,} words...")

        import time
        t0 = time.time()

        self._root = None
        self._size = 0

        for word in word_list:
            self._insert(word)

        elapsed = time.time() - t0
        self.is_ready = True

        print(f"[BKTree] ✅ Built {self._size:,} nodes in {elapsed:.1f}s")

    def _insert(self, word: str) -> None:
        """Insert a single word into the tree."""
        if self._root is None:
            self._root = (word, {})
            self._size += 1
            return

        current_word, children = self._root
        # Walk down until we find an empty slot
        while True:
            dist = levenshtein_distance(word, current_word)
            if dist == 0:
                return  # duplicate — skip
            if dist not in children:
                children[dist] = (word, {})
                self._size += 1
                return
            # Slot taken — go deeper
            current_word, children = children[dist]

    # ----------------------------------------------------------
    # SEARCH
    # ----------------------------------------------------------

    def search(self, query: str, max_distance: int) -> List[Tuple[str, int]]:
        """
        Find all words within max_distance of query.

        Uses triangle inequality to prune branches:
        If dist(query, node) = d, then any match must have
        distance in [d - max_distance, d + max_distance]
        from node — all other children are skipped entirely.

        Args:
            query:        The misspelled / OCR word to correct.
            max_distance: Maximum edit distance to accept (usually 2–4).

        Returns:
            List of (word, distance) tuples sorted by distance ascending.
            Empty list if tree not built yet.
        """
        if not self.is_ready or self._root is None:
            return []

        results: List[Tuple[str, int]] = []
        # Stack-based DFS — avoids Python recursion limit on large trees
        stack = [self._root]

        while stack:
            current_word, children = stack.pop()
            dist = levenshtein_distance(query, current_word)

            if dist <= max_distance:
                results.append((current_word, dist))

            # Triangle inequality pruning:
            # Only visit children whose edge label d satisfies
            # dist - max_distance <= d <= dist + max_distance
            low  = dist - max_distance
            high = dist + max_distance

            for edge_dist, child_node in children.items():
                if low <= edge_dist <= high:
                    stack.append(child_node)

        results.sort(key=lambda x: x[1])
        return results

    # ----------------------------------------------------------
    # UTILITY
    # ----------------------------------------------------------

    def search_words_only(self, query: str, max_distance: int) -> List[str]:
        """
        Convenience wrapper — returns only words, not (word, distance) tuples.
        Drop-in replacement for the old linear scan output format.

        Args:
            query:        The word to search for.
            max_distance: Maximum edit distance.

        Returns:
            List of candidate words sorted by distance ascending.
        """
        return [word for word, _ in self.search(query, max_distance)]

    @property
    def size(self) -> int:
        """Number of words indexed in the tree."""
        return self._size