from __future__ import annotations

from collections import deque


class PersistentAlert:
    """Require N positive frames in a rolling window, with cooldown."""

    def __init__(self, window: int = 10, required: int = 6, cooldown: int = 30):
        if not 0 < required <= window:
            raise ValueError("required must be between 1 and window")
        self.history = deque(maxlen=window)
        self.required = required
        self.cooldown = cooldown
        self.remaining = 0

    def update(self, positive: bool) -> bool:
        self.history.append(bool(positive))
        if self.remaining:
            self.remaining -= 1
            return False
        if len(self.history) == self.history.maxlen and sum(self.history) >= self.required:
            self.remaining = self.cooldown
            self.history.clear()
            return True
        return False

