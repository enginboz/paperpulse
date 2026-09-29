from collections.abc import Iterable
from datetime import date
from typing import Protocol

from paperpulse.models import Paper


class Source(Protocol):
    """A literature source. Implementations must be safe to call with overlapping windows."""

    name: str

    def fetch(self, start: date, end: date) -> Iterable[Paper]:
        """Yield papers added to the source between `start` and `end` (inclusive)."""
        ...
