"""Thread-safe immutable command snapshots for the keyboard controller."""

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSnapshot:
    sequence: int
    left_right: str
    up_down: str
    action: str


class CommandMailbox:
    """Publish and consume complete movement/action commands atomically."""

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = CommandSnapshot(0, "none", "none", "none")

    def publish(self, left_right, up_down, action):
        command = (str(left_right), str(up_down), str(action))
        with self._lock:
            current = self._snapshot
            if command == (current.left_right, current.up_down, current.action):
                return current
            self._snapshot = CommandSnapshot(
                current.sequence + 1, command[0], command[1], command[2]
            )
            return self._snapshot

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def clear(self):
        with self._lock:
            current = self._snapshot
            self._snapshot = CommandSnapshot(
                current.sequence + 1, "none", "none", "none"
            )
            return self._snapshot

    def has_pending_action(self, action):
        with self._lock:
            return self._snapshot.action == action

    def consume_action(self, sequence, action):
        """Clear an action only if no newer command replaced its snapshot."""
        with self._lock:
            current = self._snapshot
            if current.sequence != sequence or current.action != action:
                return False
            self._snapshot = CommandSnapshot(
                current.sequence + 1,
                current.left_right,
                current.up_down,
                "none",
            )
            return True
