"""Two-phase turn-before-attack scheduling for directional skills."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DirectionalAttackDecision:
    should_attack: bool
    turning: bool
    preserve_pending_attack: bool
    reason: str


class DirectionalAttackGate:
    """Require a confirmed turn before attacking a target behind the player.

    The gate does not infer facing from pixels.  It consumes the direction that
    the keyboard controller actually dispatched, along with the dispatch time.
    A target is re-evaluated every main-loop frame, so a disappeared or
    side-switching target cancels/restarts the pending turn instead of producing
    a delayed blind attack.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.pending_direction: Optional[str] = None
        self.turn_requested_at = 0.0

    def decide(
        self,
        target_direction,
        facing_direction,
        *,
        now,
        facing_changed_at=0.0,
        turn_delay=0.08,
        cooldown_ready=True,
        attack_pending=False,
    ):
        if target_direction not in ("left", "right"):
            self.reset()
            return DirectionalAttackDecision(False, False, False, "no_target")

        turn_delay = max(0.0, float(turn_delay))

        # The controller has not yet dispatched this direction.  Request only
        # the turn in this frame; the caller will detect the target again before
        # this gate is allowed to emit an attack.
        if facing_direction != target_direction:
            if self.pending_direction != target_direction:
                self.pending_direction = target_direction
                self.turn_requested_at = float(now)
            return DirectionalAttackDecision(False, True, False, "turn_requested")

        # Even if another subsystem (route/patrol) just turned the player, give
        # the game enough time to commit the facing state before attacking.
        settle_from = float(facing_changed_at or 0.0)
        if self.pending_direction == target_direction:
            settle_from = max(settle_from, self.turn_requested_at)
        elif self.pending_direction is not None:
            # The target changed to the direction we already face.  The old
            # turn request is irrelevant and must not delay this valid attack.
            self.reset()

        if settle_from > 0.0 and float(now) - settle_from < turn_delay:
            return DirectionalAttackDecision(False, True, False, "turn_settling")

        self.reset()
        if attack_pending:
            # Keep publishing the same atomic command until the keyboard thread
            # consumes it.  Duplicate publication is de-duplicated by the
            # command mailbox, so this cannot create a second attack.
            return DirectionalAttackDecision(False, False, True, "attack_pending")
        if not cooldown_ready:
            return DirectionalAttackDecision(False, False, False, "cooldown")
        return DirectionalAttackDecision(True, False, False, "ready")
