import unittest

from src.engine.DirectionalAttackGate import DirectionalAttackGate
from src.input.CommandMailbox import CommandMailbox


class DirectionalAttackGateTest(unittest.TestCase):
    def test_target_behind_requires_turn_and_redetection(self):
        gate = DirectionalAttackGate()

        requested = gate.decide(
            "left", "right", now=10.0, facing_changed_at=5.0,
            turn_delay=0.08, cooldown_ready=True,
        )
        self.assertTrue(requested.turning)
        self.assertFalse(requested.should_attack)

        settling = gate.decide(
            "left", "left", now=10.05, facing_changed_at=10.01,
            turn_delay=0.08, cooldown_ready=True,
        )
        self.assertEqual(settling.reason, "turn_settling")
        self.assertFalse(settling.should_attack)

        ready = gate.decide(
            "left", "left", now=10.10, facing_changed_at=10.01,
            turn_delay=0.08, cooldown_ready=True,
        )
        self.assertTrue(ready.should_attack)
        self.assertFalse(ready.turning)

    def test_already_facing_stable_target_attacks_immediately(self):
        gate = DirectionalAttackGate()
        decision = gate.decide(
            "right", "right", now=20.0, facing_changed_at=10.0,
            turn_delay=0.08, cooldown_ready=True,
        )
        self.assertTrue(decision.should_attack)

    def test_pending_attack_is_preserved_without_duplicate(self):
        gate = DirectionalAttackGate()
        decision = gate.decide(
            "right", "right", now=20.0, facing_changed_at=10.0,
            turn_delay=0.08, cooldown_ready=True, attack_pending=True,
        )
        self.assertFalse(decision.should_attack)
        self.assertTrue(decision.preserve_pending_attack)

    def test_missing_target_cancels_pending_turn(self):
        gate = DirectionalAttackGate()
        gate.decide("left", "right", now=10.0, cooldown_ready=True)
        decision = gate.decide(None, "left", now=10.2, cooldown_ready=True)
        self.assertEqual(decision.reason, "no_target")
        self.assertIsNone(gate.pending_direction)


class CommandMailboxTest(unittest.TestCase):
    def test_snapshot_keeps_direction_and_action_together(self):
        mailbox = CommandMailbox()
        attack = mailbox.publish("left", "none", "attack")
        mailbox.publish("right", "none", "none")

        self.assertFalse(mailbox.consume_action(attack.sequence, "attack"))
        current = mailbox.snapshot()
        self.assertEqual(
            (current.left_right, current.up_down, current.action),
            ("right", "none", "none"),
        )

    def test_duplicate_publish_does_not_create_a_second_attack(self):
        mailbox = CommandMailbox()
        first = mailbox.publish("left", "none", "attack")
        duplicate = mailbox.publish("left", "none", "attack")
        self.assertEqual(first.sequence, duplicate.sequence)
        self.assertTrue(mailbox.consume_action(first.sequence, "attack"))
        self.assertFalse(mailbox.has_pending_action("attack"))

    def test_clear_invalidates_an_unconsumed_action(self):
        mailbox = CommandMailbox()
        attack = mailbox.publish("left", "none", "attack")

        mailbox.clear()

        self.assertFalse(mailbox.consume_action(attack.sequence, "attack"))
        current = mailbox.snapshot()
        self.assertEqual(
            (current.left_right, current.up_down, current.action),
            ("none", "none", "none"),
        )


if __name__ == "__main__":
    unittest.main()
