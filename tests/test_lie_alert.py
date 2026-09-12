import threading
import unittest

from src.utils.LieAlert import LocalSoundAlert


class LocalSoundAlertTest(unittest.TestCase):
    def test_disabled_alert_is_not_scheduled(self):
        calls = []
        alert = LocalSoundAlert(enabled=False, play_once=lambda: calls.append(1))

        self.assertFalse(alert.notify())
        self.assertEqual(calls, [])

    def test_alert_runs_asynchronously_and_repeats(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def play_once():
            calls.append(1)
            started.set()
            release.wait(1.0)

        alert = LocalSoundAlert(
            repeat=2, interval_seconds=0.0, play_once=play_once
        )

        self.assertTrue(alert.notify())
        self.assertTrue(started.wait(1.0))
        # Duplicate submissions while playback is active are coalesced.
        self.assertFalse(alert.notify())
        release.set()
        alert.wait(1.0)
        self.assertEqual(len(calls), 2)

    def test_invalid_numeric_config_falls_back_without_disabling_alert(self):
        calls = []
        alert = LocalSoundAlert(
            repeat="invalid",
            interval_seconds=None,
            play_once=lambda: calls.append(1),
        )

        self.assertTrue(alert.notify())
        alert.wait(1.0)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
