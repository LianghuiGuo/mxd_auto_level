import ast
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


class LieDetectorIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "src/engine/MapleStoryAutoLevelUp.py"
        source = ast.parse(path.read_text(encoding="utf-8"))
        engine = next(
            node for node in source.body
            if isinstance(node, ast.ClassDef) and node.name == "MapleStoryAutoBot"
        )
        methods = {
            "start", "_start_threads", "terminate_threads",
            "_on_lie_worker_result", "_lie_target_to_window_coord",
            "_handle_lie_detector",
        }
        # Isolate lifecycle methods from platform-specific input and UI imports.
        engine.body = [
            node for node in engine.body
            if isinstance(node, ast.FunctionDef) and node.name in methods
        ]
        cls.engine_code = compile(
            ast.Module(body=[engine], type_ignores=[]), str(path), "exec"
        )

    def setUp(self):
        self.order = []
        self.runtime = Mock()
        self.runtime.warm_up.side_effect = lambda: self.order.append("warm_up")
        self.worker = Mock()
        self.worker.is_alive = False
        self.keyboard = Mock()
        self.capture = Mock()
        self.capture.window_title = "authorized test client"
        self.capture.window_hwnd = 0
        self.health = Mock()
        self.namespace = {
            "time": time,
            "threading": SimpleNamespace(Thread=Mock()),
            "np": np,
            "logger": Mock(),
            "LieDetectorRuntime": Mock(return_value=self.runtime),
            "LieDetectorWorker": Mock(return_value=self.worker),
            "KeyBoardController": Mock(side_effect=self._create_keyboard),
            "GameWindowCapturor": Mock(return_value=self.capture),
            "LocalSoundAlert": Mock(),
            "HealthMonitor": Mock(return_value=self.health),
            "Profiler": Mock(),
            "RuneSolver": Mock(),
            "click_in_game_window": Mock(),
            "move_mouse_in_game_window": Mock(),
        }
        exec(self.engine_code, self.namespace)
        self.bot = self.namespace["MapleStoryAutoBot"]()
        self.bot.cfg = {
            "lie_detector": {"enabled": True, "backend": "trtyolo"},
            "health_monitor": {"enable": False},
            "bot": {"mode": "normal"},
            "game_window": {"title_bar_height": 59},
        }
        self.bot.args = SimpleNamespace(test_image="", init_state="")
        self.bot.lie_detector_runtime = None
        self.bot.lie_detector_worker = None
        self.bot.kb = None
        self.bot.capture = None
        self.bot.health_monitor = None
        self.bot.is_disable_control = False
        self.bot.is_terminated = True
        self.bot._directional_attack_gate = Mock()
        self.bot.fsm = Mock()
        self.bot._lie_worker_frame_source = Mock()
        self.bot.loop = Mock()
        self.bot.stop_record = Mock()

    def _create_keyboard(self, _config):
        self.order.append("keyboard")
        return self.keyboard

    def test_sdk_initializes_before_automation_and_worker_owns_cleanup(self):
        self.bot.start()
        self.assertEqual(self.order, ["warm_up", "keyboard"])
        kwargs = self.namespace["LieDetectorWorker"].call_args.kwargs
        self.assertEqual(kwargs["on_stop"], self.runtime.close)
        self.assertEqual(kwargs["target_fps"], 33)
        self.worker.start.assert_called_once()
        self.assertFalse(self.bot.is_terminated)

    def test_authorization_failure_blocks_start_and_closes_runtime(self):
        self.runtime.warm_up.side_effect = RuntimeError("machine code not authorized")
        with self.assertRaisesRegex(RuntimeError, "not authorized"):
            self.bot.start()
        self.namespace["KeyBoardController"].assert_not_called()
        self.namespace["GameWindowCapturor"].assert_not_called()
        self.namespace["LieDetectorWorker"].assert_not_called()
        self.runtime.close.assert_called_once()
        self.assertTrue(self.bot.is_terminated)

    def test_capture_start_failure_closes_sdk_and_stops_keyboard(self):
        self.namespace["GameWindowCapturor"].side_effect = RuntimeError("no window")
        with self.assertRaisesRegex(RuntimeError, "no window"):
            self.bot.start()
        self.runtime.close.assert_called_once()
        self.assertTrue(self.keyboard.is_terminated)
        self.assertTrue(self.bot.is_terminated)

    def test_pause_waits_before_close_and_preserves_recording(self):
        self.bot.start()
        self.worker.stop.side_effect = lambda: self.order.append("worker_stop")
        self.runtime.close.side_effect = lambda: self.order.append("runtime_close")
        self.bot.terminate_threads(finalize_recording=False)
        self.assertEqual(self.order[-2:], ["worker_stop", "runtime_close"])
        self.bot.stop_record.assert_not_called()
        self.assertIsNone(self.bot.lie_detector_worker)
        self.capture.stop.assert_called_once()
        self.assertTrue(self.keyboard.is_terminated)

    def test_stop_timeout_does_not_release_inflight_sdk_handle(self):
        self.bot.start()
        self.worker.is_alive = True
        self.bot.terminate_threads(finalize_recording=False)
        self.runtime.close.assert_not_called()
        self.assertIs(self.bot.lie_detector_worker, self.worker)
        self.assertTrue(self.bot.is_terminated)

    def test_restart_is_blocked_while_old_worker_is_alive(self):
        self.bot.start()
        self.worker.is_alive = True
        with self.assertRaisesRegex(RuntimeError, "still stopping"):
            self.bot.start()
        self.runtime.close.assert_not_called()
        self.assertEqual(self.namespace["LieDetectorRuntime"].call_count, 1)

    def test_restart_creates_a_fresh_tracker(self):
        self.bot.start()
        self.bot.terminate_threads(finalize_recording=False)
        fresh = Mock()
        self.namespace["LieDetectorRuntime"].return_value = fresh
        self.bot.start()
        fresh.warm_up.assert_called_once()
        self.assertIs(self.bot.lie_detector_runtime, fresh)
        kwargs = self.namespace["LieDetectorWorker"].call_args.kwargs
        self.assertEqual(kwargs["on_stop"], fresh.close)

    def test_disabled_tracking_does_not_require_sdk(self):
        self.bot.cfg["lie_detector"]["enabled"] = False
        self.bot.start()
        self.namespace["LieDetectorRuntime"].assert_not_called()
        self.namespace["LieDetectorWorker"].assert_not_called()

    def test_pointer_uses_content_coordinates_with_title_offset_once(self):
        self.bot.start()
        self.bot.lie_sound_alert = None
        value = SimpleNamespace(
            panel_just_confirmed=False,
            confirm_click=None,
            target_frame=(640.0, 336.0),
            engaged=True,
            phase="panel",
            panel_roi=(400, 100, 600, 420),
            tracking=None,
        )
        self.bot._on_lie_worker_result(SimpleNamespace(value=value))
        self.namespace["move_mouse_in_game_window"].assert_called_once_with(
            self.capture.window_title, (640, 395)
        )
        value.target_frame = None
        self.bot._on_lie_worker_result(SimpleNamespace(value=value))
        self.namespace["move_mouse_in_game_window"].assert_called_once()

    def test_confirm_click_uses_existing_window_mapping(self):
        self.bot.start()
        value = SimpleNamespace(
            panel_just_confirmed=False,
            confirm_click=(750.0, 415.0),
            target_frame=None,
            engaged=True,
        )
        self.bot._on_lie_worker_result(SimpleNamespace(value=value))
        self.namespace["click_in_game_window"].assert_called_once_with(
            self.capture.window_title, (750, 474)
        )

    def test_tracking_hold_still_suspends_keyboard_automation(self):
        self.bot.start()
        value = SimpleNamespace(engaged=True)
        self.worker.latest.return_value = SimpleNamespace(value=value)
        self.bot._lie_content_frame = Mock(return_value=(None, 0))
        self.bot._handle_lie_detector()
        self.assertTrue(self.keyboard.automation_suspended)
        self.keyboard.set_command.assert_called_with("none none none")
        self.keyboard.release_all_key.assert_called_once()
        self.health.disable.assert_called_once()

    def _panel_snapshot(self, target=None):
        return SimpleNamespace(value=SimpleNamespace(
            panel_just_confirmed=False, confirm_click=None,
            target_frame=target, engaged=True, phase="panel",
            panel_roi=(300, 100, 600, 420),
            tracking=SimpleNamespace(acquired=False, actionable=False, center=None),
        ))

    def test_mouse_hold_is_logged_at_most_every_five_seconds(self):
        self.bot.start()
        log = self.namespace["logger"]
        log.reset_mock()
        with patch("time.monotonic", return_value=10.0):
            self.bot._on_lie_worker_result(self._panel_snapshot())
            self.bot._on_lie_worker_result(self._panel_snapshot())
        with patch("time.monotonic", return_value=15.0):
            self.bot._on_lie_worker_result(self._panel_snapshot())
        messages = [call.args[0] for call in log.info.call_args_list]
        self.assertEqual(sum("Mouse HOLD" in text for text in messages), 2)
        self.assertIn("acquired=False", messages[0])
        self.assertIn("actionable=False", messages[0])
        self.namespace["move_mouse_in_game_window"].assert_not_called()

    def test_first_mouse_target_and_return_are_logged_once(self):
        self.bot.start()
        log = self.namespace["logger"]
        log.reset_mock()
        self.namespace["move_mouse_in_game_window"].return_value = (700, 450)
        for _ in range(5):
            self.bot._on_lie_worker_result(self._panel_snapshot((640.0, 336.0)))
        messages = [call.args[0] for call in log.info.call_args_list]
        self.assertEqual(sum("First mouse target" in text for text in messages), 1)
        self.assertEqual(sum("Mouse move call returned" in text for text in messages), 1)
        self.assertIn("requested_screen_position=(700, 450)", messages[-1])
        self.assertEqual(self.namespace["move_mouse_in_game_window"].call_count, 5)

    def test_disabled_control_is_logged_without_moving_mouse(self):
        self.bot.start()
        self.bot.is_disable_control = True
        self.bot._on_lie_worker_result(self._panel_snapshot((640.0, 336.0)))
        self.namespace["move_mouse_in_game_window"].assert_not_called()
        self.assertIn(
            "reason=disable_control",
            self.namespace["logger"].warning.call_args.args[0],
        )

    def test_jump_guard_logs_rejected_target_without_changing_guard(self):
        self.bot.start()
        self.bot._lie_last_pointer_target = (100.0, 100.0)
        self.bot._on_lie_worker_result(self._panel_snapshot((600.0, 300.0)))
        self.namespace["move_mouse_in_game_window"].assert_not_called()
        self.assertIn(
            "reason=jump_guard", self.namespace["logger"].warning.call_args.args[0]
        )
        self.bot._on_lie_worker_result(self._panel_snapshot((600.0, 300.0)))
        self.namespace["move_mouse_in_game_window"].assert_called_once()

    def test_mouse_request_diagnostics_include_location_and_source_timestamp(self):
        self.bot.start()
        self.namespace["move_mouse_in_game_window"].return_value = (700, 450)
        snapshot = self._panel_snapshot((640.0, 336.0))
        snapshot.captured_at = time.monotonic()
        self.bot._on_lie_worker_result(snapshot)
        timestamp, action, location, call_ms, completed_at = self.runtime.record_mouse_action.call_args.args
        self.assertEqual(timestamp, snapshot.captured_at)
        self.assertEqual(action, "move_requested")
        self.assertEqual(location, (700, 450))
        self.assertGreaterEqual(call_ms, 0.0)
        self.assertGreaterEqual(completed_at, timestamp)

    def test_jump_rejection_is_recorded_in_recent_diagnostics(self):
        self.bot.start()
        self.bot._lie_last_pointer_target = (100.0, 100.0)
        snapshot = self._panel_snapshot((600.0, 300.0))
        snapshot.captured_at = time.monotonic()
        self.bot._on_lie_worker_result(snapshot)
        self.assertEqual(self.runtime.record_mouse_action.call_args.args[1], "jump_guard")
        self.assertIsNone(self.runtime.record_mouse_action.call_args.args[2])
        self.namespace["move_mouse_in_game_window"].assert_not_called()

    def test_mouse_failure_is_recorded_and_still_raised(self):
        self.bot.start()
        snapshot = self._panel_snapshot((640.0, 336.0))
        snapshot.captured_at = time.monotonic()
        self.namespace["move_mouse_in_game_window"].side_effect = RuntimeError("mouse failed")
        with self.assertRaisesRegex(RuntimeError, "mouse failed"):
            self.bot._on_lie_worker_result(snapshot)
        self.assertEqual(self.runtime.record_mouse_action.call_args.args[1], "move_error")


if __name__ == "__main__":
    unittest.main()
