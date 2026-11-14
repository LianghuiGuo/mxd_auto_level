import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QCheckBox, QLineEdit
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("PySide6 is not installed in this test environment") from exc

from src.utils.ui import (
    create_advance_setting_gbox,
    update_advance_setting_gbox,
)


class AdvancedSettingsNestedConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.cfg = {
            "lie_detector": {
                "enabled": True,
                "alert": {
                    "enabled": True,
                    "sound": True,
                    "sound_repeat": 2,
                    "sound_interval_seconds": 0.18,
                    "sound_file": "",
                },
            }
        }
        self.gbox = create_advance_setting_gbox(
            "lie_detector", self.cfg
        )

    def test_nested_alert_fields_are_visible_and_write_back(self):
        refs = self.gbox._field_refs
        self.assertIn("alert.enabled", refs)
        self.assertIn("alert.sound_repeat", refs)
        self.assertIn("alert.sound_file", refs)

        self.assertIsInstance(refs["alert.enabled"], QCheckBox)
        self.assertIsInstance(refs["alert.sound_repeat"], QLineEdit)
        refs["alert.enabled"].setChecked(False)
        refs["alert.sound_repeat"].setText("4")
        refs["alert.sound_file"].setText(r"C:\Windows\Media\Alarm01.wav")

        self.assertFalse(self.cfg["lie_detector"]["alert"]["enabled"])
        self.assertEqual(
            self.cfg["lie_detector"]["alert"]["sound_repeat"], 4
        )
        self.assertEqual(
            self.cfg["lie_detector"]["alert"]["sound_file"],
            r"C:\Windows\Media\Alarm01.wav",
        )

    def test_nested_alert_fields_refresh_after_loading_config(self):
        values = {
            "enabled": True,
            "alert": {
                "enabled": False,
                "sound": False,
                "sound_repeat": 3,
                "sound_interval_seconds": 0.4,
                "sound_file": r"C:\alert.wav",
            },
        }

        update_advance_setting_gbox(self.gbox, values)

        refs = self.gbox._field_refs
        self.assertFalse(refs["alert.enabled"].isChecked())
        self.assertFalse(refs["alert.sound"].isChecked())
        self.assertEqual(refs["alert.sound_repeat"].text(), "3")
        self.assertEqual(
            refs["alert.sound_interval_seconds"].text(), "0.4"
        )
        self.assertEqual(refs["alert.sound_file"].text(), r"C:\alert.wav")

    def test_arbitrary_mapping_remains_hidden(self):
        cfg = {
            "route": {
                "search_range": 10,
                "color_code": {"255,0,0": "left none none"},
            }
        }

        gbox = create_advance_setting_gbox("route", cfg)

        self.assertIn("search_range", gbox._field_refs)
        self.assertNotIn("color_code.255,0,0", gbox._field_refs)


class LoggingSettingsUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from src.utils.logger import default_logging_config
        self.cfg = {"logging": default_logging_config()}
        self.gbox = create_advance_setting_gbox("logging", self.cfg)
        self.addCleanup(self.gbox.close)

    @patch("src.utils.ui.logger")
    def test_category_checkbox_updates_config_and_logger_immediately(self, log):
        refs = self.gbox._field_refs
        for field in ("categories.lie_detector", "categories.monsters", "categories.input_heartbeat"):
            self.assertIsInstance(refs[field], QCheckBox)
        refs["categories.monsters"].setChecked(False)
        self.assertFalse(self.cfg["logging"]["categories"]["monsters"])
        log.configure.assert_called_with(self.cfg["logging"])
        self.assertTrue(self.cfg["logging"]["categories"]["lie_detector"])

    @patch("src.utils.ui.logger")
    def test_loaded_logging_configuration_refreshes_widgets(self, log):
        self.cfg["logging"]["console"] = False
        self.cfg["logging"]["categories"]["input_heartbeat"] = False
        update_advance_setting_gbox(self.gbox, self.cfg["logging"])
        self.assertFalse(self.gbox._field_refs["console"].isChecked())
        self.assertFalse(self.gbox._field_refs["categories.input_heartbeat"].isChecked())
        log.configure.assert_called_with(self.cfg["logging"])
        restored = yaml.safe_load(yaml.safe_dump(self.cfg))
        self.assertEqual(restored, self.cfg)

    def test_qt_log_handler_emits_formatted_messages(self):
        import logging
        from src.utils.ui import QtLogHandler
        handler = QtLogHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        messages = []
        handler.log_signal.connect(lambda message, level: messages.append((message, level)))
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "SDK ready", (), None)
        handler.handle(record)
        self.app.processEvents()
        self.assertEqual(messages, [("INFO SDK ready", logging.INFO)])
        handler.close()

    def test_logging_settings_remain_editable_while_bot_runs(self):
        from PySide6.QtWidgets import QGroupBox
        path = Path(__file__).resolve().parents[1] / "src/ui/ui.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
        method = next(node for node in window.body if isinstance(node, ast.FunctionDef) and node.name == "set_gbox_enabled")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        controls = [QGroupBox() for _ in range(4)]
        other = QGroupBox()
        instance = SimpleNamespace(
            attack_gbox=controls[0], key_binding_gbox=controls[1],
            pet_skill_gbox=controls[2], map_selection_gbox=controls[3],
            advance_settings_gboxes={"logging": self.gbox, "other": other},
        )
        namespace["set_gbox_enabled"](instance, False)
        self.assertTrue(self.gbox.isEnabled())
        self.assertTrue(self.gbox._field_refs["categories.monsters"].isEnabled())
        self.assertFalse(other.isEnabled())
        self.assertFalse(controls[0].isEnabled())
        namespace["set_gbox_enabled"](instance, True)
        self.assertTrue(other.isEnabled())


if __name__ == "__main__":
    unittest.main()
