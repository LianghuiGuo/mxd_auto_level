import os
import unittest

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


if __name__ == "__main__":
    unittest.main()
