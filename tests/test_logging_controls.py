import io
import logging
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import uuid

import yaml

from src.utils.logger import LOG_CATEGORIES, MSLogger, default_logging_config


class LoggingControlsTest(unittest.TestCase):
    def setUp(self):
        self.file_stream = io.StringIO()
        self.console_stream = io.StringIO()
        with patch("src.utils.logger.logging.FileHandler", return_value=logging.StreamHandler(self.file_stream)):
            with patch("sys.stderr", self.console_stream):
                self.log = MSLogger("test_logging_" + uuid.uuid4().hex)
        self.ui_stream = io.StringIO()
        self.log.addHandler(logging.StreamHandler(self.ui_stream))
        self.addCleanup(self._close_handlers)

    def _close_handlers(self):
        for handler in list(self.log._logger.handlers):
            self.log._logger.removeHandler(handler)
            handler.close()

    def _clear(self):
        for stream in (self.file_stream, self.console_stream, self.ui_stream):
            stream.seek(0)
            stream.truncate(0)

    def test_all_categories_enabled_by_default(self):
        for category in LOG_CATEGORIES:
            self.log.for_category(category).info("category=" + category)
        for stream in (self.file_stream, self.console_stream, self.ui_stream):
            for category in LOG_CATEGORIES:
                self.assertIn("category=" + category, stream.getvalue())

    def test_every_category_can_be_disabled_on_all_outputs(self):
        for category in LOG_CATEGORIES:
            with self.subTest(category=category):
                self._clear()
                self.log.configure({"categories": {category: False}})
                self.log.for_category(category).info("hidden")
                self.log.info("hidden explicit", category=category)
                for stream in (self.file_stream, self.console_stream, self.ui_stream):
                    self.assertEqual(stream.getvalue(), "")

    def test_only_lie_detector_enabled_hides_other_info(self):
        config = default_logging_config()
        config["categories"] = {category: category == "lie_detector" for category in LOG_CATEGORIES}
        self.log.configure(config)
        for category in LOG_CATEGORIES:
            self.log.info("category=" + category, category=category)
        self.assertIn("category=lie_detector", self.console_stream.getvalue())
        self.assertNotIn("category=monsters", self.console_stream.getvalue())
        self.assertNotIn("category=input_heartbeat", self.console_stream.getvalue())

    def test_disabled_categories_keep_warnings_and_errors_by_default(self):
        self.log.configure({"categories": {"monsters": False}})
        self.log.info("hidden", category="monsters")
        self.log.warning("warning retained", category="monsters")
        self.log.error("error retained", category="monsters")
        for stream in (self.file_stream, self.console_stream, self.ui_stream):
            self.assertNotIn("hidden", stream.getvalue())
            self.assertIn("warning retained", stream.getvalue())
            self.assertIn("error retained", stream.getvalue())

    def test_warning_protection_can_be_disabled(self):
        self.log.configure({"keep_warnings_errors": False, "categories": {"monsters": False}})
        self.log.warning("hidden", category="monsters")
        self.log.error("hidden", category="monsters")
        self.assertEqual(self.console_stream.getvalue(), "")
        self.assertEqual(self.file_stream.getvalue(), "")
        self.assertEqual(self.ui_stream.getvalue(), "")

    def test_master_switch_disables_every_level(self):
        self.log.configure({"enabled": False})
        self.log.info("hidden")
        self.log.warning("hidden")
        self.log.error("hidden")
        for stream in (self.file_stream, self.console_stream, self.ui_stream):
            self.assertEqual(stream.getvalue(), "")

    def test_output_channels_are_independent(self):
        streams = {"file": self.file_stream, "console": self.console_stream, "ui": self.ui_stream}
        for channel, stream in streams.items():
            with self.subTest(channel=channel):
                self._clear()
                self.log.configure({channel: False})
                self.log.error("channel test")
                self.assertEqual(stream.getvalue(), "")
                for other, other_stream in streams.items():
                    if other != channel:
                        self.assertIn("channel test", other_stream.getvalue())

    def test_minimum_level_applies_to_existing_and_new_handlers(self):
        self.log.configure({"level": "ERROR"})
        new_stream = io.StringIO()
        self.log.addHandler(logging.StreamHandler(new_stream), output="ui")
        self.log.warning("hidden")
        self.log.error("visible")
        for stream in (self.file_stream, self.console_stream, self.ui_stream, new_stream):
            self.assertNotIn("hidden", stream.getvalue())
            self.assertIn("visible", stream.getvalue())

    def test_category_adapter_can_override_its_default(self):
        self.log.configure({"categories": {"input": False}})
        adapter = self.log.for_category("input")
        adapter.info("hidden")
        adapter.info("heartbeat visible", category="input_heartbeat")
        self.assertNotIn("hidden", self.console_stream.getvalue())
        self.assertIn("heartbeat visible", self.console_stream.getvalue())

    def test_runtime_reconfiguration_and_new_handlers_follow_switches(self):
        self.log.configure({"categories": {"lie_detector": False}})
        new_stream = io.StringIO()
        self.log.addHandler(logging.StreamHandler(new_stream), output="console")
        self.log.info("hidden", category="lie_detector")
        self.log.configure({"categories": {"lie_detector": True}})
        self.log.info("visible", category="lie_detector")
        self.assertEqual(new_stream.getvalue(), "visible\n")

    def test_source_function_categories_do_not_depend_on_message_text(self):
        self.log.configure({"keep_warnings_errors": False, "categories": {"monsters": False, "resources": False}})
        for filename, function in (("MapleStoryAutoLevelUp.py", "_ensure_yolo_model"), ("common.py", "load_image")):
            record = logging.LogRecord("test", logging.INFO, filename, 1, "unrelated text", (), None, function)
            self.log._logger.handle(record)
        self.assertEqual(self.file_stream.getvalue(), "")
        self.assertEqual(self.console_stream.getvalue(), "")

    def test_ui_source_logs_are_classified(self):
        self.log.configure({"categories": {"ui": False}})
        record = logging.LogRecord("test", logging.INFO, "src/ui/ui.py", 1, "ui hidden", (), None, "load_ui_state")
        self.log._logger.handle(record)
        self.assertEqual(self.console_stream.getvalue(), "")

    def test_wrapper_records_the_real_caller_for_source_classification(self):
        namespace = {"logger": self.log}
        code = compile(
            "def _ensure_yolo_model():\n    logger.info('hidden model log')\n",
            "src/engine/MapleStoryAutoLevelUp.py", "exec",
        )
        exec(code, namespace)
        self.log.configure({"categories": {"monsters": False}})
        namespace["_ensure_yolo_model"]()
        self.assertEqual(self.console_stream.getvalue(), "")
        self.log.configure({"categories": {"monsters": True}})
        namespace["_ensure_yolo_model"]()
        self.assertIn("hidden model log", self.console_stream.getvalue())

    def test_unknown_categories_follow_general_switch(self):
        self.log.configure({"categories": {"general": False}})
        self.log.info("uncategorized hidden")
        self.log.info("unknown hidden", category="unknown")
        self.assertEqual(self.console_stream.getvalue(), "")

    def test_multiline_logs_are_filtered_as_one_record(self):
        self.log.configure({"categories": {"minimap": False}})
        self.log.info("minimap\nArtifacts written to log/example.png", category="minimap")
        self.assertEqual(self.console_stream.getvalue(), "")

    def test_default_yaml_exposes_every_category(self):
        config = yaml.safe_load(Path("config/config_default.yaml").read_text(encoding="utf-8"))["logging"]
        self.assertEqual(config, default_logging_config())

    def test_shared_logger_remains_usable_from_multiple_threads(self):
        self.log.configure({"categories": {"monsters": False}})
        threads = [threading.Thread(target=lambda: self.log.info("visible", category="lie_detector")) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(self.ui_stream.getvalue().count("visible"), 5)


if __name__ == "__main__":
    unittest.main()
