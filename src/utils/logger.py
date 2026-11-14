'''
Global Logger
'''
# Standard Import
import logging
import datetime
import os

LOG_CATEGORIES = (
    "lie_detector", "monsters", "player", "minimap", "input", "input_heartbeat",
    "buff", "route", "fsm", "chair_rest", "watchdog", "health", "rune",
    "capture", "recording", "performance", "channel", "party", "email",
    "config", "resources", "ui", "general",
)

_MODULE_CATEGORIES = {
    "ui.py": "ui",
    "AutoBotController.py": "ui",
    "main.py": "ui",
}

_SOURCE_CATEGORIES = {
    "MapleStoryAutoLevelUp.py": {
        "_ensure_nametag_template_loaded": "player",
        "_detect_player_location_cascade": "player",
        "_is_nametag_false_positive": "player",
        "get_player_location_by_nametag": "player",
        "_is_party_red_bar_false_positive": "player",
        "get_player_location_by_party_red_bar": "player",
        "get_player_location_on_global_map": "minimap",
        "get_nearest_color_code": "route",
        "_ensure_yolo_model": "monsters",
        "_run_yolo_full_frame": "monsters",
        "get_monsters_in_range": "monsters",
        "get_img_frame": "capture",
        "is_player_stuck": "watchdog",
        "update_cmd_by_random": "watchdog",
        "screenshot_img_frame": "recording",
        "update_info_on_img_frame_debug": "monsters",
        "update_cmd_by_mob_detection": "monsters",
        "update_cmd_by_route": "route",
        "_dump_minimap_debug": "minimap",
        "check_reach_goal": "route",
        "_on_lie_worker_result": "lie_detector",
        "_handle_lie_detector": "lie_detector",
        "load_config": "config",
        "enable_viz": "ui",
        "disable_viz": "ui",
        "start_record": "recording",
        "stop_record": "recording",
        "_write_recording_frame": "recording",
        "get_login_button_location": "capture",
        "channel_change": "channel",
        "is_need_change_channel": "channel",
        "ensure_is_in_party": "party",
        "loop": "ui",
    },
    "common.py": {
        "load_yaml": "config", "load_yaml_with_comments": "config", "save_yaml": "config",
        "load_image": "resources", "screenshot": "recording",
        "get_player_location_on_minimap": "minimap",
        "get_all_other_player_locations_on_minimap": "minimap",
        "debug_minimap_colors": "minimap", "mask_route_colors": "route",
        "normalize_pixel_coordinate": "config", "get_window_region_mac": "capture",
        "activate_game_window": "capture", "resize_window": "capture",
        "click_in_game_window": "input",
        "send_email": "email", "check_inbox": "email",
    },
}


def default_logging_config():
    return {
        "enabled": True,
        "level": "INFO",
        "console": True,
        "ui": True,
        "file": True,
        "keep_warnings_errors": True,
        "categories": {category: True for category in LOG_CATEGORIES},
    }


class _OutputFilter(logging.Filter):
    def __init__(self, owner, output):
        super().__init__()
        self.owner = owner
        self.output = output

    def filter(self, record):
        config = self.owner._config
        if not config["enabled"] or not config[self.output]:
            return False
        if config["keep_warnings_errors"] and record.levelno >= logging.WARNING:
            return True
        category = getattr(record, "category", None)
        filename = os.path.basename(record.pathname)
        if category is None:
            category = _SOURCE_CATEGORIES.get(filename, {}).get(
                record.funcName, _MODULE_CATEGORIES.get(filename, "general")
            )
        return config["categories"].get(category, config["categories"]["general"])


class _CategoryLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        category = kwargs.pop("category", self.extra["category"])
        kwargs["extra"] = {**kwargs.get("extra", {}), "category": category}
        return msg, kwargs


class MSLogger:
    '''
    MapleStory AutoBot Logger
    '''
    def __init__(self, name="MSBot"):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._config = default_logging_config()
        self._handlers = []

        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        os.makedirs("log", exist_ok=True)
        file_handler = logging.FileHandler(f"log/{name}_{now_str}.log", mode='w', encoding="utf-8")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.addHandler(file_handler, output="file")
        self.addHandler(console_handler, output="console")

        self._file_handler = file_handler
        self._console_handler = console_handler

    def configure(self, config):
        values = default_logging_config()
        for key in values:
            if key != "categories" and key in config:
                values[key] = config[key]
        values["categories"].update(config.get("categories", {}))
        self.set_level(values["level"])
        self._config = values

    def set_level(self, level):
        self._logger.setLevel(level)
        for handler in self._handlers:
            handler.setLevel(level)

    def for_category(self, category):
        return _CategoryLogger(self._logger, {"category": category})

    def info(self, msg, *, category=None):
        self._logger.info(msg, extra={"category": category}, stacklevel=2)

    def warning(self, msg, *, category=None):
        self._logger.warning(msg, extra={"category": category}, stacklevel=2)

    def error(self, msg, *, category=None):
        self._logger.error(msg, extra={"category": category}, stacklevel=2)

    def debug(self, msg, *, category=None):
        self._logger.debug(msg, extra={"category": category}, stacklevel=2)

    def addHandler(self, handle, *, output="ui"):
        handle.setLevel(self._logger.level)
        handle.addFilter(_OutputFilter(self, output))
        self._handlers.append(handle)
        self._logger.addHandler(handle)

# Initialize shared logger instance for global import
logger = MSLogger()
