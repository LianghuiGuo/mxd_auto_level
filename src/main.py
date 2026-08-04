# Standard Import
import sys
import os

# Pyside
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import qInstallMessageHandler, QtMsgType

# Load Import
from src.ui.ui import MainWindow
from src.ui.AutoBotController import AutoBotController


# Keywords that identify the harmless DPI awareness warning printed by Qt6
# on Windows when the process DPI context is already locked (e.g. by Python.exe
# manifest, compatibility settings, or the host shell). Suppressing this one
# message is safer than fighting Qt/Windows with competing DPI API calls.
_DPI_WARNING_KEYWORDS = (
    "SetProcessDpiAwarenessContext",
    "DPI_AWARENESS_CONTEXT",
    "highdpi.html#configuring-windows",
)


def _qt_message_handler(msg_type, context, message):
    """Custom Qt message handler. Silently drops DPI-awareness warnings.

    Works with both PySide6 enum access styles:
      - QtMsgType.QtWarningMsg  (enum-member style)
      - QtWarningMsg             (legacy int-constant style)
    and avoids hard-coding enum names so it stays robust across versions.
    """
    msg = str(message)
    # Only DPI-related warnings are swallowed. Everything else is forwarded
    # so we never hide real Qt issues.
    if any(kw in msg for kw in _DPI_WARNING_KEYWORDS):
        return
    # Determine a short prefix for the remaining (non-DPI) messages based on
    # the numeric value of msg_type (QtMsgType is IntEnum under the hood).
    #   0=Debug, 1=Info, 2=Warning, 3=Critical, 4=Fatal
    numeric = int(msg_type) if hasattr(msg_type, "__int__") else -1
    level_prefix = {0: "D", 1: "I", 2: "W", 3: "C", 4: "F"}.get(numeric, "?")
    print(f"[Qt{level_prefix}] {msg}", file=sys.stderr)


def main():
    '''
    Main Function
    Run: python -m src.main
    '''
    # Install the handler BEFORE creating QApplication, so Qt's very first
    # messages (including the DPI one during platform plugin init) are filtered.
    qInstallMessageHandler(_qt_message_handler)

    app = QApplication(sys.argv)

    autoBotController = AutoBotController()
    ui = MainWindow(autoBotController)

    autoBotController.update_signal(ui)

    ui.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
