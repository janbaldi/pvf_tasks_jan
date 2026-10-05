"""One logger, and a record of everything it was told.

Modules import ``log`` and call ``log.step`` / ``info`` / ``success`` / ``warn``
/ ``error``. Every event is kept in memory as well as printed, so the report can
show the run without the pipeline having to run twice.

The console gets aligned columns and colour when stdout is a terminal: light
blue for the structure, green, yellow and red only for OK, warnings and errors.
Set ``NO_COLOR`` to turn colour off, or ``FORCE_COLOR`` to keep it when piping.
No icons: the structure is carried by the rules, the columns and the colour.
While a step is computing and nothing has been printed for a moment, a small
progress bar animates under the last line; it only runs on a real terminal, so a
log file or a pipe never sees it. The stage rules, section rules and result blocks
(``stage``, ``section``, ``facts``, ``names``, ``summary``) are printed only.
They repeat what the events and the report already hold, so they are not stored
as events.
"""

import atexit
import logging
import os
import shutil
import sys
import textwrap
import threading
import time
from datetime import datetime
from typing import Optional

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[94m",  # light blue: the default colour for everything structural
}

#: level → (label on screen, colour of the label, colour of the message)
_LEVELS = {
    "STEP": ("STEP", "blue", ""),
    "INFO": ("INFO", "dim", ""),
    "SUCCESS": ("OK", "green", ""),
    "WARN": ("WARN", "yellow", "yellow"),
    "ERROR": ("ERROR", "red", "red"),
}
_MODULE_WIDTH = 11
_INDENT = " " * (8 + 2 + 5 + 2 + _MODULE_WIDTH + 1)

#: The rule characters, and plain ASCII for a console whose code page lacks
#: them: a Windows console on cp1252 has no ━, and writing one raises.
GLYPHS = {"heavy": "━", "light": "─", "times": "×"}
ASCII = {"heavy": "=", "light": "-", "times": "x"}


def _encodes(text: str, encoding: str) -> bool:
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _colour_enabled(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not (hasattr(stream, "isatty") and stream.isatty()):
        return False
    if os.name == "nt":
        # A Windows console shows ANSI codes as text until virtual terminal
        # processing is switched on for it.
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # stdout
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except (AttributeError, OSError):
            return False
    return True


class _Scanner:
    """A bar sliding to and fro, ``[  -=#   ] working 4s``, while the console is quiet.

    Plain ASCII, so every console can draw it. It runs on a thread of its own
    and only draws after ``DELAY`` seconds of silence, so a quick step never
    flickers. Anything the logger prints stops it first and wipes its line, so
    the two never write over each other.
    """

    DELAY = 0.5
    TICK = 0.1
    TRACK = 8

    def __init__(self, logger: "PipelineLogger"):
        self._logger = logger
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def frame(self, tick: int) -> str:
        span = self.TRACK - 1
        position = tick % (2 * span)
        direction = 1 if position < span else -1
        head = position if direction == 1 else 2 * span - position
        cells = [" "] * self.TRACK
        for offset, glyph in ((2, "-"), (1, "="), (0, "#")):
            cell = head - offset * direction
            if 0 <= cell < self.TRACK:
                cells[cell] = glyph
        return "".join(cells)

    def start(self) -> None:
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        started = time.monotonic()
        if self._stop.wait(self.DELAY):
            return
        tick, width = 0, 0
        while not self._stop.is_set():
            elapsed = int(time.monotonic() - started)
            text = f"{_INDENT}[{self.frame(tick)}] working {elapsed}s"
            width = len(text)
            sys.stdout.write("\r" + self._logger._safe(self._logger._paint(text, "blue")))
            sys.stdout.flush()
            tick += 1
            self._stop.wait(self.TICK)
        # Spaces rather than an erase code: a console without ANSI support
        # would print the code as text.
        sys.stdout.write("\r" + " " * width + "\r")
        sys.stdout.flush()


class PipelineEvent:
    """A single structured log event."""

    __slots__ = ("ts", "level", "module", "message", "detail")

    def __init__(
        self,
        level: str,
        module: str,
        message: str,
        detail: Optional[str] = None,
    ):
        self.ts = datetime.now()
        self.level = level  # STEP | INFO | SUCCESS | WARN | ERROR
        self.module = module
        self.message = message
        self.detail = detail


class PipelineLogger:
    """
    Thin wrapper around stdlib logging that also stores structured events
    for later HTML report generation.
    """

    def __init__(self):
        self.events: list[PipelineEvent] = []
        self._logger = logging.getLogger("pvf_pipeline")
        self._logger.setLevel(logging.DEBUG)

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

        # Prevent duplicate output via root logger
        self._logger.propagate = False
        self.colour = _colour_enabled(sys.stdout)
        self.animate = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        # Decided once, from the stream itself, so it holds however the run was
        # started: the pvf command, python -m pvf, a notebook or a script.
        self._encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        self.glyph = GLYPHS if _encodes("".join(GLYPHS.values()), self._encoding) else ASCII
        self._scanner = _Scanner(self)
        atexit.register(self._scanner.stop)

    # ── public API ─────────────────────────────────────────────────────────────

    def step(self, module: str, message: str, detail: str | None = None):
        self._emit("STEP", module, message, detail)

    def info(self, module: str, message: str, detail: str | None = None):
        self._emit("INFO", module, message, detail)

    def success(self, module: str, message: str, detail: str | None = None):
        self._emit("SUCCESS", module, message, detail)

    def warn(self, module: str, message: str, detail: str | None = None):
        self._emit("WARN", module, message, detail)

    def error(self, module: str, message: str, detail: str | None = None):
        self._emit("ERROR", module, message, detail)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _paint(self, text: str, *styles: str) -> str:
        codes = "".join(_ANSI[s] for s in styles if s)
        return f"{codes}{text}{_ANSI['reset']}" if self.colour and codes else text

    @staticmethod
    def _width() -> int:
        return max(60, min(shutil.get_terminal_size((100, 20)).columns, 140))

    def _safe(self, text: str) -> str:
        """The text as this console can print it: what its code page lacks becomes "?"."""
        return text.encode(self._encoding, "replace").decode(self._encoding)

    def _print(self, text: str = "") -> None:
        self._scanner.stop()
        # Messages carry arrows and the like from all over the pipeline; one
        # that the console cannot encode must not become a logging error.
        self._logger.info(self._safe(text))

    def _rule(self, head: str, char: str, *styles: str) -> None:
        self._print()
        self._print(self._paint(head + char * max(3, self._width() - len(head)), *styles))

    def _emit(self, level: str, module: str, message: str, detail: str | None):
        event = PipelineEvent(level, module, message, detail)
        self.events.append(event)

        label, label_colour, message_colour = _LEVELS[level]
        prefix = (
            f"{self._paint(event.ts.strftime('%H:%M:%S'), 'dim')}  "
            f"{self._paint(f'{label:<5}', label_colour, 'bold')}  "
            f"{self._paint(f'{module:<{_MODULE_WIDTH}}', 'dim')} "
        )
        self._print(prefix + self._paint(message, message_colour))
        if detail:
            wrapped = textwrap.fill(
                str(detail),
                width=self._width(),
                initial_indent=_INDENT,
                subsequent_indent=_INDENT,
                break_on_hyphens=False,
                break_long_words=False,  # a path stays one piece, even past the edge
            )
            self._print(self._paint(wrapped, "dim"))
        if self.animate:
            # Whatever comes next may take a while; the scanner fills the wait.
            self._scanner.start()

    def mark(self) -> int:
        """Where the log stands now.

        A stage passes this to :meth:`since` when it builds its report, so the
        events of an earlier stage — or of an earlier call in the same process —
        cannot turn up in it and be counted as this run's warnings.
        """
        return len(self.events)

    def since(self, mark: int) -> list["PipelineEvent"]:
        """Every event logged after a mark. This run's log, and only it."""
        return self.events[mark:]

    def stage(self, title: str):
        """A heavy rule opening a stage."""
        heavy = self.glyph["heavy"]
        self._rule(f"{heavy * 2} {title} ", heavy, "blue", "bold")

    def section(self, title: str):
        """A light rule between the steps of a stage."""
        light = self.glyph["light"]
        self._rule(f"{light * 2} {title} ", light, "blue")

    def facts(self, title: str, rows: list[tuple[str, str]]):
        """A titled block of label/value lines, labels aligned."""
        self._print()
        self._print(self._paint(title, "blue", "bold"))
        width = max((len(label) for label, _ in rows), default=0)
        for label, value in rows:
            self._print(f"  {self._paint(f'{label:<{width}}', 'dim')}  {value}")

    def names(self, title: str, names: list[str], level: str = "WARN"):
        """A heading with its count, then one name per line under it."""
        colour = _LEVELS[level][2] or "green"
        self._print()
        self._print(self._paint(f"{title} ({len(names):,})", colour, "bold"))
        for name in names:
            self._print(f"  {self._paint('-', colour)} {name}")

    def summary(self, mark: int = 0):
        """How the stage went: duration, counts, and each warning or error again."""
        events = self.events[mark:]
        problems = [e for e in events if e.level in ("WARN", "ERROR")]
        warns = sum(1 for e in problems if e.level == "WARN")
        errors = len(problems) - warns
        elapsed = str(datetime.now() - events[0].ts).split(".")[0] if events else "0:00:00"

        colour = "red" if errors else "yellow" if warns else "green"
        heavy = self.glyph["heavy"]
        self._rule(
            f"{heavy * 2} Finished in {elapsed} · {warns} warnings · {errors} errors ",
            heavy,
            colour,
            "bold",
        )
        # A warning raised once per fold or per site is listed once, with its count.
        repeats: dict[tuple[str, str, str], int] = {}
        for event in problems:
            key = (event.level, event.module, event.message)
            repeats[key] = repeats.get(key, 0) + 1
        for (level, module, message), count in repeats.items():
            label, label_colour, _ = _LEVELS[level]
            times = self._paint(f"  ({self.glyph['times']}{count})", "dim") if count > 1 else ""
            self._print(
                f"  {self._paint(f'{label:<5}', label_colour, 'bold')}  "
                f"{self._paint(f'{module:<{_MODULE_WIDTH}}', 'dim')} {message}{times}"
            )


# Singleton — import and use everywhere
log = PipelineLogger()
