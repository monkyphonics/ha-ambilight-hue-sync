"""Deterministic tests for physical Hue colour scheduling."""

import math
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from custom_components.hue_entertainment import backends
from custom_components.hue_entertainment.const import COLOR_SPACE_RGB
from custom_components.hue_entertainment.entertainment import ChannelColor


class Clock:
    def __init__(self):
        self.now = 1.0
        self.pending = []

    def time(self):
        return self.now

    def call_at(self, deadline, callback):
        handle = SimpleNamespace(cancelled=False)
        handle.cancel = lambda: setattr(handle, "cancelled", True)
        self.pending.append((deadline, callback, handle))
        return handle

    def run_at(self, now):
        self.now = now
        pending, self.pending = self.pending, []
        for deadline, callback, handle in pending:
            if handle.cancelled:
                continue
            if deadline <= now:
                callback()
            else:
                self.pending.append((deadline, callback, handle))


class TestPhysicalOutput(TestCase):
    def setUp(self):
        self.clock = Clock()
        self.allowed = True
        self.frames = []
        commands = patch.object(
            backends, "LightColorCommand", lambda **values: SimpleNamespace(**values)
        )
        commands.start()
        self.addCleanup(commands.stop)
        get_loop = patch.object(backends.asyncio, "get_running_loop", return_value=self.clock)
        get_loop.start()
        self.addCleanup(get_loop.stop)

    def make_backend(self, smoothing_ms):
        backend = backends.HueEntertainmentBackend(
            "bridge",
            "key",
            "client",
            "area",
            {1: 9, 2: 42},
            smoothing_ms=smoothing_ms,
            output_allowed=lambda: self.allowed,
        )
        backend._session = SimpleNamespace(is_streaming=True, send=self.frames.append)
        return backend

    def test_latest_frame_survives_cap_and_no_burst_after_stall(self):
        backend = self.make_backend(0)
        backend.send_frame([ChannelColor(1, 0, 0, 0)], COLOR_SPACE_RGB)
        self.clock.run_at(1.0)
        for value in (12000, 65535):
            backend.send_frame([ChannelColor(1, value, 0, 0)], COLOR_SPACE_RGB)
        self.clock.run_at(1.01)
        self.assertEqual(len(self.frames), 1)
        self.clock.run_at(1.2)
        self.assertEqual([frame[0].red for frame in self.frames], [0, 255])

    def test_smoothing_uses_elapsed_time_and_preserves_other_channels(self):
        backend = self.make_backend(60)
        backend.send_frame(
            [ChannelColor(1, 0, 0, 0), ChannelColor(2, 0, 65535, 0)], COLOR_SPACE_RGB
        )
        self.clock.run_at(1.0)
        self.clock.run_at(1.1)
        backend.send_frame([ChannelColor(1, 65535, 0, 0)], COLOR_SPACE_RGB)
        self.clock.run_at(1.12)
        self.clock.run_at(1.16)
        self.assertEqual(len(self.frames), 3)
        for frame, elapsed in zip(self.frames[1:], (0.02, 0.06), strict=True):
            self.assertEqual([command.channel_id for command in frame], [9, 42])
            self.assertEqual(frame[0].red, round(255 * (1 - math.exp(-elapsed / 0.06))))
            self.assertEqual(frame[1].green, 255)

    def test_suppression_clears_pending_target_and_filter(self):
        backend = self.make_backend(60)
        backend.send_frame([ChannelColor(1, 65535, 0, 0)], COLOR_SPACE_RGB)
        self.allowed = False
        self.clock.run_at(1.02)
        self.assertEqual(self.frames, [])
        self.allowed = True
        backend.send_frame([ChannelColor(1, 0, 0, 0)], COLOR_SPACE_RGB)
        self.clock.run_at(1.04)
        self.assertEqual(self.frames[0][0].red, 0)

    def test_near_black_commands_use_monotonic_wire_bytes(self):
        backend = self.make_backend(0)
        for index, value in enumerate((0, 1, 254, 255, 256, 257, 65535)):
            backend.send_frame([ChannelColor(1, value, 0, 0)], COLOR_SPACE_RGB)
            self.clock.run_at(2.0 + index * 0.1)
        self.assertEqual([frame[0].red for frame in self.frames], [0, 1, 255])
