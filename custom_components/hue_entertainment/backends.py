"""Output backends for normalized Ambilight channel frames.

Legacy HueStream and JointSpace inputs share these destinations, keeping input
transport independent from the physical output transport.
"""

from __future__ import annotations

import asyncio
import colorsys
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib import import_module
from typing import Any

from .const import COLOR_SPACE_XY, DEFAULT_STREAM_FPS
from .entertainment import ChannelColor, EntertainmentEngine

_LOGGER = logging.getLogger(__name__)

# Home Assistant installs manifest requirements before loading this module and
# imports integrations in its import executor. Resolve the external package once
# here so no import work can occur later in an async session/frame path. getattr
# avoids the static-name collision with this integration's internal package.
_HUE_LIBRARY = import_module("hue_entertainment")
EntertainmentSession = getattr(_HUE_LIBRARY, "EntertainmentSession")
LightColorCommand = getattr(_HUE_LIBRARY, "LightColorCommand")


class EntertainmentOutputBackend(ABC):
    """A destination for normalized HueStream channel colours."""

    @abstractmethod
    async def async_start(self) -> None: ...

    @abstractmethod
    def send_frame(self, channels: list[ChannelColor], color_space: int) -> None: ...

    @abstractmethod
    async def async_stop(self) -> None: ...

    def clear_pending(self) -> None:
        """Discard any scheduled output when sync is suppressed."""

    async def async_close(self) -> None:
        """Release resources. Backends without resources need no override."""

    @property
    def available(self) -> bool:
        """Whether this output has enough configuration to run."""
        return True

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether this output currently has an active connection/session."""

    def handle_light_command(self, light_id: int, body: dict[str, Any]) -> None:
        """Accept a classic Hue REST command when supported by the backend."""


class HomeAssistantLightBackend(EntertainmentOutputBackend):
    """The legacy HA/ZHA output path, including state restore and coalescing."""

    def __init__(self, engine: EntertainmentEngine) -> None:
        self.engine = engine

    async def async_start(self) -> None:
        await self.engine.async_snapshot_lights()

    def send_frame(self, channels: list[ChannelColor], color_space: int) -> None:
        self.engine.handle_channels(channels, color_space)

    async def async_stop(self) -> None:
        await self.engine.async_restore_lights()

    @property
    def connected(self) -> bool:
        return self.engine.is_active or self.engine.is_driving_lights

    def handle_light_command(self, light_id: int, body: dict[str, Any]) -> None:
        """Keep the non-streaming Hue v1 compatibility path on HA lights."""
        self.engine.handle_light_command(light_id, body)


class DisabledOutputBackend(EntertainmentOutputBackend):
    """Safe placeholder while a staged physical Hue setup is incomplete."""

    async def async_start(self) -> None:
        return

    def send_frame(self, channels: list[ChannelColor], color_space: int) -> None:
        return

    async def async_stop(self) -> None:
        return

    @property
    def available(self) -> bool:
        return False

    @property
    def connected(self) -> bool:
        return False

    @property
    def stats(self) -> dict[str, Any]:
        return {"configured": False, "type": "incomplete_hue_output"}


class HueEntertainmentBackend(EntertainmentOutputBackend):
    """Native DTLS Hue Entertainment output to a physical Hue Bridge.

    ``EntertainmentSession.send`` is intentionally non-blocking. It delegates
    the socket work to the library's sender thread, so this backend never turns
    input frames into Home Assistant ``light.turn_on`` service calls.
    """

    def __init__(
        self,
        host: str,
        app_key: str,
        client_key: str,
        area_id: str,
        channel_map: dict[int, int],
        *,
        fps_cap: int = DEFAULT_STREAM_FPS,
        smoothing_ms: int = 0,
        output_allowed: Callable[[], bool] = lambda: True,
        brightness_multiplier: float = 1.0,
        saturation_multiplier: float = 1.0,
    ) -> None:
        self._host = host
        self._app_key = app_key
        self._client_key = client_key
        self._area_id = area_id
        self.channel_map = channel_map
        self._fps_cap = max(1, fps_cap)
        self._brightness_multiplier = max(0.0, brightness_multiplier)
        self._saturation_multiplier = max(0.0, saturation_multiplier)
        self._session: Any = None
        self._tau = smoothing_ms / 1000
        self._output_allowed = output_allowed
        self._targets: dict[int, tuple[int, int, int]] = {}
        self._values: dict[int, tuple[float, float, float]] = {}
        self._last_output: dict[int, tuple[int, int, int]] = {}
        self._last_step: float | None = None
        self._next_send = 0.0
        self._last_submission = 0.0
        self._tick: asyncio.TimerHandle | None = None
        self._targets_replaced = 0
        self._target_pending = False
        self._frames_sent = 0
        self._reconnects = 0
        self._restart_required = False
        self._start_lock = asyncio.Lock()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "host": self._host,
            "area_id": self._area_id,
            "channel_mapping": self.channel_map,
            "frames_sent": self._frames_sent,
            "targets_replaced": self._targets_replaced,
            "reconnects": self._reconnects,
            "streaming": bool(self._session and self._session.is_streaming),
        }

    @property
    def connected(self) -> bool:
        return bool(self._session and self._session.is_streaming and not self._restart_required)

    async def async_start(self) -> None:
        async with self._start_lock:
            if self.connected:
                return
            self.clear_pending()
            if self._restart_required and self._session is not None:
                await self._session.aclose()
                self._session = None
            if self._session is None:
                self._session = EntertainmentSession(
                    self._host, self._app_key, self._client_key, idle_timeout=0
                )
            try:
                await self._session.start(self._area_id)
            except Exception as err:  # bridge may be offline; next TV start retries
                self._reconnects += 1
                self._restart_required = True
                _LOGGER.warning("Unable to start physical Hue Entertainment stream: %s", err)
                raise
            self._restart_required = False
            _LOGGER.debug("Physical Hue Entertainment session started for area %s", self._area_id)

    def send_frame(self, channels: list[ChannelColor], color_space: int) -> None:
        if not self.connected or not self._output_allowed():
            self.clear_pending()
            return

        now = asyncio.get_running_loop().time()
        self._advance(now)
        targets = {}
        for channel in channels:
            physical_channel = self.channel_map.get(channel.channel_id)
            if physical_channel is None:
                continue
            rgb = self._adjust(*_as_rgb(channel, color_space))
            targets[physical_channel] = rgb
            self._values.setdefault(physical_channel, (float(rgb[0]), float(rgb[1]), float(rgb[2])))

        if not targets:
            return
        if self._target_pending:
            self._targets_replaced += 1
        self._target_pending = True
        self._targets.update(targets)
        self._schedule_tick(now)

    def _advance(self, now: float) -> None:
        previous, self._last_step = self._last_step, now
        if not self._tau or previous is None:
            return
        alpha = -math.expm1(-(now - previous) / self._tau)

        def blend(current: float, target: int) -> float:
            value = current + alpha * (target - current)
            return float(target) if abs(value - target) < 0.5 else value

        for channel, (red, green, blue) in self._targets.items():
            r, g, b = self._values[channel]
            self._values[channel] = blend(r, red), blend(g, green), blend(b, blue)

    def _schedule_tick(self, now: float) -> None:
        if self._tick is None:
            self._tick = asyncio.get_running_loop().call_at(
                max(now, self._next_send), self._send_tick
            )

    def _send_tick(self) -> None:
        self._tick = None
        self._target_pending = False
        if not self.connected or not self._output_allowed():
            self.clear_pending()
            return
        now = asyncio.get_running_loop().time()
        self._advance(now)
        values = self._values if self._tau else self._targets
        output = {
            channel: (_wire_component(r), _wire_component(g), _wire_component(b))
            for channel, (r, g, b) in values.items()
        }
        self._next_send = now + 1 / self._fps_cap
        if output != self._last_output or now - self._last_submission >= 1.0:
            commands = [
                LightColorCommand(channel_id=channel, red=r, green=g, blue=b)
                for channel, (r, g, b) in output.items()
            ]
            try:
                self._session.send(commands)
            except Exception:
                self._restart_required = True
                self.clear_pending()
                _LOGGER.debug("Hue Entertainment send failed; session will be recreated", exc_info=True)
                return
            self._last_submission = now
            self._last_output = output
            self._frames_sent += 1
        if self._tau and self._values != self._targets:
            self._schedule_tick(now)

    def clear_pending(self) -> None:
        """Forget pending colours and reset the filter at session boundaries."""
        if self._tick is not None:
            self._tick.cancel()
            self._tick = None
        self._target_pending = False
        self._targets.clear()
        self._values.clear()
        self._last_output.clear()
        self._last_step = None
        self._next_send = 0.0

    async def async_stop(self) -> None:
        self.clear_pending()
        if self._session is not None:
            await self._session.stop()
            self._restart_required = False
            _LOGGER.debug("Physical Hue Entertainment session stopped")

    async def async_close(self) -> None:
        self.clear_pending()
        if self._session is not None:
            await self._session.aclose()
            self._session = None
            self._restart_required = False

    def _adjust(self, red: int, green: int, blue: int) -> tuple[int, int, int]:
        if self._brightness_multiplier == self._saturation_multiplier == 1.0:
            return red, green, blue
        hue, saturation, value = colorsys.rgb_to_hsv(red / 65535, green / 65535, blue / 65535)
        adjusted_red, adjusted_green, adjusted_blue = colorsys.hsv_to_rgb(
            hue,
            min(1.0, saturation * self._saturation_multiplier),
            min(1.0, value * self._brightness_multiplier),
        )
        return (
            round(adjusted_red * 65535),
            round(adjusted_green * 65535),
            round(adjusted_blue * 65535),
        )


def _wire_component(value: float) -> int:
    """Pass an explicit byte to the pinned encoder, including near black."""
    # 0.1.2 interprets values <= 255 as byte values, and shifts larger values.
    # Quantize once here to avoid a near-black 255 becoming full intensity.
    return round(value / 257)


def _as_rgb(channel: ChannelColor, color_space: int) -> tuple[int, int, int]:
    """Convert source RGB or CIE xy + brightness to 16-bit sRGB."""
    if color_space != COLOR_SPACE_XY:
        red, green, blue = channel.r, channel.g, channel.b
    else:
        x, y, bri = channel.r / 65535, channel.g / 65535, channel.b / 65535
        if y <= 0:
            return (0, 0, 0)
        # CIE xyY -> linear RGB (D65), then gamma encode.
        y_lum = bri
        x_val, z_val = x * y_lum / y, (1 - x - y) * y_lum / y
        linear = (
            3.2406 * x_val - 1.5372 * y_lum - 0.4986 * z_val,
            -0.9689 * x_val + 1.8758 * y_lum + 0.0415 * z_val,
            0.0557 * x_val - 0.2040 * y_lum + 1.0570 * z_val,
        )
        rgb = [max(0.0, value) for value in linear]
        peak = max(rgb, default=0.0)
        if peak > 1:
            rgb = [value / peak for value in rgb]
        red, green, blue = (round(_gamma(value) * 65535) for value in rgb)
    return (red, green, blue)


def _gamma(value: float) -> float:
    return 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055
