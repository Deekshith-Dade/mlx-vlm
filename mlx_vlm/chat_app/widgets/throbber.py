from time import monotonic
from typing import Callable

from rich.segment import Segment
from rich.style import Style as RichStyle

from textual.visual import Visual
from textual.color import Color, Gradient

from textual.style import Style
from textual.strip import Strip
from textual.visual import RenderOptions
from textual.widget import Widget
from textual.css.styles import RulesMap


COLORS = [
    "#000000",
    "#0a0a0a",
    "#1a1a1a",
    "#2a2a2a",
    "#3a3a3a",
    "#4a4a4a",
    "#5a5a5a",
    "#6a6a6a",
    "#7a7a7a",
    "#8a8a8a",
    "#9a9a9a",
    "#aaaaaa",
    "#bbbbbb",
    "#cccccc",
    "#dddddd",
    "#eeeeee",
    "#ffffff",
    "#eeeeee",
    "#dddddd",
    "#cccccc",
    "#bbbbbb",
    "#aaaaaa",
    "#9a9a9a",
    "#8a8a8a",
    "#7a7a7a",
    "#6a6a6a",
    "#5a5a5a",
    "#4a4a4a",
    "#3a3a3a",
    "#2a2a2a",
    "#1a1a1a",
    "#0a0a0a",
]


class ThrobberVisual(Visual):
    """A Textual 'Visual' object.

    Analogous to a Rich renderable, but with support for transparency.

    """

    gradient = Gradient.from_colors(*[Color.parse(color) for color in COLORS])

    def render_strips(
        self, width: int, height: int | None, style: Style, options: RenderOptions
    ) -> list[Strip]:
        """Render the Visual into an iterable of strips.

        Args:
            width: Width of desired render.
            height: Height of desired render or `None` for any height.
            style: The base style to render on top of.
            options: Additional render options.

        Returns:
            An list of Strips.
        """

        time = monotonic()
        gradient = self.gradient
        background = style.rich_style.bgcolor

        strips = [
            Strip(
                [
                    Segment(
                        "━",
                        RichStyle.from_color(
                            gradient.get_rich_color((offset / width - time) % 1.0),
                            background,
                        ),
                    )
                    for offset in range(width)
                ],
                width,
            )
        ]
        return strips

    def get_optimal_width(self, rules: RulesMap, container_width: int) -> int:
        return container_width

    def get_height(self, rules: RulesMap, width: int) -> int:
        return 1


class Throbber(Widget):
    def on_mount(self) -> None:
        self.auto_refresh = 1 / 15

    def render(self) -> ThrobberVisual:
        return ThrobberVisual()