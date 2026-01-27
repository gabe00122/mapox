import pygame
from abc import ABC, abstractmethod

from mapox.renderer import GridworldRenderer, GridRenderState
from mapox.timestep import TimeStep


class EnvironmentClient[State](ABC):
    @abstractmethod
    def render(self, state: State, timestep: TimeStep): ...

    @abstractmethod
    def save_video(self): ...


class GridworldClient:
    """EnvironmentClient that renders via GridworldRenderer using per-env adapters."""

    def __init__(
        self, env, screen_width: int = 960, screen_height: int = 960, fps: int = 10
    ):
        assert hasattr(env, "get_render_state"), (
            "Env must implement get_render_state(state)"
        )
        assert hasattr(env, "get_render_settings"), (
            "Env must implement get_render_settings()"
        )
        self.env = env
        self.renderer = GridworldRenderer(
            screen_width=screen_width, screen_height=screen_height, fps=fps
        )
        self.renderer.set_env(env.get_render_settings())

    def render(self, state, timestep):
        rs: GridRenderState = self.env.get_render_state(state)
        self.renderer.render(rs)

    def render_pov(self, state, timestep):
        """Render only the focused agent's point-of-view, filling the screen."""
        rs: GridRenderState = self.env.get_render_state(state)
        self.renderer.render_agent_view(rs)

    def handle_event(self, event: pygame.event.Event) -> bool:
        return self.renderer.handle_event(event)

    def record_frame(self):
        self.renderer.record_frame()

    def save_video(self, file_name: str):
        self.renderer.save_video(file_name)

    def focus_agent(self, agent_id: int | None):
        self.renderer.focus_agent(agent_id)
