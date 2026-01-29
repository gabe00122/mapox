from mapox.environment import EnvState, Environment
from mapox.renderer import GridworldRenderer
from mapox.timestep import TimeStep

import pygame
from typing import Generic


class GridworldClient(Generic[EnvState]):
    """EnvironmentClient that renders via GridworldRenderer using per-env adapters."""

    def __init__(self, env: Environment[EnvState], screen_width: int = 960, screen_height: int = 960, fps: int = 10):
        self.env = env
        self.renderer = GridworldRenderer(screen_width=screen_width, screen_height=screen_height, fps=fps)
        self.renderer.set_env(env.get_render_settings())

    def render(self, state: EnvState, timestep: TimeStep, pov: bool = False) -> None:
        render_state = self.env.get_render_state(state)
        if pov:
            self.renderer.render_agent_view(render_state)
        else:
            self.renderer.render(render_state)

    def handle_event(self, event: pygame.event.Event) -> bool:
        return self.renderer.handle_event(event)

    def record_frame(self):
        self.renderer.record_frame()

    def save_video(self, file_name: str):
        self.renderer.save_video(file_name)

    def focus_agent(self, agent_id: int | None):
        self.renderer.focus_agent(agent_id)
