import sys
import ctypes
import argparse
import abc
import time
import os
import numpy as np
import retro
import pyglet
from pyglet import gl
from pyglet.window import key as keycodes

from rominfo import *

radius = 6


def printState(state):
    state_n = np.reshape(state, (2*radius + 1, 2*radius + 1))
    state_str = '\n'.join(','.join(str(cell) for cell in row) for row in state_n)
    print(state_str)
    _ = os.system("clear")
    mm = {'0': '  ', '1': '$$', '-1': '@@'}
    for i, l in enumerate(state_n):
        line = list(map(lambda x: mm[str(x)], l))
        if i == radius + 1:
            line[radius] = 'X'

def getRam(env):
    ram = []
    for k, v in env.data.memory.blocks.items():
        ram += list(v)
    return np.array(ram)

class RetroInteractive(abc.ABC):
    """
    Interactive setup for retro games
    """
    def __init__(self, game, state, scenario, record):
        env = retro.make(game=game, state=state, scenario=scenario, record=record)
        self._buttons = env.buttons
        self.env = env
        self.tps = 60
        self.aspect_ratio = 4/3

        obs = env.reset()
        self._image = self.get_image(obs, env)
        assert len(self._image.shape) == 3 and self._image.shape[2] == 3, 'must be an RGB image'
        image_height, image_width = self._image.shape[:2]

        if self.aspect_ratio is None:
            self.aspect_ratio = image_width / image_height

        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        max_win_width = screen.width * 0.9
        max_win_height = screen.height * 0.9
        win_width = image_width
        win_height = int(win_width / self.aspect_ratio)

        while win_width > max_win_width or win_height > max_win_height:
            win_width //= 2
            win_height //= 2
        while win_width < max_win_width / 2 and win_height < max_win_height / 2:
            win_width *= 2
            win_height *= 2

        win = pyglet.window.Window(width=win_width, height=win_height)

        self._key_handler = pyglet.window.key.KeyStateHandler()
        win.push_handlers(self._key_handler)
        win.on_close = self._on_close

        gl.glEnable(gl.GL_TEXTURE_2D)
        self._texture_id = gl.GLuint(0)
        gl.glGenTextures(1, ctypes.byref(self._texture_id))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, image_width, image_height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        self._win = win
        self._key_previous_states = {}

        self._steps = 0
        self._episode_steps = 0
        self._episode_returns = 0
        self._prev_episode_returns = 0

        self._current_time = 0
        self._sim_time = 0
        self._max_sim_frames_per_update = 4

    def _update(self, dt):
        max_dt = self._max_sim_frames_per_update / self.tps
        if dt > max_dt:
            dt = max_dt

        self._current_time += dt
        while self._sim_time < self._current_time:
            self._sim_time += 1 / self.tps

            keys_pressed = set()
            for key_code, pressed in self._key_handler.items():
                if pressed:
                    keys_pressed.add(key_code)

                self._key_previous_states[key_code] = pressed

            if keycodes.ESCAPE in keys_pressed:
                self._on_close()

            inputs = keys_pressed

            keys = []
            for keycode in inputs:
                for name in dir(keycodes):
                    if getattr(keycodes, name) == keycode:
                        keys.append(name)

            act = self.keys_to_act(keys)

            if act is not None:
                obs, rew, done, _info = self.env.step(act)

                saved_inputs = np.array(act)
                saved_inputs = saved_inputs.astype(int)

                ram = getRam(self.env)
                state, x, y = getInputs(ram)

                state = np.reshape(state, (13, 13))
                printState(state)

                self._image = self.get_image(obs, self.env)
                self._episode_returns += rew
                self._steps += 1
                self._episode_steps += 1

                if self._steps % self.tps == 0 or done:
                    episode_returns_delta = self._episode_returns - self._prev_episode_returns
                    self._prev_episode_returns = self._episode_returns
                    mess = 'steps={self._steps} episode_steps={self._episode_steps} episode_returns_delta={episode_returns_delta} episode_returns={self._episode_returns}'.format(
                        **locals()
                    )
                    print(mess)

                if done:
                    self.env.reset()
                    self._episode_steps = 0
                    self._episode_returns = 0
                    self._prev_episode_returns = 0

    def _draw(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        video_buffer = ctypes.cast(self._image.tobytes(), ctypes.POINTER(ctypes.c_short))
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self._image.shape[1], self._image.shape[0], gl.GL_RGB, gl.GL_UNSIGNED_BYTE, video_buffer)

        x = 0
        y = 0
        w = self._win.width
        h = self._win.height

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ('v2f', [x, y, x + w, y, x + w, y + h, x, y + h]),
            ('t2f', [0, 1, 1, 1, 1, 0, 0, 0]),
        )

    def _on_close(self):
        self.env.close()
        sys.exit(0)

    @abc.abstractmethod
    def get_image(self, obs, venv):
        """
        Given an observation and the Env object, return an RGB array to display to the user
        """
        pass

    @abc.abstractmethod
    def keys_to_act(self, keys):
        """
        Given a list of keys that the user has input, produce a gym action to pass to the environment

        For sync environments, keys is a list of keys that have been pressed since the last step
        For async environments, keys is a list of keys currently held down
        """
        pass

    def run(self):
        """
        Run the interactive window until the user quits
        """
        prev_frame_time = time.time()
        while True:
            self._win.switch_to()
            self._win.dispatch_events()
            now = time.time()
            self._update(now - prev_frame_time)
            prev_frame_time = now
            self._draw()
            self._win.flip()

class MyRetroInteractive(RetroInteractive):
    def __init__(self, game, state, scenario, record):
        super().__init__(game=game, state=state, scenario=scenario, record=record)
        self.inputs = []
        self.states = []

    def get_image(self, obs, env):
        return env.render(mode='rgb_array')

    def keys_to_act(self, keys):
        inputs = {
            None: False,

            'BUTTON': 'Z' in keys,
            'A': 'Z' in keys,
            'B': 'X' in keys,

            'C': 'C' in keys,
            'X': 'A' in keys,
            'Y': 'S' in keys,
            'Z': 'D' in keys,

            'L': 'Q' in keys,
            'R': 'W' in keys,

            'UP': 'UP' in keys,
            'DOWN': 'DOWN' in keys,
            'LEFT': 'LEFT' in keys,
            'RIGHT': 'RIGHT' in keys,

            'MODE': 'TAB' in keys,
            'SELECT': 'TAB' in keys,
            'RESET': 'ENTER' in keys,
            'START': 'ENTER' in keys,
        }
        return [inputs[b] for b in self.env.buttons]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='SuperMarioWorld-Snes')
    parser.add_argument('--state', default='YoshiIsland2')
    parser.add_argument('--scenario', default=None)
    parser.add_argument('--record', default=True, nargs='?', const=True)
    args = parser.parse_args()

    ia = MyRetroInteractive(game=args.game, state=args.state, scenario=args.scenario, record=args.record)
    ia.run()

if __name__ == '__main__':
    main()

