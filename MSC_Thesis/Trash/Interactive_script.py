#importing necessary packages
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
import matplotlib.pyplot as plt

#Getting the rom info
from rominfo import *
from baselines.common.retro_wrappers import *


def getRamI(env):
    ram = []
    for k, v in env.data.memory.blocks.items():
        ram += list(v)
    return np.array(ram)

class RetroInteractive(abc.ABC):
    """
    Interactive setup for retro games
    """
    def __init__(self, env, sync=True, tps=60, aspect_ratio=None):
        self.inputs = []
        self.states = []
        self.statearray = []
        self.env = env
        self.tps = 60
        self.aspect_ratio = 4/3
        obs = env.reset()
        self._image = self.get_image(obs, env)
        assert len(self._image.shape) == 3 and self._image.shape[2] == 3, 'must be an RGB image'
        image_height, image_width = self._image.shape[:2]

        if aspect_ratio is None:
            aspect_ratio = image_width / image_height

        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        max_win_width = screen.width * 0.9
        max_win_height = screen.height * 0.9
        win_width = image_width
        win_height = int(win_width / aspect_ratio)

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

        self._env = env
        self._win = win

        # self._render_human = render_human
        self._key_previous_states = {}

        self._steps = 0
        self._episode_steps = 0
        self._episode_returns = 0
        self._prev_episode_returns = 0

        self._tps = tps
        self._sync = sync
        self._current_time = 0
        self._sim_time = 0
        self._max_sim_frames_per_update = 4

    def _update(self, dt):
        # cap the number of frames rendered so we don't just spend forever trying to catch up on frames
        # if rendering is slow
        max_dt = self._max_sim_frames_per_update / self._tps
        if dt > max_dt:
            dt = max_dt

        

        # catch up the simulation to the current time
        self._current_time += dt
        while self._sim_time < self._current_time:
            self._sim_time += 1 / self._tps

            keys_clicked = set()
            keys_pressed = set()
            for key_code, pressed in self._key_handler.items():
                if pressed:
                    keys_pressed.add(key_code)

                if not self._key_previous_states.get(key_code, False) and pressed:
                    keys_clicked.add(key_code)
                self._key_previous_states[key_code] = pressed

            if keycodes.ESCAPE in keys_pressed:
                self._on_close()

            # assume that for async environments, we just want to repeat keys for as long as they are held
            inputs = keys_pressed
            if self._sync:
                inputs = keys_clicked

            keys = []
            for keycode in inputs:
                for name in dir(keycodes):
                    if getattr(keycodes, name) == keycode:
                        keys.append(name)

            act = self.keys_to_act(keys)

            if not self._sync or act is not None:
                obs, rew, done, _info = self._env.step(act)

                #Important code

                saved_inputs = np.array(act)
                saved_inputs = saved_inputs.astype(int)
                self.inputs.append(saved_inputs)
                #print(saved_inputs)
                #print(saved_inputs.shape)
                
                
                saved_output = np.array(obs)
                saved_output = saved_output.astype(int)
                self.states.append(obs) #saved the image observations
                #np.save('data2.npy', dataset)
                #print(act)

                ram = getRamI(self._env)
                state, x, y = getInputs(ram)
                state = np.reshape(state, (13, 13))
                self.statearray.append(state)
                #print(state)
                #print(state.shape)
                #print(state)

                self._image = self.get_image(obs, self._env)
                self._episode_returns += rew
                self._steps += 1
                self._episode_steps += 1
                np.set_printoptions(precision=2)
                self._episode_rewards.append(self._episode_returns)
                if self._sync:
                    done_int = int(done)  # shorter than printing True/False
                    mess = 'steps={self._steps} episode_steps={self._episode_steps} rew={rew} episode_returns={self._episode_returns} done={done_int}'.format(
                        **locals()
                    )
                    #print(mess)
                elif self._steps % self._tps == 0 or done:
                    episode_returns_delta = self._episode_returns - self._prev_episode_returns
                    self._prev_episode_returns = self._episode_returns
                    mess = 'steps={self._steps} episode_steps={self._episode_steps} episode_returns_delta={episode_returns_delta} episode_returns={self._episode_returns}'.format(
                        **locals()
                    )
                    #print(mess)

                if done:
                    self._env.reset()
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
        #dataset = np.array((self.states,self.inputs))
        #np.save('data.npy', dataset)

        #print(self.inputs)

        #print("Size of inputs:", len(self.inputs))
        #print("Data type of inputs:", type(self.inputs))
        #print("Size of states:", len(self.states))
        #print("Data type of states:", type(self.states))

        inputs_array = np.array(self.inputs)
        states_obs = np.array(self.states)
        states_array = np.array(self.statearray)

        #np.save('/Users/sankalpssss/Documents/marioenv/mario/mario2/Data/Input_session1.npy',inputs_array)
        #np.save('/Users/sankalpssss/Documents/marioenv/mario/mario2/Data/States_obs_session1.npy',states_obs)
        #np.save('/Users/sankalpssss/Documents/marioenv/mario/mario2/Data/States_array_session1.npy',states_array)

    
        # Now you have inputs_array and states_array as NumPy arrays
    
        print("Size of inputs array:", inputs_array.shape)
        print("Data type of inputs array:", inputs_array.dtype)
        print(inputs_array[0])
        print("Size of states obs array:", states_obs.shape)
        print("Data type of states array:", states_obs.dtype)
        print(states_obs[0])
        print("Size of states array array:", states_array.shape)
        print("Data type of states array:", states_array.dtype)
        print(states_array[0])

        #dataset = np.array((states_array,inputs_array))
        #np.save('/Users/sankalpssss/Documents/marioenv/mario/mario2/Data.npy', dataset)

        # Plot total rewards against time

        
        if self._episode_rewards:
            episode_durations = range(len(self._episode_rewards))
            plt.plot(episode_durations, self._episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Total Rewards')
            plt.title('Total Rewards Over Episodes')
            plt.show()
        
        self._env.close()
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
        env = retro.make(game=game, state=state, scenario=scenario, record=record)
        self._buttons = env.buttons
        super().__init__(env=env, sync=False, tps=60, aspect_ratio=4/3)
        self._episode_rewards = []

    def get_image(self, _obs, env):
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
        return [inputs[b] for b in self._buttons]


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