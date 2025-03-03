""""
This script defines two classes, Interactive and RetroInteractive, 
along with a main() function to run the interactive environment.
"""

"""
---------------------------------------------------------------------------------------------------
"""

"""
- These lines import necessary modules and libraries required for the script.
- sys, ctypes, argparse, abc, time, os: Standard Python libraries.
- numpy: For numerical computations.
- retro: Retro Gym environment for playing retro games.
- pyglet: Library for windowing and multimedia in Python.
- gl and keycodes from pyglet: Specific components for OpenGL graphics and keyboard input handling.
- rominfo and baselines.common.retro_wrappers: Additional custom modules being imported.
- The imports are needed for various functionalities like handling user inputs, rendering the game environment, 
defining abstract base classes, and interacting with the Retro Gym environment.

IMPORTANT: It depends which RAM you are using. I am using the one with 0,1,-1
For that I will be easier to understand as of when to jump if the object is close to the agent.
in contrast to the script that only writes 0,1.
"""

#Import necessary modules and libraries.
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
from baselines.common.retro_wrappers import *


"""
---------------------------------------------------------------------------------------------------
"""


class Interactive(abc.ABC):
    """
    Define a base class Interactive as an abstract base class (ABC) 
    for making gym environments interactive for human use.
    """
    def __init__(self, env, sync=True, tps=60, aspect_ratio=None): #constructor method for the Interactive class.
        #Initialize lists to store inputs and states.
        self.inputs = []
        self.states = []
        self.states_array = []
        obs = env.reset()

        """
        These lines initialize the game environment by resetting it (env.reset()) and capturing the initial observation (obs).
        It then retrieves an RGB image representation of the observation using the get_image() method.
        The assert statement ensures that the image has three dimensions (RGB) and raises an error if not.
        python

        """
        self._image = self.get_image(obs, env)
        assert len(self._image.shape) == 3 and self._image.shape[2] == 3, 'must be an RGB image'
        image_height, image_width = self._image.shape[:2]

        if aspect_ratio is None:
            aspect_ratio = image_width / image_height

        """
        calculate the optimal window size based on the screen resolution and the aspect ratio of the captured image.
        It ensures that the window size is not too small or too large.
        """
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        max_win_width = screen.width * 0.9
        max_win_height = screen.height * 0.9
        win_width = image_width
        win_height = int(win_width / aspect_ratio)

        #If the window size is too large, it is halved, and if it's too small, it is doubled.
        while win_width > max_win_width or win_height > max_win_height:
            win_width //= 2
            win_height //= 2
        while win_width < max_win_width / 2 and win_height < max_win_height / 2:
            win_width *= 2
            win_height *= 2

        #Create a Pyglet window with the calculated width and height.
        win = pyglet.window.Window(width=win_width, height=win_height)
        
        """
        These lines set up a key event handler to capture keyboard inputs (self._key_handler) 
        and specify a close event handler (self._on_close) for the window.
        """
        self._key_handler = pyglet.window.key.KeyStateHandler()
        win.push_handlers(self._key_handler)
        win.on_close = self._on_close

        #OpenGL settings are initialized to enable 2D textures and generate a texture ID for rendering.
        gl.glEnable(gl.GL_TEXTURE_2D)
        self._texture_id = gl.GLuint(0)
        gl.glGenTextures(1, ctypes.byref(self._texture_id))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, image_width, image_height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        #References to the environment and window are stored for further use.
        self._env = env
        self._win = win

        # This dictionary will store the previous states of keys to determine if a key has been clicked.
        self._key_previous_states = {}

        #These variables track various metrics related to the interaction with the environment.
        self._steps = 0
        self._episode_steps = 0
        self._episode_returns = 0
        self._prev_episode_returns = 0

        self._tps = tps
        self._sync = sync
        self._current_time = 0
        self._sim_time = 0
        self._max_sim_frames_per_update = 4


    #This method is responsible for updating the environment based on user input and rendering the updated state.
    def _update(self, dt):
        """
        This section limits the time delta (dt) to ensure that the number of frames rendered per update 
        does not exceed a certain threshold (_max_sim_frames_per_update).
        """
        max_dt = self._max_sim_frames_per_update / self._tps
        if dt > max_dt:
            dt = max_dt

        # These lines ensure that the simulation time (self._sim_time) keeps pace with 
        #the current time (self._current_time) based on the ticks per second (_tps).
        self._current_time += dt
        while self._sim_time < self._current_time:
            self._sim_time += 1 / self._tps

            """
            This section detects keys that have been clicked or are currently pressed, 
            distinguishing between newly clicked keys and those held down since the last update.
            """
            keys_clicked = set()
            keys_pressed = set()
            for key_code, pressed in self._key_handler.items():
                if pressed:
                    keys_pressed.add(key_code)

                if not self._key_previous_states.get(key_code, False) and pressed:
                    keys_clicked.add(key_code)
                self._key_previous_states[key_code] = pressed

            #If the ESCAPE key is pressed, it triggers the _on_close method to close the environment.
            if keycodes.ESCAPE in keys_pressed:
                self._on_close()

            # assume that for async environments, we just want to repeat keys for as long as they are held
            inputs = keys_pressed
            if self._sync:
                inputs = keys_clicked

            """
            The inputs variable is determined based on whether the environment is synchronized (self._sync). 
            If synchronized, it uses keys clicked since the last update; otherwise, it uses keys currently pressed.
            """
            keys = []
            for keycode in inputs:
                for name in dir(keycodes):
                    if getattr(keycodes, name) == keycode:
                        keys.append(name)

            #The keys_to_act method is called to convert the detected keys into a gym action to be passed to the environment.
            act = self.keys_to_act(keys)

            #Checks if the environment is not synchronized or if there is an action to be taken.
            if not self._sync or act is not None:
                """
                The environment's step method is called with the action (act) to advance the environment by one step, 
                resulting in a new observation (obs), reward (rew), and done flag (done).
                """
                obs, rew, done, _info = self._env.step(act)

                #Storing inputs in a list of 1-D arrays
                saved_inputs = np.array(act)
                saved_inputs = saved_inputs.astype(int)
                print(saved_inputs)
                self.inputs.append(saved_inputs)

                #Save the Images
                self.states.append(obs)
                
                #Save the 13x13 RAM States
                ram = getRam(self._env)
                state, x, y = getInputs(ram)
                state = np.reshape(state, (13, 13))
                #print(state)
                saved_outputs = np.array(state.flatten())
                saved_outputs = saved_outputs.astype(int)
                self.states_array.append(saved_outputs)

                """
                The image to be displayed to the user is updated based on the new observation (obs) 
                and the environment (self._env).
                """
                
                self._image = self.get_image(obs, self._env)

                #Various counters are updated based on the received reward (rew) and the number of steps taken.
                self._episode_returns += rew
                self._steps += 1
                self._episode_steps += 1
                np.set_printoptions(precision=2)

                #If the environment is synchronized, a message is constructed with various metrics for logging purposes.
                if self._sync:
                    done_int = int(done)  # shorter than printing True/False
                    mess = 'steps={self._steps} episode_steps={self._episode_steps} rew={rew} episode_returns={self._episode_returns} done={done_int}'.format(
                        **locals()
                    )
                    #print(mess)
                
                #If the environment is asynchronous, a message is constructed and printed at regular intervals or when an episode is finished.
                elif self._steps % self._tps == 0 or done:
                    episode_returns_delta = self._episode_returns - self._prev_episode_returns
                    self._prev_episode_returns = self._episode_returns
                    mess = 'steps={self._steps} episode_steps={self._episode_steps} episode_returns_delta={episode_returns_delta} episode_returns={self._episode_returns}'.format(
                        **locals()
                    )
                    #print(mess)

                #If the episode is done, the environment is reset, and episode-related counters are reset.
                if done:
                    self._env.reset()
                    self._episode_steps = 0
                    self._episode_returns = 0
                    self._prev_episode_returns = 0

    ##This method is responsible for rendering the current state of the environment to the Pyglet window.
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

    #This method is called when the window is closed
    def _on_close(self):
        
        #Concatenating

        #RAM State Array
        final_state_array = np.empty((len(self.states_array),),dtype=object)
        for i in range(len(self.states_array)):
            final_state_array[i] = self.states_array[i]


        #Action Array
        final_action_array = np.empty((len(self.inputs),),dtype=object)
        for i in range(len(self.inputs)):
            final_action_array[i] = self.inputs[i]

        #Making a state acion pair. Remember to check the datatypes (they are objects -> arrays)
        dataset = np.array((final_state_array,final_action_array))
        #np.save("/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/All/L1_Jumps_S10",dataset)
        self._env.close()
        sys.exit(0)

    @abc.abstractmethod
    def get_image(self, obs, venv):
        """
        Given an observation and the Env object, return an rgb array to display to the user
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
        # pyglet.app.run() has issues like https://bitbucket.org/pyglet/pyglet/issues/199/attempting-to-resize-or-close-pyglet
        # and also involves inverting your code to run inside the pyglet framework
        # avoid both by using a while loop
        prev_frame_time = time.time()
        while True:
            self._win.switch_to()
            self._win.dispatch_events()
            now = time.time()
            self._update(now - prev_frame_time)
            prev_frame_time = now
            self._draw()
            self._win.flip()


class RetroInteractive(Interactive):
    """
    Interactive setup for retro games
    """
    def __init__(self, game, state, scenario, record):
        env = retro.make(game=game, state=state, scenario=scenario, record=record)
        self._buttons = env.buttons
        # Frame skip (hold an action for this many frames) and sticky actions
        #env = StochasticFrameSkip(env, 4, 0.25)
        # scale and turn RGB image to grayscale
        #env = WarpFrame(env, width=84, height=84, grayscale=True)
        super().__init__(env=env, sync=False, tps=60, aspect_ratio=4/3)

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

    ia = RetroInteractive(game=args.game, state=args.state, scenario=args.scenario, record=args.record)
    ia.run()


if __name__ == '__main__':
    main()