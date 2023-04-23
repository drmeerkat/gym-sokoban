import numpy as np
import random
import copy
from gymnasium.spaces import Box
from typing import overload

from . import render_utils as renu
from .sokoban_env import SokobanEnv
from .sokoban_env_fixed_targets import FixedTargetsSokobanEnv
from .sokoban_env_pull import PushAndPullSokobanEnv
from .sokoban_env_two_player import TwoPlayerSokobanEnv
from .boxoban_env import BoxobanEnv


class SokobanEnvColorBox(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    box_context_mapping = {
        0: (1, 2),
        1: (2, 2),
        2: (2, 3)
    }

    agent_context_mapping = {
        0: (1, 1),
        1: (2, 1),
        2: (3, 1),
        3: (3, 2),
        4: (3, 3),
        5: (1, 2),
        6: (2, 2),
        7: (2, 3)
    }

    def __init__(self, color_threshold = 30, render_mode='rgb_array', **kwargs):
        kwargs['num_boxes'] = 1
        kwargs['max_steps'] = kwargs.get('max_steps', 20)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 40)
        kwargs['dim_room'] = (5, 5)
        kwargs['reset'] = False # don't initialize the room

        self.render_mode = render_mode
        self.color_threshold = color_threshold
        self.box_color = False
        self.target_color = False
        assert isinstance(self.color_threshold, int), 'change_color_threshold must be an integer'
        assert self.color_threshold <= 100 and self.color_threshold >= 0, \
                'change_color_threshold must be in the range of [0, 100]'

        # Context = [box init pos, agent init pos, type of interventions]
        # For simplicity, we only allow the box to be in these three locations
        # w-wall, t-target, e-empty, a-agent, b-box
        # [[w, w, w, w, w],
        #  [w, e, 0, t, w],
        #  [w, e, 1, 2, w],
        #  [w, a, e, e, w],
        #  [w, w, w, w, w]]
        # And the agent to be in these 5 locations
        # [[w, w, w, w, w],
        #  [w, 0, 5, t, w],
        #  [w, 1, 6, 7, w],
        #  [w, 2, 3, 4, w],
        #  [w, w, w, w, w]]
        # There are 3 types of interventions
        # 0 - no intervention (box color = y/b, target color = box color)
        # 1 - do(box color = y, target color = random)
        # 2 - do(box color = b, target color = random)
        self._context = [0, 6, 0]
        super(SokobanEnvColorBox, self).__init__(**kwargs)
        self.observation_space = Box(low=0, high=6, shape=(1, kwargs['dim_room'][0], kwargs['dim_room'][1]), dtype=np.int64)

    def set_context(self, context):
        self._context = np.array(context)

    def get_context(self):
        return copy.deepcopy(self._context)

    context = property(get_context, set_context)

    def reset(self, second_player=False, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        room_fixed = np.array([[0, 0, 0, 0, 0],
                               [0, 1, 1, 2, 0],
                               [0, 1, 1, 1, 0],
                               [0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0]])
        room_state = copy.deepcopy(room_fixed)

        # set room_state accordingly
        room_state[self.box_context_mapping[int(np.floor(self.context[0]))]] = 4
        room_state[self.agent_context_mapping[int(np.floor(self.context[1]))]] = 5
        #  target pos: box pos
        box_mapping = {(1, 3):self.box_context_mapping[int(np.floor(self.context[0]))]}

        self.set_box_mapping(box_mapping)
        self.set_room_fixed(room_fixed)
        self.set_room_state(room_state)
        # this indicates whether box color C is fixed: 0-no, 1-blue, 2-yellow 
        self.intervention = int(np.floor(self.context[2]))

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self._set_box_color()

        if self.render_mode == 'human':
            self.render()

        return self._observe(), {}
    
    def _set_box_color(self):
        U = (random.random() <= self.color_threshold/100)
        self.target_color = U
        if self.intervention == 2:
            # fix box color to yellow
            self.box_color = False
        elif self.intervention == 1:
            # fix box color to blue
            self.box_color = True
        else:
            # this always aligns with box_color in this env
            self.box_color = self.target_color

    def step(self, action):
        # box and target color from s_t
        self.prev_box_color = self.box_color
        self.prev_target_color = self.target_color
        
        # box and target color from s_t+1
        self._set_box_color()

        # Handle other movement events
        obs, r, term, trunc, info = super(SokobanEnvColorBox, self).step(action)
        info['success'] = term and (r > 7)

        return obs, r, term, trunc, info

    def get_image(self, mode, scale=1):
        if mode.startswith('tiny_'):
            img = renu.room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale, change_color=self.box_color)
        else:
            img = renu.room_to_rgb(self.room_state, self.room_fixed, change_color=self.box_color)

        return img

    def _calc_reward(self):
        """
        Reward is confounded with self.box_color
        So pushing the box to the target is always preferrable.
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target

        game_won = self._check_if_all_boxes_on_target()
        if game_won:
            # use box and target color from s_t!
            # if self.prev_box_color == self.prev_target_color and self.prev_target_color == 0:
            #     self.reward_last += self.reward_finished
            # elif self.prev_box_color != self.prev_target_color and self.prev_target_color == 0:
            #     self.reward_last += 2*self.reward_finished
            # else:
            #     self.reward_last += -self.reward_finished

            # NEW REWARD function
            if self.prev_target_color == 0: 
                # yellow box +10
                self.reward_last += self.reward_finished
            elif self.prev_target_color == 1:
                # blue box -10, next try with -1
                self.reward_last += -self.reward_finished
            else:
                pass
            #     self.reward_last += -self.reward_finished

        self.boxes_on_target = current_boxes_on_target


class SokobanEnv_Target(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 1)
        kwargs['max_steps'] = kwargs.get('max_steps', 30)
        kwargs['dim_room'] = kwargs.get('dim_room', (5, 5))
        super(SokobanEnv_Target, self).__init__(**kwargs)
        # overwrite observation space
        self.observation_space = Box(low=0, high=5, shape=(1, kwargs['dim_room'][0], kwargs['dim_room'][1]), dtype=np.int64)

    def reset(self, second_player=False, render_mode='rgb_array', seed=12345):
        room_fixed = np.array([[0, 0, 0, 0, 0],
                               [0, 1, 1, 2, 0],
                               [0, 1, 1, 1, 0],
                               [0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0]])
        room_state = np.array([[0, 0, 0, 0, 0],
                               [0, 1, 1, 2, 0],
                               [0, 1, 4, 1, 0],
                               [0, 1, 1, 5, 0],
                               [0, 0, 0, 0, 0]])
        #  target pos: box pos
        box_mapping = {(1, 3):(2, 2)}
        self.set_box_mapping(box_mapping)
        self.set_room_fixed(room_fixed)
        self.set_room_state(room_state)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self.render_mode = render_mode
        starting_observation = self.render()

        return np.expand_dims(self.room_state, 0)

    def step(self, action, observation_mode='rgb_array'):
        observation, reward, done, info = super(SokobanEnv_Target, self).step(action, observation_mode=observation_mode)
        return np.expand_dims(self.room_state, 0), reward, done, info


class SokobanEnv_Source1(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 1)
        kwargs['max_steps'] = kwargs.get('max_steps', 30)
        kwargs['dim_room'] = kwargs.get('dim_room', (5,5))
        super(SokobanEnv_Source1, self).__init__(**kwargs)
        # overwrite observation space
        self.observation_space = Box(low=0, high=5, shape=(1, kwargs['dim_room'][0], kwargs['dim_room'][1]), dtype=np.int64)

    def reset(self, second_player=False, render_mode='rgb_array', seed=12345):
        if seed is not None:
            random.seed(seed)
        room_fixed = np.array([[0, 0, 0, 0, 0],
                               [0, 1, 1, 2, 0],
                               [0, 1, 1, 1, 0],
                               [0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0]])
        room_state1 = np.array([[0, 0, 0, 0, 0],
                               [0, 5, 4, 2, 0],
                               [0, 1, 1, 1, 0],
                               [0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0]])
        room_state2 = np.array([[0, 0, 0, 0, 0],
                               [0, 1, 4, 2, 0],
                               [0, 5, 1, 1, 0],
                               [0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0]])
        #  target pos: box pos
        box_mapping = {(1, 3):(2, 2)}
        self.set_box_mapping(box_mapping)
        self.set_room_fixed(room_fixed)
        sample = random.random()
        if sample > .5:
            self.set_room_state(room_state1)
        else:
            self.set_room_state(room_state2)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self.render_mode = render_mode
        starting_observation = self.render()

        return np.expand_dims(self.room_state, 0)

    def step(self, action, observation_mode='rgb_array'):
        observation, reward, done, info = super(SokobanEnv_Source1, self).step(action, observation_mode=observation_mode)
        return np.expand_dims(self.room_state, 0), reward, done, info


class SokobanEnv_Source2(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 1)
        kwargs['max_steps'] = kwargs.get('max_steps', 30)
        kwargs['dim_room'] = kwargs.get('dim_room', (5,5))
        super(SokobanEnv_Source2, self).__init__(**kwargs)
        # overwrite observation space
        self.observation_space = Box(low=0, high=5, shape=(1, kwargs['dim_room'][0], kwargs['dim_room'][1]), dtype=np.int64)

    def reset(self, second_player=False, render_mode='rgb_array', seed=12345):
        if seed is not None:
            random.seed(seed)
        room_fixed = np.array([[0, 0, 0, 0, 0],
                               [0, 1, 1, 2, 0],
                               [0, 1, 1, 1, 0],
                               [0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0]])
        room_state1 = np.array([[0, 0, 0, 0, 0],
                               [0, 1, 4, 2, 0],
                               [0, 1, 5, 1, 0],
                               [0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0]])
        room_state2 = np.array([[0, 0, 0, 0, 0],
                               [0, 1, 4, 2, 0],
                               [0, 5, 1, 1, 0],
                               [0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0]])
        room_state3 = np.array([[0, 0, 0, 0, 0],
                               [0, 1, 4, 2, 0],
                               [0, 1, 1, 5, 0],
                               [0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0]])
        #  target pos: box pos
        box_mapping = {(1, 3):(2, 2)}
        self.set_box_mapping(box_mapping)
        self.set_room_fixed(room_fixed)
        sample = random.random()
        if sample > .7:
            self.set_room_state(room_state3)
        elif sample > .4:
            self.set_room_state(room_state2)
        else:
            self.set_room_state(room_state1)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self.render_mode = render_mode
        starting_observation = self.render()

        return np.expand_dims(self.room_state, 0)

    def step(self, action, observation_mode='rgb_array'):
        observation, reward, done, info = super(SokobanEnv_Source2, self).step(action, observation_mode=observation_mode)
        return np.expand_dims(self.room_state, 0), reward, done, info


class SokobanEnv_Source3(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 1)
        kwargs['max_steps'] = kwargs.get('max_steps', 30)
        kwargs['dim_room'] = kwargs.get('dim_room', (5,5))
        super(SokobanEnv_Source3, self).__init__(**kwargs)
        # overwrite observation space
        self.observation_space = Box(low=0, high=5, shape=(1, kwargs['dim_room'][0], kwargs['dim_room'][1]), dtype=np.int64)

    def reset(self, second_player=False, render_mode='rgb_array', seed=12345):
        if seed is not None:
            random.seed(seed)
        room_fixed = np.array([[0, 0, 0, 0, 0],
                               [0, 1, 1, 2, 0],
                               [0, 1, 1, 1, 0],
                               [0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0]])
        room_state1 = np.array([[0, 0, 0, 0, 0],
                                [0, 1, 1, 2, 0],
                                [0, 1, 4, 1, 0],
                                [0, 1, 5, 1, 0],
                                [0, 0, 0, 0, 0]])
        room_state2 = np.array([[0, 0, 0, 0, 0],
                                [0, 1, 1, 2, 0],
                                [0, 1, 4, 1, 0],
                                [0, 5, 1, 1, 0],
                                [0, 0, 0, 0, 0]])
        room_state3 = np.array([[0, 0, 0, 0, 0],
                                [0, 1, 1, 2, 0],
                                [0, 1, 4, 1, 0],
                                [0, 1, 1, 5, 0],
                                [0, 0, 0, 0, 0]])
        #  target pos: box pos
        box_mapping = {(1, 3):(2, 2)}
        self.set_box_mapping(box_mapping)
        self.set_room_fixed(room_fixed)
        sample = random.random()
        if sample > .7:
            self.set_room_state(room_state3)
        elif sample > .4:
            self.set_room_state(room_state2)
        else:
            self.set_room_state(room_state1)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self.render_mode = render_mode
        starting_observation = self.render()

        return np.expand_dims(self.room_state, 0)
        
    def step(self, action, observation_mode='rgb_array'):
        observation, reward, done, info = super(SokobanEnv_Source3, self).step(action, observation_mode=observation_mode)
        return np.expand_dims(self.room_state, 0), reward, done, info


class SokobanEnv1(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        super(SokobanEnv1, self).__init__(**kwargs)


class SokobanEnv2(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 5)
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 40)
        super(SokobanEnv2, self).__init__(**kwargs)


class SokobanEnv_Small0(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 2)
        super(SokobanEnv_Small0, self).__init__(**kwargs)


class SokobanEnv_Small1(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        super(SokobanEnv_Small1, self).__init__(**kwargs)


class SokobanEnv_Large0(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 300)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 43)
        super(SokobanEnv_Large0, self).__init__(**kwargs)


class SokobanEnv_Large1(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 300)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 43)
        super(SokobanEnv_Large1, self).__init__(**kwargs)


class SokobanEnv_Large1(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 300)
        kwargs['num_boxes'] = kwargs.get('num_boxes',5)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 43)
        super(SokobanEnv_Large1, self).__init__(**kwargs)


class SokobanEnv_Huge0(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 13))
        kwargs['max_steps'] = kwargs.get('max_steps', 300)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 5)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(SokobanEnv_Huge0, self).__init__(**kwargs)


class FixedTargets_Env_v0(FixedTargetsSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(FixedTargets_Env_v0, self).__init__(**kwargs)


class FixedTargets_Env_v1(FixedTargetsSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(FixedTargets_Env_v1, self).__init__(**kwargs)


class FixedTargets_Env_v2(FixedTargetsSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 2)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(FixedTargets_Env_v2, self).__init__(**kwargs)


class FixedTargets_Env_v3(FixedTargetsSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(FixedTargets_Env_v3, self).__init__(**kwargs)


class PushAndPull_Env_v0(PushAndPullSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(PushAndPull_Env_v0, self).__init__(**kwargs)


class PushAndPull_Env_v1(PushAndPullSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(PushAndPull_Env_v1, self).__init__(**kwargs)


class PushAndPull_Env_v2(PushAndPullSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 2)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(PushAndPull_Env_v2, self).__init__(**kwargs)


class PushAndPull_Env_v3(PushAndPullSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(PushAndPull_Env_v3, self).__init__(**kwargs)


class PushAndPull_Env_v4(PushAndPullSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 300)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(PushAndPull_Env_v4, self).__init__(**kwargs)


class PushAndPull_Env_v5(PushAndPullSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 300)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 5)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(PushAndPull_Env_v5, self).__init__(**kwargs)


class TwoPlayer_Env0(TwoPlayerSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 2)
        super(TwoPlayer_Env0, self).__init__(**kwargs)


class TwoPlayer_Env1(TwoPlayerSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        super(TwoPlayer_Env1, self).__init__(**kwargs)


class TwoPlayer_Env2(TwoPlayerSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        super(TwoPlayer_Env2, self).__init__(**kwargs)


class TwoPlayer_Env3(TwoPlayerSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        super(TwoPlayer_Env3, self).__init__(**kwargs)


class TwoPlayer_Env4(TwoPlayerSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        super(TwoPlayer_Env4, self).__init__(**kwargs)



class TwoPlayer_Env5(TwoPlayerSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        super(TwoPlayer_Env5, self).__init__(**kwargs)

class Boxban_Env0(BoxobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'unfiltered')
        kwargs['split'] = kwargs.get('split', 'train')
        super(Boxban_Env0, self).__init__(**kwargs)

class Boxban_Env0_val(BoxobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'unfiltered')
        kwargs['split'] = kwargs.get('split', 'valid')
        super(Boxban_Env0_val, self).__init__(**kwargs)

class Boxban_Env0_test(BoxobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'unfiltered')
        kwargs['split'] = kwargs.get('split', 'test')
        super(Boxban_Env0_test, self).__init__(**kwargs)

class Boxban_Env1(BoxobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'medium')
        super(Boxban_Env1, self).__init__(**kwargs)

class Boxban_Env1_val(BoxobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'medium')
        kwargs['split'] = kwargs.get('split', 'valid')
        super(Boxban_Env1_val, self).__init__(**kwargs)
