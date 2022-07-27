from typing import overload
import numpy as np
import random

from .sokoban_env import SokobanEnv
from .sokoban_env_fixed_targets import FixedTargetsSokobanEnv
from .sokoban_env_pull import PushAndPullSokobanEnv
from .sokoban_env_two_player import TwoPlayerSokobanEnv
from .boxoban_env import BoxobanEnv

class SokobanEnv_Target(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 1)
        kwargs['max_steps'] = kwargs.get('max_steps', 30)
        kwargs['dim_room'] = kwargs.get('dim_room', (5,5))
        super(SokobanEnv_Target, self).__init__(**kwargs)

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
        starting_observation = self.render()

        return self.room_state

    def step(self, action, observation_mode='rgb_array'):
        observation, reward, done, info = super(SokobanEnv_Target, self).step(action, observation_mode=observation_mode)
        return self.room_state, reward, done, info


class SokobanEnv_Source1(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 1)
        kwargs['max_steps'] = kwargs.get('max_steps', 30)
        kwargs['dim_room'] = kwargs.get('dim_room', (5,5))
        super(SokobanEnv_Source1, self).__init__(**kwargs)

    def reset(self, second_player=False, render_mode='rgb_array', seed=12345):
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
        starting_observation = self.render()

        return self.room_state

    def step(self, action, observation_mode='rgb_array'):
        observation, reward, done, info = super(SokobanEnv_Source1, self).step(action, observation_mode=observation_mode)
        return self.room_state, reward, done, info


class SokobanEnv_Source2(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 1)
        kwargs['max_steps'] = kwargs.get('max_steps', 30)
        kwargs['dim_room'] = kwargs.get('dim_room', (5,5))
        super(SokobanEnv_Source2, self).__init__(**kwargs)

    def reset(self, second_player=False, render_mode='rgb_array', seed=12345):
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
        starting_observation = self.render()

        return self.room_state

    def step(self, action, observation_mode='rgb_array'):
        observation, reward, done, info = super(SokobanEnv_Source2, self).step(action, observation_mode=observation_mode)
        return self.room_state, reward, done, info


class SokobanEnv_Source3(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 1)
        kwargs['max_steps'] = kwargs.get('max_steps', 30)
        kwargs['dim_room'] = kwargs.get('dim_room', (5,5))
        super(SokobanEnv_Source3, self).__init__(**kwargs)

    def reset(self, second_player=False, render_mode='rgb_array', seed=12345):
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
        starting_observation = self.render()

        return self.room_state
        
    def step(self, action, observation_mode='rgb_array'):
        observation, reward, done, info = super(SokobanEnv_Source3, self).step(action, observation_mode=observation_mode)
        return self.room_state, reward, done, info

class SokobanEnv1(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        super(SokobanEnv1, self).__init__(**kwargs)


class SokobanEnv2(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 5)
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 40)
        super(SokobanEnv2, self).__init__(**kwargs)


class SokobanEnv_Small0(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 2)
        super(SokobanEnv_Small0, self).__init__(**kwargs)


class SokobanEnv_Small1(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        super(SokobanEnv_Small1, self).__init__(**kwargs)


class SokobanEnv_Large0(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
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
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 2)
        super(TwoPlayer_Env0, self).__init__(**kwargs)


class TwoPlayer_Env1(TwoPlayerSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        super(TwoPlayer_Env1, self).__init__(**kwargs)


class TwoPlayer_Env2(TwoPlayerSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        super(TwoPlayer_Env2, self).__init__(**kwargs)


class TwoPlayer_Env3(TwoPlayerSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        super(TwoPlayer_Env3, self).__init__(**kwargs)


class TwoPlayer_Env4(TwoPlayerSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        super(TwoPlayer_Env4, self).__init__(**kwargs)



class TwoPlayer_Env5(TwoPlayerSokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        super(TwoPlayer_Env5, self).__init__(**kwargs)

class Boxban_Env0(BoxobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'unfiltered')
        kwargs['split'] = kwargs.get('split', 'train')
        super(Boxban_Env0, self).__init__(**kwargs)

class Boxban_Env0_val(BoxobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'unfiltered')
        kwargs['split'] = kwargs.get('split', 'valid')
        super(Boxban_Env0_val, self).__init__(**kwargs)

class Boxban_Env0_test(BoxobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'unfiltered')
        kwargs['split'] = kwargs.get('split', 'test')
        super(Boxban_Env0_test, self).__init__(**kwargs)

class Boxban_Env1(BoxobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'medium')
        super(Boxban_Env1, self).__init__(**kwargs)

class Boxban_Env1_val(BoxobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'medium')
        kwargs['split'] = kwargs.get('split', 'valid')
        super(Boxban_Env1_val, self).__init__(**kwargs)
