from re import match
import gym
from gym.vector.vector_env import VectorEnv
import numpy as np

class MaskVelocityWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to mask velocity terms in
    observations. The intention is the make the MDP partially observatiable.
    """
    def __init__(self, env: gym.Env):
        super(MaskVelocityWrapper, self).__init__(env)
        if issubclass(type(env), VectorEnv):
            env_name: str = env.envs[0].unwrapped.spec.id
        else:
            env_name: str = env.unwrapped.spec.id

        if env_name == "CartPole-v0":
            self.mask = np.array([1., 0., 1., 0.])
        elif env_name == "CartPole-v1":
            self.mask = np.array([1., 0., 1., 0.])
        elif env_name == "Pendulum-v0":
            self.mask = np.array([1., 1., 0.])
        elif env_name == "LunarLander-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1,])
        elif env_name == "LunarLanderContinuous-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1,])
        else:
            raise NotImplementedError

    def observation(self, observation):
        return  observation * self.mask


class PerturbationWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to mask velocity terms in
    observations. The intention is the make the MDP partially observatiable.
    """
    def __init__(self, env: gym.Env, sigma: float):
        super(PerturbationWrapper, self).__init__(env)
        self.sigma = sigma

    def observation(self, observation):
        return  observation + np.random.randn(*(observation.shape)) * self.sigma