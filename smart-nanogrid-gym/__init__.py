from gym.envs.registration import registry, register, make, spec


register(
     id='ChargingEnv-v0',
     entry_point='smart-nanogrid-gym.envs:ChargingEnv',
     max_episode_steps=200,
)
