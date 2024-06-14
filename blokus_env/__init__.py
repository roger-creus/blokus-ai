from gymnasium.envs.registration import register

register(
    id='BlokusEnv-v0',
    entry_point='blokus_env.blokus_env:BlokusEnv',
)
