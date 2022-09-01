from gym.envs.registration import register

register(
    id='ctc-executioner-v1',
    entry_point='execution:ExecutionEnv_TwoActions'
)