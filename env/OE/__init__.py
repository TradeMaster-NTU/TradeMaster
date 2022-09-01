from gym.envs.registration import register

# register(
#     id='ctc-executioner-v0',
#     entry_point='env.OE.execution_env:ExecutionEnv'
# )

register(
    id='ctc-executioner-v1',
    entry_point='env.OE.execution_env:ExecutionEnv_TwoActions'
)
