"""Registers the 'text2sql_native_think' environment in every Python process that has this directory on PYTHONPATH
(rollout workers are separate Ray processes, and the skyrl_gym registry is per process). Never raises."""
try:
    from skyrl_gym.envs import register
    from skyrl_gym.envs.registration import registry
    if "text2sql_native_think" not in registry:
        register(id="text2sql_native_think", entry_point="sql_native_think_env:SQLNativeThinkEnv")
except Exception:
    pass
