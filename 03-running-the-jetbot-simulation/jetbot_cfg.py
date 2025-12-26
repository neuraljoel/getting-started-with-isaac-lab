# jetbot_cfg.py

import isaaclab.sim as sim_utils

# Use a raw string so Windows backslashes don't become escape sequences.
JETBOT_USD_PATH = r"C:\Users\joelm\Desktop\isaaclab\IsaacLab\assets\robots\jetbot.usd"

# Config for spawning Jetbot from a USD file.
JETBOT_CFG = sim_utils.UsdFileCfg(
    usd_path=JETBOT_USD_PATH,
    scale=(1.0, 1.0, 1.0)
)