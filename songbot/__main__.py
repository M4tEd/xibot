"""Live bot entrypoint (discord.py gateway client).

NEVER run on this machine during the mission: a Netskope agent blocks/flags
all Discord domains. Behavioral validation goes through the headless harness
(`python -m songbot.harness`).
"""
