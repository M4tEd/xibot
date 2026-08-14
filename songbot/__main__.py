"""Live bot entrypoint (discord.py gateway client) — ``python -m songbot``.

NEVER run on this machine during the mission: a Netskope agent blocks/flags
all Discord domains. Behavioral validation goes through the headless harness
(``python -m songbot.harness``); the live playtest is the user's deferred
manual step, off-mission. Importing this module has no side effects — the bot
only starts under ``__name__ == "__main__"``.
"""

from songbot.bot.client import main

if __name__ == "__main__":
    raise SystemExit(main())
