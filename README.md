# SongBot

A Heardle-style daily song-guessing game for Discord. Every day at a configured
time, SongBot posts a very short audio snippet of a song to a channel. Players
press **Hear more** to privately unlock longer snippets (worth fewer points),
press **Guess** to submit answers through a modal, and climb a persistent
leaderboard with win streaks. The next day's post opens by revealing the
previous song and its winners.

Songs come from two catalog sources behind a provider abstraction: a YouTube
playlist (via yt-dlp) and/or a local audio directory (via mutagen tags).

## Features

- **Daily post** — a new challenge every day at `DAILY_POST_TIME` in the
  configured `TIMEZONE`: an embed, a 1-second snippet attachment, and three
  buttons. Posting is idempotent (a restart never double-posts), and songs are
  never repeated until the whole catalog has been used.
- **Hear-more ladder** — each player starts at the 1s snippet (worth 100
  points) and can press **Hear more** up to 4 times to unlock 2s / 4s / 8s /
  16s snippets, worth 75 / 50 / 30 / 15 points. Unlocks are per-user and
  private (ephemeral replies); the snippet always arrives as
  `songbot-snippet.mp3` so the filename never leaks the song.
- **Modal guessing** — **Guess** opens a modal text field (no slash-command
  typing). Guesses are fuzzy-matched (rapidfuzz, threshold 85 after
  normalization, plus a per-token fallback that forgives a typo in one token —
  including inside a combined title+artist guess and on short names like
  "Halo"); matching the title **or** the artist is correct (`GUESS_MATCH_MODE`
  can restrict correctness to just the title or just the artist). Feedback is
  ephemeral; wrong guesses cost nothing. Max 6 guesses per player per day.
- **Scoring & bonus** — a correct guess awards the points of the player's
  current snippet level. Naming **both** artist and title in a single guess
  earns a 1.5× bonus (rounded half-up: 75 → 113, 15 → 23; only in the default
  `either` match mode). The first correct
  guess triggers a public "🎉 @user guessed today's song…" announcement that
  never reveals the song.
- **Streaks** — consecutive calendar days (configured timezone) with at least
  one solve. Missing a day resets the current streak; your best streak is kept.
- **Leaderboard** — **Leaderboard** shows an ephemeral top 10 by total points
  (tiebreak: wins, then user id), with wins and current streaks.
- **Admin commands** — guild-scoped slash commands, gated on the invoker's
  **Manage Server** permission:
  - `/songbot-post` — post today's challenge now (idempotent)
  - `/songbot-skip` — replace today's song with a different one (refused once
    the challenge is revealed or anyone has solved it)
  - `/songbot-reload` — refresh the song catalog from its sources
  - `/songbot-fixsong` — correct the title/artist of a challenge's song when
    the catalog parsed it badly (user-uploaded video titles are messy).
    Targets the latest challenge's song by default, or a specific challenge
    with `date:` (YYYY-MM-DD); `artist:` may be omitted to keep the current
    one. The correction applies to new guesses immediately and is re-applied
    after every catalog reload (the `song_overrides` table); already-recorded
    guesses keep their original results. The ephemeral ack shows the old →
    new metadata (admin-only, ephemeral — public posts never name songs)
- **Next-day reveal** — each new daily post first reveals the previous
  challenge: the song title/artist and a winners summary (or "nobody got it").

## Architecture

Strict engine/adapter split: the game engine is pure Python and never imports
discord.py; the Discord adapter is thin and contains no game rules. A headless
harness replaces the adapter's network transport with a recorder, so the real
button/modal handlers can be driven locally without any Discord contact.

```
┌──────────────────────────── discord.py adapter (thin) ───────────────────────────┐
│  Bot client · Views (buttons) · Modals · Embeds · Admin commands · Health server │
│  Translates Discord events → engine calls; formats engine results → Discord UI   │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │ (pure Python calls, no Discord types cross this line)
┌───────────────────────────────────▼─────────────────────────────────────────────┐
│                              GAME ENGINE (pure, fully testable)                  │
│  Daily challenge lifecycle · Guess processing (rapidfuzz) · Scoring ladder ·     │
│  Snippet unlocks · Streaks · Leaderboard · Scheduler                             │
└──────────────┬───────────────────────────┬──────────────────────────┬───────────┘
               │                           │                          │
      ┌────────▼────────┐       ┌──────────▼──────────┐     ┌─────────▼─────────┐
      │ Catalog providers│       │  Snippet generator  │     │  SQLite storage   │
      │ local dir (mutagen│       │  ffmpeg exact cuts, │     │  challenges, users│ │
      │  + filename) /    │       │  random offset,     │     │  scores, streaks  │
      │  YouTube (yt-dlp) │       │  disk cache         │     │                   │
      └───────────────────┘       └─────────────────────┘     └───────────────────┘
```

```
songbot/
├── __main__.py          # LIVE bot entrypoint (python -m songbot)
├── config.py            # .env loading + validation (Settings dataclass)
├── db.py                # SQLite schema, migrations, connection helper
├── catalog/             # Song type, provider protocol, local + YouTube providers
├── snippets.py          # ffmpeg exact-duration snippet cuts + disk cache
├── matching.py          # normalization + fuzzy guess matching
├── engine.py            # GameEngine: lifecycle, guesses, scoring, streaks
├── scheduler.py         # pure time logic (next post time, day boundaries)
├── bot/                 # discord.py adapter (the ONLY package importing discord)
│   ├── client.py · views.py · modals.py · embeds.py · admin.py · health.py
└── harness/             # headless validation harness (no network)
    ├── fakes.py         # FakeInteraction + Recorder
    └── cli.py           # python -m songbot.harness <scenario>
```

## Requirements

- Python 3.12+ (managed via [uv](https://docs.astral.sh/uv/))
- ffmpeg + ffprobe on `PATH` (snippet generation) — macOS: `brew install ffmpeg`;
  Windows: `winget install ffmpeg`
- Network access to YouTube only if you enable the YouTube playlist provider

## Setup

```bash
git clone <repo-url> && cd SongBot

# Create the virtualenv and install the package + dev tools
uv venv --python 3.12
uv pip install -e . && uv pip install --group dev

# Recreate the synthetic demo music library at data/fixture-music/ (gitignored,
# so fresh clones must regenerate it; the script is idempotent and needs ffmpeg)
.venv/bin/python scripts/generate_fixture_music.py

# Configure
cp .env.example .env   # then edit .env (see below)
```

All commands below use the venv binaries directly (`.venv/bin/...`); activating
the venv (`source .venv/bin/activate`) works too. On Windows the venv layout
differs: use `.venv\Scripts\python.exe` in place of `.venv/bin/python`
(activation: `.venv\Scripts\Activate.ps1` in PowerShell,
`.venv\Scripts\activate.bat` in cmd), and set environment variables with
`$env:VAR = "value"` (PowerShell) or `set VAR=value` (cmd) instead of
`export VAR=value`.

## Configuration (`.env`)

Every key is read from `.env`, with `os.environ` taking precedence (including
empty-string overrides — e.g. `YOUTUBE_PLAYLIST_URL=""` disables that provider
for one invocation). Invalid configuration fails fast at startup with a single
error listing every problem.

| Key | Default | Meaning |
| --- | --- | --- |
| `DISCORD_BOT_TOKEN` | *(required)* | Bot token from the Discord Developer Portal. Only used by the live bot; the harness and tests never use it. |
| `DISCORD_GUILD_ID` | `""` (unset) | Optional bootstrap: ID of one server (guild) to configure at startup. Must be set together with `DISCORD_CHANNEL_ID` — both or neither. Right-click the server name → *Copy Server ID* (Developer Mode). |
| `DISCORD_CHANNEL_ID` | `""` (unset) | Optional bootstrap: ID of the channel the daily challenge is posted to in the bootstrap guild. Right-click the channel → *Copy Channel ID*. |
| `YOUTUBE_PLAYLIST_URL` | `""` (disabled) | Public/unlisted YouTube playlist used as a song catalog. Empty disables the YouTube provider. |
| `LOCAL_MUSIC_DIR` | `""` (disabled) | Directory of local audio files (mp3/m4a/flac/ogg) used as a song catalog; artist/title come from tags with an `Artist - Title.ext` filename fallback. Empty disables the local provider. For a self-contained demo point it at `./data/fixture-music` (a generated library of 8 synthetic 30s songs, gitignored — fresh clones recreate it with `.venv/bin/python scripts/generate_fixture_music.py`); for real use point it at your own music folder. |
| `DAILY_POST_TIME` | `12:00` | Daily post time, strict zero-padded `HH:MM` (00:00–23:59) in `TIMEZONE`. |
| `TIMEZONE` | `America/Halifax` | IANA timezone name governing post time, challenge dates, and streaks. |
| `MAX_GUESSES_PER_DAY` | `6` | Guesses per player per challenge (≥ 1). Wrong guesses are free but count toward the limit; the winning guess counts too. |
| `SNIPPET_LENGTHS` | `1,2,4,8,16` | Comma-separated snippet durations in seconds (the hear-more ladder). First entry ships with the daily post. |
| `SNIPPET_POINTS` | `100,75,50,30,15` | Comma-separated points per ladder level; must have the same number of entries as `SNIPPET_LENGTHS`. |
| `BOTH_CORRECT_MULTIPLIER` | `1.5` | Score multiplier when one guess matches both artist and title (> 0; rounded half-up). Only applies when `GUESS_MATCH_MODE=either`. |
| `GUESS_MATCH_MODE` | `either` | What counts as a correct guess: `either` (title OR artist), `title` (title only), or `artist` (artist only — handy for a single-artist catalog). The both-fields bonus only exists in `either` mode. |
| `DATABASE_PATH` | `./data/songbot.db` | SQLite database file (created and migrated automatically). |
| `SNIPPET_CACHE_DIR` | `./data/snippets` | On-disk cache for generated snippets (keyed by challenge). |
| `HEALTH_PORT` | `3108` | Port for the `GET /health` liveness endpoint. |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`. |
| `DISCORD_API_BASE` | `https://discord.com/api/v10` | Discord REST API base URL, applied before login. Only relevant to live runs — see the note below. |

At least one catalog provider must be enabled or posting fails with a
`catalog_empty` error.

### Creating the Discord bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
   → **New Application**.
2. **Bot** tab → **Reset Token** → copy the token into `DISCORD_BOT_TOKEN`.
   No privileged gateway intents are needed (the bot only uses the `guilds`
   intent).
3. Invite the bot with the `bot` and `applications.commands` scopes and
   permissions integer **116736** (Send Messages + Embed Links + Attach Files +
   Read Message History):

   ```
   https://discord.com/oauth2/authorize?client_id=<YOUR_APPLICATION_ID>&scope=bot+applications.commands&permissions=116736
   ```

4. Either copy your server and channel IDs (enable *User Settings → Advanced →
   Developer Mode*, then right-click) into `DISCORD_GUILD_ID` /
   `DISCORD_CHANNEL_ID` to pre-configure one server, or leave both unset and
   run `/songbot-setup` in each server after inviting the bot.

Admin slash commands are registered guild-scoped (instant availability in
every joined server) and additionally require the invoker to have the
**Manage Server** permission.

### Multiple servers

One bot instance serves any number of servers. Each server picks its own
post channel — channel names are never involved (the bot stores the channel's
unique ID), so servers may name their channels however they like:

1. Invite the bot to the server (same invite URL as above).
2. In that server, run `/songbot-setup` and pick the channel from Discord's
   channel picker (requires **Manage Server**). That's it — the next daily
   post lands there, and `/songbot-post` works immediately.

Configuration lives in the `guild_settings` table, so adding a server never
requires a redeploy. Re-running `/songbot-setup` moves future posts to the
new channel (the reveal of an already-posted challenge still goes to the
channel it was posted in). Removing the bot from a server drops that server's
configuration; its game history is kept and resumes if the bot is re-added
and set up again.

The `DISCORD_GUILD_ID`/`DISCORD_CHANNEL_ID` pair is a bootstrap for **one**
server (typically your own): it is upserted on every startup, so for that
server the env wins over `/songbot-setup` — edit `.env` to change it.

The post schedule (`DAILY_POST_TIME`/`TIMEZONE`) is global: every server
posts at the same wall-clock time.

## Running the live bot

```bash
.venv/bin/python -m songbot
```

The bot connects to the Discord gateway, syncs the admin commands, starts the
health endpoint (`GET http://127.0.0.1:3108/health` →
`{"status":"ok","mode":"live","guild":...}`), posts today's challenge if none
exists, and then posts daily at `DAILY_POST_TIME`.

> **Important:** run the live bot on a machine whose network can reach Discord.
> If your machine sits behind a network security agent that blocks or flags
> Discord domains, `python -m songbot` **must be run on a different,
> unrestricted machine** (or network path) — the bot cannot log in or post
> while Discord traffic is intercepted. On such restricted machines you can
> still run the full test suite and the headless harness (below), which make
> zero Discord requests.
>
> `DISCORD_API_BASE` exists for live deployments on networks where the Discord
> API must be reached through a different base URL (e.g. an allowlisted
> reverse proxy in front of `discord.com/api/v10`). Leave it unset for normal
> use; it only affects the live bot, never the harness or tests.

## Running the headless harness locally

The harness boots the real engine and the real view/modal callback code with a
simulated Discord transport, records every outgoing message/embed/attachment,
and prints a JSON transcript. It never constructs a Discord client and never
touches the network — perfect for local development and for trying the game
loop end to end.

```bash
.venv/bin/python -m songbot.harness <scenario> [--now "ISO-8601"]
```

Scenarios: `post` · `hear-more --user U [--times N]` · `guess --user U --text T` ·
`leaderboard --user U` · `advance-day` · `admin-setup [--channel C] |
admin-post|admin-skip|admin-reload --as-admin|--as-non-admin` ·
`admin-fixsong --title T [--artist A] [--date YYYY-MM-DD]
--as-admin|--as-non-admin` · `status` · `reset` · `serve` (health endpoint
only).

The harness drives one guild per run: the `DISCORD_GUILD_ID`/
`DISCORD_CHANNEL_ID` pair when set, else the deterministic `harness-guild`/
`harness-channel` defaults (seeded into `guild_settings` on startup, exactly
like the live env bootstrap).

`--now` injects the clock (naive timestamps are read as UTC); `--user alice`
uses the bare name as a stable user id (`--user 42:alice` sets an explicit id).

### Example: a full day of play, then the next-day reveal

Use an isolated database and cache so the demo doesn't touch real state, and a
local-only catalog. On a fresh clone, first run `.venv/bin/python
scripts/generate_fixture_music.py` to recreate `data/fixture-music/`:

```bash
export DATABASE_PATH=/tmp/songbot-demo/songbot.db
export SNIPPET_CACHE_DIR=/tmp/songbot-demo/snippets
export YOUTUBE_PLAYLIST_URL=""                 # local-only catalog
export LOCAL_MUSIC_DIR=./data/fixture-music    # 8-song synthetic demo library

# Day 1: post the daily challenge (embed + 1s snippet + buttons)
.venv/bin/python -m songbot.harness post --now "2026-08-13T15:00:00Z"

# Peek at today's song (status is a test-only surface that reveals it)
.venv/bin/python -m songbot.harness status --now "2026-08-13T16:00:00Z"

# Alice unlocks longer snippets twice (1s -> 2s -> 4s; payout drops 100 -> 75 -> 50)
.venv/bin/python -m songbot.harness hear-more --user alice --times 2 --now "2026-08-13T16:05:00Z"

# Alice guesses wrong, then right — replace <title> with the song title
# that status printed above (the daily pick is deterministic per date+guild)
.venv/bin/python -m songbot.harness guess --user alice --text "some wrong guess" --now "2026-08-13T16:06:00Z"
.venv/bin/python -m songbot.harness guess --user alice --text "<title>" --now "2026-08-13T16:07:00Z"

# Bob checks the leaderboard (ephemeral top 10)
.venv/bin/python -m songbot.harness leaderboard --user bob --now "2026-08-13T17:00:00Z"

# Day 2: reveal yesterday's song + winners, then post the next challenge
.venv/bin/python -m songbot.harness advance-day

# Admin flows (permission simulation)
.venv/bin/python -m songbot.harness admin-skip --as-admin --now "2026-08-14T16:00:00Z"
.venv/bin/python -m songbot.harness admin-reload --as-non-admin
# Correct a badly parsed song's metadata (targets the latest challenge's song)
.venv/bin/python -m songbot.harness admin-fixsong --as-admin --title "Corrected Title" --artist "Corrected Artist"

# Start over
.venv/bin/python -m songbot.harness reset
```

Each command prints the recorded Discord payloads (`channel` = daily post,
`announcement` = solve announcements and reveals, `ephemeral` = per-user
replies, `modal` = the guess modal) plus resulting state. A same-day repeat
`post` prints `{"already_posted": true, "messages": []}`.

## Development

```bash
.venv/bin/pytest -q        # full test suite (unit + integration)
.venv/bin/ruff check       # lint
.venv/bin/mypy songbot     # strict type check
```

Test notes:

- Unit tests (`tests/unit/`) are pure and deterministic (injected clocks,
  seeded RNG, tmp dirs).
- Integration tests (`tests/integration/`) exercise the real fixture music
  library at `data/fixture-music/` (skipped automatically if absent), real
  ffmpeg/ffprobe snippet cuts, and the real YouTube playlist — those need
  network access to YouTube and are hardened against transient throttling.
- Tests never touch `data/` runtime state; everything runs in tmp dirs.

## First live run: manual playtest checklist

Run through this on your server the first time you deploy the live bot. It
maps 1:1 to the core game loop.

1. **Startup** — the bot logs in and shows online; `curl
   http://127.0.0.1:3108/health` returns `{"status":"ok","mode":"live",...}`.
2. **Daily post appears** — at `DAILY_POST_TIME` (or immediately on first run),
   the channel gets the "🎵 Daily Song" embed with a `songbot-snippet.mp3`
   attachment and three buttons (Hear more / Guess / Leaderboard). Nothing in
   the post reveals the song title or artist.
3. **Hear more is ephemeral** — press **Hear more**: only you see the reply,
   with a longer snippet attached and the new (lower) point value. Press again
   to walk the ladder up to the longest snippet; a fifth press is refused.
4. **Guess modal** — press **Guess**: a modal with a single text field opens.
   Submit a wrong guess → ephemeral ❌ feedback with guesses remaining, and no
   public message. Submit empty/whitespace → ephemeral "please enter a guess"
   notice that doesn't consume a guess.
5. **Correct guess + announcement** — submit the right title or artist →
   ephemeral ✅ with your points, plus a public "🎉 @you guessed today's song
   in N guesses for P points!" message that does **not** name the song. A
   guess naming both artist and title shows the 1.5× bonus (default `either`
   match mode only). Further guesses
   and Hear-more presses from you are refused afterwards.
6. **Leaderboard** — press **Leaderboard**: an ephemeral top-10 embed shows
   points, wins, and 🔥 streaks, and includes your new score.
7. **Next-day reveal** — the following day at post time, the bot first posts
   "🎶 Yesterday's Song, Revealed" naming the song and its winners (or
   "Nobody got it"), then posts the new daily challenge. Yesterday's buttons
   now answer "This challenge has closed."
8. **Admin commands** — as a non-admin, `/songbot-post` is denied. As a
   Manage-Server admin: `/songbot-post` is idempotent on an already-posted
   day, `/songbot-skip` replaces today's song with a fresh snippet,
   `/songbot-reload` refreshes the catalog and reports per-source counts.
9. **Restart safety** — restart the bot process: no duplicate post appears and
   the existing buttons keep working.
