# SUCCESSION -- what keeps Iris alive, and what to do when the owner cannot

Written 2026-10-30 (docs/ARCHITECTURE_PLAN.md section 6, ADR 0011). Three
ledgers: the accounts and money that lapse, the physical acts a person must
do, and the runbooks for the gaps the software does not close by itself.
HANDBOOK.md is the day-to-day page; this one is for the month the owner is
away, or the year after.

Nothing secret is written here. Credentials live in
configs/config_private.py on each machine (never in git) and in the
owner's password manager; this page says which exist and what each is for.

## Ledger 1: accounts and money

| What | Where | What dies when it lapses | Where the credential is |
|---|---|---|---|
| Domain irisscience.org | the registrar in the owner's password manager | the public site and the subscriber mail | registrar login |
| Web host | a Linux host on the tailnet (100.91.17.119, user taylor); Caddy serves /srv/iris-site, pulled from GitHub every 5 minutes; `iris-subscribe` holds the subscriber list and sends announcements | the site and the announcements | ssh key `~/.ssh/id_ed25519_iris` on the PC only |
| Cloudflare tunnel | the public face of the web host | the public site (the tailnet copy still works) | Cloudflare login |
| Tailscale | every machine: PC, Spark, web host, the owner's phone | the chat from anywhere, the Spark's sync, publishing, the site pull trigger | Tailscale admin login; each machine is enrolled |
| GitHub | taylorhogan/srt (this repo; CI gates `release`) and taylorhogan/taylorhogan.github.io (the site) | `update` stops pulling; publishing stops; CI stops | the owner's GitHub account; both machines hold push rights through their git credentials |
| Pushover | the phone app and one application token | every alert | `cfg["pushover"]` in config_private.py |
| Open-Meteo / NWS | no account | nothing: the forecast has no key | - |
| `cfg["weather"]["api_key"]` | a legacy weather-service key still in config_private.py; nothing reads it (checked 2026-10-30) | nothing | may be left blank |
| Anthropic API key | writes the gallery prose on the Spark (D2) | the cards say "facts only"; nothing stops | `cfg["anthropic"]` in the Spark's config_private.py (not yet present on 2026-10-30; the `anthropic` package in the Spark's venv) |
| Mastodon | optional mirror, off | nothing | `cfg["mastodon"]` |
| OneDrive | IrisBackup: the daily zip of the irreplaceable state (scripts/backup_state.py) | restores after a disk loss | the owner's Microsoft account |
| Electricity and internet at the observatory | the house | everything | - |

The order to renew when money is short: Tailscale and the domain first, the
web host second, the API key last.

## Ledger 2: physical acts

docs/MAINTENANCE.md is the schedule and the procedures (roof track, guide
wheels, gasket and ice, mirror, collimation, humidity, power, flats, disks,
cameras). Do not duplicate it here; keep its log current.

Two acts are not maintenance and belong to whoever is on site:

- **Closing the roof when the software cannot** (HANDBOOK section 5). The
  gate opener has a manual release; the scope must be parked first.
- **Replacing a camera, a plug or the PC.** Each is a constant somewhere in
  configs/config_public.py or local/ (the camera poses, the parked-scope
  reference, the roof tag references). CLAUDE.md names each file. Re-record
  the references with the scripts named there before trusting the roof.

## Ledger 3: the gaps, as runbooks

**Who gets the alerts.** Pushover goes to the owner's phone. To add a
successor: install Pushover on their phone, add their user key as a second
delivery in the owner's Pushover application (or a group key), and put the
group key in `cfg["pushover"]["user"]`. Until then an unacknowledged
priority-2 alert repeats on one phone only.

**The merge after the owner.** CI fast-forwards `release` only on green.
Anyone with push rights to taylorhogan/srt can land a fix: commit to `main`,
wait for the green run, `update` in the chat. A successor needs push rights
granted in GitHub by the owner now, not later.

**Per-hardware degraded behaviour** (what already happens by itself):

| If this fails | The machine | You |
|---|---|---|
| the north roof camera | falls back to the inside camera for the roof; a read neither answers is unknown, which refuses to move | fix the camera; a hold may need `resolve!` |
| the inside camera | the north camera decides alone | fix it |
| the mount (PWI4 unreachable) | pre-flight fails, a priority-2 push, no roof open | PWI4 on the PC; Pegasus Unity |
| N.I.N.A mid-run | the night ends normally (CAPTURE_LOST) | open N.I.N.A in the morning |
| the scheduler or conductor process | relaunched alone, one push | nothing; read iris.log |
| the chat server | start_srt restarts everything; the watchdogs push directly | restart start_srt if it stays down |
| the Spark | no renders, no publishing, no suggestions; imaging continues | restart it; morning_trigger catches up on the next run |
| the PC's disk | the night is called off at noon below the floor, priority 2 | `purge`, `purge bad go` |
| the Spark's disk | sync and render refuse below the floor, priority 2 | `scripts/spark_prune.py --go` |
| the forecast | no night is planned without one | wait |
| SIMBAD | the queue carries positions and types; suggestions use the catalogue | nothing |

**Restoring from the backup.** OneDrive IrisBackup holds daily zips of
local/ minus the bulk captures. After a disk loss: clone the repo, restore
config_private.py from the password manager, unzip the newest backup into
local/, run scripts/start_srt.bat. The sky solution and the parked-scope
reference are in the zip; without them the roof will not move until they
are re-recorded.

**Scheduled jobs** are the only things not in the repo: docs/tasks/README.md
(the PC's schtasks, with the commands to recreate each) and
docs/tasks/spark_crontab.txt (the Spark's crontab, paste with `crontab -`).
The test suite checks that every job names a script that exists.

## The three flags, for the record

As of 2026-10-30 `planner.auto_stop`, `nina.filter_advisor` and
`publish.gate` are in shadow. The owner flips each after judging its noon
lines; HANDBOOK section 7 lists them. A successor who finds them still in
shadow may leave them there: in shadow the observatory images every clear
night and publishes nothing, which is the safe default.
