# ADR 0009: Publishers and Spark jobs stay cron-shaped

Status: accepted 2026-08-28

The 5-minute publishers and the Spark morning jobs are periodic,
idempotent and stateless; daemonizing them buys supervision burden and
nothing else. They become CLIENTS of the conductor's API instead of
readers of state files. Windows Scheduled Tasks and Spark cron survive --
they are also the most heir-comprehensible scheduling technology
available.
