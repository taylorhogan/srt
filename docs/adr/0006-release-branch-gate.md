# ADR 0006: CI-gated release branch is the deploy

Status: accepted 2026-08-28

GitHub Actions runs the suite on every push to main; on green it
fast-forwards the release branch; both observatory pull points (boot and
the update command) do fetch + merge --ff-only origin/release. A red main
means the observatory keeps running yesterday's green -- the correct
failure mode for an unattended telescope. ff-only keeps the dev==prod
checkout untangled: local work ahead of release simply runs as-is.
Emergency bypass: push a sha to release by hand.
