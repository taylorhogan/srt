# ADR 0010: The website grows; the webchat stays an operator console

Status: accepted 2026-08-28

Everything a VIEWER wants (global state, target board, finished pictures)
goes to irisscience.org, fed from the same journal the chat renders. New
features default to the website; chat receives only what an operator
needs (safety commands, diagnostics, cancels). Auto-published gallery
entries are generated and committed by the publish service; authored
prose remains human.
