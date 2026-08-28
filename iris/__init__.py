"""iris — the observatory's formal core.

Grown strangler-fig style beside the legacy layout (see
docs/ARCHITECTURE_PLAN.md). Nothing in here may import from cmd_processing,
end_points, or scripts: dependency arrows point INWARD. Doers emit journal
events; consoles (chat, website) render them.
"""
