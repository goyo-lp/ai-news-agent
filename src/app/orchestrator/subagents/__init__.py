"""Isolated adaptive workers the coordinator delegates to via ``task()``.

Each subagent is a :class:`deepagents.SubAgent` spec (name, description, system
prompt, tools, model) run in its own context; only its compressed on-disk
artifact crosses back to the coordinator.
"""
