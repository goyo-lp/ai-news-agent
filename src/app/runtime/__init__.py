"""Per-run environment and workspace lifecycle for the `propose` command.

These are the steps that surround a spine run — making the search backend
available, giving the run a clean workspace, resolving the writer's voice
input, and shipping the results — extracted out of :mod:`app.main` so the CLI
module stays parse → dispatch → report.
"""
