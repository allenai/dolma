"""Mixture-agnostic helpers shared by resharding campaigns.

Nothing here knows about a particular mixture. `dispatch` builds poormanray
command lines for the worker lifecycle; the campaign-specific planning and
materialization workflow for the Dolma 3.5 14T mix lives in
`scripts/dolma3p5_resharding/`, which consumes these builders.
"""
