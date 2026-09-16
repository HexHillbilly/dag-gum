"""dag-gum — the weight-free steering sandwich.

SCP/SSP condition the surface *before* generation (top bread), AVG governs the
*middle* (the meat), and the two-stage synthesis closes it *after* (bottom bread).
The "gum" is the binder: this package wires pre-gen → mid-gen → synthesis behind
one `steer()` call, and routes tool-needing turns through the persona-drop gate
so the persona never corrupts tool cognition.
"""
from gum.sandwich import Sandwich

__all__ = ["Sandwich"]
