"""OPERATOR AI OS - JARVIS.

A personal, multi-agent, multi-model AI operating system.

Layering (each layer only depends on the ones above it):

    config / logging      -> process-wide settings and structured audit logging
    security              -> permissions, risk levels, human approval
    models                -> provider abstraction + model router + cost engine
    agents                -> declarative agent registry + runtimes
    core                  -> task engine, planner, orchestrator, JARVIS
"""

__version__ = "0.1.0"
