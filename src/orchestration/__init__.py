"""Reusable application orchestration for CivicLens interfaces."""

__all__ = ["BACKEND_NOT_READY_MESSAGE", "route_question"]


def __getattr__(name: str):
    """Preserve package exports without eagerly importing the router."""

    if name in __all__:
        from src.orchestration import question_router

        return getattr(question_router, name)
    raise AttributeError(name)
