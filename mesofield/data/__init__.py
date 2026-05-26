try:
    from .writer import CustomWriter, CV2Writer
except ImportError:  # pymmcore-plus not installed (analysis-only env)
    CustomWriter = None  # type: ignore[assignment,misc]
    CV2Writer = None  # type: ignore[assignment,misc]

__all__ = ["CustomWriter", "CV2Writer"]
