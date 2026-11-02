"""Rule-based scam detection package."""
from .engine import score_transcript

__all__ = ["score_transcript"]