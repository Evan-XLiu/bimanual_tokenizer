"""BTT ablation: convention retained, arm normalization and weights untied."""

from praxis.codecs.tokens.btt_independent.config import BTTIndependentConfig
from praxis.codecs.tokens.btt_independent.encoder import BTTIndependentEncoder
from praxis.codecs.tokens.btt_independent.tokenizer import BTTIndependentTokenizer

__all__ = ["BTTIndependentConfig", "BTTIndependentEncoder", "BTTIndependentTokenizer"]
