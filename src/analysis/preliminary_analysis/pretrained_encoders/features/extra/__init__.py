"""相関で絞る前の、新しい特徴量の候補（design.md §13）。

既存の 15 個（feature_builder.FEATURE_NAMES）とは分けて置く。相関で絞ったあと、残ったものを
FEATURE_NAMES に一斉に足す（design.md §12.5・§13.8）。
"""
