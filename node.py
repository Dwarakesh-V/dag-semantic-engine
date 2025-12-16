# Built-in
from typing import List

# Downloaded
import numpy as np

class node:
    def __init__(self, idx: str, intent: str, examples: List[str] = None):
        self.id = idx
        self.intent = intent
        self.examples = examples or []
        self.sp = False
        self.children: List["node"] = []

        # cached embeddings
        self.example_embeddings: np.ndarray | None = None
