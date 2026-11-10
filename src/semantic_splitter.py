""" This document handles semantic splitting of text into chunks for embedding and retrieval. 
    It:
        - Detects sentence boundaries and extracts sentences
        - Vectorizes sentences using a transformer model
        - Computes cosine similarity between sentences to determine semantic similarity and creates chunks of semantically similar sentences
        - Calculates the semantic distance
        - Splits text where semantic distance exceeds a threshold
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from networkx.algorithms import threshold

class SemanticSplitter:
    def __init__(self, model_name="all-MiniLM-L6-v2", distance_threshold_percentile: float = 90.0):
        """ 
            Use the model and matrix operations to find semantic boundaries in the text.
            Arguments:
                model_name (str): The name of the transformer model to use for embedding sentences.
                distance_threshold_percentile (float): The percentile of the semantic distance distribution to use as a threshold for splitting text into chunks. Sentences with a semantic distance above this threshold will be split into separate chunks.
        """
        self.model = SentenceTransformer(model_name)
        self.percentile = distance_threshold_percentile
        
    def _split_into_sentences(self, text: str) -> list:
        """ Split the text into sentences using regex. """
        sentence_endings = re.compile(r'(?<=[.!?]) +')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str) -> list:
        """ 
            Split the text into semantically similar chunks based on cosine similarity of sentence embeddings.
            Arguments:
                text (str): The input text to be chunked.
            Returns:
                list: A list of semantically similar chunks of text.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
    
        if len(sentences) == 1:
            return [{"chunk": sentences[0], "sentence_count": 1}]
        
        # 1. Batch encode sentences to get embeddings
        embeddings = self.model.encode(sentences,
                                       batch_size=32,
                                       show_progress_bar=True,
                                       normalize_embeddings=True)
        # 2. Compute cosine similarity matrix | Dot product of normalized embeddings
        similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        distances = 1 - similarities
        
        # 3. Determine threshold for splitting based on the specified percentile
        threshold = np.percentile(distances, self.percentile)
        
        # 4. Find split indices where the semantic distance exceeds the threshold
        split_indices = np.where(distances > threshold)[0] + 1  # +1 to get the index of the next sentence after the split point
        
        # 5. Create chunks based on the split indices
        chunks = []
        start_idx = 0
        for split_idx in list(split_indices) + [len(sentences)]:
            chunk_sentences = sentences[start_idx:split_idx]
            chunk_text = " ".join(chunk_sentences)
            chunks.append({
                "chunk": chunk_text, 
                "sentence_count": len(chunk_sentences),
                "start_sentence_index": start_idx,
                "end_sentence_index": split_idx - 1
            })
            start_idx = split_idx
        return chunks