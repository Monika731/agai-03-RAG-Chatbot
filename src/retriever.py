import csv
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from src.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QA_CONFIDENCE_THRESHOLD = 0.35
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
QA_FILE = 'data/processed/qa_dataset.csv'


class HybridRetriever:
    """
    Two-stage retrieval:
    1. In-memory cosine similarity search on Q/A pairs (no ChromaDB)
    2. If no confident match, fall back to ChromaDB vector search on course documents
    """

    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.qa_pairs = []
        self.qa_embeddings = None
        self._load_qa()

    def _load_qa(self):
        try:
            with open(QA_FILE, 'r', encoding='utf-8') as f:
                self.qa_pairs = list(csv.DictReader(f))
            if self.qa_pairs:
                questions = [row['question'] for row in self.qa_pairs]
                self.qa_embeddings = self.model.encode(questions, convert_to_numpy=True)
                logger.info(f"Loaded {len(self.qa_pairs)} Q/A pairs into memory.")
        except FileNotFoundError:
            logger.warning("Q/A dataset not found — skipping in-memory Q/A search.")

    def _search_qa(self, query: str, n_results: int = 3) -> list:
        if self.qa_embeddings is None or len(self.qa_pairs) == 0:
            return []

        query_emb = self.model.encode([query], convert_to_numpy=True)

        # Cosine similarity = dot product of normalized vectors
        qa_norms = self.qa_embeddings / np.linalg.norm(self.qa_embeddings, axis=1, keepdims=True)
        q_norm = query_emb / np.linalg.norm(query_emb)
        similarities = (qa_norms @ q_norm.T).flatten()

        # Convert similarity to distance (1 - similarity) to stay consistent with ChromaDB
        distances = 1 - similarities
        top_indices = np.argsort(distances)[:n_results]

        return [
            {
                'question': self.qa_pairs[i]['question'],
                'answer': self.qa_pairs[i]['answer'],
                'source_page': self.qa_pairs[i].get('source_page', ''),
                'course_title': self.qa_pairs[i].get('course_title', ''),
                'distance': float(distances[i]),
            }
            for i in top_indices
        ]

    def retrieve(self, query: str) -> dict:
        qa_results = self._search_qa(query, n_results=3)
        doc_results = self.vector_store.search_docs(query, n_results=3)

        best_qa = qa_results[0] if qa_results else None
        used_qa = bool(best_qa and best_qa['distance'] < QA_CONFIDENCE_THRESHOLD)

        return {
            'used_qa': used_qa,
            'qa_results': qa_results,
            'doc_results': doc_results,
            'best_qa': best_qa,
            'primary_source': 'Q/A Dataset' if used_qa else 'Vector Search',
        }
