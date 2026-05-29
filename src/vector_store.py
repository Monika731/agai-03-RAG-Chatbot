import os
import json
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHROMA_DIR = 'data/chroma_db'


class VectorStoreManager:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.docs_col = self.client.get_or_create_collection(
            name="coursera_docs",
            metadata={"hnsw:space": "cosine"},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "],
        )

    def embed(self, texts: list) -> list:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def index_documents(self, courses_file: str = 'data/raw/all_courses.json'):
        """Chunk and embed all course documents into Chroma."""
        with open(courses_file, 'r', encoding='utf-8') as f:
            courses = json.load(f)

        chunks, ids, metas = [], [], []

        for course in courses:
            content = (
                f"Title: {course.get('title', '')}\n"
                f"Description: {course.get('description', '')}\n"
                f"Level: {course.get('level', '')}\n"
                f"Duration: {course.get('duration', '')}\n"
                f"Rating: {course.get('rating', '')}\n"
                f"Instructor: {course.get('instructor', '')}\n"
                f"Skills: {', '.join(course.get('skills', []))}\n"
                f"Content: {course.get('raw_text', '')}"
            )
            slug = course.get('url', '').rstrip('/').split('/')[-1]
            for j, chunk in enumerate(self.splitter.split_text(content)):
                chunks.append(chunk)
                ids.append(f"{slug}_chunk_{j}")
                metas.append({
                    'url': course.get('url', ''),
                    'title': course.get('title', ''),
                })

        self._batch_add(self.docs_col, chunks, ids, metas)
        logger.info(f"Indexed {len(chunks)} document chunks from {len(courses)} courses.")

    def _batch_add(self, collection, texts, ids, metas, batch_size=50):
        for i in range(0, len(texts), batch_size):
            t = texts[i:i+batch_size]
            collection.add(
                documents=t,
                embeddings=self.embed(t),
                ids=ids[i:i+batch_size],
                metadatas=metas[i:i+batch_size],
            )

    def search_docs(self, query: str, n_results: int = 3) -> list:
        results = self.docs_col.query(
            query_embeddings=self.embed([query]),
            n_results=n_results,
        )
        return [
            {'text': doc, 'metadata': meta, 'distance': dist}
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0],
            )
        ]

    def doc_count(self) -> int:
        return self.docs_col.count()


def build_vector_store():
    """Index course documents into ChromaDB. Q/A pairs are searched in-memory at runtime."""
    manager = VectorStoreManager()
    logger.info("Indexing course documents...")
    manager.index_documents()
    logger.info("Vector store ready.")


if __name__ == "__main__":
    build_vector_store()
