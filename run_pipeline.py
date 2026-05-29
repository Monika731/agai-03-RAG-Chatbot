"""
Run the full RAG pipeline in sequence:
  Phase 1: Scrape Coursera AI/ML course pages
  Phase 2: Generate synthetic Q/A pairs using Groq
  Phase 3: Build Chroma vector store
"""
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run():
    logger.info("=" * 60)
    logger.info("PHASE 1: Scraping Coursera pages")
    logger.info("=" * 60)
    from src.scraper import scrape_all_courses
    courses = scrape_all_courses()
    logger.info(f"Scraped {len(courses)} courses.\n")

    if not courses:
        logger.error("No courses scraped. Check your internet connection and try again.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("PHASE 2: Generating Q/A pairs with Groq")
    logger.info("=" * 60)
    from src.qa_generator import generate_all_qa
    qa_pairs = generate_all_qa()
    logger.info(f"Generated {len(qa_pairs)} Q/A pairs.\n")

    logger.info("=" * 60)
    logger.info("PHASE 3: Building vector store (ChromaDB)")
    logger.info("=" * 60)
    from src.vector_store import build_vector_store
    build_vector_store()
    logger.info("Vector store built.\n")

    logger.info("=" * 60)
    logger.info("Pipeline complete! Run the chatbot with:")
    logger.info("  streamlit run app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
