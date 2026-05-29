import os
import logging
from groq import Groq
from dotenv import load_dotenv
from src.retriever import HybridRetriever

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a knowledgeable and friendly Coursera AI/ML course advisor.
You help learners find the right courses, understand content, prerequisites, duration,
skills gained, and make confident decisions about their learning journey.
You answer based on retrieved Coursera course data. Be accurate, encouraging, and concise.
If information is unavailable, say so honestly rather than guessing."""

ANSWER_PROMPT = """Use the retrieved course information below to answer the user's question clearly and helpfully.

Retrieved Context:
{context}

Retrieval Method: {method}

Conversation History:
{history}

User Question: {question}

Instructions:
- Answer directly and specifically
- Mention course titles where relevant
- If comparing courses, structure your answer clearly
- Keep the answer focused (3-6 sentences unless detail is needed)"""


class CourseraChatbot:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.retriever = HybridRetriever()
        self.history = []

    def _build_context(self, retrieval: dict) -> tuple[str, str]:
        """Format retrieved results into a context string and source label."""
        if retrieval['used_qa']:
            best = retrieval['best_qa']
            ctx = f"Matched Q/A:\nQ: {best['question']}\nA: {best['answer']}"
            for doc in retrieval['doc_results'][:2]:
                ctx += f"\n\nSupporting context from '{doc['metadata'].get('title', '')}' ({doc['metadata'].get('url', '')}):\n{doc['text'][:300]}"
            source = f"Q/A Dataset — Course: {best.get('course_title', 'N/A')} | {best.get('source_page', '')}"
        else:
            parts = []
            for doc in retrieval['doc_results']:
                parts.append(
                    f"From '{doc['metadata'].get('title', 'course')}' ({doc['metadata'].get('url', '')}):\n{doc['text']}"
                )
            ctx = "\n\n".join(parts)
            source = "Vector Search on Course Documents"
        return ctx, source

    def _format_history(self) -> str:
        if not self.history:
            return "None"
        lines = []
        for msg in self.history[-6:]:
            role = "User" if msg['role'] == 'user' else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def chat(self, user_message: str) -> dict:
        """Process a user message and return answer + metadata."""
        retrieval = self.retriever.retrieve(user_message)
        context, source = self._build_context(retrieval)

        prompt = ANSWER_PROMPT.format(
            context=context,
            method=retrieval['primary_source'],
            history=self._format_history(),
            question=user_message,
        )

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content.strip()

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})

        return {
            'answer': answer,
            'source': source,
            'used_qa': retrieval['used_qa'],
            'retrieval': retrieval,
        }

    def clear_history(self):
        self.history = []


if __name__ == "__main__":
    bot = CourseraChatbot()
    print("Coursera AI/ML Advisor — type 'quit' to exit\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ('quit', 'exit'):
            break
        result = bot.chat(q)
        print(f"\nAssistant: {result['answer']}")
        print(f"[Source: {result['source']}]\n")
