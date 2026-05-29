import json
import os
import csv
import time
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

QA_PROMPT = """You are an educational content expert. Based on the Coursera course information below, generate {num_pairs} diverse and realistic question-answer pairs that a student might ask.

Course Information:
{course_info}

Return ONLY a JSON array in this exact format (no other text):
[
  {{"question": "...", "answer": "..."}},
  ...
]

Guidelines:
- Questions must be natural and conversational
- Answers must be 2-4 sentences, helpful, and specific to this course
- Cover a variety of topics: content, difficulty, duration, prerequisites, skills gained, instructor, rating, price, career outcomes
- Include beginner-level questions
- Make at least one question about who this course is recommended for"""


def generate_qa_for_course(course: dict, num_pairs: int = 8) -> list:
    """Generate Q/A pairs for a single course using Groq LLaMA."""
    course_info = (
        f"Title: {course.get('title', 'N/A')}\n"
        f"URL: {course.get('url', 'N/A')}\n"
        f"Description: {course.get('description', 'N/A')}\n"
        f"Instructor: {course.get('instructor', 'N/A')}\n"
        f"Rating: {course.get('rating', 'N/A')}\n"
        f"Level: {course.get('level', 'N/A')}\n"
        f"Duration: {course.get('duration', 'N/A')}\n"
        f"Skills: {', '.join(course.get('skills', [])) or 'N/A'}\n"
        f"Additional Content:\n{course.get('raw_text', '')[:1200]}"
    )

    prompt = QA_PROMPT.format(num_pairs=num_pairs, course_info=course_info)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
        )
        content = response.choices[0].message.content.strip()

        start = content.find('[')
        end = content.rfind(']') + 1
        if start == -1 or end <= start:
            logger.warning(f"No JSON found in response for: {course.get('title')}")
            return []

        pairs = json.loads(content[start:end])
        return [
            {
                'question': p.get('question', '').strip(),
                'answer': p.get('answer', '').strip(),
                'source_page': course.get('url', ''),
                'course_title': course.get('title', ''),
            }
            for p in pairs
            if p.get('question') and p.get('answer')
        ]
    except Exception as e:
        logger.error(f"Error generating QA for '{course.get('title')}': {e}")
        return []


def generate_all_qa(
    courses_file: str = 'data/raw/all_courses.json',
    output_dir: str = 'data/processed',
) -> list:
    """Generate Q/A pairs for all scraped courses and save to CSV + JSON."""
    os.makedirs(output_dir, exist_ok=True)

    with open(courses_file, 'r', encoding='utf-8') as f:
        courses = json.load(f)

    all_qa = []

    for i, course in enumerate(courses):
        logger.info(f"[{i+1}/{len(courses)}] Generating QA: {course.get('title', 'Unknown')}")
        pairs = generate_qa_for_course(course, num_pairs=2)
        all_qa.extend(pairs)
        if len(all_qa) >= 50:
            all_qa = all_qa[:50]
            logger.info(f"  Reached 50 pair cap, stopping early.")
            break
        logger.info(f"  +{len(pairs)} pairs  |  Total: {len(all_qa)}")
        time.sleep(1)

    csv_path = os.path.join(output_dir, 'qa_dataset.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'answer', 'source_page', 'course_title'])
        writer.writeheader()
        writer.writerows(all_qa)

    json_path = os.path.join(output_dir, 'qa_dataset.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)

    logger.info(f"\nTotal Q/A pairs generated: {len(all_qa)}")
    logger.info(f"Saved: {csv_path} and {json_path}")
    return all_qa


if __name__ == "__main__":
    qa = generate_all_qa()
    print(f"\nTotal Q/A pairs: {len(qa)}")
