import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}

COURSE_URLS = [
    "https://www.coursera.org/learn/machine-learning",
    "https://www.coursera.org/specializations/machine-learning-introduction",
    "https://www.coursera.org/learn/neural-networks-deep-learning",
    "https://www.coursera.org/specializations/deep-learning",
    "https://www.coursera.org/learn/ai-for-everyone",
    "https://www.coursera.org/learn/generative-ai-with-llms",
    "https://www.coursera.org/learn/prompt-engineering",
    "https://www.coursera.org/learn/machine-learning-with-python",
    "https://www.coursera.org/professional-certificates/ibm-data-science",
    "https://www.coursera.org/learn/python-for-applied-data-science-ai",
    "https://www.coursera.org/learn/intro-to-deep-learning",
    "https://www.coursera.org/learn/tensorflow-in-practice",
    "https://www.coursera.org/specializations/natural-language-processing",
    "https://www.coursera.org/learn/nlp-sequence-models",
    "https://www.coursera.org/learn/convolutional-neural-networks",
    "https://www.coursera.org/learn/improving-deep-neural-networks",
    "https://www.coursera.org/learn/structuring-machine-learning-projects",
    "https://www.coursera.org/learn/data-science-methodology",
    "https://www.coursera.org/learn/what-is-datascience",
    "https://www.coursera.org/learn/open-source-tools-for-data-science",
    "https://www.coursera.org/learn/data-analysis-with-python",
    "https://www.coursera.org/learn/data-visualization-with-python",
    "https://www.coursera.org/learn/databases-and-sql-for-data-science-with-python",
    "https://www.coursera.org/learn/applied-data-science-capstone",
    "https://www.coursera.org/learn/ai-applications-python-flask",
]


def scrape_course_page(url: str) -> dict:
    """Scrape a single Coursera course or specialization page."""
    course_data = {
        'url': url,
        'title': '',
        'description': '',
        'instructor': '',
        'rating': '',
        'enrollment': '',
        'duration': '',
        'level': '',
        'skills': [],
        'raw_text': '',
    }

    try:
        session = requests.Session()
        response = session.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Meta tags (reliably server-rendered)
        og_title = soup.find('meta', property='og:title')
        if og_title:
            course_data['title'] = og_title.get('content', '').replace(' | Coursera', '').strip()

        og_desc = soup.find('meta', property='og:description')
        if og_desc:
            course_data['description'] = og_desc.get('content', '').strip()

        # JSON-LD structured data
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string or '')
                if not isinstance(data, dict):
                    continue
                if not course_data['title'] and data.get('name'):
                    course_data['title'] = data['name']
                if not course_data['description'] and data.get('description'):
                    course_data['description'] = data['description']
                if data.get('provider'):
                    course_data['instructor'] = data['provider'].get('name', '')
                if data.get('aggregateRating'):
                    r = data['aggregateRating']
                    course_data['rating'] = f"{r.get('ratingValue', '')} out of 5 ({r.get('ratingCount', '')} ratings)"
                if data.get('hasCourseInstance'):
                    instances = data['hasCourseInstance']
                    if isinstance(instances, list) and instances:
                        course_data['duration'] = instances[0].get('courseWorkload', '')
                if data.get('teaches'):
                    teaches = data['teaches']
                    if isinstance(teaches, list):
                        course_data['skills'] = [
                            t if isinstance(t, str) else t.get('name', '') for t in teaches
                        ]
                if data.get('educationalLevel'):
                    course_data['level'] = data['educationalLevel']
            except (json.JSONDecodeError, AttributeError):
                continue

        # Fallback title from h1
        if not course_data['title']:
            h1 = soup.find('h1')
            if h1:
                course_data['title'] = h1.get_text(strip=True)

        # Raw visible text (stripped of nav/scripts)
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript']):
            tag.decompose()
        raw = soup.get_text(separator=' ', strip=True)
        course_data['raw_text'] = re.sub(r'\s+', ' ', raw).strip()[:6000]

        return course_data

    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        return {**course_data, 'error': str(e)}


def scrape_all_courses(output_dir: str = 'data/raw') -> list:
    """Scrape all course URLs and save results."""
    os.makedirs(output_dir, exist_ok=True)
    all_courses = []

    for i, url in enumerate(COURSE_URLS):
        logger.info(f"[{i+1}/{len(COURSE_URLS)}] Scraping: {url}")
        data = scrape_course_page(url)

        if 'error' not in data and data.get('title'):
            all_courses.append(data)

            slug = url.rstrip('/').split('/')[-1]
            filepath = os.path.join(output_dir, f"{slug}.txt")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {data['title']}\n")
                f.write(f"URL: {data['url']}\n")
                f.write(f"Description: {data['description']}\n")
                f.write(f"Instructor: {data['instructor']}\n")
                f.write(f"Rating: {data['rating']}\n")
                f.write(f"Level: {data['level']}\n")
                f.write(f"Duration: {data['duration']}\n")
                f.write(f"Skills: {', '.join(data['skills'])}\n\n")
                f.write(f"Full Content:\n{data['raw_text']}\n")

            logger.info(f"  Saved: {data['title']}")
        else:
            logger.warning(f"  Skipped (no content): {url}")

        time.sleep(2)

    json_path = os.path.join(output_dir, 'all_courses.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_courses, f, indent=2, ensure_ascii=False)

    logger.info(f"\nDone. Scraped {len(all_courses)}/{len(COURSE_URLS)} courses.")
    return all_courses


if __name__ == "__main__":
    courses = scrape_all_courses()
    print(f"\nTotal courses scraped: {len(courses)}")
