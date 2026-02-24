import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.processor_LLM import processor_agent
from agents.answerer import answer

questions = [
    "Tell me about Butler Court prices",
    "What are the entry requirements for Accounting and Finance ?",
    "Tell me the modules for Computer Science BSc",
    "Can I talk to current students or attend a student panel?",
    "What does a typical week on this course look like?",
    "How are classes delivered (lectures, seminars, labs, tutorials)?",
    "What is the average class size for first-year modules?",
    "How is student performance assessed and graded?",
    "Are there opportunities for placements, internships or year abroad?",
    "What career support and employability services are available?",
    "What modules are compulsory and what options are available?",
    "How flexible is the course if I want to change modules or pathway?",
    "Who will be teaching the course and what are their backgrounds?",
    "What facilities are available for practical work (labs, studios, workshops)?",
    "How accessible are library resources and research databases?",
    "What support services exist for mental health and wellbeing?",
    "How is academic support delivered (tutors, drop-ins, workshops)?",
    "What accommodation options are available and when should I apply?",
    "What are typical living costs in the area and on-campus?",
    "Are there scholarships, bursaries or financial support options?",
    "How safe is the campus and local area?",
    "What transport links are available to and from campus?",
    "What student societies and clubs would you recommend for newcomers?",
    "Is there support for international students and visa advice?",
    "Can I do part-time work while studying and how much is allowed?",
    "What are the graduate employment rates and example employers?",
    "How do I apply and what are key application deadlines?"
]

for q in questions:
    processed = processor_agent.process(q)
    resp = answer(processed)

    print("=" * 80)
    print("Q:", q)
    print("Processor:", processed.get("domain"), processed.get("intent"))
    print()
    print(resp["answer"])
    print()
    print("DEBUG:", resp.get("debug"))
    print("=" * 80)
    print()