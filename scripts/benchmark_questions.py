from __future__ import annotations

from typing import Dict, List


BENCHMARK_QUESTIONS: List[Dict[str, str]] = [
    {
        "id": "accommodation_01",
        "category": "accommodation",
        "question": "How much is Butler Court?",
        "ground_truth_answer": "Butler Court prices for 2025/26 start at GBP 126.68 per week, with higher room types costing GBP 128.16 and GBP 203.98 per week.",
    },
    {
        "id": "accommodation_02",
        "category": "accommodation",
        "question": "Does Butler Court have en-suite rooms?",
        "ground_truth_answer": "Yes. Butler Court offers en-suite room types, including Shared Twin with En-Suite and Large En-Suite, alongside a standard shared-bathroom option.",
    },
    {
        "id": "accommodation_03",
        "category": "accommodation",
        "question": "Is Butler Court catered?",
        "ground_truth_answer": "No. Butler Court is self-catered.",
    },
    {
        "id": "accommodation_04",
        "category": "accommodation",
        "question": "Where is Butler Court located?",
        "ground_truth_answer": "Butler Court is in East Park, near the School of Design and Creative Arts and the rugby pitches, at Butler Court, Loughborough University, LE11 3TS.",
    },
    {
        "id": "accommodation_05",
        "category": "accommodation",
        "question": "Which is cheaper, Butler Court or Cayley?",
        "ground_truth_answer": "Butler Court is cheaper. Its lowest listed weekly price is GBP 126.68, while Cayley's lowest listed weekly price is GBP 183.40.",
    },
    {
        "id": "accommodation_06",
        "category": "accommodation",
        "question": "Does Cayley include meals?",
        "ground_truth_answer": "Yes. Cayley is catered and includes meals Monday to Friday, listed as 15 meals per week.",
    },
    {
        "id": "accommodation_07",
        "category": "accommodation",
        "question": "What room types does Butler Court offer?",
        "ground_truth_answer": "Butler Court offers Standard, Shared Twin with En-Suite, and Large En-Suite 4ft bed room types.",
    },
    {
        "id": "accommodation_08",
        "category": "accommodation",
        "question": "How long is the Butler Court contract?",
        "ground_truth_answer": "The main Butler Court contract length is 41 weeks, with additional semester-only options of 18 and 23 weeks.",
    },
    {
        "id": "accommodation_09",
        "category": "accommodation",
        "question": "Who can stay in Butler Court?",
        "ground_truth_answer": "Butler Court rooms are listed for undergraduates and international students, with let options also shown for returning students and semester stays.",
    },
    {
        "id": "accommodation_10",
        "category": "accommodation",
        "question": "What is the highest weekly Butler Court room price?",
        "ground_truth_answer": "The highest listed weekly Butler Court room price is GBP 203.98 for the Large En-Suite 4ft bed room.",
    },
    {
        "id": "undergraduate_courses_01",
        "category": "undergraduate_courses",
        "question": "What are the entry requirements for Accounting and Finance?",
        "ground_truth_answer": "Accounting and Finance BSc has a typical offer of AAB.",
    },
    {
        "id": "undergraduate_courses_02",
        "category": "undergraduate_courses",
        "question": "What are the UK fees for Accounting and Finance?",
        "ground_truth_answer": "The UK full-time fee for Accounting and Finance is GBP 9,790 per year, with the placement year charged at about 20% of the full-time fee.",
    },
    {
        "id": "undergraduate_courses_03",
        "category": "undergraduate_courses",
        "question": "Does Accounting and Finance have a placement year?",
        "ground_truth_answer": "Yes. Accounting and Finance offers an optional placement year.",
    },
    {
        "id": "undergraduate_courses_04",
        "category": "undergraduate_courses",
        "question": "What are the entry requirements for Aeronautical Engineering BEng?",
        "ground_truth_answer": "Aeronautical Engineering BEng has a typical offer of AAB.",
    },
    {
        "id": "undergraduate_courses_05",
        "category": "undergraduate_courses",
        "question": "What are the entry requirements for Aeronautical Engineering MEng?",
        "ground_truth_answer": "Aeronautical Engineering MEng has a typical offer of A*AA.",
    },
    {
        "id": "undergraduate_courses_06",
        "category": "undergraduate_courses",
        "question": "What is the difference between Aeronautical Engineering BEng and MEng?",
        "ground_truth_answer": "The BEng has a lower typical offer and lasts 3 years or 4 with placement, while the MEng lasts 4 years or 5 with placement and includes more advanced technical and management study plus a larger final project.",
    },
    {
        "id": "undergraduate_courses_07",
        "category": "undergraduate_courses",
        "question": "What are the entry requirements for Architecture?",
        "ground_truth_answer": "Architecture BArch has a typical offer of AAA.",
    },
    {
        "id": "undergraduate_courses_08",
        "category": "undergraduate_courses",
        "question": "What are the entry requirements for Architectural Engineering BEng?",
        "ground_truth_answer": "Architectural Engineering BEng has a typical offer of ABB.",
    },
    {
        "id": "undergraduate_courses_09",
        "category": "undergraduate_courses",
        "question": "Is Architecture professionally accredited?",
        "ground_truth_answer": "Yes. The Architecture degree is accredited by the Royal Institute of British Architects (RIBA), and the course overview states it exempts students from the RIBA Part I exam.",
    },
    {
        "id": "undergraduate_courses_10",
        "category": "undergraduate_courses",
        "question": "Is Architectural Engineering triple accredited?",
        "ground_truth_answer": "Yes. Architectural Engineering is fully triple accredited through JBM, CIOB, and CIBSE accreditation.",
    },
    {
        "id": "contextual_policy_01",
        "category": "contextual_policy",
        "question": "What is an Access Loughborough Contextual Offer?",
        "ground_truth_answer": "An Access Loughborough Contextual Offer is a reduced contextual offer for eligible applicants, determined mainly from UCAS-shared contextual information such as postcode and free-school-meal data, and it can be up to two grades below the typical offer.",
    },
    {
        "id": "contextual_policy_02",
        "category": "contextual_policy",
        "question": "Does a contextual offer guarantee an offer?",
        "ground_truth_answer": "No. Meeting the eligibility criteria for an Access Loughborough Contextual Offer does not guarantee that you will receive an offer.",
    },
    {
        "id": "contextual_policy_03",
        "category": "contextual_policy",
        "question": "How much lower can a contextual offer be?",
        "ground_truth_answer": "An Access Loughborough Contextual Offer can be up to two grades lower than the typical offer.",
    },
    {
        "id": "contextual_policy_04",
        "category": "contextual_policy",
        "question": "Can I still apply if I haven’t taken IELTS yet?",
        "ground_truth_answer": "Yes. You can still apply without IELTS, but if you receive an offer then the English language qualification will be included as a condition.",
    },
    {
        "id": "contextual_policy_05",
        "category": "contextual_policy",
        "question": "What is the standard IELTS requirement?",
        "ground_truth_answer": "The standard IELTS requirement is 6.5 overall with 6.0 in each of reading, writing, listening, and speaking.",
    },
    {
        "id": "contextual_policy_06",
        "category": "contextual_policy",
        "question": "Can all postgraduate students bring dependants?",
        "ground_truth_answer": "No. Since January 2024, bringing dependants has mostly been restricted to postgraduate research students, with some exceptions such as certain government-funded students or students already holding relevant dependant permission.",
    },
    {
        "id": "contextual_policy_07",
        "category": "contextual_policy",
        "question": "Who can be dependants on a student visa?",
        "ground_truth_answer": "Eligible dependants are a spouse, civil partner, unmarried partner in a relationship similar to marriage or civil partnership for at least 2 years, and children who were under 18 when they first applied.",
    },
    {
        "id": "contextual_policy_08",
        "category": "contextual_policy",
        "question": "When should I apply for a visa extension?",
        "ground_truth_answer": "You should apply for a visa extension before your current visa expires, and not more than three months before the start of a new course.",
    },
    {
        "id": "contextual_policy_09",
        "category": "contextual_policy",
        "question": "Do I need to pay the immigration health surcharge?",
        "ground_truth_answer": "Yes. International students on courses longer than 6 months normally need to pay the immigration health surcharge as part of their visa application.",
    },
    {
        "id": "contextual_policy_10",
        "category": "contextual_policy",
        "question": "Can alumni join Powerbase?",
        "ground_truth_answer": "Yes. Alumni are one of the groups eligible to join Powerbase through the Gold membership option.",
    },
]
