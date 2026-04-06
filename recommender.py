"""
Student-Question Recommender (Fixed)
Recommends questions to students based on their weakness profile using
cosine similarity between student feature vectors and question feature vectors.

BUGS FIXED:
In the original `recommend()` function, lines 47-48 had two critical bugs:
1. The variable `student_profile` (computed as student_vector - cohort_baseline) was
   immediately overwritten with `cohort_baseline / profile_norm`. This discarded
   the per-student adjustment entirely.
2. The norm used for normalization was `np.linalg.norm(cohort_baseline)` instead of
   `np.linalg.norm(student_profile)`, so even the normalization denominator was wrong.

Result: Every student got the exact same profile (the normalized cohort average),
so all students received identical recommendations — 10/10 overlap for every pair.

Fix: Normalize the actual `student_profile` vector using its own norm.
"""

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

TOPICS = [
    "mechanics", "thermodynamics", "electrostatics", "optics",
    "modern_physics", "organic_chemistry", "inorganic_chemistry",
    "physical_chemistry", "algebra", "calculus", "coordinate_geometry",
    "trigonometry"
]
TOPIC_TO_IDX = {t: i for i, t in enumerate(TOPICS)}
DIFFICULTY_WEIGHT = {"easy": 0.5, "medium": 1.0, "hard": 1.5}


def build_feature_matrix(records: list[dict], record_type: str = "student") -> np.ndarray:
    """Build a normalized feature matrix from student or question records."""
    n_records = len(records)
    matrix = np.zeros((n_records, len(TOPICS)))

    if record_type == "student":
        for i, rec in enumerate(records):
            for topic, score in rec.get("weakness_scores", {}).items():
                if topic in TOPIC_TO_IDX:
                    matrix[i, TOPIC_TO_IDX[topic]] = score
    else:
        for i, rec in enumerate(records):
            topic = rec.get("topic", "")
            weight = DIFFICULTY_WEIGHT.get(rec.get("difficulty", "medium"), 1.0)
            if topic in TOPIC_TO_IDX:
                matrix[i, TOPIC_TO_IDX[topic]] = weight

    matrix = normalize(matrix, axis=1, norm="l2")
    return matrix


def recommend(student_matrix: np.ndarray, question_matrix: np.ndarray,
              questions: list[dict], student_idx: int, top_n: int = 10) -> list[dict]:
    """Return top-N recommended questions for a given student."""
    cohort_baseline = student_matrix.mean(axis=0)
    student_profile = student_matrix[student_idx] - cohort_baseline

    # FIX: normalize using the student_profile's own norm (not cohort_baseline's norm)
    # and keep student_profile (not overwrite with cohort_baseline)
    profile_norm = np.linalg.norm(student_profile)
    student_profile = student_profile / (profile_norm + 1e-10)

    similarities = cosine_similarity(
        student_profile.reshape(1, -1), question_matrix
    ).flatten()

    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [{
        "question_id": questions[idx]["id"],
        "topic": questions[idx]["topic"],
        "difficulty": questions[idx]["difficulty"],
        "score": round(float(similarities[idx]), 4)
    } for idx in top_indices]


def recommend_questions_for_weakness(weakness_scores: dict, question_bank: list[dict], top_n: int = 10) -> list[dict]:
    """
    Convenience wrapper: given a weakness_scores dict and a question bank,
    return top-N recommended question IDs targeting the student's weak topics.
    
    weakness_scores: {topic_name: float} where higher = weaker
    question_bank: list of dicts with keys 'id', 'topic', 'difficulty'
    """
    students = [{"weakness_scores": weakness_scores}]
    student_matrix = build_feature_matrix(students, "student")
    question_matrix = build_feature_matrix(question_bank, "question")
    return recommend(student_matrix, question_matrix, question_bank, 0, top_n)
