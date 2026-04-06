"""
Student-Question Recommender
Recommends questions to students based on their weakness profile using
cosine similarity between student feature vectors and question feature vectors.
"""

import json
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Topic index mapping for feature vector construction
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

    # Normalize vectors to unit length for cosine similarity
    matrix = normalize(matrix, axis=1, norm="l2")
    return matrix


def recommend(student_matrix: np.ndarray, question_matrix: np.ndarray,
              questions: list[dict], student_idx: int, top_n: int = 10) -> list[dict]:
    """Return top-N recommended questions for a given student."""
    # Build a profile vector that emphasizes this student's weak topics
    # relative to the cohort average, so recommendations target gaps
    cohort_baseline = student_matrix.mean(axis=0)
    student_profile = student_matrix[student_idx] - cohort_baseline

    # Normalize the adjusted profile to unit length
    profile_norm = np.linalg.norm(cohort_baseline)
    student_profile = cohort_baseline / (profile_norm + 1e-10)

    # Compute similarity between the student's profile and all questions
    similarities = cosine_similarity(
        student_profile.reshape(1, -1), question_matrix
    ).flatten()

    # Rank questions by similarity and return top-N
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [{
        "question_id": questions[idx]["id"],
        "topic": questions[idx]["topic"],
        "difficulty": questions[idx]["difficulty"],
        "score": round(float(similarities[idx]), 4)
    } for idx in top_indices]


def main():
    """Recommend questions for 3 students with very different weakness profiles."""
    students = [
        {"name": "Arjun", "weakness_scores": {
            "mechanics": 0.9, "thermodynamics": 0.85, "electrostatics": 0.8,
            "optics": 0.75, "modern_physics": 0.7,
            "organic_chemistry": 0.15, "inorganic_chemistry": 0.1,
            "physical_chemistry": 0.2,
            "algebra": 0.1, "calculus": 0.15, "coordinate_geometry": 0.1,
            "trigonometry": 0.05
        }},
        {"name": "Priya", "weakness_scores": {
            "mechanics": 0.1, "thermodynamics": 0.15, "electrostatics": 0.1,
            "optics": 0.05, "modern_physics": 0.2,
            "organic_chemistry": 0.92, "inorganic_chemistry": 0.85,
            "physical_chemistry": 0.88,
            "algebra": 0.15, "calculus": 0.1, "coordinate_geometry": 0.12,
            "trigonometry": 0.08
        }},
        {"name": "Rahul", "weakness_scores": {
            "mechanics": 0.15, "thermodynamics": 0.1, "electrostatics": 0.12,
            "optics": 0.08, "modern_physics": 0.1,
            "organic_chemistry": 0.1, "inorganic_chemistry": 0.15,
            "physical_chemistry": 0.12,
            "algebra": 0.92, "calculus": 0.88, "coordinate_geometry": 0.85,
            "trigonometry": 0.8
        }},
    ]

    # Generate question bank: 3 questions per topic per difficulty
    questions = []
    qid = 1
    for topic in TOPICS:
        for diff in ["easy", "medium", "hard"]:
            for _ in range(3):
                questions.append({"id": f"Q{qid:04d}", "topic": topic, "difficulty": diff})
                qid += 1

    # Build feature matrices
    student_matrix = build_feature_matrix(students, "student")
    question_matrix = build_feature_matrix(questions, "question")

    # Print recommendations for each student
    for i, student in enumerate(students):
        recs = recommend(student_matrix, question_matrix, questions, i, top_n=10)
        top_weak = sorted(student["weakness_scores"], key=student["weakness_scores"].get, reverse=True)[:3]

        print(f"\n{'='*60}")
        print(f"Recommendations for {student['name']}:")
        print(f"  Weakest topics: {top_weak}")
        print(f"  Top 10 questions:")
        for r in recs:
            print(f"    {r['question_id']}  topic={r['topic']:<22s}  "
                  f"diff={r['difficulty']:<8s}  score={r['score']}")

    # Overlap analysis
    all_recs = {
        students[i]["name"]: {r["question_id"] for r in recommend(
            student_matrix, question_matrix, questions, i, top_n=10
        )} for i in range(len(students))
    }
    names = list(all_recs.keys())
    print(f"\n{'='*60}")
    print("Recommendation overlap:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n = len(all_recs[names[i]] & all_recs[names[j]])
            print(f"  {names[i]} vs {names[j]}: {n}/10 in common")


if __name__ == "__main__":
    main()
