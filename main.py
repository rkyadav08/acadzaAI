"""
Acadza AI — Student Performance Analyzer & DOST Recommender
FastAPI application with endpoints for analysis, recommendation, question lookup, and leaderboard.
"""

import json
import re
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from recommender import recommend_questions_for_weakness, TOPICS, TOPIC_TO_IDX

app = FastAPI(title="Acadza AI Recommender", version="1.0.0")

BASE_DIR = Path(__file__).parent
STUDENTS_DATA: list[dict] = []
QUESTION_BANK: list[dict] = []
QUESTION_INDEX: dict[str, dict] = {}
DOST_CONFIG: dict = {}

# ─── Chapter-to-topic mapping ───────────────────────────────────────────────
CHAPTER_TO_TOPIC = {
    "Thermodynamics": "thermodynamics",
    "Electrostatics": "electrostatics",
    "Kinematics": "kinematics",
    "Optics": "optics",
    "Heat Transfer": "heat_transfer",
    "Laws of Motion": "laws_of_motion",
    "Rotational Mechanics": "rotational_mechanics",
    "Organic Chemistry": "organic_chemistry",
    "Chemical Bonding": "chemical_bonding",
    "Physical Chemistry": "physical_chemistry",
    "Algebra": "algebra",
    "Calculus": "calculus",
    "Coordinate Geometry": "coordinate_geometry",
    "Probability": "probability",
}

# Topics that map to the recommender's TOPICS list (for cosine-similarity based rec)
RECOMMENDER_TOPICS = set(TOPICS)


def load_data():
    global STUDENTS_DATA, QUESTION_BANK, QUESTION_INDEX, DOST_CONFIG

    if STUDENTS_DATA:
        return  # already loaded

    with open(BASE_DIR / "student_performance.json", encoding="utf-8") as f:
        STUDENTS_DATA.extend(json.load(f))

    with open(BASE_DIR / "question_bank.json",encoding="utf-8") as f:
        raw_questions = json.load(f)

    # Normalize question IDs and deduplicate
    seen_ids = set()
    for q in raw_questions:
        qid = normalize_question_id(q.get("_id"))
        if qid and qid not in seen_ids:
            q["_id_normalized"] = qid
            QUESTION_BANK.append(q)
            QUESTION_INDEX[qid] = q
            seen_ids.add(qid)
            if q.get("qid"):
                QUESTION_INDEX[q["qid"]] = q

    with open(BASE_DIR / "dost_config.json",encoding="utf-8") as f:
        DOST_CONFIG.update(json.load(f))


def normalize_question_id(raw_id) -> Optional[str]:
    """Handle both {'$oid': '...'} and flat string formats."""
    if isinstance(raw_id, dict):
        return raw_id.get("$oid")
    if isinstance(raw_id, str):
        return raw_id
    return None


def parse_marks(marks_raw) -> float:
    """
    Normalize the messy marks field into a single numeric score.
    Formats handled:
      - int/float: 49 → 49.0
      - "+52 -8" → 52 - 8 = 44.0
      - "39/100" → 39.0
      - "49/120 (40.8%)" → 49.0
      - "68/100" → 68.0
      - "22" → 22.0
    """
    if isinstance(marks_raw, (int, float)):
        return float(marks_raw)

    s = str(marks_raw).strip()

    # Format: "+52 -8"
    plus_minus = re.match(r'^[+]?(\d+(?:\.\d+)?)\s+[-](\d+(?:\.\d+)?)$', s)
    if plus_minus:
        return float(plus_minus.group(1)) - float(plus_minus.group(2))

    # Format: "49/120 (40.8%)" or "39/100"
    frac = re.match(r'^(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)(?:\s*\([\d.]+%\))?$', s)
    if frac:
        return float(frac.group(1))

    # Format: plain number string "22"
    try:
        return float(s)
    except ValueError:
        return 0.0


def get_max_marks_estimate(attempt: dict) -> float:
    """Estimate max possible marks from the attempt context."""
    marks_raw = attempt.get("marks")
    s = str(marks_raw).strip() if marks_raw is not None else ""

    # If format is "X/Y" or "X/Y (%)", use Y
    frac = re.match(r'^(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)(?:\s*\([\d.]+%\))?$', s)
    if frac:
        return float(frac.group(2))

    # Otherwise estimate from total_questions * 4 (JEE-style +4 per correct)
    total_q = attempt.get("total_questions", 25)
    return total_q * 4.0


def compute_percentage(attempt: dict) -> float:
    """Compute percentage score for an attempt."""
    score = parse_marks(attempt.get("marks"))
    max_marks = get_max_marks_estimate(attempt)
    if max_marks <= 0:
        return 0.0
    return round((score / max_marks) * 100, 1)


def get_student(student_id: str) -> dict:
    for s in STUDENTS_DATA:
        if s["student_id"] == student_id:
            return s
    raise HTTPException(status_code=404, detail=f"Student {student_id} not found")


def strip_html(text: str) -> str:
    """Remove HTML tags for plaintext preview."""
    return re.sub(r'<[^>]+>', '', text).strip()


def get_question_clean(q: dict) -> dict:
    """Return a clean version of a question with plaintext preview."""
    qtype = q.get("questionType", "")
    qdata = q.get(qtype, {})
    question_html = qdata.get("question", "") if isinstance(qdata, dict) else ""
    solution_html = qdata.get("solution", "") if isinstance(qdata, dict) else ""
    answer = qdata.get("answer", None) if isinstance(qdata, dict) else None

    return {
        "id": q.get("qid", q.get("_id_normalized")),
        "questionType": qtype,
        "subject": q.get("subject"),
        "topic": q.get("topic"),
        "subtopic": q.get("subtopic"),
        "difficulty": q.get("difficulty"),
        "question_preview": strip_html(question_html)[:200],
        "answer": answer,
        "has_solution": bool(solution_html),
    }


# ─── Analysis Logic ─────────────────────────────────────────────────────────

def analyze_student(student: dict) -> dict:
    """Full analysis of a student's performance across all sessions."""
    attempts = student["attempts"]
    name = student["name"]
    student_id = student["student_id"]

    # Chapter-wise aggregation
    chapter_stats = {}
    subject_stats = {}
    total_score = 0
    total_max = 0
    completion_count = 0
    abort_count = 0
    total_time = 0
    total_questions_attempted = 0
    total_questions_total = 0
    slowest_questions = []

    for att in attempts:
        score = parse_marks(att["marks"])
        max_m = get_max_marks_estimate(att)
        pct = compute_percentage(att)
        total_score += score
        total_max += max_m

        if att.get("completed"):
            completion_count += 1
        else:
            abort_count += 1

        total_time += att.get("time_taken_minutes", 0)
        total_questions_attempted += att.get("attempted", 0)
        total_questions_total += att.get("total_questions", 0)

        subj = att.get("subject", "Unknown")
        if subj not in subject_stats:
            subject_stats[subj] = {"scores": [], "attempts": 0}
        subject_stats[subj]["scores"].append(pct)
        subject_stats[subj]["attempts"] += 1

        for ch in att.get("chapters", []):
            if ch not in chapter_stats:
                chapter_stats[ch] = {"scores": [], "attempts_count": 0, "time_spent": []}
            chapter_stats[ch]["scores"].append(pct)
            chapter_stats[ch]["attempts_count"] += 1
            chapter_stats[ch]["time_spent"].append(att.get("avg_time_per_question_seconds", 0))

        if att.get("slowest_question_id"):
            slowest_questions.append({
                "question_id": att["slowest_question_id"],
                "time_seconds": att.get("slowest_question_time_seconds", 0),
                "attempt_id": att["attempt_id"],
            })

    # Compute chapter averages and identify strengths/weaknesses
    chapter_breakdown = {}
    for ch, stats in chapter_stats.items():
        avg_score = round(sum(stats["scores"]) / len(stats["scores"]), 1)
        avg_time = round(sum(stats["time_spent"]) / len(stats["time_spent"]), 1)
        chapter_breakdown[ch] = {
            "average_score_pct": avg_score,
            "attempts": stats["attempts_count"],
            "avg_time_per_question_seconds": avg_time,
        }

    sorted_chapters = sorted(chapter_breakdown.items(), key=lambda x: x[1]["average_score_pct"])
    weakest = [c[0] for c in sorted_chapters[:3]]
    strongest = [c[0] for c in sorted_chapters[-2:]]

    subject_breakdown = {}
    for subj, stats in subject_stats.items():
        subject_breakdown[subj] = {
            "average_score_pct": round(sum(stats["scores"]) / len(stats["scores"]), 1),
            "attempts": stats["attempts"],
        }

    overall_pct = round((total_score / total_max) * 100, 1) if total_max > 0 else 0
    attempt_rate = round((total_questions_attempted / total_questions_total) * 100, 1) if total_questions_total > 0 else 0

    # Trend: compare first half vs second half
    n = len(attempts)
    first_half_scores = [compute_percentage(a) for a in attempts[:n // 2]]
    second_half_scores = [compute_percentage(a) for a in attempts[n // 2:]]
    first_avg = sum(first_half_scores) / len(first_half_scores) if first_half_scores else 0
    second_avg = sum(second_half_scores) / len(second_half_scores) if second_half_scores else 0
    if second_avg > first_avg + 5:
        trend = "improving"
    elif second_avg < first_avg - 5:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "student_id": student_id,
        "name": name,
        "class": student.get("class"),
        "stream": student.get("stream"),
        "total_attempts": len(attempts),
        "overall_score_pct": overall_pct,
        "attempt_rate_pct": attempt_rate,
        "completion_rate_pct": round(completion_count / len(attempts) * 100, 1),
        "aborted_tests": abort_count,
        "total_time_minutes": total_time,
        "trend": trend,
        "subject_breakdown": subject_breakdown,
        "chapter_breakdown": chapter_breakdown,
        "weakest_chapters": weakest,
        "strongest_chapters": strongest,
        "slowest_questions": sorted(slowest_questions, key=lambda x: -x["time_seconds"])[:5],
    }


# ─── Recommendation Logic ───────────────────────────────────────────────────

def build_weakness_scores(analysis: dict) -> dict:
    """
    Convert chapter_breakdown into topic-level weakness_scores suitable
    for the recommender. Lower performance → higher weakness score.
    """
    weakness = {}
    for chapter, stats in analysis["chapter_breakdown"].items():
        topic = CHAPTER_TO_TOPIC.get(chapter)
        if topic and topic in RECOMMENDER_TOPICS:
            # Invert: 0% performance → 1.0 weakness, 100% → 0.0
            weakness[topic] = round(max(0, (100 - stats["average_score_pct"])) / 100, 3)
    return weakness


def get_questions_for_topic(topic: str, difficulty: str = None, limit: int = 5) -> list[str]:
    """Get question IDs from the bank matching a topic and optional difficulty."""
    results = []
    for q in QUESTION_BANK:
        if q.get("topic") == topic:
            if difficulty and q.get("difficulty") != difficulty:
                continue
            qtype = q.get("questionType", "")
            qdata = q.get(qtype, {})
            if isinstance(qdata, dict) and qdata.get("answer"):  # skip broken questions
                results.append(q.get("qid", q["_id_normalized"]))
            if len(results) >= limit:
                break
    return results


def recommend_dosts(student: dict, analysis: dict) -> dict:
    """
    Generate a step-by-step DOST recommendation plan for the student.
    Uses the fixed recommender module for question selection.
    """
    steps = []
    weakest = analysis["weakest_chapters"]
    strongest = analysis["strongest_chapters"]
    trend = analysis["trend"]
    overall = analysis["overall_score_pct"]
    completion_rate = analysis["completion_rate_pct"]
    attempt_rate = analysis["attempt_rate_pct"]

    # Build weakness scores for the recommender
    weakness_scores = build_weakness_scores(analysis)

    # Get recommender-based question suggestions
    # Build a question bank in the format the recommender expects
    rec_qbank = []
    for q in QUESTION_BANK:
        topic = q.get("topic", "")
        diff_val = q.get("difficulty")
        diff_str = {1: "easy", 2: "medium", 3: "hard", 4: "hard", 5: "hard"}.get(diff_val, "medium") if isinstance(diff_val, int) else "medium"
        rec_qbank.append({
            "id": q.get("qid", q["_id_normalized"]),
            "topic": topic,
            "difficulty": diff_str,
        })

    recommended_qids = []
    if weakness_scores:
        recs = recommend_questions_for_weakness(weakness_scores, rec_qbank, top_n=20)
        recommended_qids = [r["question_id"] for r in recs]

    step_num = 1

    # Step 1: Address weakest chapter with concept revision
    if weakest:
        weakest_ch = weakest[0]
        topic = CHAPTER_TO_TOPIC.get(weakest_ch, weakest_ch.lower())
        q_ids = get_questions_for_topic(topic, limit=5)
        steps.append({
            "step": step_num,
            "dost_type": "concept",
            "target_chapter": weakest_ch,
            "params": DOST_CONFIG.get("concept", {}).get("params", {}),
            "question_ids": q_ids,
            "reasoning": f"{weakest_ch} is the student's weakest chapter. Start with concept revision to build foundational understanding before practice.",
            "message_to_student": f"Let's start by reviewing the core concepts in {weakest_ch}. Understanding the theory will make problem-solving much easier!"
        })
        step_num += 1

    # Step 2: Formula revision for weak chapters
    if len(weakest) >= 2:
        steps.append({
            "step": step_num,
            "dost_type": "formula",
            "target_chapter": weakest[1],
            "params": DOST_CONFIG.get("formula", {}).get("params", {}),
            "question_ids": [],
            "reasoning": f"{weakest[1]} needs formula revision. Many errors in this chapter likely stem from formula confusion.",
            "message_to_student": f"Time to brush up on the key formulas for {weakest[1]}. Having these at your fingertips will save you time and mistakes."
        })
        step_num += 1

    # Step 3: Practice assignment on weakest topic with recommender-selected questions
    if weakest:
        weakest_topic = CHAPTER_TO_TOPIC.get(weakest[0], "")
        # Use recommender results filtered for this topic
        rec_for_topic = [qid for qid in recommended_qids if any(
            q["_id_normalized"] == qid and q.get("topic") == weakest_topic for q in QUESTION_BANK
        )][:5]
        if not rec_for_topic:
            rec_for_topic = get_questions_for_topic(weakest_topic, limit=5)

        difficulty = "easy" if overall < 30 else "medium" if overall < 60 else "hard"
        params = {**DOST_CONFIG.get("practiceAssignment", {}).get("params", {})}
        params["difficulty"] = difficulty

        steps.append({
            "step": step_num,
            "dost_type": "practiceAssignment",
            "target_chapter": weakest[0],
            "params": params,
            "question_ids": rec_for_topic,
            "reasoning": f"Targeted practice on {weakest[0]} at {difficulty} difficulty based on current performance ({overall}% overall).",
            "message_to_student": f"Now let's practice {weakest[0]} with some targeted problems. Take your time — no timer here!"
        })
        step_num += 1

    # Step 4: Speed drill if slow response time
    avg_times = [a.get("avg_time_per_question_seconds", 0) for a in student["attempts"]]
    overall_avg_time = sum(avg_times) / len(avg_times) if avg_times else 0
    if overall_avg_time > 150:  # More than 2.5 min per question is slow for JEE
        speed_topic = CHAPTER_TO_TOPIC.get(strongest[0] if strongest else weakest[0], "")
        speed_qs = get_questions_for_topic(speed_topic, "easy", limit=10)
        steps.append({
            "step": step_num,
            "dost_type": "clickingPower",
            "target_chapter": strongest[0] if strongest else weakest[0],
            "params": DOST_CONFIG.get("clickingPower", {}).get("params", {}),
            "question_ids": speed_qs[:10],
            "reasoning": f"Average time per question is {overall_avg_time:.0f}s — above the 150s threshold. Speed drill on a stronger topic to build confidence and pace.",
            "message_to_student": "Your accuracy is developing, but speed is crucial for JEE. Let's do a quick-fire round on a topic you're comfortable with!"
        })
        step_num += 1

    # Step 5: Picking power for MCQ elimination practice
    if attempt_rate < 80:
        pick_topic = CHAPTER_TO_TOPIC.get(weakest[1] if len(weakest) > 1 else weakest[0], "")
        pick_qs = get_questions_for_topic(pick_topic, limit=5)
        steps.append({
            "step": step_num,
            "dost_type": "pickingPower",
            "target_chapter": weakest[1] if len(weakest) > 1 else weakest[0],
            "params": DOST_CONFIG.get("pickingPower", {}).get("params", {}),
            "question_ids": pick_qs,
            "reasoning": f"Attempt rate is {attempt_rate}% — student is skipping too many questions. Option elimination practice will help attempt more confidently.",
            "message_to_student": "You're leaving some questions unanswered. Let's practice eliminating wrong options — even partial knowledge can earn marks!"
        })
        step_num += 1

    # Step 6: Multi-day revision if multiple weak chapters
    if len(weakest) >= 2:
        params = {**DOST_CONFIG.get("revision", {}).get("params", {})}
        params["alloted_days"] = 3
        params["strategy"] = 1
        steps.append({
            "step": step_num,
            "dost_type": "revision",
            "target_chapter": ", ".join(weakest[:2]),
            "params": params,
            "question_ids": [],
            "reasoning": f"Multiple weak chapters ({', '.join(weakest[:2])}) need systematic revision over multiple days.",
            "message_to_student": f"Let's set up a 3-day revision plan covering {' and '.join(weakest[:2])}. Consistency beats cramming!"
        })
        step_num += 1

    # Step 7: Timed practice test to build exam readiness
    if completion_rate < 70 or trend == "declining":
        difficulty = "easy" if overall < 30 else "medium"
        params = {**DOST_CONFIG.get("practiceTest", {}).get("params", {})}
        params["difficulty"] = difficulty
        params["duration_minutes"] = 45  # shorter than full test to build confidence
        test_qs = recommended_qids[:10] if recommended_qids else []
        steps.append({
            "step": step_num,
            "dost_type": "practiceTest",
            "target_chapter": weakest[0] if weakest else "Mixed",
            "params": params,
            "question_ids": test_qs,
            "reasoning": f"Completion rate ({completion_rate}%) and trend ({trend}) suggest test anxiety. A shorter, easier test builds confidence.",
            "message_to_student": "Let's try a shorter mock test. The goal isn't perfection — it's completing the paper and building your exam stamina!"
        })
        step_num += 1

    # Step 8: Speed race for competitive motivation (if student is doing OK)
    if overall > 40:
        race_topic = CHAPTER_TO_TOPIC.get(strongest[-1] if strongest else weakest[0], "")
        race_qs = get_questions_for_topic(race_topic, "medium", limit=5)
        steps.append({
            "step": step_num,
            "dost_type": "speedRace",
            "target_chapter": strongest[-1] if strongest else weakest[0],
            "params": DOST_CONFIG.get("speedRace", {}).get("params", {}),
            "question_ids": race_qs,
            "reasoning": f"Student has decent fundamentals ({overall}%). A competitive race adds motivation and tests speed under pressure.",
            "message_to_student": "Ready for a challenge? Race against the bot on a topic you know well. Let's see how fast you can go!"
        })
        step_num += 1

    return {
        "student_id": student["student_id"],
        "name": student["name"],
        "total_steps": len(steps),
        "plan": steps,
    }


# ─── Leaderboard Logic ──────────────────────────────────────────────────────

def compute_student_score(student: dict, analysis: dict) -> float:
    """
    Composite scoring formula:
      40% overall_score_pct
    + 20% completion_rate
    + 15% attempt_rate
    + 15% trend_bonus (improving=15, stable=8, declining=0)
    + 10% consistency (inverse of score variance)
    """
    overall = analysis["overall_score_pct"]
    completion = analysis["completion_rate_pct"]
    attempt_rate = analysis["attempt_rate_pct"]
    trend = analysis["trend"]

    trend_bonus = {"improving": 15, "stable": 8, "declining": 0}.get(trend, 5)

    # Consistency: low variance across attempts = high consistency
    attempt_pcts = [compute_percentage(a) for a in student["attempts"]]
    if len(attempt_pcts) > 1:
        variance = sum((x - sum(attempt_pcts) / len(attempt_pcts)) ** 2 for x in attempt_pcts) / len(attempt_pcts)
        consistency = max(0, 10 - (variance ** 0.5) / 5)  # scale down
    else:
        consistency = 5

    score = (overall * 0.4) + (completion * 0.2) + (attempt_rate * 0.15) + (trend_bonus * 0.15) + (consistency * 0.10)
    return round(score, 2)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    load_data()


@app.post("/analyze/{student_id}")
def analyze(student_id: str):
    student = get_student(student_id)
    return analyze_student(student)


@app.post("/recommend/{student_id}")
def recommend_endpoint(student_id: str):
    student = get_student(student_id)
    analysis = analyze_student(student)
    return recommend_dosts(student, analysis)


@app.get("/question/{question_id}")
def get_question(question_id: str):
    q = QUESTION_INDEX.get(question_id)
    if not q:
        raise HTTPException(status_code=404, detail=f"Question {question_id} not found")
    return get_question_clean(q)


@app.get("/leaderboard")
def leaderboard():
    entries = []
    for student in STUDENTS_DATA:
        analysis = analyze_student(student)
        score = compute_student_score(student, analysis)
        entries.append({
            "rank": 0,
            "student_id": student["student_id"],
            "name": student["name"],
            "score": score,
            "overall_pct": analysis["overall_score_pct"],
            "strength": analysis["strongest_chapters"][0] if analysis["strongest_chapters"] else "N/A",
            "weakness": analysis["weakest_chapters"][0] if analysis["weakest_chapters"] else "N/A",
            "focus_area": analysis["weakest_chapters"][0] if analysis["weakest_chapters"] else "N/A",
            "trend": analysis["trend"],
        })

    entries.sort(key=lambda x: -x["score"])
    for i, e in enumerate(entries):
        e["rank"] = i + 1

    return {"leaderboard": entries}
