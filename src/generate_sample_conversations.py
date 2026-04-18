"""
Generate 50 sample conversations covering diverse cases, then score them
against a representative subset of facets and save results to JSON.
"""

import json
import os
import sys
import csv
import time
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

sys.path.insert(0, os.path.dirname(__file__))
from scorer import score_turn, OLLAMA_MODEL, OLLAMA_BASE_URL

class Turn(BaseModel):
    turn_id: int
    speaker: str
    text: str

class ConversationData(BaseModel):
    turns: list[Turn]

# ── Conversation Templates ──────────────────────────────────────────────────

CONVERSATION_SEEDS = [
    # Emotional support scenarios
    {"id": 1, "theme": "grief_support", "desc": "Person grieving loss of a parent"},
    {"id": 2, "theme": "anxiety_therapy", "desc": "Anxious student before exam"},
    {"id": 3, "theme": "depression_check", "desc": "Feeling hopeless and unmotivated"},
    {"id": 4, "theme": "breakup_support", "desc": "Going through a breakup"},
    {"id": 5, "theme": "anger_management", "desc": "Venting about workplace injustice"},
    # Professional / cognitive scenarios
    {"id": 6, "theme": "technical_help", "desc": "Debugging a software problem"},
    {"id": 7, "theme": "career_advice", "desc": "Career change decision"},
    {"id": 8, "theme": "negotiation", "desc": "Salary negotiation practice"},
    {"id": 9, "theme": "critical_feedback", "desc": "Giving critical feedback on a report"},
    {"id": 10, "theme": "problem_solving", "desc": "Solving a logistics problem"},
    # Social / interpersonal
    {"id": 11, "theme": "conflict_resolution", "desc": "Mediating a family argument"},
    {"id": 12, "theme": "friendship_repair", "desc": "Reconnecting after a fight"},
    {"id": 13, "theme": "romantic_intent", "desc": "Flirtatious conversation"},
    {"id": 14, "theme": "parenting_advice", "desc": "Struggling parent seeking advice"},
    {"id": 15, "theme": "social_anxiety", "desc": "Fear of social situations"},
    # Safety edge cases
    {"id": 16, "theme": "passive_aggression", "desc": "Subtle passive-aggressive exchange"},
    {"id": 17, "theme": "manipulation_attempt", "desc": "Manipulative persuasion attempt"},
    {"id": 18, "theme": "gaslighting", "desc": "One party denying the other's reality"},
    {"id": 19, "theme": "boundary_violation", "desc": "Pushing against personal boundaries"},
    {"id": 20, "theme": "crisis_intervention", "desc": "Someone expressing hopelessness"},
    # Linguistic quality
    {"id": 21, "theme": "eloquent_speech", "desc": "Articulate philosophical discussion"},
    {"id": 22, "theme": "poor_grammar", "desc": "Low literacy, informal conversation"},
    {"id": 23, "theme": "bilingual_code_switch", "desc": "Mixing two languages mid-conversation"},
    {"id": 24, "theme": "formal_register", "desc": "Highly formal business communication"},
    {"id": 25, "theme": "slang_heavy", "desc": "Heavy use of slang and informal language"},
    # Spiritual / mindfulness
    {"id": 26, "theme": "mindfulness_practice", "desc": "Discussing meditation experience"},
    {"id": 27, "theme": "religious_doubt", "desc": "Questioning religious beliefs"},
    {"id": 28, "theme": "spiritual_seeking", "desc": "Looking for meaning and purpose"},
    {"id": 29, "theme": "grief_spiritual", "desc": "Using spirituality to cope with loss"},
    {"id": 30, "theme": "interfaith_dialogue", "desc": "Respectful cross-faith conversation"},
    # Motivation / self-improvement
    {"id": 31, "theme": "goal_setting", "desc": "Setting ambitious career goals"},
    {"id": 32, "theme": "procrastination", "desc": "Struggling to start important tasks"},
    {"id": 33, "theme": "resilience_test", "desc": "Bouncing back from failure"},
    {"id": 34, "theme": "self_doubt", "desc": "Imposter syndrome at work"},
    {"id": 35, "theme": "productivity_coaching", "desc": "Building better work habits"},
    # Cognitive / reasoning
    {"id": 36, "theme": "logical_debate", "desc": "Structured argument on climate policy"},
    {"id": 37, "theme": "misinformation", "desc": "Spreading and correcting false beliefs"},
    {"id": 38, "theme": "ethical_dilemma", "desc": "Trolley problem variant discussion"},
    {"id": 39, "theme": "statistical_misuse", "desc": "Misinterpreting data statistics"},
    {"id": 40, "theme": "creative_brainstorm", "desc": "Brainstorming a new product idea"},
    # Health / wellbeing
    {"id": 41, "theme": "chronic_illness", "desc": "Living with a chronic condition"},
    {"id": 42, "theme": "health_anxiety", "desc": "Excessive worry about symptoms"},
    {"id": 43, "theme": "diet_advice", "desc": "Seeking nutritional guidance"},
    {"id": 44, "theme": "exercise_motivation", "desc": "Getting back into fitness"},
    {"id": 45, "theme": "sleep_issues", "desc": "Struggling with insomnia"},
    # Relationship dynamics
    {"id": 46, "theme": "jealousy_relationship", "desc": "Jealousy in a relationship"},
    {"id": 47, "theme": "trust_rebuilding", "desc": "Rebuilding trust after betrayal"},
    {"id": 48, "theme": "assertiveness", "desc": "Learning to say no"},
    {"id": 49, "theme": "loneliness", "desc": "Feeling isolated and disconnected"},
    {"id": 50, "theme": "celebration", "desc": "Sharing exciting good news"},
]


def generate_conversation(seed: dict, variation: int = 1) -> list[dict]:
    """Use ChatOllama to generate realistic 3-4 turn conversations via structured outputs."""
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.7,
    ).with_structured_output(ConversationData)

    prompt = f"""Generate a realistic, natural conversation with 3-4 turns about: {seed['desc']}
Theme: {seed['theme']}
Variation Instruction: This is variant #{variation} of this theme. Adjust the tone, context, background, or character attitudes slightly so it provides uniquely diverse data compared to other versions.

Rules:
- Make it emotionally authentic and varied.
- Include subtle psychological cues relevant to the theme.
"""
    response = llm.invoke(prompt)
    if not response or not getattr(response, 'turns', None):
        return []
    return [{"turn_id": t.turn_id, "speaker": t.speaker, "text": t.text} for t in response.turns]


def load_facets() -> list[dict]:
    """Load all observable facets for scoring sample conversations."""
    facets = []
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'facets_cleaned.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("observable_in_text", "True") == "True":
                facets.append({
                    "facet_id": int(row["facet_id"]),
                    "facet_name": row["facet_name"],
                    "category": row["category"],
                })
    return facets


def main():
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'conversations')
    os.makedirs(output_dir, exist_ok=True)

    # Use all applicable facets dynamically
    facets = load_facets()
    print(f"Scoring against {len(facets)} facets per turn")

    all_results = []

    for variation in range(1, 7):  # 6 variations of 50 seeds = 300 total points
        for i, seed in enumerate(CONVERSATION_SEEDS):
            current_idx = (variation - 1) * 50 + i + 1
            print(f"[{current_idx}/300] Generating: {seed['theme']} (Variation {variation}) ...")
            try:
                turns = generate_conversation(seed, variation)
                print(f"  → {len(turns)} turns generated, scoring turn 1...")

                # Score only the first user turn for the sample (keeps demo fast)
                user_turn = next((t for t in turns if t["speaker"] == "User"), turns[0])
                turn_text = f"{user_turn['speaker']}: {user_turn['text']}"
                scores = score_turn(turn_text, facets)

                result = {
                    "conversation_id": current_idx,
                    "theme": seed["theme"],
                    "description": seed["desc"],
                    "variation": variation,
                    "turns": turns,
                    "scored_turn": {
                        "turn_id": user_turn["turn_id"],
                        "speaker": user_turn["speaker"],
                        "text": user_turn["text"],
                        "scores": scores,
                    }
                }
                all_results.append(result)

                # Save individual file
                fname = os.path.join(output_dir, f"conv_{current_idx:03d}_{seed['theme']}_v{variation}.json")
                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)

                time.sleep(0.5)

            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({
                    "conversation_id": current_idx, 
                    "theme": seed["theme"], 
                    "variation": variation, 
                    "error": str(e)
                })

    # Save combined file
    combined_path = os.path.join(output_dir, "all_conversations_scored.json")
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone! {len(all_results)} conversations saved to {output_dir}")
    print(f"Combined file: {combined_path}")


if __name__ == "__main__":
    main()
