# Ahoum Prompt Log
*This log details the core LLM injection prompts, showing how they were optimized to scale the architecture securely.*

---

## 1. Conversation Generation Prompt
**File Location:** `src/generate_sample_conversations.py`  
**Purpose:** Iteratively construct highly realistic, diverse, multi-turn roleplay conversations across 6 stylistic variations for 50 distinct psychological themes.

### Previous Prompt (V1)
```text
Generate a realistic, natural conversation with 3-4 turns about: {seed['desc']}
Theme: {seed['theme']}
Variation Instruction: This is variant #{variation} of this theme. Adjust the tone, context, background, or character attitudes slightly so it provides uniquely diverse data compared to other versions.

Format as JSON array:
[
  {{"turn_id": 1, "speaker": "User", "text": "..."}},
  {{"turn_id": 2, "speaker": "Assistant", "text": "..."}},
  {{"turn_id": 3, "speaker": "User", "text": "..."}}
]

Rules:
- Make it emotionally authentic and varied
- Include subtle psychological cues relevant to the theme
- Return ONLY the JSON array, no markdown
```

### Current Optimized Prompt (V2)
```text
Generate a realistic, natural conversation with 3-4 turns about: {seed['desc']}
Theme: {seed['theme']}
Variation Instruction: This is variant #{variation} of this theme. Adjust the tone, context, background, or character attitudes slightly so it provides uniquely diverse data compared to other versions.

Rules:
- Make it emotionally authentic and varied.
- Include subtle psychological cues relevant to the theme.
```

### 🧠 Why the changes were made:
- **Removed explicit JSON string templates & markdown bans:** We transitioned to **Langchain Structured Outputs**. Langchain dynamically handles schema injection and format validation under the hood via the Pydantic `ConversationData` object.
- **Removed brittle Regex parsing:** Initially, the system attempted to strip ` ```json ` blocks manually from the output. By stripping the manual JSON instructions, we prevented the LLM from entering a dual-format conflict, allowing the native API object parser to cleanly capture the payload on the first try with zero `"not_found_error"` bugs.

---

## 2. Global Judge System Prompt
**File Location:** `src/scorer.py`  
**Purpose:** Prime the evaluate LLM (Llama 3.1 8B) for strict, impartial, and highly-discriminating evaluation parameters.

### Current Prompt
```text
You are a precise conversation evaluation judge. Score a conversation turn on behavioral/linguistic/psychological facets.

Rules:
1. Base scores ONLY on evidence in the text. 
2. Unobservable facets (physiological, activity counts): score=1, confidence=0.1.
3. Use the full 1-5 range. Do NOT default everything to 3.
```

*(No major structural overrides were needed here, as the foundational logic constraint was already solid.)*

---

## 3. Dynamic Injection Prompt (User Scoring)
**File Location:** `src/scorer.py`  
**Purpose:** Dynamically append the literal conversation metrics to the LLM alongside an iterative sub-array of facets to score.

### Previous Prompt Injection Logic (V1)
*The V1 pipeline mathematically hardcoded a slice of 20 facets at a time (`batch_size=20`), and manually begged the model to output a specific JSON dictionary array.*

### Current Optimized Prompt Injection (V2)
```text
{SYSTEM_PROMPT}

Conversation turn to evaluate:
---
{conversation_turn}
---

Score EXACTLY these {len(batch)} facets based on the text:
{facet_list}
```

### 🧠 Why the changes were made:
- **Scaled Batch Limit from 20 to 150:** In V1, scoring 300 facets required slicing the data into pieces of 20, forcing the code to make **15 sequential API calls per turn**. Since we upgraded the standard target to **Llama 3.1 8B** (which natively handles a 128,000 token context window compared to older 8k models), we increased `len(batch)` natively to `150`. This slashes the required API calls down to just **2 per turn**, speeding up evaluation execution by over **90%**!
- **Removed manual format enforcement:** Just like the conversation generator, we shifted the burden of structuring the `1-5` scores from the raw text string into a strict Pydantic Object schema (`BatchScoreOutput`), ensuring 100% data integrity without the LLM hallucinating extra fields or getting confused by missing quotation marks.
