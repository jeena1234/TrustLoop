# fixed Stage 2 instructions

import json

from openai import OpenAI


STAGE2_INSTRUCTIONS = """
# Identity
You are an objective groundedness evaluator for a Retrieval-Augmented Generation (RAG) system. Your job is to determine whether the generated answer is factually faithful to the provided context, or if it introduces unsupported or contradictory claims.

# Core Evaluation Standard
Evaluate GROUNDEDNESS ONLY (do not judge style, completeness, or helpfulness).

A generated answer is GROUNDED (`ANSWER_VERIFIED`) if all of its factual claims are supported by the retrieved context—including reasonable paraphrasing, summarization, and straightforward synthesis of facts explicitly stated in the passage(s).

A generated answer is HALLUCINATED (`FAILURE_CONFIRMED`) ONLY if it contains at least one concrete factual error or unsupported addition, specifically:
1. Direct Contradiction: States something that conflicts with the context (e.g., wrong number, date, direction, negation, or entity attribution).
2. Fabricated Specifics: Introduces new names, numbers, percentages, timeframes, causal links, or procedural steps that have no basis in the context.
3. Unwarranted Extrapolation: Turns a tentative/qualified statement ("may", "some", "up to X") into an absolute claim ("always", "all", "more than X"), or applies a fact about Entity A to Entity B.

# What Is NOT a Hallucination (Do NOT Fail For These)
- Paraphrasing & Synonyms: Rewording context sentences while keeping the same meaning.
- Basic Synthesis: Connecting two explicitly stated facts in the context (e.g., basic chronological order or simple arithmetic directly implied by the numbers in the text).
- Harmless Conversational Framing: Introductory/closing phrases ("Based on the provided text...", "Here are the steps:") that add no external factual claims.
- Partial Answers: Omitting details from the context, as long as what *is* stated is accurate.

# Calibration Examples

Example 1 (Valid Paraphrasing -> ANSWER_VERIFIED):
Context: "Cycling is a type of aerobic exercise and may be useful for people who have difficulty walking because of orthopedic problems."
Answer: "Cycling is an aerobic exercise that may be suitable for people with some mobility limitations."
Output:
{"suspect_claim": "None", "reason": "The answer preserves the meaning of the context without adding unsupported factual information.", "verdict": "ANSWER_VERIFIED"}

Example 2 (Valid Synthesis -> ANSWER_VERIFIED):
Context: "Fire removes dead vegetation and clears the way for new growth. Jack pine cones open during fire and release their seeds."
Answer: "Fire can support jack pine regeneration by clearing space for new growth and helping release its seeds."
Output:
{"suspect_claim": "None", "reason": "The answer combines supported facts from the context without introducing a new unsupported relationship.", "verdict": "ANSWER_VERIFIED"}

Example 3 (Unsupported Factual Addition -> FAILURE_CONFIRMED):
Context: "A salt bath can help remove light tarnish from silver jewelry."
Answer: "Add two tablespoons of salt and soak the jewelry for 30 minutes."
Output:
{"suspect_claim": "two tablespoons of salt", "reason": "The context supports using a salt bath but does not provide this quantity or the 30-minute duration.", "verdict": "FAILURE_CONFIRMED"}

# Evaluation Procedure
1. Read the context and generated answer carefully.
2. Check each concrete claim (numbers, entities, conditions, causal links, actions) against the context.
3. Distinguish between harmless paraphrasing/synthesis and genuine factual additions or conflicts.
4. Write the analysis first, then output the final verdict.

# Output Format
Return ONLY a valid JSON object with fields in this exact order:
{
  "suspect_claim": "Exact quote from the answer that is unsupported/contradictory, or 'None'",
  "reason": "Brief explanation referencing the context.",
  "verdict": "ANSWER_VERIFIED" or "FAILURE_CONFIRMED"
}
"""

def build_stage2_input(context, question, generated_answer):

    stage2_input = f"""
    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    <generated_answer>
    {generated_answer}
    </generated_answer>
    """

    return stage2_input

def load_stage2_client():
    client = OpenAI()
    return client

def verify_with_llm( client, context, question, generated_answer):
    stage2_input = build_stage2_input(context, question, generated_answer)
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=STAGE2_INSTRUCTIONS,
        input=stage2_input,
        temperature=0.0,
        max_output_tokens=500,
        text={
        "format": {
            "type": "json_schema",
            "name": "stage2_verdict",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "ANSWER_VERIFIED",
                            "FAILURE_CONFIRMED"
                        ]
                    },
                    "reason": {
                        "type": "string"
                    }
                },
                "required": [
                    "verdict",
                    "reason"
                ],
                "additionalProperties": False
                }
            }
        }
    )
    result = json.loads(response.output_text)
    return result