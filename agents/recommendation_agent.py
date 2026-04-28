from langchain_openai import ChatOpenAI
from ..helper.state import PipelineState
from ..helper.util import call_llm
import os
from dotenv import load_dotenv

load_dotenv()

RECOMMENDER_SYSTEM = """
You are a product recommendation engine.
Score each product against the original user requirements.
 
Return ONLY a valid JSON array, sorted by match_score descending:
[
  {
    "rank": <number>,
    "product": <name>,
    "match_score": <integer 0-100>,
    "price_inr": <number or null>,
    "pros": <list of strings>,
    "cons": <list of strings>,
    "reasoning": <1-2 sentence explanation>,
    "verdict": <"top pick" | "good value" | "consider if" | "skip">
  }
]
"""

llm = ChatOpenAI(
    model="minimaxai/minimax-m2.7",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.2,
)


def recommender(state: PipelineState) -> PipelineState:
    print("\n[Agent 5 — Recommender] Scoring and ranking products...")
    prompt = f"""
            Original user query: {state['user_query']}
 
            Parsed requirements:
            {json.dumps(state['parsed_spec'], indent=2)}
 
            Normalized product specs:
            {json.dumps(state['normalized_table'], indent=2)}
 
            Review signals:
            {json.dumps(state['review_signals'], indent=2)}
 
            Score and rank all products above against the user requirements.
        """
    recommendations = call_llm(llm, RECOMMENDER_SYSTEM, prompt)
    if isinstance(recommendations, dict):
        recommendations = [recommendations]
    print(f"  → Ranked {len(recommendations)} products")
    return {**state, "recommendations": recommendations}
