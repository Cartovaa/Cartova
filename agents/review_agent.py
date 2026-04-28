from langchain_openai import ChatOpenAI
from helper.state import PipelineState
from helper.utils import call_llm
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

REVIEWER_SYSTEM = """
You are a product review analyst.
Given a product name and search snippet data, extract review signals.
 
Return ONLY valid JSON with this shape:
{
  "product": <name>,
  "praised_features": <list of strings>,
  "criticisms": <list of strings>,
  "recurring_issues": <list of strings>,
  "best_fit_use_cases": <list of strings>,
  "sentiment_score": <float 0.0 to 1.0>
}
"""

llm = ChatOpenAI(
    model="minimaxai/minimax-m2.7",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.2,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

def reviewer(state: PipelineState) -> PipelineState:
    print("\n[Agent 4 — Review Analyzer] Extracting review signals...")
    table = state["normalized_table"] or []
    raw = state["raw_listings"] or []
    signals = []
 
    for product in table[:6]:  # analyze top 6 products
        name = product.get("name", "Unknown")
        # find matching snippets
        relevant = [
            r.get("content", "") for r in raw
            if name.split()[0].lower() in r.get("title", "").lower()
            or name.split()[0].lower() in r.get("content", "").lower()
        ][:3]
        snippet_text = "\n".join(relevant) if relevant else "No specific reviews found."
 
        prompt = f"Product: {name}\n\nReview data:\n{snippet_text}"
        try:
            signal = call_llm(llm, REVIEWER_SYSTEM, prompt)
            signals.append(signal)
            print(f"  → Analyzed: {name}")
        except Exception as e:
            print(f"  ⚠ Failed for {name}: {e}")
            signals.append({"product": name, "praised_features": [], "criticisms": [],
                            "recurring_issues": [], "best_fit_use_cases": [], "sentiment_score": 0.5})
 
    return {**state, "review_signals": signals}
