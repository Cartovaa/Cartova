from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from typing import List
from helper.state import PipelineState
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="meta/llama-3.1-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.2,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

RESULTS_THRESHOLD = 3
 
def run_searches(queries: List[str]) -> List[dict]:
    results = []
    for q in queries:
        try:
            res = tavily.search(query=q, max_results=5)
            results.extend(res.get("results", []))
        except Exception as e:
            print(f"  ⚠ Search error for '{q}': {e}")
    return results
 
def searcher(state: PipelineState) -> PipelineState:
    print("\n[Agent 2 — Searcher] Firing search queries...")
    spec = state["parsed_spec"]
    queries = spec["search_queries"]
    fallback_used = False
 
    listings = run_searches(queries)
    print(f"  → Got {len(listings)} results")
 
    if len(listings) < RESULTS_THRESHOLD:
        print("  ↩ Below threshold — dropping soft constraints and retrying...")
        fallback_used = True
        category = spec["hard_constraints"]["category"]
        max_price = spec["hard_constraints"].get("max_price_inr", "")
        fallback_queries = [
            f"best {category} under {max_price} INR India 2026",
            f"top {category} budget India buy online",
            f"{category} under {max_price} rupees review",
        ]
        listings = run_searches(fallback_queries)
        print(f"  → Fallback got {len(listings)} results")
 
    return {**state, "raw_listings": listings, "search_fallback_used": fallback_used}
