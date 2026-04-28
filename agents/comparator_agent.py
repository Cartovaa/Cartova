from langchain_openai import ChatOpenAI
from ..helper.state import PipelineState
from ..helper.utils import call_llm
import os
from dotenv import load_dotenv

load_dotenv()

COMPARATOR_SYSTEM = """
You are a product spec normalizer.
Given raw search result snippets about laptops (or other products),
extract and normalize a comparison table.
 
Return ONLY a valid JSON array. Each item must have:
{
  "name": <product name>,
  "price_inr": <number or null>,
  "weight_kg": <number or null>,
  "battery_hours": <number or null>,
  "ram_gb": <number or null>,
  "storage_gb": <number or null>,
  "processor": <string or null>,
  "url": <string or null>
}
 
If a spec is missing, set it to null. Do NOT invent values.
"""

llm = ChatOpenAI(
    model="abacusai/dracarys-llama-3.1-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.2,
)

def comparator(state: PipelineState) -> PipelineState:
    print("\n[Agent 3 — Comparator] Normalizing spec table...")
    listings_text = "\n\n".join([
        f"Title: {r.get('title','')}\nURL: {r.get('url','')}\nSnippet: {r.get('content','')}"
        for r in (state["raw_listings"] or [])[:15]  # cap at 15 to stay within context
    ])
    table = call_llm(COMPARATOR_SYSTEM, listings_text)
    if isinstance(table, dict):
        table = [table]
    print(f"  → Normalized {len(table)} products")
    return {**state, "normalized_table": table}
