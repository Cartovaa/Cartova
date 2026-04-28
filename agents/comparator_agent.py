from langchain_openai import ChatOpenAI
from helper.state import PipelineState
from helper.utils import call_llm
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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
    timeout=60,        # add this
    max_retries=3,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

def enrich_with_price(name: str) -> str:
    try:
        result = tavily.search(
            query=f"{name} price India site:flipkart.com OR site:amazon.in",
            max_results=3
        )
        snippets = [r.get("content", "") for r in result.get("results", [])]
        return "\n".join(snippets)
    except:
        return ""

def comparator(state: PipelineState) -> PipelineState:
    print("\n[Agent 3 — Comparator] Normalizing spec table...")

    listings_text = "\n\n".join([
        f"Title: {r.get('title','')}\nSnippet: {r.get('content','')[:300]}"
        for r in (state["raw_listings"] or [])[:8]
    ])

    # first pass — get product names
    table = call_llm(llm, COMPARATOR_SYSTEM, listings_text)
    if isinstance(table, dict):
        table = [table]

    # second pass — enrich each product with price
    for product in table:
        if not product.get("price_inr"):
            print(f"  → Fetching price for: {product['name']}")
            price_data = enrich_with_price(product["name"])
            if price_data:
                enriched = call_llm(llm,
                    "Extract only price_inr as a number from this text. Return JSON: {\"price_inr\": <number or null>}",
                    price_data
                )
                product["price_inr"] = enriched.get("price_inr")

    print(f"  → Normalized {len(table)} products")
    return {**state, "normalized_table": table}
