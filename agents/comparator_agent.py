from langchain_openai import ChatOpenAI
from ..utils import call_llm
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
    model="meta/llama-3.1-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.2,
)

def comparator():
    recommendation = call_llm(llm, COMPARATOR_SYSTEM, prompt)

