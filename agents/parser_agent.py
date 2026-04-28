from flask import json
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from ..helper.state import PipelineState
from ..helper.utils import call_llm

PARSER_SYSTEM = """
You are a product requirement parser. 
Convert the user query into a structured JSON spec.

Return ONLY valid JSON with this exact shape:
{
  "hard_constraints": {
    "max_price_inr": <number or null>,
    "min_price_inr": <number or null>,
    "category": <string>
  },
  "soft_constraints": {
    "weight": <"lightweight"|"any">,
    "use_case": <string>,
    "preferred_brands": <list of strings or []>,
    "preferred_specs": <list of strings or []>
  },
  "search_queries": <list of 3 distinct search query strings>
}
"""


load_dotenv()

llm = ChatOpenAI(
    model="abacusai/dracarys-llama-3.1-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.2,
)

def parser(state: PipelineState) -> PipelineState:
    print("\n[Agent 1 — Parser] Parsing query...")
    spec = call_llm(PARSER_SYSTEM, state["user_query"])
    print(f"  → Spec: {json.dumps(spec, indent=2)}")
    return {**state, "parsed_spec": spec}
