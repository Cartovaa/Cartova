from typing import TypedDict, List, Optional


class PipelineState(TypedDict):
    user_query: str
    parsed_spec: Optional[dict]         # Agent 1 output
    raw_listings: Optional[List[dict]]  # Agent 2 output
    normalized_table: Optional[List[dict]]  # Agent 3 output
    review_signals: Optional[List[dict]]    # Agent 4 output
    recommendations: Optional[List[dict]]   # Agent 5 output
    search_fallback_used: bool
