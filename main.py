from agents.comparator_agent import comparator
from agents.parser_agent import parser
from agents.review_agent import reviewer
from agents.searcher_agent import searcher
from agents.recommendation_agent import recommender

from helper.state import PipelineState
from langgraph.graph import StateGraph, END


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)
 
    graph.add_node("parser",      parser)
    graph.add_node("searcher",    searcher)
    graph.add_node("comparator",  comparator)
    graph.add_node("reviewer",    reviewer)
    graph.add_node("recommender", recommender)
 
    graph.set_entry_point("parser")
    graph.add_edge("parser",      "searcher")
    graph.add_edge("searcher",    "comparator")
    graph.add_edge("comparator",  "reviewer")
    graph.add_edge("reviewer",    "recommender")
    graph.add_edge("recommender", END)
 
    return graph.compile()

def run_pipeline(query: str) -> dict:
    pipeline = build_graph()
 
    initial_state: PipelineState = {
        "user_query": query,
        "parsed_spec": None,
        "raw_listings": None,
        "normalized_table": None,
        "review_signals": None,
        "recommendations": None,
        "search_fallback_used": False,
    }
 
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
 
    final_state = pipeline.invoke(initial_state)
 
    print(f"\n{'='*60}")
    print("FINAL RECOMMENDATIONS")
    print('='*60)
    for rec in (final_state["recommendations"] or []):
        print(f"\n#{rec.get('rank')} {rec.get('product')} — {rec.get('match_score')}% match")
        print(f"  Verdict  : {rec.get('verdict')}")
        print(f"  Price    : ₹{rec.get('price_inr', 'N/A')}")
        print(f"  Pros     : {', '.join(rec.get('pros', []))}")
        print(f"  Cons     : {', '.join(rec.get('cons', []))}")
        print(f"  Reasoning: {rec.get('reasoning')}")
 
    if final_state["search_fallback_used"]:
        print("\n⚠ Note: Soft constraints were dropped during search (fallback activated)")
 
    return final_state
 
 
if __name__ == "__main__":
    result = run_pipeline("budget laptop under ₹40k, lightweight, for college")
