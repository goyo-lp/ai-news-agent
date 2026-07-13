from langgraph.graph import END

from app.graph.workflow import route_after_rank


def test_route_after_rank_continues_with_articles() -> None:
    state = {"articles_selected": [{"id": "a1"}]}
    assert route_after_rank(state) == "summarize"


def test_route_after_rank_ends_when_empty() -> None:
    assert route_after_rank({"articles_selected": []}) == END
    assert route_after_rank({}) == END
