"""Graph-topology smoke tests (no DB / LLM): the chain and debate graphs wire up
the expected nodes.

Trimmed from an earlier scratch file — dropped a `test_graph_execution_with_memory`
case that wrapped everything in `except Exception: pass` (it asserted nothing and
gave false confidence) and a mode test that patched internals. Kept only the two
checks that catch a real regression: the graph structure.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import build_chain_graph, build_final_recommendation_graph  # noqa: E402


def test_chain_graph_has_expected_nodes():
    nodes = set(build_chain_graph().get_graph().nodes.keys())
    for expected in ("data", "sentiment", "valuation", "fundamental", "risk", "writer"):
        assert expected in nodes, f"chain graph missing node {expected!r}"


def test_debate_graph_routes_from_manager():
    g = build_final_recommendation_graph().get_graph()
    assert "debate_manager" in set(g.nodes.keys())
    # the manager is a routing source (conditional dispatch to the specialists)
    assert "debate_manager" in {e.source for e in g.edges}
