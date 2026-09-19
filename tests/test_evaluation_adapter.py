from fhp_escher.evaluation_adapter import _import_suite
from fhp_escher.game import load_fhp_game


def test_shared_evaluation_suite_is_available():
    suite = _import_suite()
    assert len(suite.published_rule_agents(load_fhp_game())) == 5
