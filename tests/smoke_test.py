from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import learned_operator

assert (ROOT / "data/raw/alzkb/alzkb-v2-0-0_memgraph.cypherl").exists()
assert (ROOT / "data/raw/evo2/ad_all_scores_v3_vcf.csv").exists()
model = learned_operator.LearnedScorer.load(
    ROOT / "outputs/models/learned_operator_weights.json"
)
assert len(model.weights) == learned_operator.N_FEATURES == 14
config = json.loads((ROOT / "config/default.json").read_text())
assert config["allow_heuristic_scorer"] is False
assert config["allow_synthetic_graph"] is False
print("smoke test passed: real inputs + 14-feature learned scorer")
