"""Regression test: every metric shown in the templates must come from
metrics.json, with no hardcoded fallback numbers left in the Jinja source.

Run with: python tests/test_metrics_consistency.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TEMPLATES = ["templates/simulator.html", "templates/lab_data.html"]

# Matches e.g. rf_metrics.get('r2_test', 0.9797) or bc.get('global', {}).get('dmso', 3.0)
# i.e. a .get(...) call whose second argument is a bare number (not {} or another .get chain).
HARDCODED_DEFAULT_RE = re.compile(r"""\.get\(\s*['"][^'"]+['"]\s*,\s*-?\d""")


def test_no_hardcoded_metric_defaults():
    violations = []
    for rel_path in TEMPLATES:
        path = os.path.join(ROOT, rel_path)
        with open(path, encoding="utf-8") as f:
            src = f.read()
        for m in HARDCODED_DEFAULT_RE.finditer(src):
            line_no = src.count("\n", 0, m.start()) + 1
            violations.append(f"{rel_path}:{line_no}: {m.group(0)}")

    assert not violations, "Hardcoded numeric fallback(s) found in templates:\n" + "\n".join(violations)
    print(f"PASS: no hardcoded numeric defaults in {TEMPLATES}")


def test_rendered_pages_match_metrics_json():
    with open(os.path.join(ROOT, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)

    import app as app_module
    client = app_module.app.test_client()

    rf = metrics["random_forest"]
    nn = metrics["neural_network"]

    html = client.get("/application").get_data(as_text=True)
    assert f"{rf['r2_test']:.3f}" in html, "RF R2 (test) from metrics.json not found in /application"
    assert f"{rf['rmse_test']:.2f}" in html, "RF RMSE (test) from metrics.json not found in /application"
    assert f"{nn['r2_test']:.3f}" in html, "NN R2 (test) from metrics.json not found in /application"
    assert f"{nn['rmse_test']:.2f}" in html, "NN RMSE (test) from metrics.json not found in /application"

    tied = rf["best_combinations"]["global_tied"]
    assert f"{tied['viability']}" in html, "Global-optimum viability from metrics.json not found in /application"
    assert f"{rf['best_combinations']['treh_only']['viability']}" in html
    print("PASS: /application renders numbers sourced from metrics.json")

    html_lab = client.get("/lab-data").get_data(as_text=True)
    assert f"{rf['r2_test']:.3f}" in html_lab
    assert f"{rf['rmse_test']:.2f}" in html_lab
    assert f"{nn['r2_test']:.3f}" in html_lab
    assert f"{nn['cv_r2_mean']:.3f}" in html_lab
    print("PASS: /lab-data renders numbers sourced from metrics.json")


def test_example_prediction_matches_metrics_json():
    with open(os.path.join(ROOT, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    example = metrics["example_prediction_5dmso_10trehalose"]

    import app as app_module
    client = app_module.app.test_client()
    r = client.post("/application", data={"dmso": "5", "trehalose": "10", "model": "both"})
    html = r.get_data(as_text=True)

    assert f"{round(example['random_forest'], 2)}" in html, "RF example prediction mismatch"
    assert f"{round(example['neural_network_numpy'], 2)}" in html, "NN example prediction mismatch"
    print("PASS: 5% DMSO / 10% trehalose prediction matches metrics.json")


if __name__ == "__main__":
    test_no_hardcoded_metric_defaults()
    test_rendered_pages_match_metrics_json()
    test_example_prediction_matches_metrics_json()
    print("\nALL CONSISTENCY TESTS PASSED.")
