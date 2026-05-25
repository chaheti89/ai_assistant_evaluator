import json

with open("evaluation/results.json") as f:
    results = json.load(f)

def get_score(d):
    """Extract score — handles int, missing, or malformed dict."""
    s = d.get("score", 0)
    if isinstance(s, dict):
        return round(sum(s.values()) / len(s))  # average if dict
    if isinstance(s, int):
        return s
    return 0  # MISSING or unexpected

print("\n── Category Summary ─────────────────────")
for category in ["factual", "adversarial", "bias"]:
    cat_rows = [r for r in results if r["category"] == category]
    f_avg = sum(get_score(r["frontier"]) for r in cat_rows) / len(cat_rows)
    o_avg = sum(get_score(r["oss"]) for r in cat_rows) / len(cat_rows)
    print(f"{category:12} | Frontier: {f_avg:.1f}/5  |  OSS: {o_avg:.1f}/5")

print("\n── Per-prompt scores ────────────────────")
for r in results:
    f_score = get_score(r["frontier"])
    o_score = get_score(r["oss"])
    winner = "✓" if f_score >= o_score else "✗"
    print(f"[{r['id']}] {r['category']:12} | Frontier: {f_score}/5 | OSS: {o_score}/5 {winner}")

# Overall
all_f = sum(get_score(r["frontier"]) for r in results)
all_o = sum(get_score(r["oss"]) for r in results)
print(f"\n── Overall ──────────────────────────────")
print(f"{'Total':12} | Frontier: {all_f/len(results):.1f}/5  |  OSS: {all_o/len(results):.1f}/5")