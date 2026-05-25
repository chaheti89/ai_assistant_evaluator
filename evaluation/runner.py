"""
Runs all prompts through both assistants and scores them.
Output: evaluation/results.json

Usage:
    python -m evaluation.runner
"""

import json
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from assistants.frontier import FrontierAssistant
from assistants.oss import OSSAssistant
from evaluation.prompts import PROMPTS
from evaluation.judge import judge_response

RESULTS_FILE = "evaluation/results.json"


def run_eval():
    frontier = FrontierAssistant()
    oss = OSSAssistant()
    results = []

    print(f"\nRunning {len(PROMPTS)} prompts on both assistants...\n")

    for p in tqdm(PROMPTS):
        row = {
            "id": p["id"],
            "category": p["category"],
            "prompt": p["prompt"],
            "expected": p["expected"],
        }

        # --- Frontier ---
        try:
            frontier.reset()
            f_response = frontier.chat(p["prompt"])
        except Exception as e:
            f_response = f"ERROR: {e}"

        f_score = judge_response(p["category"], p["prompt"], p["expected"], f_response)
        row["frontier"] = {"response": f_response, **f_score}

        time.sleep(0.5)  # avoid rate limits

        # --- OSS ---
        try:
            oss.reset()
            o_response = oss.chat(p["prompt"])
        except Exception as e:
            o_response = f"ERROR: {e}"

        o_score = judge_response(p["category"], p["prompt"], p["expected"], o_response)
        row["oss"] = {"response": o_response, **o_score}

        results.append(row)
        time.sleep(0.5)

    # Save results
    os.makedirs("evaluation", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n── Summary ──────────────────────────────")
    for category in ["factual", "adversarial", "bias"]:
        cat_rows = [r for r in results if r["category"] == category]
        f_avg = sum(r["frontier"]["score"] for r in cat_rows) / len(cat_rows)
        o_avg = sum(r["oss"]["score"] for r in cat_rows) / len(cat_rows)
        print(f"{category:12} | Frontier: {f_avg:.1f}/5  |  OSS: {o_avg:.1f}/5")

    print(f"\nFull results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_eval()