#!/usr/bin/env python3
"""
Headless benchmark for the free/light pipeline.
- Uses AdvancedTradingAnalyzer in light mode (no paid/rate-limited data)
- Measures end-to-end runtime and basic result stats for N symbols
Usage:
  python scripts/benchmark_free_pipeline.py [N]
Defaults to N=1000 if not provided.
"""
import sys
import os
import time
import statistics as stats

# Ensure project root is on sys.path when running from scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from advanced_analyzer import AdvancedTradingAnalyzer


def main():
    # Parse N
    try:
        N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    except Exception:
        N = 1000

    print(f"Running free/light benchmark for N={N} symbols...")

    analyzer = AdvancedTradingAnalyzer(enable_training=False, data_mode="light")
    symbols = analyzer.stock_universe[:N]

    t0 = time.time()
    results = analyzer.run_advanced_analysis(max_stocks=len(symbols), symbols=symbols)
    t1 = time.time()

    total_time = t1 - t0
    count = len(results)
    avg_time_per_symbol = total_time / max(1, len(symbols))

    print("---- Benchmark Results ----")
    print(f"Symbols requested: {len(symbols)}")
    print(f"Results obtained: {count}")
    print(f"Total time (s): {total_time:.2f}")
    print(f"Avg time per symbol (ms): {avg_time_per_symbol * 1000:.1f}")

    if count > 0:
        preds = [r.get('prediction', 0) for r in results]
        confs = [r.get('confidence', 0) for r in results]
        print(f"Prediction mean (pct pts): {stats.mean(preds):.2f}")
        print(f"Confidence mean: {stats.mean(confs):.2f}")

    # Basic success criterion
    if count == 0:
        print("ERROR: No results produced. Investigate data sources/fallbacks.")
        sys.exit(2)

    print("OK: Free/light pipeline produced results without external limits.")


if __name__ == "__main__":
    main()
