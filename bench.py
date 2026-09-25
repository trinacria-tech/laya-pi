"""Benchmark laya-multilingual on this machine (CPU).

Usage:
    python bench.py [torch|onnx] [--batch]
"""

import argparse
import os
import resource
import statistics
import time

import laya

REPO = "convaiinnovations/laya"
SUBFOLDER = "multilingual"

WARMUP = 3
RUNS = 10

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds, charges",
            "technical": "bugs, outages, system errors, login failures",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "soon", "critical"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel?",
    },
}

STATES = [
    ("en", "I was charged twice this month and support never replied. Cancel my account."),
    ("hi", "मुझसे इस महीने दो बार शुल्क लिया गया और किसी ने जवाब नहीं दिया।"),
    ("de", "Mein Konto wurde doppelt belastet und der Support meldet sich nicht."),
    ("ar", "تم خصم المبلغ مرتين هذا الشهر ولم يرد أحد على رسائلي."),
    ("zh", "这个月我被扣了两次费用，客服一直没有回复我。"),
    ("ru", "С меня дважды списали деньги в этом месяце, поддержка молчит."),
    ("es", "Me cobraron dos veces este mes y nadie del soporte me respondió."),
]


def rss_mb():
    # ru_maxrss is KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def timed(fn, runs):
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true", help="also benchmark predict_batch")
    args = ap.parse_args()

    # laya 0.3.20's load() takes no `backend` argument, so torch CPU is the only path.
    print(f"backend=torch-cpu  threads={os.cpu_count()}")

    t0 = time.perf_counter()
    agent = laya.load(REPO, subfolder=SUBFOLDER)
    load_s = time.perf_counter() - t0
    print(f"load: {load_s:.1f}s   rss after load: {rss_mb():.0f} MB\n")

    for _lang, text in STATES[:1]:
        timed(lambda: agent.predict({"body": text}, QUESTIONS), WARMUP)

    print(f"{'lang':<6} {'median':>9} {'mean':>9} {'min':>9} {'max':>9}   decision")
    print("-" * 72)
    all_samples = []
    for lang, text in STATES:
        state = {"body": text}
        samples = timed(lambda: agent.predict(state, QUESTIONS), RUNS)
        all_samples += samples
        result = agent.predict(state, QUESTIONS)
        ans = result["answers"]
        decision = (
            f"{ans['department']['choice']}"
            f" / urg={ans['urgency']['score']:.2f}"
            f" / churn={ans['churn_risk']['noul']:.2f}"
        )
        print(
            f"{lang:<6} {statistics.median(samples):>8.1f}ms {statistics.mean(samples):>8.1f}ms "
            f"{min(samples):>8.1f}ms {max(samples):>8.1f}ms   {decision}"
        )

    print("-" * 72)
    print(
        f"overall median {statistics.median(all_samples):.1f}ms  "
        f"p90 {statistics.quantiles(all_samples, n=10)[8]:.1f}ms  "
        f"n={len(all_samples)}"
    )
    print(f"peak rss: {rss_mb():.0f} MB")

    if args.batch:
        states = [{"body": t} for _, t in STATES]
        timed(lambda: agent.predict_batch(states, QUESTIONS), 1)
        samples = timed(lambda: agent.predict_batch(states, QUESTIONS), 5)
        per_item = statistics.median(samples) / len(states)
        print(
            f"\nbatch({len(states)}): median {statistics.median(samples):.1f}ms "
            f"-> {per_item:.1f}ms/item"
        )
        print(f"peak rss: {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
