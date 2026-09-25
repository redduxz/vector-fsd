#!/usr/bin/env python3
"""Rapid micro-commit generator for research notes — one commit per log line."""
import os, random, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "docs", "research-log.md")

PREFIXES = ["docs", "chore", "research", "dev", "note", "wip", "log"]
VERBS = [
    "log planning iteration", "record sim observation", "note tuning delta",
    "jot safety review", "log perception note", "record checkpoint",
    "update research log", "mark test run", "log control tweak",
    "note latency sample", "append session line", "log loop iteration",
]

env = dict(os.environ)
env.update({
    "GIT_AUTHOR_NAME": "redduxz",
    "GIT_AUTHOR_EMAIL": "224149905+redduxz@users.noreply.github.com",
    "GIT_COMMITTER_NAME": "redduxz",
    "GIT_COMMITTER_EMAIL": "224149905+redduxz@users.noreply.github.com",
})

def commit(msg):
    subprocess.run(["git", "add", LOG], cwd=ROOT, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "commit", "-qm", msg], cwd=ROOT, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
if not os.path.exists(LOG):
    open(LOG, "w").write("# research log\n\n")

start = time.time()
for i in range(1, n + 1):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"iteration {i}: {random.choice(VERBS)}\n")
    commit(f"{random.choice(PREFIXES)}: {random.choice(VERBS)} ({i})")
    if i % 500 == 0:
        print(f"{i} commits in {time.time()-start:.0f}s", flush=True)
print(f"done: {n} commits in {time.time()-start:.0f}s")
