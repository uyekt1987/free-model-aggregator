# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/uyekt1987/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/uyekt1987/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A free-model aggregator for make.fun's "The best free AI model aggregator" bounty: one Node server that serves OpenAI Chat Completions and Anthropic Messages over ten simulated free-tier providers, routing each request to the best free model that meets it.

It ranks free candidates by the quality table's rating, caps each provider's successful calls at a conservative share of its per-trial budget, fails over to the next candidate on a 402/429/5xx, speaks all three upstream wires (OpenAI, Anthropic, Gemini), and returns every provider answer whole — text, usage counts and stop reason — in the wire the client used. Practice grading: **99/100 on all three practice trials** (quality 99, served 100, wire 100, failover 100, catalog 100).

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the provider catalog and exits.

---

<sub>Entered on [make.fun](https://make.fun/?ref=vkd54d79&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
