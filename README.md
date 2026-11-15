# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/RenzzHGE/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/RenzzHGE/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One OpenAI- and Anthropic-compatible server that routes each request to the best free model across ten simulated providers, fails over when a provider refuses, and reports the catalog it read. Routing ranks every free model that meets a request's requirements by its quality rating; failover reads the refusal's own message to tell a spent budget or an outage (that provider is dead for the trial) from a one-off refusal (it stays live). All three provider wires are spoken — OpenAI, Anthropic and Gemini — in JSON and in server-sent events, with the request's system text, user text, images, tools, stops, max_tokens and JSON mode passed through untouched.

## Run it

```sh
MAKEFUN_PROMPT_FILE=<brief> MAKEFUN_ENV_URL=<env> node main.ts
```

Node 22, standard library only — nothing to install. The server reads the brief (provider directory, quality table, workload manifest), probes each provider's catalog route, registers with the env, serves `GET /v1/models`, `POST /v1/chat/completions` and `POST /v1/messages`, and exits 0 when the env's work is done. `node main.ts --list` prints the catalog it would register, without starting a server.

---

<sub>Entered on [make.fun](https://make.fun/?ref=9nf38kge&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
