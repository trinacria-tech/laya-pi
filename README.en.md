# laya-pi

[Italiano](README.md) · **English**

Run [Laya](https://github.com/NandhaKishorM/laya) — single-pass, typed
"System-1" decisions — entirely on a **Raspberry Pi 4**, CPU only, no cloud.

The stack ships as one Docker image with four roles:

| Service      | Port   | What it does                                                                 |
|--------------|--------|------------------------------------------------------------------------------|
| `laya-onnx`  | `8001` | `/v1/systemone` API backed by ONNX Runtime (fp32), ~1.3–1.65× faster than torch |
| `snake`      | `8080` | Autonomous Snake in three modes (easy Laya, hard Laya, RL net), streamed to browsers over SSE; FAQ at `/faq` |
| `showcase`   | `8091` | Web playground with five example decisions (email routing, moderation, router, triage, multilingual) |
| `laya`       | `8000` | Stock `laya-serve` on torch CPU (optional, `torch` profile)                  |

All services speak the same `POST /v1/systemone` + `GET /health` wire protocol,
so any Laya client can point at either backend unchanged.

## Quick start

```sh
docker compose up -d --build
```

The first boot downloads model weights from Hugging Face into the `laya-models`
volume, so give it a few minutes. Then open, from any machine on the LAN:

- `http://<pi-address>:8080/` — watch the snake play
- `http://<pi-address>:8091/` — try the decision API

To also run the torch-based `laya-serve`:

```sh
docker compose --profile torch up -d laya
```

## Snake models

| Mode | Model | Input |
|------|-------|-------|
| EASY | [`soyelmismo/laya-multilingual-onnx`](https://huggingface.co/soyelmismo/laya-multilingual-onnx) | moves pre-labelled by the planner (Hamiltonian cycle), with shield |
| HARD | [`trinacratech/snake-rl-room-onnx`](https://huggingface.co/trinacratech/snake-rl-room-onnx) | head-relative board facts, no shield; Laya fine-tuned with the RL net as teacher |
| HARD (dropdown) | [`trinacratech/snake-trap-onnx`](https://huggingface.co/trinacratech/snake-trap-onnx) | same input; earlier fine-tune, rule-based teacher |
| RL | [`trinacratech/snake-rl-ppo`](https://huggingface.co/trinacratech/snake-rl-ppo) | the board as 6 grids; 5 MB PPO net, runs inside the snake container |

Models download on first use into the `laya-models` volume. Scores and explanations are on the FAQ
page (`http://<pi-address>:8080/faq`, Italian).

## Calling the API

```sh
curl -s http://<pi-address>:8001/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
        "state": "I was charged twice this month. Cancel my account.",
        "questions": {
          "department": {
            "type": "choice",
            "instructions": "Which department should handle this?",
            "criteria": {"billing": "payments, refunds", "technical": "bugs, outages", "other": "everything else"}
          },
          "urgency":    {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]},
          "churn_risk": {"type": "noul",  "instructions": "Does the user threaten to cancel?"}
        }
      }'
```

The three primitives are `choice` (pick one option), `score` (position on an
ordered scale) and `noul` (yes/no probability). Everything is answered in a
single forward pass — no generated tokens.

## Repository layout

```
Dockerfile            one image for every service (CPU torch + laya[serve] + onnxruntime)
docker-compose.yml    service wiring, ports, thread budgets, healthchecks
laya_onnx_serve.py    laya-serve-compatible server on ONNX Runtime
laya_capped.py        laya-serve with language auto-routing but one resident checkpoint
laya_single.py        laya-serve pinned to a single checkpoint
snakeweb/             Snake game, Laya policy + safety shield, SSE spectator server
showcase/             API playground: example catalogue + same-origin proxy
bench.py              latency / memory benchmark across seven languages
smoke.py              quick "does the checkpoint load" check
onnx_probe.py         ONNX export sanity check and latency probe
scripts/              exhibition-wifi.sh: isolate the hotspot (exhibition mode)
docs/                 extra guides
```

## Raspberry Pi notes

Things learned the hard way on a Pi 4 (Cortex-A72, 4 cores, no swap):

- **Use the CPU torch wheel.** The default PyPI `torch` for aarch64 is the CUDA
  build, which targets ARMv8.2-A and crashes with `SIGILL` on the A72. The
  Dockerfile installs from `https://download.pytorch.org/whl/cpu`.
- **fp32 ONNX, not int8.** int8 is smaller and faster but visibly degrades the
  `noul` heads; fp32 matches torch numerically.
- **Tune ONNX Runtime threads.** `ONNXAgent` builds its session with default
  options; `laya_onnx_serve.py` rebuilds it with capped intra-op threads,
  sequential execution and full graph optimisation.
- **Watch RAM.** Holding two torch checkpoints alongside the ONNX service leaves
  under 400 MB free, so `laya_capped.py` keeps only one checkpoint resident.
- **Idle means idle.** The snake only plays while at least one spectator is
  connected, so the Pi does no inference when nobody is watching.
- **Exhibition mode.** `sudo scripts/exhibition-wifi.sh on|off|status` isolates
  the Wi-Fi hotspot from the LAN to simulate the exhibition; see
  [docs/exhibition-wifi.en.md](docs/exhibition-wifi.en.md).

Typical latency on the Pi: ~2.2 s per decision with ONNX (4 threads), ~4.2 s
with torch.

## Configuration

| Variable        | Default                              | Used by               |
|-----------------|--------------------------------------|-----------------------|
| `LAYA_URL`      | `http://localhost:8000` / `:8001`    | snake, showcase       |
| `LAYA_MODEL`    | `multilingual`                       | laya, snake           |
| `LAYA_HOST`     | `0.0.0.0`                            | servers               |
| `LAYA_PORT`     | `8000` / `8001`                      | servers               |
| `LAYA_THREADS`  | —                                    | laya (torch)          |
| `LAYA_API_KEY`  | —                                    | snake client          |
| `ONNX_REPO`     | `soyelmismo/laya-multilingual-onnx`  | laya-onnx             |
| `ONNX_FILE`     | `model-fp32.onnx`                    | laya-onnx             |
| `ORT_THREADS`   | cores − 1                            | laya-onnx             |
| `SNAKE_MODE_MODELS` | `{"easy": "multilingual", "hard": "snake-rl-room"}` | snake (model per mode) |
| `SNAKE_RL_ONNX` | `/models/snake-rl/snake-rl.onnx` | snake (RL mode): local file, if present |
| `SNAKE_RL_REPO` | `trinacratech/snake-rl-ppo`      | snake (RL mode): otherwise downloaded from here |
| `SNAKE_RL_THREADS` | `3`                               | snake (RL mode)       |
| `SNAKE_RL_TICK` | `0.03` s per move                    | snake (RL mode)       |

## Credits

- [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations — the model and `laya` package.
- [laya-mlx](https://github.com/mizorewww/laya-mlx) — Snake rules and decision policy, ported here from MLX to CPU torch/ONNX (see `snakeweb/NOTICE`).
- [soyelmismo/laya-multilingual-onnx](https://huggingface.co/soyelmismo/laya-multilingual-onnx) — community ONNX export.
- Fonts: [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) and [Play](https://fonts.google.com/specimen/Play), both under the SIL Open Font License 1.1 (see `LICENSES/`).

## License

[Apache License 2.0](LICENSE). Bundled fonts are licensed under the SIL OFL 1.1.
