# laya-pi

**Italiano** · [English](README.en.md)

Esegui [Laya](https://github.com/NandhaKishorM/laya) — decisioni tipizzate
"System-1" in un solo passaggio — interamente su un **Raspberry Pi 4**, solo CPU,
senza cloud.

Lo stack è un'unica immagine Docker con quattro ruoli:

| Servizio     | Porta  | Cosa fa                                                                        |
|--------------|--------|--------------------------------------------------------------------------------|
| `laya-onnx`  | `8001` | API `/v1/systemone` su ONNX Runtime (fp32), ~1,3–1,65× più veloce di torch     |
| `snake`      | `8080` | Snake autonomo in tre modalità (Laya facile, Laya difficile, rete RL), trasmesso ai browser via SSE; FAQ su `/faq` |
| `showcase`   | `8091` | Vetrina web con cinque decisioni di esempio (smistamento email, moderazione, router, triage, multilingue) |
| `laya`       | `8000` | `laya-serve` standard su torch CPU (opzionale, profilo `torch`)                |

Tutti i servizi parlano lo stesso protocollo `POST /v1/systemone` + `GET /health`,
quindi qualsiasi client Laya può puntare a entrambi i backend senza modifiche.

## Avvio rapido

```sh
docker compose up -d --build
```

Al primo avvio i pesi del modello vengono scaricati da Hugging Face nel volume
`laya-models`: servono alcuni minuti. Poi apri, da qualsiasi macchina della LAN:

- `http://<indirizzo-pi>:8080/` — guarda il serpente giocare
- `http://<indirizzo-pi>:8091/` — prova l'API decisionale

Per avviare anche `laya-serve` basato su torch:

```sh
docker compose --profile torch up -d laya
```

## Modelli dello Snake

| Modalità  | Modello | Cosa riceve |
|-----------|---------|-------------|
| FACILE    | [`soyelmismo/laya-multilingual-onnx`](https://huggingface.co/soyelmismo/laya-multilingual-onnx) | mosse etichettate dal pianificatore (ciclo hamiltoniano), con scudo |
| DIFFICILE | [`trinacratech/snake-rl-room-onnx`](https://huggingface.co/trinacratech/snake-rl-room-onnx) | fatti sul tabellone rispetto alla testa, senza scudo; fine-tune di Laya con la rete RL come insegnante |
| DIFFICILE (menu) | [`trinacratech/snake-trap-onnx`](https://huggingface.co/trinacratech/snake-trap-onnx) | stesso input; fine-tune precedente, insegnante a regole |
| RL        | [`trinacratech/snake-rl-ppo`](https://huggingface.co/trinacratech/snake-rl-ppo) | il tabellone come 6 griglie; rete PPO da 5 MB, gira dentro il container dello snake |

I modelli vengono scaricati al primo utilizzo nel volume `laya-models`. Punteggi e spiegazioni
nella pagina FAQ (`http://<indirizzo-pi>:8080/faq`).

## Altri modelli decisionali

Oltre a Laya, il servizio `laya-onnx` offre due modelli decisionali con la stessa API
`/v1/systemone`, eseguiti in GGUF Q8_0 da `llama-server` di [llama.cpp](https://github.com/ggml-org/llama.cpp)
(compilato nell'immagine). Si scelgono dal menu modello dello snake e della vetrina; resta
sempre un solo modello in memoria.

| Nome | Modello | Licenza | Mossa snake | Email (3 domande) | RAM |
|------|---------|---------|-------------|-------------------|-----|
| `multilingual` | Laya multilingue, ONNX fp32 | Apache-2.0 | ~1 s | ~5 s | ~2,6 GB |
| `d1-omni` | [`LiquidAI/d1-omni-600M-GGUF`](https://huggingface.co/LiquidAI/d1-omni-600M-GGUF) | LFM Open License v1.0 | ~3,5 s | ~17 s | ~0,85 GB |
| `julia-1` | [`ggml-org/Julia-1-GGUF`](https://huggingface.co/ggml-org/Julia-1-GGUF) ([Supersonic Labs](https://huggingface.co/SupersonicLabs/Julia-1)) | Apache-2.0 | ~0,5 s | ~2 s | ~0,7 GB |

Misurato su Raspberry Pi 4, 4 thread; RAM = intero servizio `laya-onnx`. d1-omni è il più accurato sulle nostre prove (mosse
etichettate 12/12, vetrina sensata) ma ~3 volte più lento di Laya; le quantizzazioni più basse non
aiutano (sul Cortex-A72 il limite è il calcolo: Q4_0 3,8 s, Q5_K_M 4,8 s; ONNX q8/q4 6,3/4,8 s).
Julia-1 è il più veloce ma meno affidabile: 8/12 mosse e segnala come spam/phishing (0,8) un'email
di rimborso legittima, anche in BF16. Qui si usa solo il testo di d1-omni (niente immagini/audio).

## Chiamare l'API

```sh
curl -s http://<indirizzo-pi>:8001/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
        "state": "Mi avete addebitato due volte questo mese. Disdico l'\''abbonamento.",
        "questions": {
          "reparto": {
            "type": "choice",
            "instructions": "Quale reparto dovrebbe gestire questa richiesta?",
            "criteria": {"fatturazione": "pagamenti, rimborsi", "tecnico": "bug, guasti", "altro": "tutto il resto"}
          },
          "urgenza":        {"type": "score", "instructions": "Quanto è urgente?", "criteria": ["bassa", "media", "alta"]},
          "rischio_disdetta": {"type": "noul",  "instructions": "L'\''utente minaccia di disdire?"}
        }
      }'
```

Le tre primitive sono `choice` (scegli un'opzione), `score` (posizione su una
scala ordinata) e `noul` (probabilità sì/no). Tutto viene risolto in un unico
passaggio in avanti del modello — nessun token generato.

## Struttura del repository

```
Dockerfile            un'immagine per tutti i servizi (torch CPU + laya[serve] + onnxruntime + llama-server)
docker-compose.yml    collegamento dei servizi, porte, thread, healthcheck
laya_onnx_serve.py    server compatibile con laya-serve su ONNX Runtime
laya_capped.py        laya-serve con instradamento per lingua ma un solo checkpoint in memoria
laya_single.py        laya-serve vincolato a un unico checkpoint
snakeweb/             gioco Snake, policy Laya + scudo di sicurezza, server SSE per spettatori
showcase/             vetrina dell'API: catalogo di esempi + proxy same-origin
bench.py              benchmark di latenza / memoria su sette lingue
smoke.py              verifica rapida che il checkpoint si carichi
onnx_probe.py         controllo dell'export ONNX e misura di latenza
scripts/              exhibition-wifi.sh: isola l'hotspot (modalità fiera)
docs/                 guide aggiuntive
```

## Note sul Raspberry Pi

Lezioni imparate sul campo con un Pi 4 (Cortex-A72, 4 core, senza swap):

- **Usa la wheel torch per CPU.** Il `torch` predefinito di PyPI per aarch64 è la
  build CUDA, pensata per ARMv8.2-A, che sull'A72 termina con `SIGILL`. Il
  Dockerfile installa da `https://download.pytorch.org/whl/cpu`.
- **ONNX fp32, non int8.** int8 è più piccolo e veloce ma degrada visibilmente le
  teste `noul`; fp32 coincide numericamente con torch.
- **Regola i thread di ONNX Runtime.** `ONNXAgent` crea la sessione con le opzioni
  predefinite; `laya_onnx_serve.py` la ricrea con thread intra-op limitati,
  esecuzione sequenziale e ottimizzazione completa del grafo.
- **Occhio alla RAM.** Due checkpoint torch insieme al servizio ONNX lasciano
  meno di 400 MB liberi, quindi `laya_capped.py` ne tiene in memoria uno solo.
- **A riposo, davvero a riposo.** Il serpente gioca solo quando almeno uno
  spettatore è connesso: senza pubblico il Pi non esegue inferenza.
- **Modalità fiera.** `sudo scripts/exhibition-wifi.sh on|off|status` isola
  l'hotspot Wi-Fi dalla LAN per simulare la fiera; vedi
  [docs/exhibition-wifi.md](docs/exhibition-wifi.md).

Latenza tipica sul Pi: ~2,2 s per decisione con ONNX (4 thread), ~4,2 s con torch.

## Configurazione

| Variabile       | Predefinito                          | Usata da              |
|-----------------|--------------------------------------|-----------------------|
| `LAYA_URL`      | `http://localhost:8000` / `:8001`    | snake, showcase       |
| `LAYA_MODEL`    | `multilingual`                       | laya, snake           |
| `LAYA_HOST`     | `0.0.0.0`                            | server                |
| `LAYA_PORT`     | `8000` / `8001`                      | server                |
| `LAYA_THREADS`  | —                                    | laya (torch)          |
| `LAYA_API_KEY`  | —                                    | client dello snake    |
| `ONNX_REPO`     | `soyelmismo/laya-multilingual-onnx`  | laya-onnx             |
| `ONNX_FILE`     | `model-fp32.onnx`                    | laya-onnx             |
| `ORT_THREADS`   | core − 1                             | laya-onnx             |
| `SNAKE_MODE_MODELS` | `{"easy": "multilingual", "hard": "snake-rl-room"}` | snake (modello per modalità) |
| `SNAKE_RL_ONNX` | `/models/snake-rl/snake-rl.onnx` | snake (modalità RL): file locale, se presente |
| `SNAKE_RL_REPO` | `trinacratech/snake-rl-ppo`      | snake (modalità RL): altrimenti scaricato da qui |
| `SNAKE_RL_THREADS` | `3`                               | snake (modalità RL)   |
| `SNAKE_RL_TICK` | `0.03` s per mossa                   | snake (modalità RL)   |

## Crediti

- [Laya](https://github.com/NandhaKishorM/laya) di Convai Innovations — il modello e il pacchetto `laya`.
- [laya-mlx](https://github.com/mizorewww/laya-mlx) — regole di Snake e policy decisionale, portate qui da MLX a torch/ONNX su CPU (vedi `snakeweb/NOTICE`).
- [soyelmismo/laya-multilingual-onnx](https://huggingface.co/soyelmismo/laya-multilingual-onnx) — export ONNX della community.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — `llama-server`, per i modelli GGUF (d1-omni di Liquid AI, Julia-1 di Supersonic Labs).
- Font: [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) e [Play](https://fonts.google.com/specimen/Play), entrambi con licenza SIL Open Font License 1.1 (vedi `LICENSES/`).

## Licenza

[Apache License 2.0](LICENSE). I font inclusi sono distribuiti con licenza SIL OFL 1.1.
