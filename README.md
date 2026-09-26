# laya-pi

**Italiano** · [English](README.en.md)

Esegui [Laya](https://github.com/NandhaKishorM/laya) — decisioni tipizzate
"System-1" in un solo passaggio — interamente su un **Raspberry Pi 4**, solo CPU,
senza cloud.

Lo stack è un'unica immagine Docker con quattro ruoli:

| Servizio     | Porta  | Cosa fa                                                                        |
|--------------|--------|--------------------------------------------------------------------------------|
| `laya-onnx`  | `8001` | API `/v1/systemone` su ONNX Runtime (fp32), ~1,3–1,65× più veloce di torch     |
| `snake`      | `8080` | Snake autonomo: ogni mossa è una decisione Laya dal vivo, trasmessa ai browser via SSE |
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
Dockerfile            un'immagine per tutti i servizi (torch CPU + laya[serve] + onnxruntime)
docker-compose.yml    collegamento dei servizi, porte, thread, healthcheck
laya_onnx_serve.py    server compatibile con laya-serve su ONNX Runtime
laya_capped.py        laya-serve con instradamento per lingua ma un solo checkpoint in memoria
laya_single.py        laya-serve vincolato a un unico checkpoint
snakeweb/             gioco Snake, policy Laya + scudo di sicurezza, server SSE per spettatori
showcase/             vetrina dell'API: catalogo di esempi + proxy same-origin
bench.py              benchmark di latenza / memoria su sette lingue
smoke.py              verifica rapida che il checkpoint si carichi
onnx_probe.py         controllo dell'export ONNX e misura di latenza
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

## Crediti

- [Laya](https://github.com/NandhaKishorM/laya) di Convai Innovations — il modello e il pacchetto `laya`.
- [laya-mlx](https://github.com/mizorewww/laya-mlx) — regole di Snake e policy decisionale, portate qui da MLX a torch/ONNX su CPU (vedi `snakeweb/NOTICE`).
- [soyelmismo/laya-multilingual-onnx](https://huggingface.co/soyelmismo/laya-multilingual-onnx) — export ONNX della community.
- Font: [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) e [Play](https://fonts.google.com/specimen/Play), entrambi con licenza SIL Open Font License 1.1 (vedi `LICENSES/`).

## Licenza

[Apache License 2.0](LICENSE). I font inclusi sono distribuiti con licenza SIL OFL 1.1.
