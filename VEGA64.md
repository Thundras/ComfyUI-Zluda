# ComfyUI-Zluda auf der Radeon RX Vega 64

Fork von [patientx/ComfyUI-Zluda](https://github.com/patientx/ComfyUI-Zluda) (Branch `pre24patched`), angepasst für die Vega 64 (gfx900) mit ZLUDA, ROCm 5.7 und Torch 2.2.

## Repo-Aufbau

| Remote | Ziel |
|---|---|
| `origin` | https://github.com/Thundras/ComfyUI-Zluda (eigener Fork) |
| `zluda` | patientx/ComfyUI-Zluda |
| `upstream` | Comfy-Org/ComfyUI |

| Branch | Zweck |
|---|---|
| `vega64` | Arbeitsbranch, folgt `origin/vega64` |
| `backup/original-changes` | nur lokal: ursprünglicher, unaufgeräumter Stand inkl. Patch-Skripten und Diagnose-Code |

Commits laufen über die Noreply-Adresse `10862226+Thundras@users.noreply.github.com` (in der lokalen Repo-Config gesetzt). Mit der Gmail-Adresse lehnt GitHub den Push wegen des E-Mail-Schutzes ab.

## Änderungen gegenüber `pre24patched`

| Änderung | Datei(en) | Grund |
|---|---|---|
| `comfy_kitchen` optional | `attention.py`, `llama.py`, `qwen35.py`, `model_prefetch.py`, `quant_ops.py`, `joyimage/model.py`, `seedvr/vae.py`, `na_diffusion_decoder.py` | comfy-kitchen lässt sich mit Torch < 2.4 nicht importieren. Fängt `ImportError` und `AttributeError` ab. |
| RoPE-Fallback ohne kitchen | `comfy/text_encoders/llama.py` | elementweise 2×2-Rotation in fp32, Details unten |
| `small_input=x.shape[1] <= 512` | `comfy/text_encoders/llama.py` | Lange Prompts liefen sonst immer über die Basic-/PyTorch-Attention, die die volle Matrix aufbaut, statt über quad cross attention. |
| `checkpoint_pickle`-Import | `comfy/utils.py` | Der Import innerhalb der Funktion machte `comfy` zu einer lokalen Variable, das führte zu einem `UnboundLocalError`. |
| YuE2-Tokens speichern und fortsetzen | `comfy/text_encoders/yue2.py`, `custom_nodes/yue2_token_backup/` | Lange Token-Generierungen müssen nicht wiederholt werden. |
| Große Tensoren stückweise auf die CPU kopieren | `comfy/model_patcher.py` | Eine einzelne große GPU→CPU-Kopie (YuE2 lm_head, ~750 MB) stürzt unter ZLUDA ab. Tensoren über 256 MB werden in 16-MB-Blöcken kopiert. |
| `--disable-mmap` liest ohne mmap | `comfy/utils.py` | Liest Tensor für Tensor in den RAM, die Spitze bleibt bei der Modellgröße. |
| Launcher | `comfyui-user.bat` | ROCm 5.7, `--use-quad-cross-attention --reserve-vram 0.9 --disable-async-offload --disable-pinned-memory --disable-mmap` |

### Messwerte (Vega 64, ZLUDA)

RoPE pro Layer, YuE2-Formen (16 Q-Heads, 8 KV-Heads, head_dim 128, 28 Layer):

| Sequenzlänge | alt (stack + matmul) | neu (fp32, elementweise) |
|---|---|---|
| 1 | 0,43 ms | 0,42 ms |
| 2048 | 236 ms | 1,5 ms |
| 8192 | 943 ms | 7,2 ms |

Die Ergebnisse sind bitidentisch. Batched `matmul` auf 2×2-Matrizen ist unter ZLUDA extrem langsam.

Laden von `yue2_3b_bf16.safetensors`:

| Variante | Zeit | RAM-Spitze |
|---|---|---|
| safetensors mit mmap und Kopie | 8,7 s | 14,85 GiB |
| alter Lese-Code (Dateiname fest eingetragen) | 3,8 s | 8,28 GiB |
| jetzt: `--disable-mmap` | 2,7 s | 7,58 GiB |

DynamicVRAM/aimdo ist hier ohnehin aus, weil es Torch ≥ 2.8 voraussetzt.

## YuE2-Token-Backup

Node: **YuE2 Generate or Load Backup** (`audio/YuE2`)

- `token_backup = Kein Backup - neu generieren`: generiert neu und speichert nach `output/yue2_tokens/<Titel>_mode-…_seed-…_<Zeit>.json`.
- Wählt man eine JSON-Datei aus, wird die Token-Generierung übersprungen und die Generierungseinstellungen kommen aus der Datei.
- Gespeichert wird nur über diesen Node, nicht über die Standard-YuE2-Nodes.
- `custom_nodes/` steht in `.gitignore`. Der Node wurde mit `git add -f` aufgenommen, und Änderungen daran müssen ebenfalls mit `git add -f` hinzugefügt werden.

## Bekannte Grenzen

- Ohne comfy-kitchen stürzen Modelle ab, die zur Laufzeit `comfy.quant_ops.ck.*` oder `comfy_kitchen.*` aufrufen (Flux, LTX, Lumina, Cosmos, Ernie, JoyImage …). Abgedeckt sind nur Imports und der Llama-/YuE2-Pfad.
- `comfy_extras/nodes_sparse_attention.py` lässt sich ohne comfy-kitchen nicht importieren. Die Startwarnung ist harmlos.
- fp8/fp4 ist nicht verfügbar.

## Alltag

### `comfy/zluda.py`

`comfyui-user.bat` kopiert bei jedem Start `comfy/customzluda/zluda-default.py` nach `comfy/zluda.py`. Dadurch ist die Datei immer als geändert markiert. Diese Änderung nie committen.

Optional kann man Git anweisen, die Datei lokal zu ignorieren:

```
git update-index --skip-worktree comfy/zluda.py
git ls-files -v | grep ^S          # markierte Dateien anzeigen
```

Ändert patientx die Datei, bricht ein Merge ab. Dann:

```
git update-index --no-skip-worktree comfy/zluda.py
git checkout -- comfy/zluda.py
git merge zluda/pre24patched
git update-index --skip-worktree comfy/zluda.py
```

### Updates von patientx holen

Der `git pull` im Launcher holt nur von `origin/vega64`. Neue Versionen von patientx kommen so dazu:

```
git checkout -- comfy/zluda.py
git fetch zluda
git merge zluda/pre24patched
git push
```

### Lokale Dateien, die Git ignoriert

Über `.git/info/exclude` (gilt nur lokal): `*.before-*`, `patch-*.py`, `test-*.py`, `zluda-diagnose.py`, `rocblas-library-backup/`, `rocblas.dll.backup`, `.claude-flow/`, `zluda/`.

Das Backup ist zu löschen, sobald es nicht mehr gebraucht wird: `git branch -D backup/original-changes`.
