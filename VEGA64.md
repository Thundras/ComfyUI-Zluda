# ComfyUI-Zluda auf der Radeon RX Vega 64

Fork von [patientx/ComfyUI-Zluda](https://github.com/patientx/ComfyUI-Zluda) (Branch `pre24patched`), angepasst für die Vega 64 (gfx900) mit ZLUDA, ROCm 5.7 und Torch 2.2 (`comfyui-user.bat`) oder Torch 2.7 mit comfy-kitchen (`comfyui-t24.bat`).

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
| Flash-Decode von comfy-kitchen unter ZLUDA aus | `comfy/customzluda/zluda-default.py` | Nur für Torch 2.7 relevant, siehe unten. |
| Launcher Torch 2.7 | `comfyui-t24.bat` | wie `comfyui-user.bat`, aber `venv-t24`, zusätzlich `--fp16-unet` und ohne `git pull` |

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

### Wo die Zeit bei YuE2 hingeht

Ein Token braucht ~180 ms. Davon entfallen ~75 % auf Matrix-Vektor-Multiplikationen (MLP, Projektionen, lm_head), ~13 % auf RMSNorm und <1 % auf das Sampling. rocBLAS erreicht über ZLUDA auf gfx900 nur ~100–200 GB/s von 484 GB/s. Das ist die eigentliche Grenze, keine einzelne langsame Stelle.

- Rechnen in fp16 statt fp32 ist für den YuE2-Text-Encoder numerisch kaputt (Überlauf).
- fp16-Gewichte mit fp32-Rechnung wären bitidentisch und ~13 % schneller. Nicht umgesetzt: `--fp16-text-enc` greift nicht, weil `pick_weight_dtype` bf16 aus dem Checkpoint behält.
- ZLUDA meldet sm 8.8, deshalb rechnet ComfyUI YuE2 in bf16, obwohl die Vega kein natives bf16 hat.
- Die „Pausen“ beim Music Sampling alle 16 Token sind das Zurücklesen der Token von der GPU. Die Gesamtzeit ändert sich dadurch nicht.

## Torch 2.7 + comfy-kitchen (`venv-t24`)

Torch 2.4 bis 2.7 laufen mit ZLUDA 3.9.5 (rocm5) und HIP 5.7 auf der Vega. comfy-kitchen 0.2.37 lädt erst ab Torch 2.7 (`custom_op` mit `list[int]`).

| | Torch 2.2 (`comfyui-user.bat`) | Torch 2.7 (`comfyui-t24.bat`) |
|---|---|---|
| YuE2, 750 Token | 2:16 min, 181,7 s gesamt | 2:15 min, 177,1 s gesamt, identische Token |
| YuE2-Diffusion in fp16 | – | klingt gleich wie bf16 (Korrelation 0,9989, Spektrum identisch) |
| SD 1.5, 512², 20 Schritte | 1,47 it/s | 1,44 it/s, optisch identisches Bild |
| Lumina 2.0, 768², 36 Schritte | läuft nicht (braucht comfy-kitchen) | 5,9 s/Schritt, ~3:40 min pro Bild |

Torch 2.7 ist nicht schneller, kann aber Modelle ausführen, die comfy-kitchen brauchen.

Unter ZLUDA zu beachten:
- Nur das `eager`-Backend von comfy-kitchen funktioniert. Die CUDA-Erweiterung ist für CUDA 12/13 (PTX 9.0, sm_75–sm_120) gebaut und schlägt mit „CUDA driver version is insufficient“ fehl. ComfyUI schaltet sie bei Torch cu118 selbst ab, aber `flash_attention_decode` umgeht das. Deshalb setzt `zluda-default.py` `flash_attention_decode_is_available` unter ZLUDA auf `False`.
- int8-Checkpoints bringen nichts: `cublasLt` mit int8 kann ZLUDA nicht, und Entpacken nach fp32 ist 1,5–2× langsamer als bf16 (lm_head stürzt dabei ab).
- Lumina 2.0 erzeugt in bf16 ab Schritt 10 NaN (schwarzes Bild), in fp32 im letzten Schritt. In fp16 ist es sauber, deshalb `--fp16-unet`.
- Der erste Lauf jeder neuen Torch-Version und jedes neuen Modells übersetzt GPU-Kernels (10–30 min). Der Cache liegt in `%LOCALAPPDATA%\ZLUDA\ComputeCache\zluda.db`.
- Torch 2.7 cu118 ist die letzte Version mit cu118-Wheels.

`venv-t24` neu aufsetzen (pip-Cache auf D:, C: ist knapp):

```
set PIP_CACHE_DIR=D:\ComfyUI-Zluda\.tmp-pip\cache
set TMP=D:\ComfyUI-Zluda\.tmp-pip\tmp
set TEMP=D:\ComfyUI-Zluda\.tmp-pip\tmp
python -m venv venv-t24
venv-t24\Scripts\python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
venv-t24\Scripts\python -m pip install -r requirements.txt
venv-t24\Scripts\python -m pip install --force-reinstall numpy==1.26.4 transformers==4.51.3 safetensors==0.5.3 onnxruntime
copy zluda\cublas.dll venv-t24\Lib\site-packages\torch\lib\cublas64_11.dll /y
copy zluda\cusparse.dll venv-t24\Lib\site-packages\torch\lib\cusparse64_11.dll /y
copy venv-t24\Lib\site-packages\torch\lib\nvrtc64_112_0.dll venv-t24\Lib\site-packages\torch\lib\nvrtc_cuda.dll /y
copy zluda\nvrtc.dll venv-t24\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y
```

Lumina 2.0 (Comfy-Org/Lumina_Image_2.0_Repackaged, `split_files`): `diffusion_models/lumina_2_model_bf16.safetensors`, `text_encoders/gemma_2_2b_fp16.safetensors`, `vae/ae.safetensors`. Empfohlen sind 36 Schritte mit Shift 6, `res_multistep`/`simple`, CFG 4.

## YuE2-Token-Backup

Node: **YuE2 Generate or Load Backup** (`audio/YuE2`)

- `token_backup = Kein Backup - neu generieren`: generiert neu und speichert nach `output/yue2_tokens/<Titel>_mode-…_seed-…_<Zeit>.json`.
- Wählt man eine JSON-Datei aus, wird die Token-Generierung übersprungen und die Generierungseinstellungen kommen aus der Datei.
- Gespeichert wird nur über diesen Node, nicht über die Standard-YuE2-Nodes.
- `custom_nodes/` steht in `.gitignore`. Der Node wurde mit `git add -f` aufgenommen, und Änderungen daran müssen ebenfalls mit `git add -f` hinzugefügt werden.

## Bekannte Grenzen

- Unter Torch 2.2 stürzen Modelle ab, die zur Laufzeit `comfy.quant_ops.ck.*` oder `comfy_kitchen.*` aufrufen (Flux, LTX, Lumina, Cosmos, Ernie, JoyImage …). Abgedeckt sind nur Imports und der Llama-/YuE2-Pfad. Für diese Modelle `comfyui-t24.bat` verwenden.
- Unter Torch 2.2 lässt sich `comfy_extras/nodes_sparse_attention.py` nicht importieren. Die Startwarnung ist harmlos.
- fp8/fp4/int8 sind auf der Vega nicht nutzbar.

## Alltag

### `comfy/zluda.py`

Alle Start-Skripte kopieren bei jedem Start `comfy/customzluda/zluda-default.py` nach `comfy/zluda.py`. Die Datei im Repo ist ein veralteter Rest: Seit dem 12.06.2025 pflegt patientx nur noch `zluda-default.py`.

Lokal ist `skip-worktree` gesetzt, damit die Kopie nie in `git status` auftaucht und nicht mitcommittet wird:

```
git update-index --skip-worktree comfy/zluda.py      # einschalten (z. B. nach einem frischen Clone)
git ls-files -v | grep ^S                            # markierte Dateien anzeigen
git update-index --no-skip-worktree comfy/zluda.py   # ausschalten
```

Sollte ein Merge doch einmal wegen dieser Datei abbrechen: ausschalten, `git checkout -- comfy/zluda.py`, mergen, wieder einschalten.

### Updates von patientx holen

Der `git pull` im Launcher holt nur von `origin/vega64`. Neue Versionen von patientx kommen so dazu:

```
git fetch zluda
git merge zluda/pre24patched
git push
```

### Lokale Dateien, die Git ignoriert

Über `.git/info/exclude` (gilt nur lokal): `*.before-*`, `patch-*.py`, `test-*.py`, `zluda-diagnose.py`, `rocblas-library-backup/`, `rocblas.dll.backup`, `.claude-flow/`, `zluda/`, `venv-t24/`, `.tmp-pip/`.

`rocblas-library-backup/` ist die originale rocBLAS-Bibliothek aus HIP 5.7 ohne gfx900-Kernel. Die installierte Bibliothek unter `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library` enthält zusätzlich gfx900- und gfx803-Kernel aus Community-Builds, ohne die die Vega nicht läuft.

Das Backup ist zu löschen, sobald es nicht mehr gebraucht wird: `git branch -D backup/original-changes`.
