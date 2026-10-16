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
| Launcher | `comfyui-user.bat` | ROCm 5.7, `--use-quad-cross-attention --reserve-vram 0.9 --disable-async-offload --disable-pinned-memory --disable-mmap --disable-smart-memory` |
| Flash-Decode von comfy-kitchen unter ZLUDA aus | `comfy/customzluda/zluda-default.py` | Nur für Torch 2.7 relevant, siehe unten. |
| Startmenü | `start.bat` | Auswahl zwischen `comfyui-t24.bat` (Torch 2.7) und `comfyui-user.bat` (Torch 2.2) |
| Launcher Torch 2.7 | `comfyui-t24.bat` | wie `comfyui-user.bat`, aber `venv-t24`, zusätzlich `--fp16-unet` und ohne `git pull` |
| Audio-VAE kachelt lange Songs direkt | `comfy/ldm/audio/autoencoder.py` | Siehe „VAE-Absturz bei langen Songs“. |
| MP3 mit Tags speichern | `custom_nodes/mp3_tags/` | Titel, Interpret, Album, Jahr, Genre, Lyrics und Cover. |

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
- Lange Songs: Der Text-Encoder (4,1 GB) und der KV-Cache konkurrieren um den VRAM. Mit ~3,7 GB des Text-Encoders im VRAM laufen ~3,3 Token/s, komplett ausgelagert nur ~1,25 Token/s. Nach einer langen Sitzung mit vielen Modellen wurde der Text-Encoder einmal komplett ausgelagert. Deshalb starten beide Launcher mit `--disable-smart-memory`: Modelle wandern nach jedem Job in den RAM, und jeder Job beginnt mit freiem VRAM. Das Neuladen dauert etwa eine Sekunde.
- Lange Songs über den Backup-Node: Er lädt Token-Generierung und akustische Phase getrennt. Für die Generierung liegt der Text-Encoder komplett im VRAM (~5 Token/s, bei 600 s Obergrenze gemessen). Vor der akustischen Phase lädt ComfyUI mit der großzügigen Schätzung neu und lagert wieder aus. Eine Variante, die beide Phasen mit knapperer Schätzung in einem Ladevorgang erledigte, lief nach 8.127 Token in der akustischen Phase in einen Out-of-Memory-Fehler: ~1,5 GiB lagen außerhalb von PyTorch, 864 MiB waren fragmentiert.
- `max_duration` (25 Token pro Sekunde Audio) großzügig lassen, z. B. 600 s. Bei den Standard-YuE2-Nodes ohne Backup-Node laufen beide Phasen in einem Ladevorgang. Dort schützt erst die hohe Schätzung bei großen Obergrenzen die akustische Phase; knappe Obergrenzen von 5–9 Minuten können dort zum Out-of-Memory-Fehler führen.
- Die Diffusion langer Songs läuft auf 8 GB ausgelagert (~28 s pro Schritt bei 316 s Audio statt ~1 s bei 30 s). ComfyUI schätzt für 316 s ~5,7 GB Rechenbedarf (`memory_usage_factor` 4.0 mit der Formel ohne Flash-Attention, plus ~1 GB Konditionierung). Die Schätzung ist nicht zu hoch: Mit erzwungen komplett geladenem Modell stürzt der erste Schritt unter ZLUDA mit einem Segmentation Fault ab statt mit einem Out-of-Memory-Fehler. Nicht ändern. Weniger Sampling-Schritte sind der einzige Hebel.
- HBCC (AMD-Software, „HBCC-Speichersegment“) an lassen. HIP/ZLUDA und damit ComfyUI sehen in jedem Fall nur 8.176 MB, HBCC hilft aber beim teilweise ausgelagerten YuE2-Text-Encoder:

  | HBCC | YuE2 Token/s (langer Song, bf16) | SD 1.5 it/s |
  |---|---|---|
  | aus | 2,98 (zweimal gemessen) | 1,40–1,49 |
  | 12 GB | 3,41 | 1,45 |
  | 16 GB | 3,34 | 1,47 |

  Die Größe des Segments spielt keine Rolle. Der Treiber setzt beim Umschalten zurück (Bildschirm flackert). Der Wechsel von an auf aus ließ ZLUDA alle Kernels neu übersetzen (~55 min), 12 → 16 GB nicht.
- VAE-Absturz bei langen Songs: ComfyUI versuchte die Audio-Dekodierung erst am Stück, obwohl die eigene Schätzung (43,6 GiB bei 8.124 Frames) weit über dem freien VRAM lag. Nach diesem echten Out-of-Memory-Fehler stürzte die Kachel-Dekodierung unter ZLUDA mit „Windows fatal exception: access violation“ ab (nachgestellt, Stack in `decode_tiled_1d`). `AudioOobleckVAE` setzt jetzt `comfy_decode_estimate_is_reliable`, damit lange Songs direkt in Kacheln dekodiert werden: 5 von 5 Läufen ohne Fehler, ~55 s für 325 s Audio. Die Warnung „Ran out of memory when regular VAE decoding“ erscheint trotzdem, das ist nur der Text für diesen Weg.
- Lange Songs über den Backup-Node, gemessen: 8.127 Token in 33:02 min (4,10 Token/s im Schnitt), Diffusion mit 32 Schritten 13:59 min.
- Hardwarebeschleunigung im Browser ausschalten, wenn er nebenbei läuft: Mit Video im Browser fiel YuE2 von ~2,7 auf ~1,9 Token/s.
- Pinned Memory (`cudaHostRegister`) funktioniert auf der Vega (13,4 statt 8,9 GB/s RAM→GPU) und brachte beim Auslagern +8 %. Ist noch nicht aktiviert, weil patientx es für ZLUDA standardmäßig abschaltet.

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

- `token_backup = Kein Backup - neu generieren`: generiert neu und speichert die Token **vor** der akustischen Phase nach `output/yue2_tokens/<Titel>_mode-…_seed-…_<Zeit>.json`. Stürzt danach etwas ab, lässt sich der Lauf aus dem Backup fortsetzen.
- Wählt man eine JSON-Datei aus, wird die Token-Generierung übersprungen und die Generierungseinstellungen kommen aus der Datei.
- Gespeichert wird nur über diesen Node, nicht über die Standard-YuE2-Nodes.
- `custom_nodes/` steht in `.gitignore`. Der Node wurde mit `git add -f` aufgenommen, und Änderungen daran müssen ebenfalls mit `git add -f` hinzugefügt werden.

## MP3 mit Tags

Node: **Save MP3 with Tags** (`audio`)

- Speichert wie „Save Audio (MP3)“ nach `output/<filename_prefix>_00001.mp3` und schreibt ID3v2.3-Tags: Titel, Interpret, Album, Jahr, Genre, Lyrics und optional ein Cover (`IMAGE`-Eingang, als JPEG eingebettet). Leere Felder werden weggelassen. Der Windows-Explorer zeigt die Tags an.
- Die Lyrics dürfen dieselben sein wie im YuE2-Node: Tags in eckigen Klammern am Zeilenanfang (`[Chorus: …]`) entfernt der Node. Ein komplett klein geschriebenes Genre bekommt große Anfangsbuchstaben.
- Genre aus dem Style: „Generate Text“ mit `text_encoders/qwen3.5_2b_bf16.safetensors` (Comfy-Org/Qwen3.5, 4,5 GB), `sampling_mode` off, `max_length` 16, System-Prompt „You classify music. Reply with only the main genre of the described music in 1 to 3 words, like a genre tag in a music library. No explanation.“ Läuft unter Torch 2.7 mit ~1 s pro Song (erster Lauf 28 s). Ergebnisse: DnB, pirate shanties, acoustic ballad, schlager. Strengere Prompts machten das 2B-Modell schlechter (Schlager → Pop).
- Braucht `mutagen` (`custom_nodes/mp3_tags/requirements.txt`), in `venv` und `venv-t24` installiert.
- Den Workflow speichert der Node nicht in die Datei.
- Wie der Backup-Node mit `git add -f` aufgenommen.

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
