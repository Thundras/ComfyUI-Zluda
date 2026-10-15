import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import folder_paths
from comfy.text_encoders.yue2 import CODEC_OFFSET, CODEC_SIZE, FRAMES_PER_SECOND

NEW = "Kein Backup - neu generieren"
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def backup_directory():
    return Path(folder_paths.get_output_directory()) / "yue2_tokens"


def backup_path(filename):
    root = backup_directory().resolve()
    path = (root / filename).resolve()
    if path.parent != root or path.suffix.lower() != ".json":
        raise ValueError("Bitte eine JSON aus output/yue2_tokens auswählen.")
    if not path.is_file():
        raise FileNotFoundError(f"Token-Sicherung fehlt: {path}")
    return path


def load_backup(path):
    saved = json.loads(path.read_text(encoding="utf-8-sig"))
    if saved.get("format_version") != 1:
        raise ValueError("Unbekanntes Format der Token-Sicherung.")
    if not isinstance(saved.get("request_tokens"), dict):
        raise ValueError("Generierungseinstellungen fehlen.")
    semantic = saved.get("semantic")
    if not isinstance(semantic, list) or not semantic:
        raise ValueError("Die Sicherung enthält keine Musik-Tokens.")
    if not all(type(t) is int and CODEC_OFFSET <= t < CODEC_OFFSET + CODEC_SIZE for t in semantic):
        raise ValueError("Ungültige Musik-Tokens in der Sicherung.")
    tokens = saved["request_tokens"]
    tokens["semantic"] = semantic
    tokens["semantic_truncated"] = bool(saved.get("semantic_truncated", False))
    return tokens


def save_backup(tokens, settings):
    semantic = tokens["semantic"]
    safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", settings["title"]).strip(" .")[:60] or "YuE2"
    if safe_title.upper().split(".")[0] in RESERVED_NAMES:
        safe_title = "Song_" + safe_title
    planning = "on" if settings["abc_planning"] else "off"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = (f"{safe_title}_mode-{settings['mode']}_abc-{planning}"
                f"_max-{settings['max_duration']:g}s_seed-{settings['seed']}_{stamp}_{uuid.uuid4().hex[:8]}.json")
    directory = backup_directory()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    payload = {
        "format_version": 1,
        "title": settings["title"],
        "generation_settings": settings,
        "actual_duration": len(semantic) / FRAMES_PER_SECOND,
        "semantic": semantic,
        "semantic_truncated": tokens["semantic_truncated"],
        "request_tokens": {k: v for k, v in tokens.items() if k not in ("semantic", "semantic_truncated")},
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return path


class YuE2LoadTokenBackup:
    @classmethod
    def INPUT_TYPES(cls):
        root = backup_directory()
        files = sorted(p.name for p in root.glob("*.json")) if root.exists() else []
        return {
            "required": {
                "clip": ("CLIP",),
                "token_backup": ([NEW] + files,),
                "title": ("STRING", {"default": "Mein Song"}),
                "style": ("STRING", {"multiline": True, "default": ""}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["full", "melody"],),
                "abc_planning": ("BOOLEAN", {"default": False}),
                "max_duration": ("FLOAT", {
                    "default": 30.0, "min": 0.04, "max": 900.0, "step": 0.04
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xffffffffffffffff
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "FLOAT", "STRING")
    RETURN_NAMES = ("conditioning", "seconds", "info")
    FUNCTION = "run"
    CATEGORY = "audio/YuE2"

    @classmethod
    def IS_CHANGED(cls, clip, token_backup, **kwargs):
        if token_backup == NEW:
            return "new-generation"
        return hashlib.sha256(backup_path(token_backup).read_bytes()).hexdigest()

    def run(self, clip, token_backup, title, style, lyrics, mode,
            abc_planning, max_duration, seed):
        if token_backup != NEW:
            path = backup_path(token_backup)
            tokens = load_backup(path)
            description = f"Backup: {path.name}"
            used_seed = tokens.get("seed")
        else:
            if not style.strip():
                raise ValueError("Für eine neue Generierung bitte Style eingeben.")

            abc = ""
            if abc_planning:
                plan_tokens = clip.tokenize(
                    style, lyrics=lyrics, cot=mode, seed=seed,
                    max_tokens=8192, penalty_window=100,
                )
                ids = clip.generate(
                    plan_tokens, max_length=8192,
                    temperature=0.7, top_p=0.9, top_k=30,
                    repetition_penalty=1.005, seed=seed,
                )
                abc = clip.decode(ids)

            generation_mode = mode if abc.strip() else "off"
            tokens = clip.tokenize(
                style, lyrics=lyrics, cot=generation_mode,
                seed=seed, abc=abc,
                max_tokens=max(1, round(max_duration * FRAMES_PER_SECOND)),
                temperature=1.0, top_p=0.95, top_k=100,
                repetition_penalty=1.2, cfg_scale=1.0,
            )
            # Generate the music tokens first and back them up before the acoustic pass,
            # which then loads like a resumed backup.
            tokens["semantic"] = clip.generate(dict(tokens, music_only=True))
            tokens["semantic_truncated"] = len(tokens["semantic"]) >= tokens["max_tokens"]
            path = save_backup(tokens, {
                "title": title.strip() or "Mein Song",
                "mode": mode,
                "abc_planning": bool(abc_planning),
                "max_duration": max_duration,
                "seed": seed,
            })
            description = f"Neue Generierung\nGesichert: {path.name}"
            used_seed = seed

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        frames = conditioning[0][1]["yue2_frames"]
        seconds = frames / FRAMES_PER_SECOND
        info = (
            f"{description}\n"
            f"{frames} Musik-Tokens, {seconds:.2f} Sekunden\n"
            f"Seed: {used_seed}"
        )
        return (conditioning, seconds, info)


NODE_CLASS_MAPPINGS = {
    "YuE2LoadTokenBackup": YuE2LoadTokenBackup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2LoadTokenBackup": "YuE2 Generate or Load Backup",
}
