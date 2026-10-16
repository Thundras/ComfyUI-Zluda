import os
import re
from io import BytesIO

import numpy as np
from mutagen.id3 import APIC, ID3, TALB, TCON, TDRC, TIT2, TPE1, USLT
from PIL import Image

import folder_paths
from comfy_api.latest._io import FolderType
from comfy_api.latest._ui import AudioSaveHelper


def plain_lyrics(lyrics):
    # Drop YuE2 section and style tags such as "[Chorus: gritty female vocal]".
    lines = [re.sub(r"^\s*\[[^\]]*\]\s*", "", line).rstrip() for line in lyrics.splitlines()]
    lines = [line for raw, line in zip(lyrics.splitlines(), lines) if line or not raw.strip()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def cover_jpeg(image):
    pixels = np.clip(image[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels).convert("RGB").save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


class SaveMP3WithTags:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/YuE2"}),
                "quality": (["V0", "320k", "128k"],),
                "title": ("STRING", {"default": ""}),
                "artist": ("STRING", {"default": ""}),
                "album": ("STRING", {"default": ""}),
                "year": ("STRING", {"default": ""}),
                "genre": ("STRING", {"default": ""}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "cover": ("IMAGE",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def save(self, audio, filename_prefix, quality, title, artist, album, year, genre, lyrics, cover=None):
        lyrics = plain_lyrics(lyrics)
        genre = genre.strip().rstrip(".")
        if genre.islower():
            genre = genre.title()
        results = AudioSaveHelper.save_audio(audio, filename_prefix, FolderType.output, None, format="mp3", quality=quality)
        frames = [frame for value, frame in (
            (title, TIT2(encoding=3, text=title.strip())),
            (artist, TPE1(encoding=3, text=artist.strip())),
            (album, TALB(encoding=3, text=album.strip())),
            (year, TDRC(encoding=3, text=year.strip())),
            (genre, TCON(encoding=3, text=genre)),
            (lyrics, USLT(encoding=3, lang="XXX", desc="", text=lyrics)),
        ) if value.strip()]
        if cover is not None:
            frames.append(APIC(encoding=3, mime="image/jpeg", type=3, desc="Cover", data=cover_jpeg(cover)))
        for result in results:
            tags = ID3()
            for frame in frames:
                tags.add(frame)
            # ID3v2.3 so Windows Explorer and older players show the tags.
            tags.save(os.path.join(folder_paths.get_output_directory(), result.subfolder, result.filename), v2_version=3)
        return {"ui": {"audio": results}}


NODE_CLASS_MAPPINGS = {
    "SaveMP3WithTags": SaveMP3WithTags,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveMP3WithTags": "Save MP3 with Tags",
}
