import os
import re
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
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


class OverlayLogo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "logo": ("IMAGE",),
                "logo_mask": ("MASK", {"tooltip": "MASK output of Load Image, from the logo's transparency."}),
                "width": ("INT", {"default": 600, "min": 8, "max": 8192}),
                "center_x": ("BOOLEAN", {"default": True}),
                "x": ("INT", {"default": 0, "min": -8192, "max": 8192, "tooltip": "Ignored when center_x is on."}),
                "y": ("INT", {"default": 48, "min": -8192, "max": 8192}),
                "outline": ("INT", {"default": 3, "min": 0, "max": 64, "tooltip": "Black outline in pixels so the logo stands out on bright images."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "overlay"
    CATEGORY = "image"

    def overlay(self, image, logo, logo_mask, width, center_x, x, y, outline):
        # Load Image returns 1 - alpha as the mask.
        alpha = 1.0 - logo_mask[:1]
        if alpha.shape[1:] != logo.shape[1:3]:
            raise ValueError("The logo needs a transparent background: connect the MASK output of Load Image for a PNG with alpha.")
        # Crop the transparent border so width, x and y refer to the visible logo.
        rows = (alpha[0] > 0.01).any(1).nonzero()
        cols = (alpha[0] > 0.01).any(0).nonzero()
        if len(rows) == 0:
            return (image,)
        top, bottom, left, right = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        logo, alpha = logo[:1, top:bottom, left:right, :3], alpha[:, None, top:bottom, left:right]

        height = max(1, round(logo.shape[1] * width / logo.shape[2]))
        rgb = F.interpolate(logo.movedim(-1, 1), size=(height, width), mode="bilinear", antialias=True)
        alpha = F.interpolate(alpha, size=(height, width), mode="bilinear", antialias=True)
        rgb = F.pad(rgb, (outline,) * 4).to(image.device)
        alpha = F.pad(alpha, (outline,) * 4).to(image.device)
        edge = F.max_pool2d(alpha, 2 * outline + 1, stride=1, padding=outline) if outline else torch.zeros_like(alpha)

        if center_x:
            x = (image.shape[2] - width) // 2
        x, y = x - outline, y - outline
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + rgb.shape[3], image.shape[2]), min(y + rgb.shape[2], image.shape[1])
        out = image.clone()
        if x1 <= x0 or y1 <= y0:
            return (out,)
        region = out[:, y0:y1, x0:x1, :3]
        crop = (slice(None), slice(None), slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
        a = alpha[crop].movedim(1, -1)
        e = edge[crop].movedim(1, -1)
        region.mul_(1.0 - e)
        region.mul_(1.0 - a).add_(rgb[crop].movedim(1, -1) * a)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "SaveMP3WithTags": SaveMP3WithTags,
    "OverlayLogo": OverlayLogo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveMP3WithTags": "Save MP3 with Tags",
    "OverlayLogo": "Overlay Logo",
}
