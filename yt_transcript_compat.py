"""
yt_transcript_compat.py ─ Utilidad compatible para obtener transcripciones de YouTube

Soporta distintas versiones de `youtube-transcript-api`:
- API por instancia (entornos antiguos):
    ytt = YouTubeTranscriptApi(); tlist = ytt.list(video_id)
- API por clase (entornos nuevos):
    YouTubeTranscriptApi.list_transcripts(video_id)
- Camino clásico:
    YouTubeTranscriptApi.get_transcript(video_id, languages=[...])

Devuelve segmentos normalizados (dicts con keys: text, start, duration) o
texto concatenado. No instancia manualmente excepciones de la librería.

Autor: JCMM + GPT, 2025-09-12
"""
from __future__ import annotations

from typing import Iterable, List, Dict, Optional
import re

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

__all__ = [
    "NoTranscriptAvailable",
    "DEFAULT_PREF_LANGS",
    "extract_video_id",
    "get_segments",
    "get_transcript_text",
]


class NoTranscriptAvailable(Exception):
    """Señala que no existe una transcripción disponible tras agotar los intentos."""


# Conjunto razonable de idiomas preferentes (puedes editar)
DEFAULT_PREF_LANGS: List[str] = ["es", "es-ES", "es-419", "en", "en-US"]


# ---------------------------- Helpers ----------------------------
_YT_PATTERNS = [
    r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&]+)",
    r"(?:https?://)?youtu\.be/([^?]+)",
    r"(?:https?://)?(?:www\.)?youtube\.com/embed/([^?]+)",
]

def extract_video_id(url_or_id: str) -> str:
    """Devuelve el video_id a partir de una URL o retorna el valor si ya es un id.
    No valida la existencia del vídeo.
    """
    if not url_or_id:
        return ""
    if "://" not in url_or_id and "/" not in url_or_id:
        # Probablemente ya es un id
        return url_or_id
    for pattern in _YT_PATTERNS:
        m = re.search(pattern, url_or_id)
        if m:
            return m.group(1).split("?")[0]
    # Si no casa, devolvemos lo que llegó (por si era un id raro)
    return url_or_id


def _normalize_segments(snippets: Iterable) -> List[Dict[str, Optional[float]]]:
    """Normaliza a una lista de dicts con keys: text, start, duration.

    Acepta listas de dicts (API clásica) o listas de objetos con atributos .text/.start/.duration
    (p.ej., FetchedTranscriptSnippet / TranscriptSegment en versiones nuevas).
    """
    norm: List[Dict[str, Optional[float]]] = []
    for s in snippets:
        if isinstance(s, dict):
            text = s.get("text", "")
            start = s.get("start")
            duration = s.get("duration")
        else:
            text = getattr(s, "text", "")
            start = getattr(s, "start", None)
            duration = getattr(s, "duration", None)
        # normaliza a float cuando existe
        start_f = float(start) if start is not None else None
        duration_f = float(duration) if duration is not None else None
        norm.append({"text": text or "", "start": start_f, "duration": duration_f})
    return norm


# ---------------------------- Core fetchers ----------------------------

def _try_instance_api(video_id: str, pref_langs: List[str]) -> Optional[List[Dict]]:
    """Intenta con la API por *instancia* (entornos antiguos).
    Retorna segmentos normalizados o None si no pudo resolverse aquí.
    """
    try:
        ytt = YouTubeTranscriptApi()
        if not hasattr(ytt, "list"):
            return None
        tlist = ytt.list(video_id)
        # Buscar por lotes de idiomas preferidos
        for langs in (pref_langs, ["es", "es-ES", "es-419"], ["en", "en-US"]):
            try:
                t = tlist.find_transcript(langs)
                return _normalize_segments(t.fetch())
            except NoTranscriptFound:
                continue
            except Exception:
                continue
        # Fallback: primer transcript disponible
        try:
            for t in tlist:
                try:
                    return _normalize_segments(t.fetch())
                except Exception:
                    continue
        except TypeError:
            # Por si tlist no fuera iterable en alguna versión muy antigua
            pass
        return None
    except (TranscriptsDisabled, VideoUnavailable):
        raise
    except Exception:
        return None


def _try_class_api(video_id: str, pref_langs: List[str]) -> Optional[List[Dict]]:
    """Intenta con la API por *clase* (entornos nuevos)."""
    try:
        if not hasattr(YouTubeTranscriptApi, "list_transcripts"):
            return None
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        for langs in (pref_langs, ["es", "es-ES", "es-419"], ["en", "en-US"]):
            try:
                t = transcripts.find_transcript(langs)
                return _normalize_segments(t.fetch())
            except NoTranscriptFound:
                continue
            except Exception:
                continue
        try:
            for t in transcripts:
                try:
                    return _normalize_segments(t.fetch())
                except Exception:
                    continue
        except TypeError:
            pass
        return None
    except (TranscriptsDisabled, VideoUnavailable):
        raise
    except Exception:
        return None


def _try_classic_get(video_id: str, pref_langs: List[str]) -> Optional[List[Dict]]:
    """Intenta con el camino clásico get_transcript(...)."""
    for langs in (pref_langs, ["es", "es-ES", "es-419"], ["en", "en-US"], None):
        try:
            if langs is None:
                snippets = YouTubeTranscriptApi.get_transcript(video_id)
            else:
                snippets = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
            return _normalize_segments(snippets)
        except NoTranscriptFound:
            continue
        except (TranscriptsDisabled, VideoUnavailable):
            raise
        except Exception:
            continue
    return None


# ---------------------------- Public API ----------------------------

def get_segments(video_id_or_url: str, pref_langs: Optional[List[str]] = None) -> List[Dict]:
    """Devuelve la transcripción en segmentos normalizados.

    Args:
        video_id_or_url: ID de vídeo o URL completa de YouTube.
        pref_langs: Lista de códigos de idioma preferidos. Si no se indica, usa DEFAULT_PREF_LANGS.

    Raises:
        TranscriptsDisabled, VideoUnavailable: si el vídeo no permite transcripciones o está no disponible.
        NoTranscriptAvailable: si no se encontró ninguna transcripción en los intentos realizados.
    """
    video_id = extract_video_id(video_id_or_url)
    langs = list(pref_langs or DEFAULT_PREF_LANGS)

    # 1) API por instancia
    seg = _try_instance_api(video_id, langs)
    if seg:
        return seg

    # 2) API por clase
    seg = _try_class_api(video_id, langs)
    if seg:
        return seg

    # 3) Camino clásico
    seg = _try_classic_get(video_id, langs)
    if seg:
        return seg

    # Nada funcionó
    raise NoTranscriptAvailable("No transcript available in the tried languages.")


def get_transcript_text(video_id_or_url: str, pref_langs: Optional[List[str]] = None) -> str:
    """Devuelve la transcripción como texto concatenado (con espacios).

    Raises las mismas excepciones que `get_segments`.
    """
    segments = get_segments(video_id_or_url, pref_langs=pref_langs)
    return " ".join(s.get("text", "") for s in segments if s.get("text")).strip()


# ---------------------------- CLI de prueba ----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python yt_transcript_compat.py <VIDEO_ID_o_URL> [idiomas separados por coma]")
        sys.exit(1)

    vid_or_url = sys.argv[1]
    langs = None
    if len(sys.argv) >= 3:
        langs = [p.strip() for p in sys.argv[2].split(',') if p.strip()]

    try:
        txt = get_transcript_text(vid_or_url, pref_langs=langs)
        print(txt[:2000] + ("..." if len(txt) > 2000 else ""))
    except TranscriptsDisabled:
        print("Transcripciones deshabilitadas para este vídeo.")
    except VideoUnavailable:
        print("Vídeo no disponible.")
    except NoTranscriptAvailable as e:
        print(str(e))
    except Exception as e:
        print(f"Error inesperado: {e}")
