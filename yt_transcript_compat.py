# yt_transcript_compat.py
# 2025-09-13 v1
# Utilidad robusta y compatible para obtener transcripciones de YouTube
# Soporta youtube-transcript-api 0.6.x y 1.x, con manejo de IP bans y proxies.
from __future__ import annotations

from typing import Iterable, List, Dict, Optional, Tuple
import os
import re

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

# Intentamos importar excepciones nuevas / internas (1.x)
try:
    # En 1.x viven en _errors
    from youtube_transcript_api._errors import RequestBlocked, IpBlocked  # type: ignore
    _IP_BLOCK_EXC: Tuple[type, ...] = (RequestBlocked, IpBlocked)
except Exception:
    _IP_BLOCK_EXC = tuple()

# Proxies (1.x)
try:
    from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig  # type: ignore
except Exception:
    WebshareProxyConfig = GenericProxyConfig = None  # type: ignore

__all__ = [
    "NoTranscriptAvailable",
    "DEFAULT_PREF_LANGS",
    "extract_video_id",
    "get_segments",
    "get_transcript_text",
]

class NoTranscriptAvailable(Exception):
    """Señala que no se pudo obtener transcripción tras agotar intentos.
    Incluye detalle cuando la IP está bloqueada por YouTube.
    """

DEFAULT_PREF_LANGS: List[str] = ["es", "es-ES", "es-419", "en", "en-US"]

# ---------------------------- Helpers ----------------------------
_YT_PATTERNS = [
    r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&]+)",
    r"(?:https?://)?youtu\.be/([^?]+)",
    r"(?:https?://)?(?:www\.)?youtube\.com/embed/([^?]+)",
    r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([^?]+)",
]

def extract_video_id(url_or_id: str) -> str:
    if not url_or_id:
        return ""
    if "://" not in url_or_id and "/" not in url_or_id:
        return url_or_id
    for pattern in _YT_PATTERNS:
        m = re.search(pattern, url_or_id)
        if m:
            return m.group(1).split("?")[0]
    return url_or_id

def _normalize_segments(snippets: Iterable) -> List[Dict[str, Optional[float]]]:
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
        start_f = float(start) if start is not None else None
        duration_f = float(duration) if duration is not None else None
        norm.append({"text": text or "", "start": start_f, "duration": duration_f})
    return norm

def _make_api() -> YouTubeTranscriptApi:
    """Crea la instancia teniendo en cuenta proxies si existen."""
    # Webshare (credenciales en env opcionales)
    if WebshareProxyConfig and os.getenv("YTA_WEBSHARE_USER") and os.getenv("YTA_WEBSHARE_PASS"):
        return YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=os.environ["YTA_WEBSHARE_USER"],
                proxy_password=os.environ["YTA_WEBSHARE_PASS"],
            )
        )
    # HTTP(S) proxy genérico desde env estándar
    if GenericProxyConfig and (os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")):
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(
                http_url=os.getenv("HTTP_PROXY"),
                https_url=os.getenv("HTTPS_PROXY"),
            )
        )
    return YouTubeTranscriptApi()

# ---------------------------- Core fetchers (nuevo primero) ----------------------------
def _try_fetch(video_id: str, pref_langs: List[str]) -> Optional[List[Dict]]:
    """Rama nueva (1.x): ytt.fetch(video_id, languages=[...])."""
    try:
        ytt = _make_api()
        # Probaremos en grupos razonables
        for langs in (pref_langs, ["es", "es-ES", "es-419"], ["en", "en-US"], []):
            kwargs = {}
            if langs:  # lista no vacía
                kwargs["languages"] = langs
            fetched = ytt.fetch(video_id, **kwargs)  # FetchedTranscript (iterable)
            return _normalize_segments(fetched)
        return None
    except _IP_BLOCK_EXC as e:  # YouTube bloquea IPs de cloud: ver docs
        raise NoTranscriptAvailable(
            f"Bloqueo de YouTube por IP (RequestBlocked/IpBlocked). Configura un proxy residencial o rota IP: {e}."
        )
    except (TranscriptsDisabled, VideoUnavailable):
        raise
    except NoTranscriptFound:
        return None
    except Exception:
        return None

def _try_instance_api(video_id: str, pref_langs: List[str]) -> Optional[List[Dict]]:
    """Compatibilidad: lista y fetch por TranscriptList."""
    try:
        ytt = _make_api()
        if not hasattr(ytt, "list"):
            return None
        tlist = ytt.list(video_id)

        # 1) Idiomas preferidos
        for langs in (pref_langs, ["es", "es-ES", "es-419"], ["en", "en-US"]):
            try:
                t = tlist.find_transcript(langs)
                return _normalize_segments(t.fetch())
            except NoTranscriptFound:
                continue

        # 2) Traducir a español/inglés si es posible
        for target in ("es", "en"):
            try:
                for t in tlist:
                    if getattr(t, "is_translatable", False):
                        try:
                            return _normalize_segments(t.translate(target).fetch())
                        except Exception:
                            continue
            except TypeError:
                pass

        # 3) Primer transcript disponible
        try:
            for t in tlist:
                try:
                    return _normalize_segments(t.fetch())
                except Exception:
                    continue
        except TypeError:
            pass
        return None
    except _IP_BLOCK_EXC as e:
        raise NoTranscriptAvailable(
            f"Bloqueo de YouTube por IP (RequestBlocked/IpBlocked). Configura un proxy residencial o rota IP: {e}."
        )
    except (TranscriptsDisabled, VideoUnavailable):
        raise
    except Exception:
        return None

def _try_classic_get(video_id: str, pref_langs: List[str]) -> Optional[List[Dict]]:
    """Compat con 0.6.x: get_transcript(...)."""
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
    video_id = extract_video_id(video_id_or_url)
    langs = list(pref_langs or DEFAULT_PREF_LANGS)

    # 1) Nuevo fetch()
    seg = _try_fetch(video_id, langs)
    if seg:
        return seg

    # 2) API por instancia (lista)
    seg = _try_instance_api(video_id, langs)
    if seg:
        return seg

    # 3) Camino clásico
    seg = _try_classic_get(video_id, langs)
    if seg:
        return seg

    raise NoTranscriptAvailable(
        "No se encontró ninguna transcripción en los intentos realizados. "
        "Si estás en un servidor o cloud, es probable un bloqueo por IP; prueba con un proxy residencial."
    )

def get_transcript_text(video_id_or_url: str, pref_langs: Optional[List[str]] = None) -> str:
    segments = get_segments(video_id_or_url, pref_langs=pref_langs)
    return " ".join(s.get("text", "") for s in segments if s.get("text")).strip()

if __name__ == "__main__":
    import sys
    vid_or_url = sys.argv[1] if len(sys.argv) >= 2 else ""
    langs = [p.strip() for p in sys.argv[2].split(",")] if len(sys.argv) >= 3 else None
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
