from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

YOUTUBE_AUDIO_TRANSCRIPTION_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_AUDIO_CACHE_DIR = ".cache/youtube-audio"
DEFAULT_TRANSCRIPT_CACHE_DIR = ".cache/youtube-transcripts"
DEFAULT_MAX_SECONDS = 28_800
DEFAULT_CHUNK_SECONDS = 1_800
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 900
DEFAULT_TRANSCRIPTION_TIMEOUT_SECONDS = 0


@dataclass(frozen=True)
class YouTubeAudioMetadata:
    video_id: str
    url: str
    title: str
    channel: str
    duration: int | None


@dataclass(frozen=True)
class YouTubeAudioTranscriptionConfig:
    audio_cache_dir: Path
    transcript_cache_dir: Path
    max_seconds: int | None
    chunk_seconds: int
    download_timeout_seconds: int
    transcription_timeout_seconds: int | None
    keep_audio: bool
    language: str | None = None
    model: str = YOUTUBE_AUDIO_TRANSCRIPTION_MODEL


@dataclass(frozen=True)
class YouTubeAudioTranscriptionResult:
    ok: bool
    status: str
    message: str
    content: str | None = None
    transcript_path: str | None = None
    metadata: dict[str, Any] | None = None
    cached: bool = False


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw_value = os.getenv(name)
    try:
        value = int(raw_value) if raw_value is not None and raw_value.strip() else default
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _resolve_path(raw_path: str | None, default: str) -> Path:
    path = Path(raw_path or default).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_config_from_env() -> YouTubeAudioTranscriptionConfig:
    max_seconds = _env_int("YOUTUBE_AUDIO_TRANSCRIPTION_MAX_SECONDS", DEFAULT_MAX_SECONDS, minimum=0)
    timeout_seconds = _env_int(
        "YOUTUBE_AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS",
        DEFAULT_TRANSCRIPTION_TIMEOUT_SECONDS,
        minimum=0,
    )
    language = (os.getenv("YOUTUBE_AUDIO_TRANSCRIPTION_LANGUAGE") or "").strip() or None
    return YouTubeAudioTranscriptionConfig(
        audio_cache_dir=_resolve_path(os.getenv("YOUTUBE_AUDIO_CACHE_DIR"), DEFAULT_AUDIO_CACHE_DIR),
        transcript_cache_dir=_resolve_path(
            os.getenv("YOUTUBE_TRANSCRIPT_CACHE_DIR"),
            DEFAULT_TRANSCRIPT_CACHE_DIR,
        ),
        max_seconds=max_seconds or None,
        chunk_seconds=_env_int("YOUTUBE_AUDIO_CHUNK_SECONDS", DEFAULT_CHUNK_SECONDS, minimum=60),
        download_timeout_seconds=_env_int(
            "YOUTUBE_AUDIO_DOWNLOAD_TIMEOUT_SECONDS",
            DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
            minimum=1,
        ),
        transcription_timeout_seconds=timeout_seconds or None,
        keep_audio=_env_flag("YOUTUBE_AUDIO_KEEP_FILES", False),
        language=language,
    )


def _canonical_youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _run_subprocess(args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _check_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} is not available on PATH")


def fetch_youtube_audio_metadata(video_id: str) -> YouTubeAudioMetadata:
    from yt_dlp import YoutubeDL

    url = _canonical_youtube_url(video_id)
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)

    duration = info.get("duration")
    return YouTubeAudioMetadata(
        video_id=video_id,
        url=url,
        title=str(info.get("title") or video_id),
        channel=str(info.get("channel") or info.get("uploader") or ""),
        duration=int(duration) if isinstance(duration, (int, float)) else None,
    )


def format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "알 수 없음"
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes}분 {secs}초"
    if minutes:
        return f"{minutes}분 {secs}초"
    return f"{secs}초"


def validate_youtube_audio_request(
    metadata: YouTubeAudioMetadata,
    config: YouTubeAudioTranscriptionConfig,
) -> tuple[bool, str]:
    if config.max_seconds is not None and metadata.duration is not None and metadata.duration > config.max_seconds:
        return (
            False,
            f"영상 길이가 {format_duration(metadata.duration)}라 설정 한도 {format_duration(config.max_seconds)}를 넘습니다.",
        )
    return True, ""


def _metadata_cache_path(config: YouTubeAudioTranscriptionConfig, video_id: str) -> Path:
    return config.transcript_cache_dir / video_id / "metadata.json"


def _model_cache_key(model: str, language: str | None) -> str:
    digest = hashlib.sha1(f"{model}:{language or 'auto'}".encode("utf-8")).hexdigest()[:12]
    return digest


def transcript_cache_path(config: YouTubeAudioTranscriptionConfig, video_id: str) -> Path:
    key = _model_cache_key(config.model, config.language)
    return config.transcript_cache_dir / video_id / f"{key}.txt"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _download_youtube_audio(
    video_id: str,
    config: YouTubeAudioTranscriptionConfig,
) -> Path:
    _check_binary("ffmpeg")
    _check_binary("ffprobe")

    from yt_dlp import YoutubeDL

    output_dir = config.audio_cache_dir / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "source.%(ext)s")
    options = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
        "socket_timeout": config.download_timeout_seconds,
    }
    with YoutubeDL(options) as ydl:
        ydl.download([_canonical_youtube_url(video_id)])

    audio_path = output_dir / "source.m4a"
    if not audio_path.exists():
        candidates = sorted(output_dir.glob("source.*"))
        if candidates:
            return candidates[0]
        raise RuntimeError("audio download did not produce an output file")
    return audio_path


def _split_audio(audio_path: Path, video_id: str, config: YouTubeAudioTranscriptionConfig) -> list[Path]:
    chunks_dir = config.audio_cache_dir / video_id / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    existing_chunks = sorted(chunks_dir.glob("chunk_*.m4a"))
    if existing_chunks:
        return existing_chunks

    output_pattern = chunks_dir / "chunk_%04d.m4a"
    _run_subprocess(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-f",
            "segment",
            "-segment_time",
            str(config.chunk_seconds),
            "-c",
            "copy",
            str(output_pattern),
        ],
        timeout=config.download_timeout_seconds,
    )
    chunks = sorted(chunks_dir.glob("chunk_*.m4a"))
    if not chunks:
        raise RuntimeError("audio chunking did not produce chunk files")
    return chunks


def _transcribe_chunk(chunk_path: Path, config: YouTubeAudioTranscriptionConfig) -> str:
    import mlx_whisper

    kwargs: dict[str, Any] = {"path_or_hf_repo": config.model}
    if config.language:
        kwargs["language"] = config.language
    result = mlx_whisper.transcribe(str(chunk_path), **kwargs)
    text = str(result.get("text") or "").strip()
    return text


def _transcribe_chunks(
    chunks: list[Path],
    video_id: str,
    config: YouTubeAudioTranscriptionConfig,
) -> str:
    chunk_text_dir = config.transcript_cache_dir / video_id / "chunks"
    chunk_text_dir.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    for index, chunk_path in enumerate(chunks):
        chunk_text_path = chunk_text_dir / f"{chunk_path.stem}.txt"
        if chunk_text_path.exists():
            text = chunk_text_path.read_text(encoding="utf-8").strip()
        else:
            text = _transcribe_chunk(chunk_path, config)
            chunk_text_path.write_text(text, encoding="utf-8")
        if text:
            texts.append(text)
        print(
            json.dumps(
                {"event": "chunk_done", "video_id": video_id, "index": index + 1, "total": len(chunks)},
                ensure_ascii=False,
            ),
            flush=True,
        )
    return "\n\n".join(texts).strip()


def transcribe_youtube_audio(
    video_id: str,
    config: YouTubeAudioTranscriptionConfig | None = None,
) -> YouTubeAudioTranscriptionResult:
    config = config or load_config_from_env()
    transcript_path = transcript_cache_path(config, video_id)
    if transcript_path.exists():
        text = transcript_path.read_text(encoding="utf-8").strip()
        if text:
            return YouTubeAudioTranscriptionResult(
                ok=True,
                status="ok",
                message="cached transcript",
                content=f"[YouTube Transcript]\n{text}",
                transcript_path=str(transcript_path),
                cached=True,
            )

    try:
        metadata = fetch_youtube_audio_metadata(video_id)
        allowed, reason = validate_youtube_audio_request(metadata, config)
        if not allowed:
            return YouTubeAudioTranscriptionResult(
                ok=False,
                status="duration_limit_exceeded",
                message=reason,
                metadata=asdict(metadata),
            )

        _write_json(_metadata_cache_path(config, video_id), asdict(metadata))
        audio_path = _download_youtube_audio(video_id, config)
        chunks = _split_audio(audio_path, video_id, config)
        text = _transcribe_chunks(chunks, video_id, config)
        if not text:
            return YouTubeAudioTranscriptionResult(
                ok=False,
                status="empty_transcription",
                message="오디오 전사 결과가 비어 있습니다.",
                metadata=asdict(metadata),
            )

        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(text, encoding="utf-8")
        if not config.keep_audio:
            shutil.rmtree(config.audio_cache_dir / video_id, ignore_errors=True)

        return YouTubeAudioTranscriptionResult(
            ok=True,
            status="ok",
            message="transcribed",
            content=f"[YouTube Transcript]\n{text}",
            transcript_path=str(transcript_path),
            metadata=asdict(metadata),
        )
    except subprocess.TimeoutExpired as exc:
        return YouTubeAudioTranscriptionResult(
            ok=False,
            status="timeout",
            message=f"오디오 전사 작업이 제한 시간 안에 끝나지 않았습니다: {exc}",
        )
    except Exception as exc:
        return YouTubeAudioTranscriptionResult(
            ok=False,
            status="error",
            message=f"오디오 전사 중 오류가 발생했습니다: {exc}",
        )


def prewarm_model(model: str = YOUTUBE_AUDIO_TRANSCRIPTION_MODEL) -> None:
    import mlx_whisper

    temp_dir = Path(os.getenv("TMPDIR") or "/tmp") / "telegram-llm-bot-whisper-prewarm"
    temp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = temp_dir / "silence.wav"
    silence = b"\x00\x00" * 16_000
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(silence)
    mlx_whisper.transcribe(str(wav_path), path_or_hf_repo=model)


def _print_result(result: YouTubeAudioTranscriptionResult) -> int:
    print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="YouTube audio transcription worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    transcribe_parser = subparsers.add_parser("transcribe")
    transcribe_parser.add_argument("video_id")
    metadata_parser = subparsers.add_parser("metadata")
    metadata_parser.add_argument("video_id")
    subparsers.add_parser("prewarm")

    args = parser.parse_args(argv)
    config = load_config_from_env()
    if args.command == "transcribe":
        return _print_result(transcribe_youtube_audio(args.video_id, config))
    if args.command == "metadata":
        metadata = fetch_youtube_audio_metadata(args.video_id)
        allowed, reason = validate_youtube_audio_request(metadata, config)
        print(
            json.dumps(
                {"ok": allowed, "reason": reason, "metadata": asdict(metadata)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0 if allowed else 1
    if args.command == "prewarm":
        prewarm_model(config.model)
        print(json.dumps({"ok": True, "model": config.model}, ensure_ascii=False), flush=True)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
