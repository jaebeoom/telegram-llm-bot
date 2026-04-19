import ipaddress
import socket
from functools import lru_cache
from urllib.parse import urljoin, urlparse

import requests

from .extractors_common import MAX_FETCH_BYTES, MAX_REDIRECTS, REQUEST_HEADERS


@lru_cache(maxsize=1024)
def _is_private_hostname(hostname: str) -> bool:
    lowered = hostname.lower()
    if lowered in {"localhost"} or lowered.endswith(".local"):
        return True

    try:
        ip = ipaddress.ip_address(lowered)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return True

    for info in infos:
        resolved_ip = ipaddress.ip_address(info[4][0])
        if (
            resolved_ip.is_private
            or resolved_ip.is_loopback
            or resolved_ip.is_link_local
            or resolved_ip.is_multicast
            or resolved_ip.is_reserved
            or resolved_ip.is_unspecified
        ):
            return True

    return False


def _validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("http/https URL만 허용됩니다.")
    if not parsed.hostname:
        raise ValueError("유효한 호스트가 없습니다.")
    if _is_private_hostname(parsed.hostname):
        raise ValueError("로컬/사설 네트워크 주소에는 접근할 수 없습니다.")
    return url


def _download_public_url(url: str) -> bytes:
    current_url = _validate_public_url(url)

    with requests.Session() as session:
        for _ in range(MAX_REDIRECTS + 1):
            response = session.get(
                current_url,
                timeout=30,
                headers=REQUEST_HEADERS,
                allow_redirects=False,
                stream=True,
            )

            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("리다이렉트 Location 헤더가 없습니다.")
                current_url = _validate_public_url(urljoin(current_url, location))
                response.close()
                continue

            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_FETCH_BYTES:
                raise ValueError("다운로드 가능한 최대 크기를 초과했습니다.")

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_FETCH_BYTES:
                    raise ValueError("다운로드 가능한 최대 크기를 초과했습니다.")
                chunks.append(chunk)

            return b"".join(chunks)

    raise ValueError("리다이렉트가 너무 많습니다.")
