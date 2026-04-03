from extractors import (
    _clean_extracted_text,
    extract_tweet_from_url,
    extract_web_text,
)


def test_clean_extracted_text_filters_browser_gate_noise():
    text = "You can see a list of supported browsers in our Help Center"

    assert _clean_extracted_text(text, 200) is None


def test_extract_web_text_skips_x_urls():
    assert extract_web_text("https://x.com/example/status/123") is None


def test_extract_tweet_from_url_falls_back_to_html(monkeypatch):
    monkeypatch.setattr("extractors.extract_tweet", lambda tweet_id: None)
    monkeypatch.setattr(
        "extractors._download_public_url",
        lambda url: '<meta property="og:description" content="실제 X 게시물 본문">'.encode("utf-8"),
    )

    text = extract_tweet_from_url("https://x.com/example/status/123")

    assert text == "[X Post]\n실제 X 게시물 본문"
