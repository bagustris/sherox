"""Tests for sherox.utils module."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sherox import SherpaError
from sherox.utils import download_file


class TestDownloadFile:
    """Tests for download_file function."""

    def test_downloads_from_scratch_when_no_file_exists(self, tmp_path):
        """Test that download starts from beginning when file doesn't exist."""
        dest = tmp_path / "test.txt"
        url = "http://example.com/test.txt"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Length": "12"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b"chunk1", b"chunk2", b""]

        with patch("urllib.request.urlopen", return_value=mock_response):
            download_file(url, dest)

        assert dest.exists()
        assert dest.read_text() == "chunk1chunk2"

    def test_accepts_string_destination(self, tmp_path):
        """Test that string paths are accepted as documented."""
        dest = str(tmp_path / "test.txt")
        url = "http://example.com/test.txt"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Length": "5"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b"hello", b""]

        with patch("urllib.request.urlopen", return_value=mock_response):
            download_file(url, dest)

        assert Path(dest).read_text() == "hello"

    def test_resumes_download_when_partial_file_exists(self, tmp_path):
        """Test that download resumes from existing file position."""
        dest = tmp_path / "test.txt"
        url = "http://example.com/test.txt"

        # Create partial file with existing content
        dest.write_bytes(b"existing")

        mock_response = MagicMock()
        mock_response.status = 206
        mock_response.headers = {"Content-Range": "bytes 8-12/13"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b"chunk", b""]

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            download_file(url, dest)

        # Verify Range header was added
        call_args = mock_urlopen.call_args
        assert call_args[0][0].headers.get("Range") == "bytes=8-"

        # Verify file was appended to, not overwritten
        assert dest.read_bytes() == b"existingchunk"

    def test_handles_missing_content_length(self, tmp_path):
        """Test that download works when Content-Length is missing."""
        dest = tmp_path / "test.txt"
        url = "http://example.com/test.txt"

        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b"chunk1", b""]

        with patch("urllib.request.urlopen", return_value=mock_response):
            download_file(url, dest)

        assert dest.exists()
        assert dest.read_text() == "chunk1"


def _response(chunks, headers):
    r = MagicMock()
    r.status = 200
    r.headers = headers
    r.__enter__ = MagicMock(return_value=r)
    r.__exit__ = MagicMock(return_value=False)
    r.read.side_effect = chunks
    return r


class TestDownloadFileRobustness:
    def test_truncated_download_is_an_error_and_keeps_partial(self, tmp_path):
        dest = tmp_path / "f.bin"
        resp = _response([b"abc", b""], {"Content-Length": "10"})
        with patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(SherpaError, match="incomplete"):
                download_file("http://example.com/f", dest)
        assert dest.read_bytes() == b"abc"      # kept so a retry can resume

    def test_passes_a_finite_timeout(self, tmp_path):
        resp = _response([b"abc", b""], {"Content-Length": "3"})
        with patch("urllib.request.urlopen", return_value=resp) as m:
            download_file("http://example.com/f", tmp_path / "f")
        assert 0 < m.call_args.kwargs["timeout"] < 600

    def test_to_stderr_keeps_stdout_clean(self, tmp_path, capsys):
        resp = _response([b"abc", b""], {"Content-Length": "3"})
        with patch("urllib.request.urlopen", return_value=resp):
            download_file("http://example.com/f", tmp_path / "f", to_stderr=True)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Downloading from" in captured.err and "100%" in captured.err
