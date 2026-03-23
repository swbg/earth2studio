#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from earth2studio.serve.client import fsspec_utils as fsspec_utils_mod
from earth2studio.serve.client.fsspec_utils import (
    SignedURLFileSystem,
    create_azure_blob_mapper,
    create_cloudfront_mapper,
    get_mapper,
)
from earth2studio.serve.client.models import (
    InferenceRequestResults,
    RequestStatus,
    StorageType,
)


class TestSignedURLFileSystemInit:
    """Test SignedURLFileSystem initialization."""

    def test_init_stores_params(self) -> None:
        """Init stores base_fs, query_params, base_url and builds query_string."""
        base_fs = Mock()
        query_params = {"Policy": "p", "Signature": "s", "Key-Pair-Id": "k"}
        base_url = "https://example.com/store"
        fs = SignedURLFileSystem(base_fs, query_params, base_url)
        assert fs._fs is base_fs
        assert fs._query_params == query_params
        assert fs._base_url == base_url
        assert "Policy=p" in fs._query_string
        assert "Signature=s" in fs._query_string
        assert "Key-Pair-Id=k" in fs._query_string


class TestSignedURLFileSystemMakeSignedPath:
    """Test _make_signed_path."""

    def test_relative_path_appends_to_base_url(self) -> None:
        """Relative path is joined with base_url and query string appended."""
        base_fs = Mock()
        fs = SignedURLFileSystem(
            base_fs, {"Policy": "p", "Signature": "s"}, "https://example.com/zarr"
        )
        out = fs._make_signed_path("subdir/.zmetadata")
        assert out.startswith("https://example.com/zarr/subdir/.zmetadata")
        assert "Policy=p" in out
        assert "Signature=s" in out
        assert "?" in out

    def test_relative_path_empty_uses_base_url_only(self) -> None:
        """Empty relative path uses base_url only."""
        base_fs = Mock()
        fs = SignedURLFileSystem(base_fs, {"Policy": "p"}, "https://example.com/")
        out = fs._make_signed_path("")
        assert out in ("https://example.com/?Policy=p", "https://example.com?Policy=p")

    def test_path_starting_with_http_unchanged_with_query_appended(self) -> None:
        """Path starting with http is used as full URL; query params appended."""
        base_fs = Mock()
        fs = SignedURLFileSystem(base_fs, {"Policy": "p"}, "https://example.com")
        out = fs._make_signed_path("https://cdn.example.com/path")
        assert out.startswith("https://cdn.example.com/path")
        assert "Policy=p" in out

    def test_url_with_existing_query_uses_ampersand(self) -> None:
        """If URL already has ?, separator is &."""
        base_fs = Mock()
        fs = SignedURLFileSystem(base_fs, {"Policy": "p"}, "https://example.com?foo=1")
        out = fs._make_signed_path("x")
        assert "&Policy=" in out or "&policy=" in out.lower()


class TestSignedURLFileSystemHandle403:
    """Test _handle_403."""

    def test_403_raises_file_not_found(self) -> None:
        """Exception containing 403 raises FileNotFoundError."""
        base_fs = Mock()
        fs = SignedURLFileSystem(base_fs, {}, "https://example.com")
        with pytest.raises(FileNotFoundError, match="File not found: /path"):
            fs._handle_403(Exception("HTTP 403 Forbidden"), "/path")

    def test_forbidden_text_raises_file_not_found(self) -> None:
        """Exception containing 'forbidden' raises FileNotFoundError."""
        base_fs = Mock()
        fs = SignedURLFileSystem(base_fs, {}, "https://example.com")
        with pytest.raises(FileNotFoundError, match="File not found: /x"):
            fs._handle_403(Exception("Access forbidden"), "/x")

    def test_other_exception_reraises(self) -> None:
        """Non-403 exception is reraised."""
        base_fs = Mock()
        fs = SignedURLFileSystem(base_fs, {}, "https://example.com")
        with pytest.raises(ValueError, match="other"):
            fs._handle_403(ValueError("other"), "/path")


class TestSignedURLFileSystemOpen:
    """Test _open."""

    def test_open_success_returns_file_like(self) -> None:
        """Successful _open returns result from base fs _open."""
        base_fs = Mock()
        base_fs._open.return_value = MagicMock()
        fs = SignedURLFileSystem(base_fs, {"Policy": "p"}, "https://example.com")
        result = fs._open("sub/file", mode="rb")
        assert result is base_fs._open.return_value
        base_fs._open.assert_called_once()
        call_url = base_fs._open.call_args[0][0]
        assert (
            "sub/file" in call_url
            or "sub%2Ffile" in call_url
            or call_url.endswith("sub/file")
        )
        assert base_fs._open.call_args[1]["mode"] == "rb"

    def test_open_403_raises_file_not_found(self) -> None:
        """Base _open raising 403 leads to FileNotFoundError."""
        base_fs = Mock()
        base_fs._open.side_effect = Exception("403 Forbidden")
        fs = SignedURLFileSystem(base_fs, {}, "https://example.com")
        with pytest.raises(FileNotFoundError, match="File not found"):
            fs._open("x")


class TestSignedURLFileSystemCatFile:
    """Test cat_file."""

    def test_cat_file_success(self) -> None:
        """cat_file success returns content from base fs."""
        base_fs = Mock()
        base_fs.cat_file.return_value = b"content"
        fs = SignedURLFileSystem(base_fs, {"Policy": "p"}, "https://example.com")
        result = fs.cat_file("path/to/file")
        assert result == b"content"
        base_fs.cat_file.assert_called_once()
        assert "path/to/file" in base_fs.cat_file.call_args[0][0] or "path" in str(
            base_fs.cat_file.call_args
        )

    def test_cat_file_with_start_end(self) -> None:
        """cat_file passes start/end to base fs."""
        base_fs = Mock()
        base_fs.cat_file.return_value = b"xx"
        fs = SignedURLFileSystem(base_fs, {}, "https://example.com")
        fs.cat_file("f", start=0, end=10)
        base_fs.cat_file.assert_called_once()
        assert base_fs.cat_file.call_args[1]["start"] == 0
        assert base_fs.cat_file.call_args[1]["end"] == 10

    def test_cat_file_403_raises_file_not_found(self) -> None:
        """cat_file 403 raises FileNotFoundError."""
        base_fs = Mock()
        base_fs.cat_file.side_effect = Exception("403")
        fs = SignedURLFileSystem(base_fs, {}, "https://example.com")
        with pytest.raises(FileNotFoundError):
            fs.cat_file("x")


class TestSignedURLFileSystemCatFileAsync:
    """Test _cat_file (delegates to cat_file)."""

    def test_cat_file_delegates_to_cat_file(self) -> None:
        """_cat_file calls cat_file with same args."""
        base_fs = Mock()
        base_fs.cat_file.return_value = b"data"
        fs = SignedURLFileSystem(base_fs, {}, "https://example.com")
        result = fs._cat_file("a", start=1, end=2)
        assert result == b"data"
        base_fs.cat_file.assert_called_once()
        assert base_fs.cat_file.call_args[1]["start"] == 1
        assert base_fs.cat_file.call_args[1]["end"] == 2


class TestSignedURLFileSystemInfo:
    """Test info."""

    def test_info_success(self) -> None:
        """info returns metadata from base fs."""
        base_fs = Mock()
        base_fs.info.return_value = {"size": 100, "type": "file"}
        fs = SignedURLFileSystem(base_fs, {}, "https://example.com")
        result = fs.info("path")
        assert result == {"size": 100, "type": "file"}

    def test_info_403_raises_file_not_found(self) -> None:
        """info 403 raises FileNotFoundError."""
        base_fs = Mock()
        base_fs.info.side_effect = Exception("Forbidden")
        fs = SignedURLFileSystem(base_fs, {}, "https://example.com")
        with pytest.raises(FileNotFoundError):
            fs.info("x")


class TestSignedURLFileSystemExists:
    """Test exists."""

    def test_exists_true(self) -> None:
        """exists returns True when base fs returns True."""
        base_fs = Mock()
        base_fs.exists.return_value = True
        fs = SignedURLFileSystem(base_fs, {}, "https://example.com")
        assert fs.exists("path") is True

    def test_exists_403_returns_false(self) -> None:
        """exists returns False when base raises 403."""
        base_fs = Mock()
        base_fs.exists.side_effect = Exception("403 Forbidden")
        fs = SignedURLFileSystem(base_fs, {}, "https://example.com")
        assert fs.exists("path") is False

    def test_exists_false_when_base_returns_false(self) -> None:
        """exists returns False when base fs returns False."""
        base_fs = Mock()
        base_fs.exists.return_value = False
        fs = SignedURLFileSystem(base_fs, {}, "https://example.com")
        assert fs.exists("path") is False

    def test_exists_other_exception_reraises(self) -> None:
        """exists re-raises non-403 exceptions."""
        base_fs = Mock()
        base_fs.exists.side_effect = RuntimeError("network error")
        fs = SignedURLFileSystem(base_fs, {}, "https://example.com")
        with pytest.raises(RuntimeError, match="network error"):
            fs.exists("path")


class TestCreateCloudfrontMapper:
    """Test create_cloudfront_mapper."""

    def test_create_cloudfront_mapper_builds_mapper(self) -> None:
        """create_cloudfront_mapper parses URL and returns FSMap with SignedURLFileSystem."""
        signed_url = "https://cdn.example.com/bucket/path?Policy=pol&Signature=sig&Key-Pair-Id=kid"
        with patch("earth2studio.serve.client.fsspec_utils.fsspec") as mock_fsspec:
            mock_fs = Mock()
            mock_fsspec.filesystem.return_value = mock_fs
            mock_map = Mock()
            mock_fsspec.mapping.FSMap = mock_map
            result = create_cloudfront_mapper(signed_url)
        mock_fsspec.filesystem.assert_called_once_with("https")
        mock_map.assert_called_once()
        call_kwargs = mock_map.call_args[1]
        assert call_kwargs["root"] == ""
        assert call_kwargs["check"] is False
        assert call_kwargs["create"] is False
        assert isinstance(call_kwargs["fs"], SignedURLFileSystem)
        assert result is mock_map.return_value

    def test_create_cloudfront_mapper_with_zarr_path(self) -> None:
        """create_cloudfront_mapper with zarr_path appends to base URL."""
        signed_url = "https://cdn.example.com/bucket?Policy=p&Signature=s&Key-Pair-Id=k"
        with patch("earth2studio.serve.client.fsspec_utils.fsspec") as mock_fsspec:
            mock_fs = Mock()
            mock_fsspec.filesystem.return_value = mock_fs
            mock_map = Mock()
            mock_fsspec.mapping.FSMap = mock_map
            create_cloudfront_mapper(signed_url, zarr_path="results.zarr")
        fs_arg = mock_map.call_args[1]["fs"]
        assert isinstance(fs_arg, SignedURLFileSystem)
        assert "results.zarr" in fs_arg._base_url

    def test_create_cloudfront_mapper_strips_trailing_wildcard(self) -> None:
        """URL path with trailing /* or * is stripped."""
        signed_url = (
            "https://cdn.example.com/bucket/*?Policy=p&Signature=s&Key-Pair-Id=k"
        )
        with patch("earth2studio.serve.client.fsspec_utils.fsspec") as mock_fsspec:
            mock_fs = Mock()
            mock_fsspec.filesystem.return_value = mock_fs
            mock_fsspec.mapping.FSMap = Mock()
            create_cloudfront_mapper(signed_url)
        fs_arg = mock_fsspec.mapping.FSMap.call_args[1]["fs"]
        assert "*" not in fs_arg._base_url
        assert fs_arg._base_url.rstrip("/").endswith("bucket")


class TestParseAzureRemotePath:
    """Test _parse_azure_remote_path."""

    def test_container_and_prefix(self) -> None:
        c, p = fsspec_utils_mod._parse_azure_remote_path(
            "azure://mycontainer/outputs/wf/exec"
        )
        assert c == "mycontainer"
        assert p == "outputs/wf/exec"

    def test_container_only(self) -> None:
        c, p = fsspec_utils_mod._parse_azure_remote_path("azure://mycontainer")
        assert c == "mycontainer"
        assert p == ""

    def test_invalid_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="azure://"):
            fsspec_utils_mod._parse_azure_remote_path("s3://bucket/key")


class TestCreateAzureBlobMapper:
    """Test create_azure_blob_mapper."""

    def test_builds_fsmap_with_adlfs(self) -> None:
        mock_fs_inst = Mock()
        mock_abfs = Mock(return_value=mock_fs_inst)
        mock_fsmap = Mock()
        cred = object()
        with patch("adlfs.AzureBlobFileSystem", mock_abfs):
            with patch(
                "earth2studio.serve.client.fsspec_utils.fsspec.mapping.FSMap",
                mock_fsmap,
            ):
                out = create_azure_blob_mapper(
                    "acct",
                    "cont",
                    "prefix/p",
                    zarr_path="results.zarr",
                    credential=cred,
                )
        mock_abfs.assert_called_once_with(account_name="acct", credential=cred)
        mock_fsmap.assert_called_once()
        assert mock_fsmap.call_args[1]["root"] == "cont/prefix/p/results.zarr"
        assert mock_fsmap.call_args[1]["fs"] is mock_fs_inst
        assert out is mock_fsmap.return_value


class TestGetMapper:
    """Test get_mapper."""

    def test_get_mapper_server_returns_none(self) -> None:
        """get_mapper returns None for SERVER storage."""
        result = InferenceRequestResults(
            request_id="r1",
            status=RequestStatus.COMPLETED,
            output_files=[],
            completion_time=datetime.now(),
            storage_type=StorageType.SERVER,
        )
        assert get_mapper(result) is None

    def test_get_mapper_s3_with_signed_url_returns_mapper(self) -> None:
        """get_mapper with S3 and signed_url returns mapper from create_cloudfront_mapper."""
        result = InferenceRequestResults(
            request_id="r1",
            status=RequestStatus.COMPLETED,
            output_files=[],
            completion_time=datetime.now(),
            storage_type=StorageType.S3,
            signed_url="https://cdn.example.com/path?Policy=p&Signature=s&Key-Pair-Id=k",
        )
        with patch(
            "earth2studio.serve.client.fsspec_utils.create_cloudfront_mapper"
        ) as mock_create:
            mock_create.return_value = Mock()
            out = get_mapper(result)
            assert out is mock_create.return_value
            mock_create.assert_called_once_with(result.signed_url, "")

    def test_get_mapper_s3_with_zarr_path_passes_through(self) -> None:
        """get_mapper passes zarr_path to create_cloudfront_mapper."""
        result = InferenceRequestResults(
            request_id="r1",
            status=RequestStatus.COMPLETED,
            output_files=[],
            completion_time=datetime.now(),
            storage_type=StorageType.S3,
            signed_url="https://cdn.example.com?Policy=p&Signature=s&Key-Pair-Id=k",
        )
        with patch(
            "earth2studio.serve.client.fsspec_utils.create_cloudfront_mapper"
        ) as mock_create:
            mock_create.return_value = Mock()
            get_mapper(result, zarr_path="results.zarr")
            mock_create.assert_called_once_with(result.signed_url, "results.zarr")

    def test_get_mapper_s3_without_signed_url_raises(self) -> None:
        """get_mapper with S3 and no signed_url raises ValueError."""
        result = InferenceRequestResults(
            request_id="r1",
            status=RequestStatus.COMPLETED,
            output_files=[],
            completion_time=datetime.now(),
            storage_type=StorageType.S3,
            signed_url=None,
        )
        with pytest.raises(ValueError, match="S3 storage type requires a signed URL"):
            get_mapper(result)

    def test_get_mapper_azure_with_remote_path_and_account_returns_mapper(self) -> None:
        """get_mapper with AZURE calls create_azure_blob_mapper with parsed paths."""
        result = InferenceRequestResults(
            request_id="r1",
            status=RequestStatus.COMPLETED,
            output_files=[],
            completion_time=datetime.now(),
            storage_type=StorageType.AZURE,
            remote_path="azure://mycontainer/outputs/prefix",
            azure_account_name="myacct",
        )
        with patch(
            "earth2studio.serve.client.fsspec_utils.create_azure_blob_mapper"
        ) as mock_create:
            mock_create.return_value = Mock()
            out = get_mapper(result)
            assert out is mock_create.return_value
            mock_create.assert_called_once_with(
                "myacct",
                "mycontainer",
                "outputs/prefix",
                zarr_path="",
            )

    def test_get_mapper_azure_with_zarr_path_passes_through(self) -> None:
        """get_mapper passes zarr_path to create_azure_blob_mapper."""
        result = InferenceRequestResults(
            request_id="r1",
            status=RequestStatus.COMPLETED,
            output_files=[],
            completion_time=datetime.now(),
            storage_type=StorageType.AZURE,
            remote_path="azure://c/p",
            azure_account_name="a",
        )
        with patch(
            "earth2studio.serve.client.fsspec_utils.create_azure_blob_mapper"
        ) as mock_create:
            mock_create.return_value = Mock()
            get_mapper(result, zarr_path="results.zarr")
            mock_create.assert_called_once_with("a", "c", "p", zarr_path="results.zarr")

    def test_get_mapper_azure_account_from_blob_url(self) -> None:
        """get_mapper resolves account from blob_url when azure_account_name is absent."""
        result = InferenceRequestResults(
            request_id="r1",
            status=RequestStatus.COMPLETED,
            output_files=[],
            completion_time=datetime.now(),
            storage_type=StorageType.AZURE,
            remote_path="azure://cont/pre",
            azure_account_name=None,
            blob_url="https://myacct.blob.core.windows.net/cont/blob.nc",
        )
        with patch(
            "earth2studio.serve.client.fsspec_utils.create_azure_blob_mapper"
        ) as mock_create:
            mock_create.return_value = Mock()
            get_mapper(result)
            mock_create.assert_called_once_with("myacct", "cont", "pre", zarr_path="")

    def test_get_mapper_azure_without_remote_path_raises(self) -> None:
        """get_mapper with AZURE and no remote_path raises ValueError."""
        result = InferenceRequestResults(
            request_id="r1",
            status=RequestStatus.COMPLETED,
            output_files=[],
            completion_time=datetime.now(),
            storage_type=StorageType.AZURE,
            remote_path=None,
            azure_account_name="x",
        )
        with pytest.raises(ValueError, match="remote_path"):
            get_mapper(result)

    def test_get_mapper_azure_without_account_raises(self) -> None:
        """get_mapper with AZURE and no way to resolve account raises ValueError."""
        result = InferenceRequestResults(
            request_id="r1",
            status=RequestStatus.COMPLETED,
            output_files=[],
            completion_time=datetime.now(),
            storage_type=StorageType.AZURE,
            remote_path="azure://c/p",
            azure_account_name=None,
            blob_url=None,
        )
        with pytest.raises(ValueError, match="azure_account_name or blob_url"):
            get_mapper(result)

    def test_get_mapper_unsupported_storage_raises(self) -> None:
        """get_mapper with unsupported storage type raises ValueError."""
        mock_result = Mock()
        mock_result.storage_type = "unsupported_type"
        with pytest.raises(ValueError, match="Unsupported storage type"):
            get_mapper(mock_result)
