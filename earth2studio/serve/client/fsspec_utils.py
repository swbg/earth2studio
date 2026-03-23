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


from typing import Any, NoReturn
from urllib.parse import parse_qs, urlencode, urlparse

import fsspec

from earth2studio.serve.client.models import InferenceRequestResults, StorageType


class SignedURLFileSystem(fsspec.AbstractFileSystem):
    """
    FSSpec filesystem that appends signed params to requests.

    Wraps a base HTTP filesystem and URL; converts 403 responses to
    FileNotFoundError for compatibility with consumers like Zarr.
    """

    def __init__(
        self, base_fs: Any, query_params: dict[str, str], base_url: str
    ) -> None:
        """
        Wrap a base filesystem and URL with query params for signed requests.

        Parameters
        ----------
        base_fs : Any
            Base fsspec filesystem (e.g. HTTP).
        query_params : dict[str, str]
            Query parameters to append (e.g. Policy, Signature, Key-Pair-Id).
        base_url : str
            Base URL for the store.
        """
        super().__init__()
        self._fs = base_fs
        self._query_params = query_params
        self._base_url = base_url
        self._query_string = urlencode(query_params, safe="~")

    def _make_signed_path(self, path: str) -> str:
        """
        Return the path with signed query parameters appended.

        Parameters
        ----------
        path : str
            Relative or absolute path/URL.

        Returns
        -------
        str
            Full URL with signed query string.
        """
        if path.startswith("http"):
            full_url = path
        else:
            clean_path = path.lstrip("/")
            full_url = (
                f"{self._base_url}/{clean_path}" if clean_path else self._base_url
            )
        separator = "&" if "?" in full_url else "?"
        return f"{full_url}{separator}{self._query_string}"

    def _handle_403(self, e: BaseException, path: str) -> "NoReturn":
        """
        Convert 403 errors to FileNotFoundError; re-raise others.

        Parameters
        ----------
        e : BaseException
            Caught exception.
        path : str
            Path that was accessed (for error message).
        """
        error_str = str(e).lower()
        if "403" in str(e) or "forbidden" in error_str:
            raise FileNotFoundError(f"File not found: {path}") from None
        raise e

    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        """
        Open a path with signed URL; 403 is converted to FileNotFoundError.

        Parameters
        ----------
        path : str
            Path or URL to open.
        mode : str, optional
            Open mode (e.g. "rb"). Default is "rb".
        **kwargs : Any
            Passed to the underlying filesystem.

        Returns
        -------
        Any
            File-like object from the base filesystem.

        Raises
        ------
        FileNotFoundError
            If the server returns 403 (forbidden).
        """
        try:
            return self._fs._open(self._make_signed_path(path), mode=mode, **kwargs)
        except Exception as e:
            self._handle_403(e, path)

    def cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> Any:
        """
        Read file contents with signed URL; 403 is converted to FileNotFoundError.

        Parameters
        ----------
        path : str
            Path or URL to read.
        start : int, optional
            Start byte offset.
        end : int, optional
            End byte offset.
        **kwargs : Any
            Passed to the underlying filesystem.

        Returns
        -------
        Any
            File contents (typically bytes).

        Raises
        ------
        FileNotFoundError
            If the server returns 403.
        """
        try:
            return self._fs.cat_file(
                self._make_signed_path(path), start=start, end=end, **kwargs
            )
        except Exception as e:
            self._handle_403(e, path)

    def _cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> Any:
        """
        Async cat_file used by Zarr; delegates to cat_file.

        Parameters
        ----------
        path : str
            Path or URL to read.
        start : int, optional
            Start byte offset.
        end : int, optional
            End byte offset.
        **kwargs : Any
            Passed to cat_file.

        Returns
        -------
        Any
            File contents.
        """
        return self.cat_file(path, start=start, end=end, **kwargs)

    def info(self, path: str, **kwargs: Any) -> Any:
        """
        Return metadata for path with signed URL; 403 becomes FileNotFoundError.

        Parameters
        ----------
        path : str
            Path or URL.
        **kwargs : Any
            Passed to the underlying filesystem.

        Returns
        -------
        Any
            Metadata dict for the path.

        Raises
        ------
        FileNotFoundError
            If the server returns 403.
        """
        try:
            return self._fs.info(self._make_signed_path(path), **kwargs)
        except Exception as e:
            self._handle_403(e, path)

    def exists(self, path: str, **kwargs: Any) -> bool:
        """
        Return True if path exists; 403 is treated as not found.

        Parameters
        ----------
        path : str
            Path or URL to check.
        **kwargs : Any
            Passed to the underlying filesystem.

        Returns
        -------
        bool
            True if path exists, False if not found or 403.
        """
        try:
            return self._fs.exists(self._make_signed_path(path), **kwargs)
        except Exception as e:
            try:
                self._handle_403(e, path)
            except FileNotFoundError:
                return False


def create_cloudfront_mapper(signed_url: str, zarr_path: str = "") -> Any:
    """
    Create an fsspec mapper for a CloudFront signed Zarr URL.

    Parameters
    ----------
    signed_url : str
        CloudFront signed URL with Policy, Signature, and Key-Pair-Id query params.
    zarr_path : str, optional
        Subpath within the signed URL to the Zarr store. Default is "".

    Returns
    -------
    fsspec.mapping.FSMap
        Mapper suitable for use with xarray.open_zarr().
    """
    # Parse the URL
    parsed = urlparse(signed_url)

    # Extract query parameters
    query_params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

    # Get base path (remove trailing /* if present)
    base_path = parsed.path.rstrip("/*").rstrip("*")

    # Reconstruct base URL
    if zarr_path != "":
        base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}/{zarr_path}"
    else:
        base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}"

    # Create HTTP filesystem
    fs = fsspec.filesystem("https")

    signed_fs = SignedURLFileSystem(fs, query_params, base_url)
    mapper = fsspec.mapping.FSMap(root="", fs=signed_fs, check=False, create=False)

    return mapper


def _parse_azure_remote_path(remote_path: str) -> tuple[str, str]:
    """
    Parse ``azure://<container>/<blob_prefix>`` into container name and prefix.

    Parameters
    ----------
    remote_path : str
        URI from result metadata, e.g. ``azure://mycontainer/outputs/wf/exec``.

    Returns
    -------
    tuple[str, str]
        ``(container_name, blob_prefix)`` — prefix may be empty.
    """
    if not remote_path.startswith("azure://"):
        raise ValueError(
            f"Expected remote_path to start with 'azure://', got {remote_path!r}"
        )
    rest = remote_path[len("azure://") :]
    slash = rest.find("/")
    if slash == -1:
        return rest, ""
    return rest[:slash], rest[slash + 1 :]


def _azure_account_name_from_blob_url(blob_url: str) -> str | None:
    """Return storage account name from ``https://<account>.blob.<host>/...``."""
    parsed = urlparse(blob_url)
    host = parsed.netloc
    if ".blob." not in host:
        return None
    return host.split(".blob.", 1)[0]


def create_azure_blob_mapper(
    account_name: str,
    container_name: str,
    blob_prefix: str,
    zarr_path: str = "",
    credential: Any | None = None,
) -> Any:
    """
    Create an fsspec mapper for Azure Blob Storage using DefaultAzureCredential.

    Assumes the caller's identity (or configured credential chain) may read blobs under
    ``container_name/blob_prefix``. No SAS or server-issued signed URLs are used.

    Parameters
    ----------
    account_name : str
        Azure storage account name.
    container_name : str
        Blob container name.
    blob_prefix : str
        Key prefix for uploaded results (no leading slash).
    zarr_path : str, optional
        Subpath to the Zarr store under the prefix. Default is "".
    credential : Any, optional
        Azure credential, e.g. from ``azure.identity``. Default is DefaultAzureCredential.

    Returns
    -------
    fsspec.mapping.FSMap
        Mapper suitable for ``xarray.open_zarr()``.
    """
    try:
        from adlfs import AzureBlobFileSystem
        from azure.identity import DefaultAzureCredential
    except ImportError as e:
        raise ImportError(
            "Azure blob access requires 'adlfs' and 'azure-identity'. "
            "Install with the serve extra or: pip install adlfs azure-identity"
        ) from e

    if credential is None:
        credential = DefaultAzureCredential()

    segments = [container_name, blob_prefix.strip("/")]
    if zarr_path:
        segments.append(zarr_path.strip("/"))
    root = "/".join(s for s in segments if s)

    fs = AzureBlobFileSystem(account_name=account_name, credential=credential)
    return fsspec.mapping.FSMap(root=root, fs=fs, check=False, create=False)


def get_mapper(
    request_result: InferenceRequestResults, zarr_path: str = ""
) -> Any | None:
    """
    Return an fsspec mapper for the result, or None for SERVER storage.

    Parameters
    ----------
    request_result : InferenceRequestResults
        For S3, ``signed_url`` (CloudFront). For Azure, ``remote_path`` (``azure://...``)
        plus ``azure_account_name`` or ``blob_url`` to resolve the account.
    zarr_path : str, optional
        Subpath to the Zarr store within the result. Default is "".

    Returns
    -------
    fsspec.mapping.FSMap or None
        Mapper for S3 (HTTPS signed URL) or Azure (DefaultAzureCredential), or None
        for SERVER storage.

    Raises
    ------
    ValueError
        If required fields for the storage type are missing, or storage type is unsupported.
    """
    if request_result.storage_type == StorageType.S3:
        if request_result.signed_url is None:
            raise ValueError("S3 storage type requires a signed URL")
        return create_cloudfront_mapper(request_result.signed_url, zarr_path)
    elif request_result.storage_type == StorageType.AZURE:
        if not request_result.remote_path:
            raise ValueError(
                "Azure storage type requires remote_path in result metadata"
            )
        account = request_result.azure_account_name
        if not account and request_result.blob_url:
            account = _azure_account_name_from_blob_url(request_result.blob_url)
        if not account:
            raise ValueError(
                "Azure storage requires azure_account_name or blob_url in result metadata "
                "to locate the storage account"
            )
        container_name, blob_prefix = _parse_azure_remote_path(
            request_result.remote_path
        )
        return create_azure_blob_mapper(
            account, container_name, blob_prefix, zarr_path=zarr_path
        )
    elif request_result.storage_type == StorageType.SERVER:
        return None
    else:
        raise ValueError(f"Unsupported storage type: {request_result.storage_type}")
