"""IAEA Nuclear Data Service client for downloading ENDF files."""

import io
import re
import zipfile

import httpx

from kika._constants import (
    ATOMIC_NUMBER_TO_SYMBOL,
    SYMBOL_TO_ATOMIC_NUMBER,
    ZAID_TO_ENDF_MAT,
)

from .cache import ENDFCache, get_cache
from .constants import (
    IAEA_BASE_URL,
    get_library_path,
    list_available_libraries,
    normalize_library_name,
)
from .exceptions import (
    IsotopeNotFoundError,
    LibraryNotFoundError,
    NetworkError,
)


def parse_isotope(isotope: str | int) -> tuple[int, int, str]:
    """
    Parse an isotope specification into Z, A, and element symbol.

    Supports formats:
    - "Fe56", "Fe-56", "fe56" (element symbol + mass)
    - "U235", "U-235", "u235"
    - 26056 (ZAID integer)
    - "26056" (ZAID string)

    Parameters
    ----------
    isotope : str or int
        Isotope specification

    Returns
    -------
    tuple[int, int, str]
        (Z, A, symbol) e.g., (26, 56, "Fe")
    """
    if isinstance(isotope, int) or (isinstance(isotope, str) and isotope.isdigit()):
        # ZAID format: ZZZAAA
        zaid = int(isotope)
        z = zaid // 1000
        a = zaid % 1000
        symbol = ATOMIC_NUMBER_TO_SYMBOL.get(z)
        if symbol is None:
            raise ValueError(f"Unknown atomic number: {z}")
        return z, a, symbol

    # String format: Element + mass (e.g., "Fe56", "Fe-56", "U235")
    isotope = isotope.strip()

    # Try to match element-mass pattern
    match = re.match(r"([A-Za-z]+)-?(\d+)", isotope)
    if match:
        symbol = match.group(1).capitalize()
        a = int(match.group(2))
        z = SYMBOL_TO_ATOMIC_NUMBER.get(symbol)
        if z is None:
            raise ValueError(f"Unknown element symbol: {symbol}")
        return z, a, symbol

    raise ValueError(f"Cannot parse isotope: {isotope}")


def build_iaea_url(z: int, a: int, symbol: str, library: str, particle: str = "n") -> str:
    """
    Build the IAEA download URL for an ENDF file.

    URL pattern:
    https://www-nds.iaea.org/public/download-endf/{LIBRARY}/{particle}/n_{ZZZ}-{Element}-{Mass}_{MAT}.zip

    Parameters
    ----------
    z : int
        Atomic number
    a : int
        Mass number
    symbol : str
        Element symbol
    library : str
        Library name
    particle : str
        Particle type (default: "n")

    Returns
    -------
    str
        Full download URL
    """
    library_path = get_library_path(library)
    zaid = z * 1000 + a

    # Get MAT number from ZAID
    mat = ZAID_TO_ENDF_MAT.get(zaid)
    if mat is None:
        raise ValueError(f"No MAT number found for ZAID {zaid}")

    # Format: n_ZZZ-Element-Mass_MAT.zip
    filename = f"{particle}_{z:03d}-{symbol}-{a}_{mat}.zip"
    return f"{IAEA_BASE_URL}/{library_path}/{particle}/{filename}"


class IAEAClient:
    """Client for downloading ENDF files from IAEA Nuclear Data Service."""

    def __init__(
        self,
        cache: ENDFCache | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the IAEA client.

        Parameters
        ----------
        cache : ENDFCache, optional
            Cache instance. If None, uses the global cache.
        timeout : float
            Request timeout in seconds (default: 30.0)
        """
        self.cache = cache or get_cache()
        self.timeout = timeout

    def download(
        self,
        isotope: str | int,
        library: str = "endfb8.1",
        particle: str = "n",
        use_cache: bool = True,
        force_download: bool = False,
    ) -> bytes:
        """
        Download an ENDF file from IAEA.

        Parameters
        ----------
        isotope : str or int
            Isotope specification (e.g., "Fe56", 26056, "U-235")
        library : str
            Library name (default: "endfb8.1")
        particle : str
            Particle type (default: "n")
        use_cache : bool
            Whether to use local cache (default: True)
        force_download : bool
            Force re-download even if cached (default: False)

        Returns
        -------
        bytes
            ENDF file content

        Raises
        ------
        LibraryNotFoundError
            If the library is not recognized
        IsotopeNotFoundError
            If the isotope is not found in the library
        NetworkError
            If the download fails
        """
        # Validate and normalize library
        try:
            canonical_lib = normalize_library_name(library)
        except KeyError:
            raise LibraryNotFoundError(library, list_available_libraries())

        # Parse isotope
        try:
            z, a, symbol = parse_isotope(isotope)
        except ValueError as e:
            raise IsotopeNotFoundError(str(isotope), library) from e

        zaid = z * 1000 + a

        # Check cache first
        if use_cache and not force_download:
            cached_path = self.cache.get(zaid, canonical_lib, particle)
            if cached_path:
                return cached_path.read_bytes()

        # Build URL and download
        try:
            url = build_iaea_url(z, a, symbol, canonical_lib, particle)
        except ValueError as e:
            raise IsotopeNotFoundError(str(isotope), library) from e

        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
        except httpx.TimeoutException:
            raise NetworkError(f"Request timed out after {self.timeout}s", url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise IsotopeNotFoundError(str(isotope), library) from e
            raise NetworkError(f"HTTP error {e.response.status_code}: {e}", url) from e
        except httpx.RequestError as e:
            raise NetworkError(f"Request failed: {e}", url) from e

        # Extract ENDF file from ZIP
        try:
            content = self._extract_endf_from_zip(response.content)
        except Exception as e:
            raise NetworkError(f"Failed to extract ENDF from ZIP: {e}", url) from e

        # Cache the result
        if use_cache:
            self.cache.put(zaid, canonical_lib, content, particle)

        return content

    def _extract_endf_from_zip(self, zip_content: bytes) -> bytes:
        """Extract the ENDF file from a ZIP archive."""
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            # Find the ENDF file in the archive
            for name in zf.namelist():
                # Skip directories
                if name.endswith("/"):
                    continue
                # The ENDF file is typically the only file or has no extension
                if not name.endswith((".zip", ".gz", ".tar")):
                    return zf.read(name)
            raise ValueError("No ENDF file found in ZIP archive")


# Module-level client instance
_client = None


def get_client() -> IAEAClient:
    """Get the global IAEA client instance."""
    global _client
    if _client is None:
        _client = IAEAClient()
    return _client
