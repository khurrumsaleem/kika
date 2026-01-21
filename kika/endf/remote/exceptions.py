"""Custom exceptions for ENDF remote download functionality."""


class ENDFRemoteError(Exception):
    """Base exception for ENDF remote operations."""

    pass


class IsotopeNotFoundError(ENDFRemoteError):
    """Raised when the requested isotope is not found in the library."""

    def __init__(self, isotope: str, library: str):
        self.isotope = isotope
        self.library = library
        super().__init__(f"Isotope '{isotope}' not found in library '{library}'")


class LibraryNotFoundError(ENDFRemoteError):
    """Raised when the requested library is not recognized."""

    def __init__(self, library: str, available: list[str]):
        self.library = library
        self.available = available
        super().__init__(
            f"Unknown library '{library}'. Available libraries: {', '.join(available)}"
        )


class NetworkError(ENDFRemoteError):
    """Raised when a network operation fails."""

    def __init__(self, message: str, url: str | None = None):
        self.url = url
        super().__init__(message)


class CacheError(ENDFRemoteError):
    """Raised when a cache operation fails."""

    pass
