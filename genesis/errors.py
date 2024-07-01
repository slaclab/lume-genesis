class Genesis4RunFailure(Exception):
    """Genesis 4 failed to run."""


class NamelistAccessError(ValueError):
    """Error accessing namelist from the main input."""


class NoSuchNamelistError(NamelistAccessError):
    """No such namelist of the given type is in the main input."""


class MultipleNamelistsError(NamelistAccessError):
    """
    More than one namelist of the given type is defined.

    Access is ambiguous.
    """


class NotFlatError(Exception):
    """The beamline uses named elements more than once; it is not flat."""


class RecursiveLineError(Exception):
    """The beamline uses named elements recursively in lines."""
