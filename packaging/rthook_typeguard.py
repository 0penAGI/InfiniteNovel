# Runtime hook for frozen builds.
# typeguard instruments functions by re-parsing their source with inspect.getsource.
# Frozen modules have no .py source on disk, so that raises OSError and kills the app
# (observed via inflect inside TTS). Degrade gracefully: fall back to a "cannot
# instrument" result, which typeguard reports as a warning and skips.
import sys

if getattr(sys, "frozen", False):
    try:
        import typeguard._decorators as _tg

        _orig_instrument = _tg.instrument

        def _patched_instrument(f):
            try:
                return _orig_instrument(f)
            except Exception:
                return "could not instrument function in frozen application"

        _tg.instrument = _patched_instrument
    except Exception:
        pass