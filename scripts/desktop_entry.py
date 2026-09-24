"""Frozen backend entrypoint. Multiprocessing hooks must precede server startup."""

import multiprocessing

from desktop_backend.server import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
