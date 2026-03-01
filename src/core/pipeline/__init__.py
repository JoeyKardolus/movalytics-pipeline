"""Pipeline orchestration module.

Provides high-level pipeline utilities:
- cleanup: Output directory organization
"""

from .cleanup import cleanup_output_directory

__all__ = [
    "cleanup_output_directory",
]
