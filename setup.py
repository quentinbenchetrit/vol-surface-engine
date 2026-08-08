"""Build the optional C++ kernel.

The extension is a speed-up, not a dependency: if pybind11 is missing or the
compiler fails, the build still succeeds and the package falls back to the
NumPy implementation at import time. That keeps `pip install` working on a
machine with no toolchain, and it is also what makes the parity test possible,
since both implementations stay present.
"""
from __future__ import annotations

import sys

from setuptools import setup

ext_modules = []
cmdclass = {}

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext_modules = [
        Pybind11Extension(
            "volsurface._core",
            ["src/volsurface/_core.cpp"],
            cxx_std=17,
            extra_compile_args=["-O3"] if sys.platform != "win32" else ["/O2"],
        )
    ]

    class OptionalBuildExt(build_ext):
        """Let the wheel build succeed even when the compiler does not."""

        def run(self):
            try:
                super().run()
            except Exception as err:  # noqa: BLE001
                print(f"warning: skipping the C++ kernel, falling back to NumPy ({err})")

        def build_extension(self, ext):
            try:
                super().build_extension(ext)
            except Exception as err:  # noqa: BLE001
                print(f"warning: could not build {ext.name}, falling back to NumPy ({err})")

    cmdclass = {"build_ext": OptionalBuildExt}
except ImportError:
    print("warning: pybind11 not available, building without the C++ kernel")

setup(ext_modules=ext_modules, cmdclass=cmdclass)
