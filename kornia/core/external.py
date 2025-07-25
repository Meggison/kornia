# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import importlib
import logging
import subprocess
import sys
from types import ModuleType
from typing import List, Optional

from kornia.config import InstallationMode, kornia_config

logger = logging.getLogger(__name__)


class LazyLoader:
    """A class that implements lazy loading for Python modules.

    This class defers the import of a module until an attribute of the module is accessed.
    It helps in reducing the initial load time and memory usage of a script, especially when
    dealing with large or optional dependencies that might not be used in every execution.

    Attributes:
        module_name: The name of the module to be lazily loaded.
        module: The actual module object, initialized to None and loaded upon first access.

    """

    auto_install: bool = False

    def __init__(self, module_name: str, dev_dependency: bool = False) -> None:
        """Initialize the LazyLoader with the name of the module.

        Args:
            module_name: The name of the module to be lazily loaded.
            dev_dependency: If the dependency is required in the dev environment.
                If True, the module will be loaded in the dev environment.
                If False, the module will not be loaded in the dev environment.

        """
        self.module_name = module_name
        self.module: Optional[ModuleType] = None
        self.dev_dependency = dev_dependency
        # Avoid repeated hasattr/assignment for the auto_install flag
        self.auto_install = False

    def _install_package(self, module_name: str) -> None:
        logger.info(f"Installing `{module_name}` ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", module_name], shell=False, check=False)  # noqa: S603

    def _load(self) -> None:
        """Load the module if it hasn't been loaded yet.

        This method is called internally when an attribute of the module is accessed for the first time. It attempts to
        import the module and raises an ImportError with a custom message if the module is not installed.
        """
        # Fast exit for doctest and sphinx
        if not self.dev_dependency:
            # Micro-optimization: cache sys.argv to a local
            argv = sys.argv
            if _DOCTEST_ARG in argv:
                logger.info(f"Doctest detected, skipping loading of '{self.module_name}'")
                return
            if self._is_sphinx_build():
                logger.info(f"Sphinx detected, skipping loading of '{self.module_name}'")
                return

        # Only import once
        if self.module is not None:
            return

        try:
            self.module = importlib.import_module(self.module_name)
            return
        except ImportError as e:
            installation_mode = kornia_config.lazyloader.installation_mode
            # Fast-path for AUTO
            if installation_mode == _AUTO or self.auto_install:
                self._install_package(self.module_name)
            elif installation_mode == _ASK:
                # Prompt only if needed
                prompt = (
                    f"Optional dependency '{self.module_name}' is not installed. "
                    "You may silent this prompt by `kornia_config.lazyloader.installation_mode = 'auto'`. "
                    "Do you wish to install the dependency? [Y]es, [N]o, [A]ll."
                )
                valid_yes = {"y", "yes"}
                valid_no = {"n", "no"}
                valid_all = {"a", "all"}
                while True:
                    if_install = input(prompt)
                    lower = if_install.strip().lower()
                    if lower in valid_yes:
                        self._install_package(self.module_name)
                        break
                    elif lower in valid_all:
                        self.auto_install = True
                        self._install_package(self.module_name)
                        break
                    elif lower in valid_no:
                        raise ImportError(
                            f"Optional dependency '{self.module_name}' is not installed. "
                            f"Please install it to use this functionality."
                        ) from e
                    else:
                        prompt = "Invalid input. Please enter 'Y', 'N', or 'A'."
            elif installation_mode == _RAISE:
                raise ImportError(
                    f"Optional dependency '{self.module_name}' is not installed. "
                    f"Please install it to use this functionality."
                ) from e
            # After installation, try to import again
            self.module = importlib.import_module(self.module_name)

    def __getattr__(self, item: str) -> object:
        """Load the module (if not already loaded) and returns the requested attribute.

        This method is called when an attribute of the LazyLoader instance is accessed.
        It ensures that the module is loaded and then returns the requested attribute.

        Args:
            item: The name of the attribute to be accessed.

        Returns:
            The requested attribute of the loaded module.

        """
        self._load()
        return getattr(self.module, item)

    def __dir__(self) -> List[str]:
        """Load the module (if not already loaded) and returns the list of attributes of the module.

        This method is called when the built-in dir() function is used on the LazyLoader instance.
        It ensures that the module is loaded and then returns the list of attributes of the module.

        Returns:
            list: The list of attributes of the loaded module.

        """
        # Micro-optimization: shortcut if already loaded
        module = self.module
        if module is None:
            self._load()
            module = self.module
        return dir(module)

    def _is_sphinx_build(self):
        """Fast inline check for __sphinx_build__ global, without exception overhead unless present."""
        # Micro-optimization: Only check if symbol is present, avoiding try/except unless likely needed
        frame = sys._getframe(1)
        return frame.f_globals.get("__sphinx_build__", False)


# NOTE: This section is used for lazy loading of external modules. However, sphinx
#       would also try to support lazy loading of external modules. To avoid that, we
#       may set the module name to `autodoc_mock_imports` in conf.py to avoid undesired
#       installation of external modules.
numpy = LazyLoader("numpy", dev_dependency=True)
PILImage = LazyLoader("PIL.Image", dev_dependency=True)
onnx = LazyLoader("onnx", dev_dependency=True)
diffusers = LazyLoader("diffusers")
transformers = LazyLoader("transformers")
onnxruntime = LazyLoader("onnxruntime")
boxmot = LazyLoader("boxmot")
segmentation_models_pytorch = LazyLoader("segmentation_models_pytorch")
basicsr = LazyLoader("basicsr")
requests = LazyLoader("requests")
ivy = LazyLoader("ivy")

_DOCTEST_ARG = "--doctest-modules"

_AUTO = InstallationMode.AUTO

_ASK = InstallationMode.ASK

_RAISE = InstallationMode.RAISE
