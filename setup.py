import distutils.core
import Cython.Build

distutils.core.setup(
    ext_modules=Cython.Build.cythonize(["lib/carSetup.pyx", "lib/cameraModule.pyx", "lib/controller.pyx"])
)
