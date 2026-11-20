#!/usr/bin/env bash
# Build reproduzível do OpenFHE (C++) e dos bindings Python para macOS arm64.
# O pip só distribui o OpenFHE para Linux x86_64, por isso compilamos do código-fonte.
#
# Uso: scripts/build_openfhe.sh          (JOBS=8 por padrão)
# Requisitos: cmake, libomp (brew install libomp), Python 3.12 em .venv.
# Resultado: biblioteca em tools/openfhe, módulo `openfhe` instalado no .venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/vendor"
PREFIX="$ROOT/tools/openfhe"
PY="$ROOT/.venv/bin/python"
OPENFHE_TAG=v1.6.1
OPENFHE_PY_TAG=v1.6.1.0.1
JOBS="${JOBS:-8}"

# OpenMP com o clang da Apple: aponta para a libomp do Homebrew
OMP="$(brew --prefix libomp)"
OMP_FLAGS=(
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include"
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include"
  -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp
  -DOpenMP_omp_LIBRARY="$OMP/lib/libomp.dylib"
)

mkdir -p "$SRC"
clone() {  # repo tag destino
  [ -d "$3/.git" ] || git clone -q --depth 1 --branch "$2" "https://github.com/openfheorg/$1.git" "$3"
}

# 1. Biblioteca C++
clone openfhe-development "$OPENFHE_TAG" "$SRC/openfhe-development"
# Só o cereal é necessário (testes, benchmarks e tcmalloc estão desligados); os demais
# submódulos são grandes e o clone automático do CMake pode travar.
git -C "$SRC/openfhe-development" submodule update --init --depth 1 third-party/cereal
cmake -S "$SRC/openfhe-development" -B "$SRC/openfhe-development/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DGIT_SUBMOD_AUTO=OFF \
  -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
  -DWITH_OPENMP=ON "${OMP_FLAGS[@]}"
cmake --build "$SRC/openfhe-development/build" --parallel "$JOBS"
cmake --install "$SRC/openfhe-development/build"

# 2. Bindings Python, instalados no site-packages do .venv
"$PY" -m pip install -q "pybind11[global]>=2.12"
clone openfhe-python "$OPENFHE_PY_TAG" "$SRC/openfhe-python"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')"
cmake -S "$SRC/openfhe-python" -B "$SRC/openfhe-python/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DPython_EXECUTABLE="$PY" -DPYTHON_EXECUTABLE="$PY" \
  -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)" \
  -DCMAKE_INSTALL_PREFIX="$SITE/openfhe" -DCMAKE_INSTALL_RPATH="$PREFIX/lib" "${OMP_FLAGS[@]}"
cmake --build "$SRC/openfhe-python/build" --parallel "$JOBS"
cmake --install "$SRC/openfhe-python/build"
# O CMake dos bindings fixa o rpath em @loader_path/lib; apontamos para a biblioteca em tools/
SO="$(ls "$SITE"/openfhe/openfhe*.so)"
otool -l "$SO" | grep -q "$PREFIX/lib" || install_name_tool -add_rpath "$PREFIX/lib" "$SO"
codesign -f -s - "$SO"

"$PY" -c "import openfhe; print('OK: openfhe importado de', openfhe.__file__)"
