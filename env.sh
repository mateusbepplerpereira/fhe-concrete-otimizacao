# Ativa o ambiente do projeto. Uso: source env.sh
#
# Contorno para macOS com SDK recente: o compilador Concrete chama `ld` do PATH com o
# SDK do CommandLineTools fixo no código (CompilerEngine.cpp:828). O ld do Xcode pode
# ser mais antigo que esse SDK ("tapi error: unknown architecture"), então colocamos
# o ld do CommandLineTools na frente do PATH.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_CLT_LD=/Library/Developer/CommandLineTools/usr/bin/ld
if [ "$(uname)" = "Darwin" ] && [ -x "$_CLT_LD" ]; then
  mkdir -p "$_ROOT/tools/bin"
  [ -L "$_ROOT/tools/bin/ld" ] || ln -s "$_CLT_LD" "$_ROOT/tools/bin/ld"
  export PATH="$_ROOT/tools/bin:$PATH"
fi
export PYTHONPATH="$_ROOT${PYTHONPATH:+:$PYTHONPATH}"
source "$_ROOT/.venv/bin/activate"
