# Redução do custo da inferência cifrada nos modelos do Concrete ML

Pesquisa acadêmica sobre o custo de servir modelos de aprendizado de máquina sobre dados cifrados com criptografia totalmente homomórfica (FHE). Partimos dos modelos do **Concrete ML**, da Zama, e propomos um motor de compilação em CKKS (`fheopt`) que reduz esse custo nas quatro famílias da biblioteca: modelos lineares, árvores, redes neurais e vizinhos mais próximos.

## Problema

Com FHE, o servidor executa o modelo sem ver a entrada nem a saída, o que viabiliza inferência como serviço sobre dados sensíveis. O obstáculo é o custo: no Concrete ML, cada inferência ocupa uma máquina inteira por milissegundos a segundos. No MLP de referência deste trabalho, isso dá cerca de 2 700 inferências por hora por máquina.

## Contexto

A [Zama](https://www.zama.ai) mantém o ecossistema TFHE de código aberto (TFHE-rs, Concrete e Concrete ML). O Concrete ML 1.9 é a linha de base desta pesquisa. No TFHE, cada operação não linear exige um bootstrapping programável (PBS), que é caro, e cada valor é cifrado separadamente. Por isso, o custo cresce com o número de PBS e com o número de amostras.

## Proposta

Levamos os modelos para o esquema CKKS, que cifra vetores de milhares de valores (*slots*) e opera sobre todos ao mesmo tempo. A ReLU é trocada por um polinômio de grau 2.

O motor `fheopt` tem dois modos:
- **vazão:** uma amostra por slot, com 8 192 amostras por lote e sem rotações;
- **latência:** uma consulta isolada, com o mínimo de rotações.

Cada otimização é um passe medido separadamente. A **fusão da ativação** absorve os coeficientes do polinômio na camada seguinte e reduz a profundidade de 7 para 5, o que faz as redes caberem num anel duas vezes menor. O **layout replicado** encadeia as camadas sem reorganizar os slots e reduz as rotações de 81 para 25.

Árvores e KNN dependem de comparações, que o motor ainda não suporta. Por enquanto, as árvores são otimizadas dentro do próprio Concrete ML por um seletor automático de precisão e probabilidade de erro.

## Pergunta e hipóteses

Quanto do custo da inferência cifrada dos modelos do Concrete ML pode ser eliminado com CKKS, mantendo 128 bits de segurança e a acurácia?

- **H1:** no modo vazão, o custo por inferência cai pelo menos duas ordens de grandeza.
- **H2:** no modo latência, uma consulta fica mais rápida nas redes neurais, e o ganho cresce com o modelo.
- **H3:** reduzir a profundidade permite anéis menores e reduz ainda mais o custo, com perda de no máximo 1 p.p.

## Metodologia

São 10 cenários fixos, iguais em todas as entregas:
- 3 lineares, 3 árvores, 3 MLPs e 1 KNN;
- nos conjuntos *digits* e *breast_cancer*;
- com divisão 60/20/20, estratificada, com seed 0.

As linhas de base são `concrete_ml`, com os padrões da biblioteca (QAT nas redes), e `concrete_ml_ptq`, a configuração mais rápida encontrada para as redes. As abordagens propostas são `fheopt_vazao`, `fheopt` e `tfhe_auto`, e `fheopt_diagonal` serve de referência em CKKS. O ganho é medido sobre a linha de base mais rápida de cada cenário.

As métricas são:
- o custo por inferência em lote;
- a latência de uma consulta;
- a acurácia;
- a fidelidade em relação à inferência em claro.

As medições usam o OpenFHE 1.6.1, num Apple M4 Pro com 8 threads.

## Resultados

Os resultados de cada entrega ficam em `results/<entrega>/README.md`. Entrega 1: **[results/entrega1/README.md](results/entrega1/README.md)**.

## Como reproduzir

Requisitos: macOS em Apple Silicon, Python 3.12, CMake e `libomp` (Homebrew).

```bash
python3.12 -m venv .venv
CMAKE_POLICY_VERSION_MINIMUM=3.5 .venv/bin/pip install -r requirements.txt
scripts/build_openfhe.sh       # o OpenFHE não tem pacote pip para macOS arm64
source env.sh

python -m benchmark.run --list                 # cenários e abordagens
python -m benchmark.run --tag entrega1         # rodada completa (~40 min)
python -m benchmark.run --tag teste --quick    # verificação rápida
python -m benchmark.report entrega1 entrega2   # comparativo entre entregas
python -m benchmark.ativacao                   # estudo da ativação polinomial
```

Cada caso roda num subprocesso próprio, e uma rodada interrompida retoma de onde parou. A cada entrega, a nova versão do motor é registrada em `benchmark/approaches.py` e o benchmark roda com uma nova tag. Os cenários existentes nunca são alterados.

## Estrutura

```
fheopt/          motor de compilação CKKS
  vazao.py         modo vazão e fusão da ativação
  compiler.py      modo latência: layouts diagonal e replicado
  backend.py       simulador de slots e backend OpenFHE
  nn.py            ativação polinomial e treino (PyTorch)
benchmark/       medições
  scenarios.py     cenários fixos e conjuntos de dados
  approaches.py    linhas de base Concrete ML e abordagens propostas
  run.py           execução (subprocessos, retomada, timeout)
  report.py        relatórios por entrega e comparativos
  ativacao.py      estudo da ativação polinomial
results/         resultados por entrega (análise, relatório e dados brutos)
scripts/         compilação do OpenFHE
```

## Status

**Entrega 1 (concluída):**
- formalização da pesquisa;
- benchmark nas 4 famílias do Concrete ML;
- motor CKKS nos modos vazão e latência;
- seletor automático para as árvores.

**Entrega 2 (planejada):**
- medir o Concrete ML com amostras em paralelo;
- perfilar o custo por operação;
- levar a fusão da ativação ao modo latência.

## Referências

- Cheon, J. H.; Kim, A.; Kim, M.; Song, Y. Homomorphic Encryption for Arithmetic of Approximate Numbers. *ASIACRYPT*, 2017.
- Chillotti, I.; Gama, N.; Georgieva, M.; Izabachène, M. TFHE: Fast Fully Homomorphic Encryption over the Torus. *Journal of Cryptology*, 2020.
- Chillotti, I.; Joye, M.; Paillier, P. Programmable Bootstrapping Enables Efficient Homomorphic Inference of Deep Neural Networks. *CSCML*, 2021.
- Stoian, A. et al. Deep Neural Networks for Encrypted Inference with TFHE. *CSCML*, 2023.
- Zama. Concrete ML. https://github.com/zama-ai/concrete-ml
- Gilad-Bachrach, R. et al. CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy. *ICML*, 2016.
- Halevi, S.; Shoup, V. Algorithms in HElib. *CRYPTO*, 2014.
- Juvekar, C.; Vaikuntanathan, V.; Chandrakasan, A. GAZELLE: A Low Latency Framework for Secure Neural Network Inference. *USENIX Security*, 2018.
- Al Badawi, A. et al. OpenFHE: Open-Source Fully Homomorphic Encryption Library. *WAHC*, 2022.
- Albrecht, M. et al. Homomorphic Encryption Standard. In: *Protecting Privacy through Homomorphic Encryption*, Springer, 2021.
