# Resultados parciais: entrega 1

Benchmark `entrega1`: 10 cenários × 6 abordagens, seed 0, Apple M4 Pro com 8 threads. Os resultados são parciais (seed única) e serão consolidados na avaliação final. Os números brutos estão em [`results.json`](results.json), e o relatório gerado automaticamente, em [`report.md`](report.md).

O ganho é sempre calculado sobre a linha de base Concrete ML **mais rápida** de cada cenário.

## Resumo por cenário

| Família | Cenário | Concrete ML (melhor) | Custo por inferência, modo vazão (tempo do lote de 8 192) | Ganho | Latência de uma consulta | Ganho | Δ acurácia |
|---|---|---|---|---|---|---|---|
| Linear | regressão logística, breast_cancer | 18,7 ms | 20,5 µs (0,17 s) | **916×** | 24,7 ms | 0,8× | 0,0 p.p. |
| Linear | regressão logística, digits | 21,1 ms | 88,9 µs (0,73 s) | **238×** | 25,5 ms | 0,8× | +0,3 p.p. |
| Linear | SVM linear, breast_cancer | 17,2 ms | 20,5 µs (0,17 s) | **838×** | 22,4 ms | 0,8× | 0,0 p.p. |
| Rede neural | MLP 64→32→32→10 | 1,32 s | 393 µs (3,2 s) | **3 359×** | 171 ms | 7,7× | +2,2 p.p. |
| Rede neural | MLP 64→64→64→10 | 3,22 s | 901 µs (7,4 s) | **3 570×** | 180 ms | 17,8× | −1,7 p.p. |
| Rede neural | MLP 30→16→16→2 | 692 ms | 108 µs (0,88 s) | **6 431×** | 142 ms | 4,9× | 0,0 p.p. |
| Árvore | árvore de decisão | 418 ms | — | — | 239 ms (`tfhe_auto`) | 1,7× | −0,9 p.p. |
| Árvore | random forest | 5,10 s | — | — | 2,74 s (`tfhe_auto`) | 1,9× | −0,9 p.p. |
| Árvore | XGBoost | 2,77 s | — | — | 1,41 s (`tfhe_auto`) | 2,0× | +0,9 p.p. |
| KNN | k = 3 | não compila | — | — | — | — | — |

## Análise

- **Custo de servir:** cai duas a três ordens de grandeza nos lineares e nas redes. No MLP de referência, uma máquina passa de cerca de 2 700 para cerca de 9 milhões de inferências por hora. O ganho vem dos slots do CKKS: cada operação custa o mesmo com 1 ou com 8 192 amostras. O empacotamento de uma amostra por slot vem do CryptoNets. A contribuição do motor é a fusão da ativação, que faz as redes caberem num anel duas vezes menor.
- **Lote obrigatório:** o modo vazão só funciona com um lote, então uma consulta isolada espera o lote inteiro, de 0,17 s a 7,4 s.
- **Latência:** nas redes neurais, o modo latência é 4,9–17,8× mais rápido que o Concrete ML, e o ganho cresce com o modelo. O motor fica entre 140 e 180 ms, enquanto o TFHE acompanha o número de PBS.
- **Lineares por consulta:** não há ganho. Sem PBS, as duas abordagens ficam no custo fixo de cifrar e decifrar.
- **Árvores:** o ganho é de cerca de 2×. O otimizador escolheu 4 bits nos três modelos (o padrão é 6), com perda de no máximo 0,9 p.p.
- **KNN:** o Concrete ML 1.9 não encontra parâmetros (`NoParametersFound`), nem com 10 pontos de treino.
- **Fidelidade:** em 100% das amostras, a classe prevista sob criptografia coincide com a prevista em claro, em todos os cenários CKKS.

## Decomposição no modelo de referência

MLP 64→32→32→10, 360 amostras de teste.

| Configuração | Rotações | Profundidade | N | Custo | Acurácia |
|---|---|---|---|---|---|
| Concrete ML, `NeuralNetClassifier` (QAT, padrão) | — | — | — | 2,41 s por consulta | 93,6% |
| Concrete ML, PTQ ajustada | — | — | — | 1,32 s por consulta | 93,9% |
| CKKS, método diagonal | 81 | 9 | 32 768 | 816 ms por consulta | 96,1% |
| CKKS, motor, modo latência | 25 | 7 | 32 768 | 171 ms por consulta | 96,1% |
| CKKS, motor, modo vazão | 0 | 5 | 16 384 | 0,39 ms por inferência (lote em 3,2 s) | 96,1% |

## Ativação polinomial

Acurácia em claro no teste do MLP de referência, com 5 seeds (`python -m benchmark.ativacao`, dados em [`../ativacao.json`](../ativacao.json)).

| Ativação | Acurácia |
|---|---|
| ReLU (referência) | 96,7% ± 0,4 |
| x² | 94,4% ± 0,6 |
| **0,125x² + 0,5x + 0,25 (adotada)** | **95,8% ± 0,2** |
| a·x² + b·x + c (coeficientes treináveis) | 95,7% ± 0,9 |

## Limitações

- **Variância:** a seed é única, e o teste do breast_cancer tem 114 amostras, de modo que uma amostra vale 0,9 p.p.
- **Concrete ML em paralelo:** no modo vazão, o Concrete ML foi medido processando uma amostra por vez, como a biblioteca executa um lote. A execução de várias amostras em paralelo ainda não foi medida.
- **Cobertura do motor:** árvores e KNN ainda não são suportados.
- **Ativação fixa:** no MLP mais largo, ela perde 1,7 p.p. para a PTQ.
- **Aritmética:** CKKS e TFHE têm modelos de aritmética diferentes (aproximada e exata), embora ambos operem com 128 bits de segurança.
