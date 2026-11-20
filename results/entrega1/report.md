# Benchmark: entrega1

Atualizado em 2026-11-19T12:03:31 · commit `43c58cd-modificado` · arm64 26.6.2 · 8 threads · concrete-ml 1.9.0 · openfhe 1.6.1.0

Latência: média ± desvio por amostra (cifrar, avaliar e decifrar), após aquecimento. Acurácia FHE: simulação FHE no Concrete ML, execução real no CKKS. Ganho: em relação à linha de base Concrete ML mais rápida do mesmo cenário.

## Modelos lineares

| Cenário | Abordagem | Acurácia (claro) | Acurácia (FHE) | Latência (ms) | Ganho | Observação |
|---|---|---|---|---|---|---|
| linear/logreg_cancer | concrete_ml | 0.982 | 0.982 | 18.7 ± 0.3 | 1.0× | 0 PBS |
| linear/logreg_cancer | concrete_ml_ptq | — | — | — | — | não suportado: apenas redes neurais |
| linear/logreg_cancer | fheopt | 0.982 | 0.982 | 24.7 ± 1.4 | 0.8× | N=16384, prof. 1, 5 rotações |
| linear/logreg_cancer | fheopt_diagonal | 0.982 | 0.982 | 25.3 ± 0.5 | 0.7× | N=16384, prof. 1, 5 rotações |
| linear/logreg_cancer | fheopt_vazao | 0.982 | 0.982 | 0.020 ± 0.000 | 916.1× | vazão: lote de 8192 em 0.17 s; N=16384, prof. 1, 0 rotações |
| linear/logreg_cancer | tfhe_auto | — | — | — | — | não suportado: aplicado a árvores e KNN |
| linear/logreg_digits | concrete_ml | 0.961 | 0.961 | 21.1 ± 0.3 | 1.0× | 0 PBS |
| linear/logreg_digits | concrete_ml_ptq | — | — | — | — | não suportado: apenas redes neurais |
| linear/logreg_digits | fheopt | 0.964 | 0.964 | 25.5 ± 0.6 | 0.8× | N=16384, prof. 1, 6 rotações |
| linear/logreg_digits | fheopt_diagonal | 0.964 | 0.964 | 60.6 ± 5.6 | 0.3× | N=16384, prof. 1, 17 rotações |
| linear/logreg_digits | fheopt_vazao | 0.964 | 0.964 | 0.089 ± 0.001 | 237.5× | vazão: lote de 8192 em 0.73 s; N=16384, prof. 1, 0 rotações |
| linear/logreg_digits | tfhe_auto | — | — | — | — | não suportado: aplicado a árvores e KNN |
| linear/linsvc_cancer | concrete_ml | 0.965 | 0.965 | 17.2 ± 0.2 | 1.0× | 0 PBS |
| linear/linsvc_cancer | concrete_ml_ptq | — | — | — | — | não suportado: apenas redes neurais |
| linear/linsvc_cancer | fheopt | 0.965 | 0.965 | 22.4 ± 0.5 | 0.8× | N=16384, prof. 1, 5 rotações |
| linear/linsvc_cancer | fheopt_diagonal | 0.965 | 0.965 | 25.3 ± 1.1 | 0.7× | N=16384, prof. 1, 5 rotações |
| linear/linsvc_cancer | fheopt_vazao | 0.965 | 0.965 | 0.021 ± 0.000 | 838.4× | vazão: lote de 8192 em 0.17 s; N=16384, prof. 1, 0 rotações |
| linear/linsvc_cancer | tfhe_auto | — | — | — | — | não suportado: aplicado a árvores e KNN |

## Árvores

| Cenário | Abordagem | Acurácia (claro) | Acurácia (FHE) | Latência (ms) | Ganho | Observação |
|---|---|---|---|---|---|---|
| tree/dtree_cancer | concrete_ml | 0.965 | 0.965 | 418.4 ± 54.7 | 1.0× | 136 PBS |
| tree/dtree_cancer | concrete_ml_ptq | — | — | — | — | não suportado: apenas redes neurais |
| tree/dtree_cancer | fheopt | — | — | — | — | não suportado: comparações (árvores/KNN) não suportadas pelo motor CKKS |
| tree/dtree_cancer | fheopt_diagonal | — | — | — | — | não suportado: comparações (árvores/KNN) não suportadas pelo motor CKKS |
| tree/dtree_cancer | fheopt_vazao | — | — | — | — | não suportado: comparações (árvores/KNN) não suportadas pelo motor CKKS |
| tree/dtree_cancer | tfhe_auto | 0.956 | 0.956 | 239.1 ± 5.4 | 1.7× | 78 PBS; escolhido n_bits=4, p_error=padrão (padrão: 136 PBS) |
| tree/rf_cancer | concrete_ml | 0.965 | 0.965 | 5 103.8 ± 38.6 | 1.0× | 2180 PBS |
| tree/rf_cancer | concrete_ml_ptq | — | — | — | — | não suportado: apenas redes neurais |
| tree/rf_cancer | fheopt | — | — | — | — | não suportado: comparações (árvores/KNN) não suportadas pelo motor CKKS |
| tree/rf_cancer | fheopt_diagonal | — | — | — | — | não suportado: comparações (árvores/KNN) não suportadas pelo motor CKKS |
| tree/rf_cancer | fheopt_vazao | — | — | — | — | não suportado: comparações (árvores/KNN) não suportadas pelo motor CKKS |
| tree/rf_cancer | tfhe_auto | 0.956 | 0.956 | 2 744.7 ± 60.8 | 1.9× | 1840 PBS; escolhido n_bits=4, p_error=0.1 (padrão: 2180 PBS) |
| tree/xgb_cancer | concrete_ml | 0.956 | 0.956 | 2 767.3 ± 135.6 | 1.0× | 1280 PBS |
| tree/xgb_cancer | concrete_ml_ptq | — | — | — | — | não suportado: apenas redes neurais |
| tree/xgb_cancer | fheopt | — | — | — | — | não suportado: comparações (árvores/KNN) não suportadas pelo motor CKKS |
| tree/xgb_cancer | fheopt_diagonal | — | — | — | — | não suportado: comparações (árvores/KNN) não suportadas pelo motor CKKS |
| tree/xgb_cancer | fheopt_vazao | — | — | — | — | não suportado: comparações (árvores/KNN) não suportadas pelo motor CKKS |
| tree/xgb_cancer | tfhe_auto | 0.965 | 0.965 | 1 413.6 ± 49.6 | 2.0× | 1000 PBS; escolhido n_bits=4, p_error=0.1 (padrão: 1280 PBS) |

## Redes neurais

| Cenário | Abordagem | Acurácia (claro) | Acurácia (FHE) | Latência (ms) | Ganho | Observação |
|---|---|---|---|---|---|---|
| nn/mlp_digits_32x2 | concrete_ml | 0.936 | 0.936 | 2 412.7 ± 56.7 | 0.5× | 224 PBS |
| nn/mlp_digits_32x2 | concrete_ml_ptq | 0.950 | 0.939 | 1 319.8 ± 58.0 | 1.0× | 64 PBS |
| nn/mlp_digits_32x2 | fheopt | 0.961 | 0.961 | 170.7 ± 0.9 | 7.7× | N=32768, prof. 7, 25 rotações |
| nn/mlp_digits_32x2 | fheopt_diagonal | 0.961 | 0.961 | 815.8 ± 13.5 | 1.6× | N=32768, prof. 9, 81 rotações |
| nn/mlp_digits_32x2 | fheopt_vazao | 0.961 | 0.961 | 0.393 ± 0.002 | 3358.8× | vazão: lote de 8192 em 3.22 s; N=16384, prof. 5, 0 rotações |
| nn/mlp_digits_32x2 | tfhe_auto | — | — | — | — | não suportado: aplicado a árvores e KNN |
| nn/mlp_digits_64x2 | concrete_ml | 0.942 | 0.942 | 4 640.1 ± 51.3 | 0.7× | 448 PBS |
| nn/mlp_digits_64x2 | concrete_ml_ptq | 0.950 | 0.958 | 3 215.3 ± 60.0 | 1.0× | 128 PBS |
| nn/mlp_digits_64x2 | fheopt | 0.942 | 0.942 | 180.2 ± 1.5 | 17.8× | N=32768, prof. 7, 28 rotações |
| nn/mlp_digits_64x2 | fheopt_diagonal | 0.942 | 0.942 | 1 480.4 ± 6.3 | 2.2× | N=32768, prof. 9, 145 rotações |
| nn/mlp_digits_64x2 | fheopt_vazao | 0.942 | 0.942 | 0.901 ± 0.003 | 3570.4× | vazão: lote de 8192 em 7.38 s; N=16384, prof. 5, 0 rotações |
| nn/mlp_digits_64x2 | tfhe_auto | — | — | — | — | não suportado: aplicado a árvores e KNN |
| nn/mlp_cancer_16x2 | concrete_ml | 0.965 | 0.965 | 691.9 ± 19.8 | 1.0× | 144 PBS |
| nn/mlp_cancer_16x2 | concrete_ml_ptq | 0.956 | 0.974 | 886.0 ± 165.7 | 0.8× | 32 PBS |
| nn/mlp_cancer_16x2 | fheopt | 0.965 | 0.965 | 141.6 ± 2.3 | 4.9× | N=32768, prof. 7, 18 rotações |
| nn/mlp_cancer_16x2 | fheopt_diagonal | 0.965 | 0.965 | 408.9 ± 7.0 | 1.7× | N=32768, prof. 9, 37 rotações |
| nn/mlp_cancer_16x2 | fheopt_vazao | 0.965 | 0.965 | 0.108 ± 0.001 | 6431.0× | vazão: lote de 8192 em 0.88 s; N=16384, prof. 5, 0 rotações |
| nn/mlp_cancer_16x2 | tfhe_auto | — | — | — | — | não suportado: aplicado a árvores e KNN |

## Vizinhos mais próximos

| Cenário | Abordagem | Acurácia (claro) | Acurácia (FHE) | Latência (ms) | Ganho | Observação |
|---|---|---|---|---|---|---|
| knn/knn_cancer | concrete_ml | — | — | — | — | erro: fase run: RuntimeError: NoParametersFound |
| knn/knn_cancer | concrete_ml_ptq | — | — | — | — | não suportado: apenas redes neurais |
| knn/knn_cancer | fheopt | — | — | — | — | não suportado: comparações (árvores/KNN) não suportadas pelo motor CKKS |
| knn/knn_cancer | fheopt_diagonal | — | — | — | — | não suportado: comparações (árvores/KNN) não suportadas pelo motor CKKS |
| knn/knn_cancer | fheopt_vazao | — | — | — | — | não suportado: comparações (árvores/KNN) não suportadas pelo motor CKKS |
| knn/knn_cancer | tfhe_auto | — | — | — | — | erro: fase run: RuntimeError: nenhuma configuração compilou (RuntimeError: NoParametersFound) |

## Resumo: latência por consulta (1 amostra)

Melhor abordagem deste projeto × melhor linha de base Concrete ML, uma consulta por vez.

| Cenário | Melhor Concrete ML | Latência (ms) | Melhor deste projeto | Latência (ms) | Ganho | Δ acurácia (FHE) |
|---|---|---|---|---|---|---|
| linear/logreg_cancer | concrete_ml | 18.7 ± 0.3 | fheopt | 24.7 ± 1.4 | 0.8× | +0.0 p.p. |
| linear/logreg_digits | concrete_ml | 21.1 ± 0.3 | fheopt | 25.5 ± 0.6 | 0.8× | +0.3 p.p. |
| linear/linsvc_cancer | concrete_ml | 17.2 ± 0.2 | fheopt | 22.4 ± 0.5 | 0.8× | +0.0 p.p. |
| tree/dtree_cancer | concrete_ml | 418.4 ± 54.7 | tfhe_auto | 239.1 ± 5.4 | 1.7× | -0.9 p.p. |
| tree/rf_cancer | concrete_ml | 5 103.8 ± 38.6 | tfhe_auto | 2 744.7 ± 60.8 | 1.9× | -0.9 p.p. |
| tree/xgb_cancer | concrete_ml | 2 767.3 ± 135.6 | tfhe_auto | 1 413.6 ± 49.6 | 2.0× | +0.9 p.p. |
| nn/mlp_digits_32x2 | concrete_ml_ptq | 1 319.8 ± 58.0 | fheopt | 170.7 ± 0.9 | 7.7× | +2.2 p.p. |
| nn/mlp_digits_64x2 | concrete_ml_ptq | 3 215.3 ± 60.0 | fheopt | 180.2 ± 1.5 | 17.8× | -1.7 p.p. |
| nn/mlp_cancer_16x2 | concrete_ml | 691.9 ± 19.8 | fheopt | 141.6 ± 2.3 | 4.9× | +0.0 p.p. |

## Resumo: vazão (custo amortizado por amostra)

Modo vazão do motor CKKS (lote de N/2 amostras) × melhor linha de base Concrete ML. O Concrete ML processa uma amostra por vez (sem empacotamento SIMD), então o custo dele por amostra é a latência.

| Cenário | Melhor Concrete ML | Latência (ms) | Melhor deste projeto | Latência (ms) | Ganho | Δ acurácia (FHE) |
|---|---|---|---|---|---|---|
| linear/logreg_cancer | concrete_ml | 18.7 ± 0.3 | fheopt_vazao | 0.020 ± 0.000 | 916.1× | +0.0 p.p. |
| linear/logreg_digits | concrete_ml | 21.1 ± 0.3 | fheopt_vazao | 0.089 ± 0.001 | 237.5× | +0.3 p.p. |
| linear/linsvc_cancer | concrete_ml | 17.2 ± 0.2 | fheopt_vazao | 0.021 ± 0.000 | 838.4× | +0.0 p.p. |
| nn/mlp_digits_32x2 | concrete_ml_ptq | 1 319.8 ± 58.0 | fheopt_vazao | 0.393 ± 0.002 | 3 358.8× | +2.2 p.p. |
| nn/mlp_digits_64x2 | concrete_ml_ptq | 3 215.3 ± 60.0 | fheopt_vazao | 0.901 ± 0.003 | 3 570.4× | -1.7 p.p. |
| nn/mlp_cancer_16x2 | concrete_ml | 691.9 ± 19.8 | fheopt_vazao | 0.108 ± 0.001 | 6 431.0× | +0.0 p.p. |
