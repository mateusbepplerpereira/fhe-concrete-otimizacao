"""fheopt: motor de compilação de inferência cifrada em CKKS (OpenFHE).

Recebe um modelo treinado (camadas lineares e ativações polinomiais) e gera um programa CKKS
otimizado em dois modos:
  compiler.py  modo latência: layouts diagonal (referência) e replicado com rotate-and-sum
  vazao.py     modo vazão: uma amostra por slot, sem rotações, com fusão da ativação
  backend.py   simulador de slots (valida layouts e conta operações) e backend OpenFHE
  nn.py        ativação polinomial e treino (PyTorch)
"""
