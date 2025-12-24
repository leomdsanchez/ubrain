uBrain
======

PoC do cerebro difusivo discreto (pensamento iterativo com IDK, budget e reward final). Detalhes conceituais em `docs/README.md`.

Setup rapido
------------
- Requer Python >=3.10.
- Instalar deps: `pip install -e .` (ou `pip install -e .[dev]` para pytest/ruff).

Estrutura
---------
- `docs/README.md`: especificacao e simulacoes de loop.
- `src/ubrain/`: codigo do cerebro (tipos, policy, scheduler, difusao, avaliador, data specs).
- `data/challenges.yaml`: dataset inicial de desafios fechados (formato YAML).
- `tasks/`: briefings de modulos e plano de execucao por time.

Execucao rapida (stub)
----------------------
- `PYTHONPATH=src python3 -m ubrain.run_stub --steps 4` (usa `DiscreteDiffusion` heuristica)  
- `PYTHONPATH=src python3 -m ubrain.run_stub --steps 4 --use-heuristic` (stub heuristico anterior)

Proximos passos
---------------
- Substituir o stub por modelo de difusao condicional real (sem usar ground truth no episodio).
- Calibrar sinais/decisao no scheduler e policy com base em entropia/estabilidade/budget.
- Expandir formatos de `output_format` (grid/som) e integrar executores/verificadores/memoria.
