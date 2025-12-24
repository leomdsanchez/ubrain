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

Proximos passos
---------------
- Implementar `_extract_candidate`, `_estimate_signals`, `_is_better` no scheduler com instrumentacao real.
- Adicionar modelo de difusao discreta concreto no modulo `diffusion`.
- Criar dataset inicial de desafios fechados e loader.
