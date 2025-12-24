Briefing - Modulo Runner/Testes
===============================

Objetivo
--------
Atualizar o runner para executar episodios end-to-end com o modelo de difusao real, o loop e a policy, calculando reward com base do desafio.

Escopo
------
- `src/ubrain/run_stub.py` (ou novo runner): carregar dataset, instanciar modelo de difusao, loop e policy, executar episodios, imprimir/registrar decisoes, candidatos, confianca, budget e reward.
- Garantir que o runner nao depende de ground truth dentro do episodio; reward so no final usando `challenge.reward` e correcao externa.
- Adicionar testes de integracao rapidos; se usar torch, documentar setup; se permanecer stub, manter sem dependencia pesada.

Entregaveis
-----------
- Runner atualizado e documentado em `README.md` (raiz) com comandos de execucao.
- Testes de integracao basicos (pode ser script ou pytest leve).

Referencias
-----------
- `docs/README.md` (fluxo completo e reward).
- Codigo atual em `src/ubrain/run_stub.py`.
