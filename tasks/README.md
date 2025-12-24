Briefing global
===============

Leia antes de qualquer tarefa
-----------------------------
- Consulte `docs/README.md` para o conceito completo: cerebro difusivo discreto, loop cognitivo com ruido/refino, sinais internos (entropia, estabilidade, satisfacao), budget, decisoes {continuar | concluir | IDK}, reward final base-custo.
- Objetivo: substituir o stub heuristico por implementacao fiel a doc, com sinais calibrados, suporte a formatos fechados e execucao end-to-end sem usar ground truth no episodio.
- Todos os modulos devem manter interfaces claras e nao depender de gabarito durante o episodio; reward so no final.
- Confiabilidade > chute: confianca precisa refletir entropia/estabilidade; penalizar erro confiante; IDK paga pouco.

Modulos e ordem sugerida
------------------------
1) Difusao
2) Scheduler/Loop
3) Policy/Decisao
4) Runner/Testes
5) Dataset/Loader (pode evoluir em paralelo)
6) Extensoes (verificadores, memoria, executores) — apos o nucleo
