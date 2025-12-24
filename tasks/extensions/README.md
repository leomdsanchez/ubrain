Briefing - Modulo Extensoes (opcional apos nucleo)
==================================================

Objetivo
--------
Implementar componentes auxiliares previstos na doc: verificadores (abas), memoria externa, executores.

Escopo
------
- Verificadores: instancias curtas que testam hipoteses {true/false/unknown} com budget proprio; cerebro agrega.
- Memoria externa: cache de atalhos (estado inicial -> prior/logits) para acelerar convergencia; nao decide.
- Executores: modelos separados para desenhar icones/gerar audio/texto curto, condicionados pelo cerebro; nao participam do pensamento.

Entregaveis
-----------
- Especificacoes e stubs para cada extensao (pode ser separado).
- Plano de integracao com o loop principal sem contaminar sinais internos.

Referencias
-----------
- `docs/README.md` (secoes de memoria, verificadores, executores).
