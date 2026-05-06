## Ajustes realizados

Este arquivo resume os ajustes feitos no projeto e no contrato do benchmark.

### 1. Ajustes nas metricas angulares

Foram corrigidos os problemas que impediam a geracao confiavel de `AAE` e `SAAE`:

- `src/flow_evaluate.py`
  - `AAE()` agora respeita os argumentos `spherical` e `of_mask`.
  - `AAE_mat()` passou a validar GT e predicao separadamente.
  - a logica de rejeicao de fluxo com magnitude invalida foi corrigida.
- `src/spherical_coordinates.py`
  - foi implementada a funcao `get_angle(...)`, usada no caminho esferico de `AAE_mat(...)`.

Com isso, o contrato deixa de depender de um fallback que produzia `saae_global = null` quando a avaliacao esferica falhava.

### 2. Ajuste da reproducao oficial do paper

O cenario `official_reproduction` do contrato deixou de medir apenas um unico par de imagens e passou a executar o protocolo completo do paper em Replica360:

- varre os subconjuntos `circ`, `line` e `rand`
- respeita as mesmas regras de amostragem do `test_replica360.py`
- agrega os resultados em `circle`, `line`, `random` e `all`
- grava comparacao direta com a Tabela 1 do paper

Arquivos adicionados ou ajustados para isso:

- `benchmark_contract/replica360_protocol.py`
- `benchmark_contract/run_inference.py`
- `benchmark_contract/evaluate.py`
- `benchmark_contract/common.py`
- `benchmark_contract/config/experiment.yaml`

Novos artefatos:

- `benchmark_contract/results/paper_reproduction_comparison.json`
- `benchmark_contract/results/raw_logs/replica360_protocol_rows.json`
- `benchmark_contract/results/raw_logs/replica360_protocol_rows.csv`
