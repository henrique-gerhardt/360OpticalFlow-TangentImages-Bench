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