Adaptacao local do contrato para o projeto `360OpticalFlow-TangentImages`.

- `run_inference.py` usa diretamente `src/flow_estimate.PanoOpticalFlow`.
- O dataset Replica 360 oficial pode ser injetado por `REPLICA360_ROOT`, sem hardcode de caminho absoluto no repositorio.
- O formato validado para a run oficial e o dataset `released`, com cenas como `hotel_0_circ`, `hotel_0_line` e `hotel_0_rand` diretamente sob o root.
- `evaluate.py` reaproveita as metricas nativas do projeto para EPE/AAE/RMSE e suas variantes esfericas.
- `profile.py` mede latencia e memoria sem assumir um modelo `torch.nn.Module`.

## Significado dos cenarios

Os tres cenarios do contrato nao representam tres algoritmos diferentes. Eles representam tres objetivos de avaliacao usando o mesmo metodo base.

### `official_reproduction`

Objetivo:

- executar o metodo o mais proximo possivel da configuracao nativa adaptada para este repositorio

Na pratica, neste projeto:

- usa o `PanoOpticalFlow` nativo
- nao faz resize para modo de eficiencia
- preserva a configuracao principal de inferencia

Esse e o cenario mais indicado quando voce quer validar a reproducao do metodo.

### `standardized_efficiency`

Objetivo:

- medir eficiencia em uma configuracao mais padronizada para comparacao com outros metodos

Na pratica, neste projeto:

- ativa `resize_for_efficiency: true`
- usa resolucao de entrada padronizada menor
- ajusta a quantidade de warmup/runs medidos no profiling

Esse e o cenario mais indicado quando o foco e latencia, throughput e consumo aproximado de memoria.

### `regional_robustness`

Objetivo:

- medir como o erro se distribui em diferentes regioes da imagem ERP, especialmente polos e faixa equatorial

Na pratica, neste projeto:

- usa o mesmo estimador base
- enfatiza a avaliacao por bandas de latitude
- reporta diferencas entre regioes polares e equatoriais

Esse e o cenario mais indicado quando o foco e robustez espacial sobre a esfera.

### Resumo pratico

- `official_reproduction`: melhor para reproducao do metodo
- `standardized_efficiency`: melhor para comparacao de eficiencia
- `regional_robustness`: melhor para analisar variacao de erro por latitude

## Build e run 

### 1. Build da imagem

Na raiz do repositorio:

```bash
docker build -f benchmark_contract/Dockerfile.benchmark -t tangent-images-benchmark .
```

### 2 Executar o benchmark com persistencia dos resultados no host

O comando abaixo monta o repositorio local em `/app` dentro do container. Isso e o mais recomendado, porque preserva no host:

- `benchmark_contract/results/`
- `benchmark_contract/outputs/`

Para a run oficial com o dataset `released`, monte tambem o dataset no container e exporte `REPLICA360_ROOT`.

Execucao do cenario principal:

```bash
docker run --rm \
  --gpus all \
  -e REPLICA360_ROOT=/datasets/released \
  -v "$PWD":/app \
  -v /caminho/no/host/released:/datasets/released:ro \
  -w /app \
  tangent-images-benchmark \
  official_reproduction
```

### 3. Rodar os outros cenarios

`standardized_efficiency`:

```bash
docker run --rm \
  --gpus all \
  -e REPLICA360_ROOT=/datasets/released \
  -v "$PWD":/app \
  -v /caminho/no/host/released:/datasets/released:ro \
  -w /app \
  tangent-images-benchmark \
  standardized_efficiency
```

`regional_robustness`:

```bash
docker run --rm \
  --gpus all \
  -e REPLICA360_ROOT=/datasets/released \
  -v "$PWD":/app \
  -v /caminho/no/host/released:/datasets/released:ro \
  -w /app \
  tangent-images-benchmark \
  regional_robustness
```

### 4. Onde ficam os artefatos

Depois da execucao, os arquivos ficam no host em:

- `benchmark_contract/results/metadata.json`
- `benchmark_contract/results/quality_metrics.json`
- `benchmark_contract/results/efficiency_metrics.json`
- `benchmark_contract/results/run_config.json`
- `benchmark_contract/results/environment.json`
- `benchmark_contract/outputs/pred_flow.flo`
- `benchmark_contract/outputs/predictions.npz`

### 5. Observacoes praticas

- Se quiser usar o mini-sample local do repositorio, ajuste `benchmark_contract/config/experiment.yaml` para uma cena compativel, como `hotel_0`, e mantenha o root padrao `data/replica_360`.
- Para a run oficial no dataset `released`, a configuracao padrao do contrato ja espera cenas como `hotel_0_circ` e root vindo de `REPLICA360_ROOT`.
- Se quiser uma execucao totalmente empacotada, sem `bind mount`, remova `-v "$PWD":/app -w /app`, mas nesse caso os resultados ficam apenas no filesystem do container.
