Adaptacao local do contrato para o projeto `360OpticalFlow-TangentImages`.

- `run_inference.py` usa diretamente `src/flow_estimate.PanoOpticalFlow`.
- O dataset Replica 360 oficial pode ser injetado por `REPLICA360_ROOT`, sem hardcode de caminho absoluto no repositorio.
- O formato validado para a run oficial e o dataset `released`, com cenas como `hotel_0_circ`, `hotel_0_line` e `hotel_0_rand` diretamente sob o root.
- `evaluate.py` reaproveita as metricas nativas do projeto para EPE/AAE/RMSE e suas variantes esfericas.
- `profile.py` mede latencia e memoria sem assumir um modelo `torch.nn.Module`.

## Build e run em Linux com Docker e GPU NVIDIA

### 1. Pre-requisitos no host

- Linux com Docker Engine instalado.
- Driver NVIDIA instalado e funcional no host.
- `nvidia-container-toolkit` instalado e configurado no Docker.
- O host pode ter suporte a CUDA 13 no driver.

Observacao:

- A imagem atual do benchmark usa a base `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`.
- Em uma maquina com driver NVIDIA recente, esse container normalmente roda sem problema mesmo quando o host ja suporta CUDA 13.
- O metodo adaptado neste repositorio nao depende de um caminho real de inferencia em GPU; o fluxo principal continua sendo essencialmente CPU-based. Ainda assim, o ambiente abaixo fica pronto para execucao em hosts com GPU NVIDIA.

### 2. Validar acesso da GPU pelo Docker

No host Linux:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Se esse comando falhar, ajuste primeiro o driver NVIDIA ou o `nvidia-container-toolkit`.

### 3. Build da imagem

Na raiz do repositorio:

```bash
docker build -f benchmark_contract/Dockerfile.benchmark -t tangent-images-benchmark .
```

### 4. Executar o benchmark com persistencia dos resultados no host

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

### 5. Rodar os outros cenarios

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

### 6. Onde ficam os artefatos

Depois da execucao, os arquivos ficam no host em:

- `benchmark_contract/results/metadata.json`
- `benchmark_contract/results/quality_metrics.json`
- `benchmark_contract/results/efficiency_metrics.json`
- `benchmark_contract/results/run_config.json`
- `benchmark_contract/results/environment.json`
- `benchmark_contract/outputs/pred_flow.flo`
- `benchmark_contract/outputs/predictions.npz`

### 7. Observacoes praticas

- Se quiser usar o mini-sample local do repositorio, ajuste `benchmark_contract/config/experiment.yaml` para uma cena compativel, como `hotel_0`, e mantenha o root padrao `data/replica_360`.
- Para a run oficial no dataset `released`, a configuracao padrao do contrato ja espera cenas como `hotel_0_circ` e root vindo de `REPLICA360_ROOT`.
- Se voce alterar os arquivos Python ou YAML do contrato, nao precisa rebuildar a imagem quando usar `bind mount`; basta executar `docker run` novamente.
- Se quiser uma execucao totalmente empacotada, sem `bind mount`, remova `-v "$PWD":/app -w /app`, mas nesse caso os resultados ficam apenas no filesystem do container.
