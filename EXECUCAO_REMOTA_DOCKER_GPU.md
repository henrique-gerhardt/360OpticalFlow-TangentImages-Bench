## Copiar o projeto para uma maquina remota e rodar o teste em Docker

Este guia assume:

- uma maquina remota Linux
- Docker instalado
- driver NVIDIA funcional
- `nvidia-container-toolkit` configurado
- GPU disponivel no host remoto

Tambem assume que o dataset Replica 360 oficial esta no formato `released`, com cenas como `hotel_0_circ`.

### 1. Validar a GPU no Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Se isso falhar, corrija primeiro o ambiente NVIDIA do host.

### 2. Build da imagem

Na raiz do repositorio remoto:

```bash
docker build -f benchmark_contract/Dockerfile.benchmark -t tangent-images-benchmark .
```

### 3. Rodar o cenario principal com o dataset oficial

O contrato agora aceita o root do dataset por variavel de ambiente `REPLICA360_ROOT`.

```bash
docker run --rm \
  --gpus all \
  -e REPLICA360_ROOT=/datasets/released \
  -v "$PWD":/app \
  -v /data/datasets/released:/datasets/released:ro \
  -w /app \
  tangent-images-benchmark \
  official_reproduction
```

### 4. Rodar os outros cenarios

`standardized_efficiency`:

```bash
docker run --rm \
  --gpus all \
  -e REPLICA360_ROOT=/datasets/released \
  -v "$PWD":/app \
  -v /data/datasets/released:/datasets/released:ro \
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
  -v /data/datasets/released:/datasets/released:ro \
  -w /app \
  tangent-images-benchmark \
  regional_robustness
```

### 5. Onde olhar os resultados

Depois da execucao, verifique:

- `benchmark_contract/results/metadata.json`
- `benchmark_contract/results/quality_metrics.json`
- `benchmark_contract/results/efficiency_metrics.json`
- `benchmark_contract/results/run_config.json`
- `benchmark_contract/results/environment.json`
- `benchmark_contract/outputs/pred_flow.flo`
- `benchmark_contract/outputs/predictions.npz`

### 6. Ajustes necessarios para a run oficial

Com os ajustes atuais, nao e necessario mudar o formato do contrato para o dataset `released`.

Os pontos importantes sao:

- definir `REPLICA360_ROOT`
- montar o dataset no container
- usar uma cena valida do dataset oficial, por exemplo `hotel_0_circ`

O contrato ja foi alinhado para isso por padrao em:

- `benchmark_contract/config/datasets.yaml`
- `benchmark_contract/config/experiment.yaml`

### 7. Se quiser trocar a cena ou o frame

Edite:

- `benchmark_contract/config/experiment.yaml`

Exemplo:

- `scene: apartment_0_circ`
- `frame_idx: 5`
- `direction: forward`

Nao e necessario rebuildar a imagem quando voce esta usando:

- `-v "$PWD":/app`

Nesse caso, basta rodar o `docker run` novamente.
