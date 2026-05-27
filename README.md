## HOST

```bash
`sudo apt-get install ffmpeg`
`uv sync`
`uv run cli/preparation.py download_yamnet_onnx`
`uv run cli/preparation.py build_calibration_set --sample_audio_files 100`
```

```bash
docker pull sophgo/tpuc_dev:v3.1

docker run --privileged --name yamnet-tpu \
    -v "$(pwd)/workspace":/workspace/yamnet \
    -it sophgo/tpuc_dev:v3.1
```

## DOCKER

```bash
git clone https://github.com/milkv-duo/tpu-mlir.git
source ./tpu-mlir/envsetup.sh
cd /workspace/yamnet
mkdir -p build && cd build
```

```bash
model_transform.py \
  --model_name yamnet \
  --model_def ../model/yamnet-onnx-float/yamnet.onnx \
  --input_shapes [[1,1,96,64]] \
  --test_input ../calibration/sample_0000.npz \
  --test_result yamnet_top_outputs.npz \
  --mlir yamnet.mlir
```

```bash
run_calibration.py yamnet.mlir \
  --dataset ../calibration \
  --input_num 100 \
  -o yamnet_cali_table
```

```bash
model_deploy.py \
  --mlir yamnet.mlir \
  --quantize BF16 \
  --calibration_table yamnet_cali_table \
  --chip cv181x \
  --test_input ../calibration/sample_0000.npz \
  --test_reference yamnet_top_outputs.npz \
  --tolerance 0.85,0.45 \
  --model yamnet.cvimodel
```

## DEVICE

```bash
mkdir -p /mnt/tpu && cd /mnt/tpu
```

## HOST

```bash
cd /workspace
git clone https://github.com/milkv-duo/tpu-sdk-sg200x.git
mv ./tpu-sdk-sg200x ./tpu-sdk

scp -r /workspace/tpu-sdk root@192.168.42.1:/mnt/tpu/
```

```bash
curl -L -o /tmp/yamnet_class_map.csv \
    https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv

ssh root@192.168.42.1 "mkdir -p /root/yamnet"

scp device/{run_yamnet.sh,run_yamnet.py} \
    workspace/build/yamnet.cvimodel \
    workspace/calibration/sample_0000.npz \
    /tmp/yamnet_class_map.csv \
    root@192.168.42.1:/root/yamnet/

ssh root@192.168.42.1 'chmod +x /root/yamnet/run_yamnet.sh'
```

## DEVICE

```bash
./run_yamnet.sh sample_0000.npz
```
