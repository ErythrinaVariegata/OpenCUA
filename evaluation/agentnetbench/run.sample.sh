#!/bin/bash

# uv pip install "openai>=1.0.0" pillow editdistance
# source .venv/bin/activate

.venv/bin/python run.py \
  --data sample_data \
  --image_dir sample_data/images \
  --output output \
  --model /data/model \
  --base_url https://ms-qlnmlmlz-100034032793-sw.gw.ap-zhongwei.ti.tencentcs.com/ms-qlnmlmlz/v1\
  --api_key my-ollama-key \
  --num_cores 10
