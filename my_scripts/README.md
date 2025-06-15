

GPU0, GPU1の二枚を使う場合のコマンド
one_vision_image.shの

- NUM_GPUS
- dataの位置

をチェックすること

また，VSCodeのターミナルで実行した場合は，exitコマンドでターミナルを落とすこと
実行したターミナルのままで，VSCodeを閉じると，実行中のプロセスがnohupだろうが止まる

```bash
CUDA_VISIBLE_DEVICES=0,1 \
nohup bash my_scripts/one_vision_image.sh > nohup.out 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
nohup bash my_scripts/example_image.sh > nohup.out 2>&1 &
```


# そのままでは評価に使えない場合がある

Projector層を学習させた場合，生成されるチェックポイントのフォルダを使うだけではエラーが発生する
その場合は以下のコマンドでスタンドアロンのモデルに変案すれば解決する
transformerのバージョンが影響してそう


```bash
python merge_lora_weights.py \
    --model_id llava-onevision-7b-ov \
    --model_path ./checkpoints/llava-onevision-7b-ov_lora-True_qlora-False_0615 \
    --model_save_path ./mymodel/onevision_finetune \
    --load_model
```

## transformerのバージョンが影響してそうだと考えた理由

以下のコードが本環境では動くが，mllm_unlearningの環境では動かない

```python
from transformers import LlavaOnevisionForConditionalGeneration
import torch

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        # "./checkpoints/llava-onevision-7b-ov_lora-True_qlora-False_0615",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda:1"
    )

print(model)
```