

GPU0, GPU1の二枚を使う場合のコマンド
one_vision_image.shの

- NUM_GPUS
- dataの位置

をチェックすること

```bash
CUDA_VISIBLE_DEVICES=0,1 \
WANDB_MODE=disabled \
nohup bash example_scripts/one_vision_image.sh > nohup.out 2>&1 &
```
