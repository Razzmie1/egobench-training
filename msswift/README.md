# Training

## MS-Swift training approach

Activate .venv and install dependencies

To convert the index in `data/train.jsonl` to the ms-swift Messages format:

```
python msswift/convert_data.py
```

The converter renders every sample frame into `msswift/smoke_data/train_converted_assets/` and writes the JSONL to `msswift/smoke_data/train_converted.jsonl`.
It takes the parameters:
    input: EgoBench index JSONL with id and sample fields (default: data/train.jsonl)
    --out: output ms-swift JSONL path (default: msswift/smoke_data/train_converted.jsonl)

After conversion adjust finetune parameters in `msswift/qwen_train.sh` and run:

```
chmod +x ./msswift/qwen_train.sh
source ./msswift/qwen_train.sh
```