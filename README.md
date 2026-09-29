# Selective Listening: Mechanism-Guided Control of Audio Influence in Large Audio-Language Models

This repository contains the anonymous reproduction code for studying how
task-irrelevant audio can change the answer produced by an audio-language model
when the text input is held fixed.

The code supports the clean text condition, the text-plus-audio interference
condition, and the optional ICAP-style late-pathway gate used for the mitigation
experiments. It is organized around the three model families and the evaluation
scripts used to compute the metrics reported in the paper.

## Reproducibility scope

The paper's main evaluation compares the following conditions:

- **Clean**: text input without audio.
- **Ungated**: text input paired with task-irrelevant audio.
- **Gated**: the same audio input with a model-specific soft gate on the late
  audio pathway.

The benchmark tasks are:

- MMLU
- ARC-Challenge

The model families used in the main comparison are Qwen2.5-Omni, Phi-4-MM,
and Voxtral. Qwen2.5-Omni is evaluated with the 7B and 3B checkpoints where
available. The paper's reported late-pathway gate settings are:

| Model family | Candidate late audio pathway | Gate scale |
| --- | --- | ---: |
| Qwen2.5-Omni | `audio_tower.layers.25` | 0.25 |
| Phi-4-MM | `audio_embed.encoder.encoders.23` | 0.10 |
| Voxtral | `audio_tower.layers.30` | 0.25 |

Module names can vary across model revisions. Use the inspection helpers in
`mechanistic/` to verify the loaded module path before applying a gate.

The interference audio uses FSD-style environmental audio or speech from a
separately obtained LibriSpeech subset. Audio files and datasets are not bundled
with this repository. Provide them locally or point the scripts to an authorized
dataset repository.

## Repository layout

```text
build_dataset/    Dataset construction and audio generation utilities
mechanistic/      Module inspection, pathway probing, and gate helpers
utils/            Dataset loading, audio handling, scoring, and evaluation helpers
figures/          Method illustrations
```

The `mechanistic/` package is required by the inference scripts and should be
kept when running the gated configurations.

## Environment setup

The dependency versions used for the reported experiments are listed in
`requirements.txt`.

```bash
pip install -r requirements.txt
```

Optional Hugging Face cache variables can be initialized with:

```bash
source env.sh
```

`env.sh` uses the current project directory and the user's standard cache
location. No machine-specific path is required.

## Dataset construction

Each builder pairs benchmark examples with randomly selected audio files and
writes the resulting dataset to a user-selected Hub repository. Supply your own
repository identifier; no repository owner is hard-coded in the code.

```bash
python build_dataset/mmlu.py \
  --audio_path ./audio_wavs \
  --repo_name <DATASET_NAMESPACE>/audio_interference_mmlu \
  --seed 0

python build_dataset/arc_challenge.py \
  --audio_path ./audio_wavs \
  --repo_name <DATASET_NAMESPACE>/audio_interference_arc_challenge \
  --seed 0
```

For speech interference, download the permitted LibriSpeech files separately and
pass the directory containing the audio files to `build_dataset/sib_mmlu.py` or
`build_dataset/sib_arc_challenge.py`. The code records source paths in the generated local
dataset; do not publish those paths if they reveal private machine information.

## Inference

All inference entry points require a dataset identifier, split, task type, model
identifier, and output path. The main comparison uses:

- `text_bench`: clean text-only inference.
- `text_bench_interference`: text paired with irrelevant audio.

The Voxtral implementation additionally accepts `audio_understanding` and
`asr` for audio-relevant tasks.

### Qwen2.5-Omni

```bash
python inference_qwen25omni.py \
  --task_id <DATASET_NAMESPACE>/audio_interference_arc_challenge \
  --task_split fsd \
  --task_type text_bench_interference \
  --model_id Qwen/Qwen2.5-Omni-7B \
  --output_path outputs/qwen25omni/fsd_arc.jsonl \
  --temperature 0 \
  --seed 0
```

The optional gate is selected with `--module_groups` and scaled with
`--gate_scale`. The default `--gate_scale 1.0` leaves matched activations
unchanged. Dynamic gate policies are available through `--dynamic_gate_mode`.

For the paper's fixed Qwen pathway, a module-group specification has the form
`late25=^audio_tower\\.layers\\.25$` and the gate scale is `0.25`. Check the
module listing first because the exact prefix depends on the checkpoint.

### Phi-4-MM

```bash
python inference_phi4mm.py \
  --task_id <DATASET_NAMESPACE>/audio_interference_mmlu \
  --task_split fsd \
  --task_type text_bench_interference \
  --model_id microsoft/Phi-4-multimodal-instruct \
  --output_path outputs/phi4mm/fsd_mmlu.jsonl \
  --temperature 0 \
  --seed 0
```

Use `--module_groups` to identify the candidate audio modules and
`--gate_scale` to apply the soft gate. Remote model code and optional speech
LoRA weights must be available in the local Hugging Face environment.

The paper uses the late Phi audio block with a scale of `0.10`; pass the
checkpoint-specific module path through `--module_groups` after inspection.

### Voxtral

```bash
python inference_voxtral.py \
  --task_id <DATASET_NAMESPACE>/audio_interference_mmlu \
  --task_split fsd \
  --task_type text_bench_interference \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --output_path outputs/voxtral/fsd_mmlu.jsonl \
  --temperature 0 \
  --seed 0
```

The Voxtral implementation supports both ordinary processor input and the
`mistral_features` path used for mechanistic gating. Dynamic policies and gate
module groups are configured with the command-line options shown by:

```bash
python inference_voxtral.py --help
```

The reported fixed Voxtral setting uses the late audio block at
`audio_tower.layers.30` with gate scale `0.25`, subject to the module names
exposed by the selected checkpoint.

## Evaluation

### Accuracy

```bash
python evaluate.py \
  --input_path outputs/qwen25omni/fsd_arc.jsonl \
  --task_id arc
```

Use `--task_id mmlu` or `--task_id arc` according to the result file. For
self-consistency outputs, add `--scs`.

### Influence rate

Influence rate (IR) compares an interference result with the corresponding
clean text result.

```bash
python influence_rate.py \
  --input_path outputs/qwen25omni/fsd_arc.jsonl \
  --target_path outputs/qwen25omni/clean_arc.jsonl \
  --task_id arc
```

### Reported metrics

The evaluation utilities expose the metrics used to compare the clean,
ungated, and gated conditions:

- **Accuracy**: fraction of examples with the correct extracted answer.
- **IR**: fraction whose correctness score differs from the clean condition.
- **Answer flip**: fraction whose extracted answer differs from the clean answer.
- **Net C2W repair**: clean-correct examples corrupted by audio and then
  repaired by the gate, minus clean-correct examples damaged by the gate.
- **Net answer restoration**: answer changes caused by audio that return to the
  clean answer under gating, minus new changes introduced by gating.

The transition analyzer writes both rates and counts to the JSON summary.

### Clean, ungated, and gated transitions

The transition analyzer computes paired answer changes and the repair metrics
used by the paper, including answer flips, clean-to-corrupted repair, and answer
restoration.

```bash
python mechanistic/analyze_gating_transitions.py \
  --clean_path outputs/model/clean.jsonl \
  --ungated_path outputs/model/ungated.jsonl \
  --gated_path outputs/model/gated.jsonl \
  --task_id arc \
  --output_summary outputs/model/gating_summary.json \
  --output_cases outputs/model/gating_cases.jsonl
```

For ASR-style JSONL outputs, use:

```bash
python evaluate_asr.py --input_path outputs/asr/results.jsonl
```


## :star: Cites:
If you found this repo is helpful, please consider citing our papers:
```
```

## 🙏 Acknowledgement: 
We  implemented our code based on [AudioInterference](https://github.com/lca0503/AudioInterference) code. We thank author for releasing the code.
