# Image Captioning — ViT + GPT2

An image captioning model built by connecting a Vision Transformer (ViT)
encoder to a GPT-2 decoder via Hugging Face's `VisionEncoderDecoderModel`.
Given an image, the model generates a natural-language caption describing it.

## Architecture

- **Encoder:** `google/vit-base-patch16-224-in21k` — pretrained ViT, extracts
  visual features from the image.
- **Decoder:** `gpt2` — pretrained GPT-2, generates the caption token by
  token, conditioned on the encoder's output via cross-attention.
- Combined with `transformers.VisionEncoderDecoderModel.from_encoder_decoder_pretrained`.

## Data

- **Flickr30k** — ~31.8K images, multiple captions each.
- **MS COCO 2017** (train + val) — ~123K images.
- Combined total: **155,070 images**, split 70/15/15 into train/validation/test
  (split at the image level, so all captions for a given image stay in the
  same split — this avoids leaking the same image between train and test).

| Split | Images |
|---|---|
| Train | 108,549 |
| Validation | 23,260 |
| Test | 23,261 |

## Training

- Batch size: 8
- Optimizer: AdamW, lr = 5e-5
- Epochs: 2 (with checkpoint/resume support so training can continue across
  Kaggle sessions)
- Loss: standard causal LM loss on caption tokens, image patches fed in via
  cross-attention
- Best model (by validation loss) saved separately from the per-epoch
  checkpoint

**Result after training:**
| Metric | Value |
|---|---|
| Train loss | 3.14 |
| Validation loss | 3.27 |

## Evaluation

Evaluated on 300 images from the held-out test set using beam search
(`num_beams=4`, `no_repeat_ngram_size=3`, `repetition_penalty=1.5`).

| Metric | Value |
|---|---|
| BLEU | 0.0 |
| METEOR | 0.284 |
| ROUGE-1 | 0.278 |
| ROUGE-2 | 0.009 |
| ROUGE-L | 0.195 |

### What these numbers mean

The model does identify the right *content* — unigram precision is 34.9%,
and captions on sample images correctly mention things like "doll", "room",
"plate", "food", "guitarist", "drummer", "cows", "grass", "hills". But word
*order* breaks down badly: trigram precision is 0%, which is why BLEU comes
out to exactly 0. ROUGE-2 (0.009) confirms this — almost no correct
two-word sequences. METEOR (0.284) looks more forgiving because it doesn't
penalize word order as heavily, and the generated captions are unusually
long relative to the references (length ratio 3.17), which inflates it.

Example generated caption on a photo of grazing cows in a field:
> `herd animals a and cows a and in field trees hills grass hill and... the is.. view a of valley. image a of`

Right words, wrong order.

## Known issues / next steps

- **No explicit end-of-sequence signal in training labels** — captions were
  padded to `max_length` and the pad tokens were masked with `-100` for the
  loss, but since GPT-2's pad token is set equal to its EOS token, the
  model never sees a "real" EOS as a target. This is a likely cause of
  captions rambling on instead of stopping cleanly.
- **Beam search may be part of the word-order problem** — a monkey-patched
  `_reorder_cache` is used to make beam search work with this
  encoder-decoder combination on this `transformers` version, alongside a
  fairly aggressive `repetition_penalty=1.5`. Worth testing generation with
  `num_beams=1` (greedy) and a lower repetition penalty to isolate whether
  this is a training issue or a generation-time issue.
- **Grammar-correction post-processing step (T5) was tried and dropped** —
  it could not fix genuinely scrambled captions and sometimes made them worse.
- **Only 2 epochs, no LR scheduler** — the model is under-trained relative
  to a typical captioning setup.
- **Evaluated on only 300 of ~23K test images**, and ROUGE was computed
  against a single reference caption per image while BLEU/METEOR used all
  references — evaluation should be scaled up and made consistent.

## How to run

This repo contains the training/evaluation notebook (`.ipynb`). To run it:

1. Open in a Kaggle notebook environment (or Jupyter with a GPU).
2. Attach the Flickr30k and MS COCO 2017 captions datasets (see paths at
   the top of the notebook — adjust if your dataset paths differ).
3. Run cells top to bottom. Training will resume automatically from
   `/kaggle/working/checkpoint.pt` if one already exists.

## Requirements

```
transformers==4.52.4
torch
pandas
scikit-learn
pillow
evaluate
rouge-score
nltk
```
