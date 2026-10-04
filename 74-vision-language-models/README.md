# Chapter 74 — Vision-Language Models (VLM)

## 1. Core Architecture

A common VLM pattern is:

~~~text
image
↓
vision encoder
↓
visual tokens/features
↓
projector / resampler
↓
language-model embedding space
↓
text + image tokens
↓
language model
↓
answer / caption / grounding
~~~

Architectures differ, but this decomposition is a useful first model.

## 2. Learning Objectives

- derive image patch/token counts
- understand vision encoders and projectors
- distinguish projector fusion from cross-attention
- reason about image-token context budgets
- understand multimodal chat formatting
- understand VLM pretraining and visual instruction tuning
- reason about high-resolution/multi-image/video inputs
- calculate bounding-box IoU
- evaluate grounding precision/recall
- understand OCR/document VLM challenges
- audit visual hallucinations
- design VLM evaluation slices

## 3. Vision Encoder

Possible backbones include ViT-like encoders or other vision networks.

The encoder converts pixels into a sequence/grid of semantic visual features.

## 4. Patch Tokens

For image H x W and patch P_h x P_w:

~~~text
rows = ceil(H / P_h)
cols = ceil(W / P_w)
patch_tokens = rows * cols
~~~

Exact preprocessing differs by model because images may be resized, cropped, tiled or dynamically packed.

## 5. Projector

If vision width d_v differs from language width d_l:

~~~text
V_projected = V @ W + b
W shape = [d_v, d_l]
~~~

A simple linear/MLP projector aligns feature dimensions; more complex models may use resamplers or cross-attention.

## 6. Fusion Strategies

Common patterns:
- prepend/insert projected image tokens into LM sequence
- cross-attention from language states to visual features
- resample many visual features into fewer latent visual tokens

No one pattern defines every VLM.

## 7. Multimodal Chat Templates

Current Transformers multimodal chat templates put modality blocks such as image/text inside each message content list, and multimodal preprocessing/chat templating is handled by a Processor rather than a text-only Tokenizer. citeturn537494search0turn537494search1

Model-specific chat templates and special image tokens must be preserved.

## 8. ImageTextToText

Current Transformers exposes high-level image-text-to-text inference and lower-level Processor + generation paths for supported VLMs. citeturn537494search0

Never assume one model's image placeholder format works for another model.

## 9. Context Budget

Visual tokens consume the same finite model context budget in architectures that insert them into the LM sequence.

~~~text
remaining context
=
max context
- text input
- visual tokens
- reserved output
~~~

High-resolution/multi-image inputs can therefore crowd out text/output budget.

## 10. High Resolution

Strategies may include:
- resize
- crop
- dynamic tiling
- multi-scale features
- resampling/compression

More visual tokens can improve detail but increase compute/memory.

## 11. Multi-Image

Multi-image tasks need explicit image ordering/identity.

Evaluate whether model confuses:
- image A vs B
- temporal/order relationships
- duplicated images

## 12. Video Foundations

Video adds a time dimension:

~~~text
frames
↓
frame/clip visual tokens
↓
temporal selection/compression
↓
language model
~~~

Naively sending every frame can explode token count.

## 13. VLM Training Stages

A common staged recipe:
1. pretrained vision encoder and language model
2. multimodal alignment/projector training
3. multimodal pretraining
4. visual instruction tuning

Some architectures train more components jointly.

## 14. Alignment Data

Examples:
- image-caption pairs
- OCR/document pairs
- visual QA
- region-text pairs
- interleaved image-text documents

Quality/provenance/duplication matter as much as quantity.

## 15. Visual Instruction Tuning

Teach model to follow multimodal conversational instructions.

Examples:
- describe
- compare images
- read text
- answer questions
- locate an object

## 16. Grounding

Grounding links generated language to image regions.

Representations:
- bounding boxes
- points
- masks
- region IDs

Grounding requires spatial evaluation, not only language quality.

## 17. Bounding Box IoU

~~~text
IoU = intersection_area / union_area
~~~

Use an explicit coordinate convention:
- pixels
- normalized [0,1]
- xyxy vs xywh

## 18. OCR / Documents

Document VLMs need:
- high-resolution text
- reading order
- tables
- charts
- layout
- small-font robustness

Caption benchmarks alone do not evaluate document understanding.

## 19. Visual Hallucination

A fluent answer may mention objects/text absent from the image.

Evaluate:
- object existence
- attribute correctness
- count
- spatial relation
- OCR exactness

## 20. Grounding Metrics

For predicted boxes and target boxes:
- IoU
- precision
- recall
- AP/mAP where appropriate

Language-only judge scores cannot replace spatial metrics for grounding tasks.

## 21. Evaluation Slices

Slice by:
- resolution
- text size
- number of images
- object count
- occlusion
- charts/documents
- language
- position in image

## 22. From Scratch

`src/vlm.py` implements:
- patch_grid
- image_token_count
- linear_project
- multimodal_context_remaining
- normalized_box_to_pixels
- bbox_iou
- grounding_counts
- grounding_precision_recall

## 23. Common Mistakes

1. raw image resolution assumed equal to model token count
2. projector dimension mismatch ignored
3. visual tokens omitted from context budgeting
4. one chat template reused across incompatible models
5. high resolution always assumed better
6. multi-image order not tested
7. OCR evaluated only with semantic similarity
8. bounding-box coordinate convention undocumented
9. visual hallucination hidden by fluent language
10. captioning score treated as complete VLM evaluation

## 24. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 25. Checklist

- [ ] vision encoder
- [ ] patch/image tokens
- [ ] projector
- [ ] multimodal chat template
- [ ] context budget
- [ ] high-res / multi-image
- [ ] grounding
- [ ] OCR/documents
- [ ] hallucination
- [ ] VLM evaluation

## 26. What's Next

Chapter 75 moves from images to real-time audio and spoken interaction.