# Chapter 44 — Object Detection & Segmentation

## 1. Beyond Classification

Image classification predicts one/few labels for an entire image.

Detection predicts:
- class
- bounding box
- confidence

Segmentation predicts labels at pixel level.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- convert bounding-box formats
- calculate box area and IoU
- implement pairwise IoU
- implement Non-Maximum Suppression
- explain confidence thresholds
- explain anchor-based vs anchor-free detection
- compare two-stage vs one-stage detectors
- explain R-CNN / Faster R-CNN / YOLO concepts
- distinguish semantic, instance and panoptic segmentation
- compute mask IoU and Dice score
- explain U-Net
- explain Mask R-CNN
- understand mAP conceptually
- build a tiny segmentation model in PyTorch

## 3. Bounding Boxes

Common formats:

### xyxy

~~~text
(x1, y1, x2, y2)
~~~

top-left and bottom-right corners.

### xywh

~~~text
(x, y, width, height)
~~~

### cxcywh

~~~text
(center_x, center_y, width, height)
~~~

Never mix formats silently.

## 4. Box Area

For xyxy:

~~~text
width = max(0, x2-x1)
height = max(0, y2-y1)

area = width * height
~~~

## 5. Intersection over Union

~~~text
IoU =
intersection area
/
union area
~~~

Range:

~~~text
0 <= IoU <= 1
~~~

Higher means stronger spatial overlap.

## 6. Pairwise IoU

For:

~~~text
N predicted boxes
M reference boxes
~~~

pairwise IoU matrix:

~~~text
(N,M)
~~~

This is used in matching/evaluation/NMS-related logic.

## 7. Non-Maximum Suppression

Goal:
- many highly overlapping predictions → keep best representative

Algorithm:

1. sort by score descending
2. keep highest-score box
3. remove lower-score boxes whose IoU with kept box exceeds threshold
4. repeat

Torchvision's current NMS follows this core interpretation for xyxy boxes.

## 8. NMS Threshold

Low threshold:
- aggressive suppression

High threshold:
- more overlapping boxes survive

Threshold choice affects duplicate detections vs missed crowded objects.

## 9. Class-Aware NMS

Normally boxes from different classes should not suppress each other.

Perform NMS:
- separately per class
or
- use class offsets/batched implementation

## 10. Confidence Threshold

Before/after NMS, discard low-confidence detections.

Trade-off:
- high threshold → precision up, recall down
- low threshold → recall up, false positives up

## 11. Anchors

Anchor-based detector starts with predefined boxes at locations/scales/aspect ratios.

Network predicts:
- objectness/class
- box offsets relative to anchors

Examples historically include Faster R-CNN and many YOLO/SSD variants.

## 12. Anchor-Free Detection

Anchor-free designs predict objects without fixed anchor boxes.

May predict:
- centers
- corners
- distances to boundaries
- heatmaps

This removes anchor-design hyperparameters.

## 13. Two-Stage Detectors

Typical pipeline:

~~~text
image
↓
backbone
↓
region proposals
↓
ROI features
↓
classification + box regression
~~~

Faster R-CNN is a canonical example.

Pros:
- strong localization/accuracy

Cons:
- often heavier/slower

## 14. Region Proposal Network

RPN predicts:
- objectness
- candidate boxes

These proposals are then processed by the second stage.

## 15. ROI Align

Feature maps are sampled/aligned for each proposed region into a fixed-size tensor.

Mask R-CNN uses ROI Align to improve spatial alignment compared with quantized ROI pooling.

## 16. One-Stage Detectors

Directly predict detections over feature maps.

Examples:
- YOLO families
- SSD
- RetinaNet-style approaches

Pros:
- efficient
- low latency

## 17. YOLO Concept

General pattern:

~~~text
image
↓
backbone
↓
multi-scale features
↓
dense detection heads
↓
boxes + class/objectness scores
↓
post-processing
~~~

Exact architecture/loss differs strongly across YOLO generations.

## 18. Focal Loss

Dense detectors have many easy negatives.

Focal loss downweights easy examples.

Binary form:

~~~text
FL(p_t)
=
-alpha_t
(1-p_t)^gamma
log(p_t)
~~~

## 19. Box Regression Loss

Possible choices:
- L1 / Smooth L1
- IoU loss
- GIoU
- DIoU
- CIoU

IoU-family losses incorporate geometry directly.

## 20. Detection Matching

Evaluation/training may match prediction to ground truth using:
- IoU threshold
- assignment rules
- Hungarian matching in set-prediction models

Do not assume all detectors use the same assignment.

## 21. Precision / Recall in Detection

Prediction is true positive when:
- correct class
- sufficient IoU
- matched to an unmatched ground-truth object

Duplicates around same object become false positives.

## 22. Average Precision

AP summarizes precision-recall behavior over confidence thresholds.

mAP averages AP across:
- classes
- sometimes multiple IoU thresholds

Always state the exact evaluation protocol.

## 23. Semantic Segmentation

Every pixel gets a semantic class:

~~~text
road
sky
person
car
...
~~~

Two persons share the same "person" semantic label.

## 24. Instance Segmentation

Distinguish individual object instances.

~~~text
person #1
person #2
~~~

Each instance has its own mask.

## 25. Panoptic Segmentation

Combines:
- semantic "stuff"
- individual "thing" instances

into a unified scene representation.

## 26. Pixel Cross-Entropy

For C classes:

~~~text
logits:
(B,C,H,W)

target:
(B,H,W)
~~~

Cross-entropy classifies every pixel.

## 27. Mask IoU

For binary masks:

~~~text
IoU =
|prediction ∩ target|
/
|prediction ∪ target|
~~~

## 28. Dice Coefficient

~~~text
Dice =
2|A∩B|
/
(|A|+|B|)
~~~

Related to F1 for binary segmentation.

## 29. U-Net

~~~text
input
↓
encoder/downsample
↓
bottleneck
↓
decoder/upsample
↑ skip connections from encoder
↓
pixel logits
~~~

Skip connections restore fine spatial details.

## 30. Mask R-CNN

Extends Faster R-CNN with a per-instance mask head.

Pipeline:
- RPN
- ROI Align
- class/box heads
- mask head

## 31. Multi-Scale Features

Objects appear at different sizes.

Feature Pyramid Networks combine:
- deep semantic features
- higher-resolution features

to support multi-scale detection.

## 32. Data Augmentation

Detection/segmentation transforms must update labels consistently.

If image is flipped:
- boxes must flip
- masks must flip

A geometry transform applied only to image silently corrupts supervision.

## 33. From Scratch

src/detection_segmentation_numpy.py includes:

- box_area
- box_iou
- pairwise_iou
- nms
- mask_iou
- dice_score

## 34. Common Mistakes

1. xyxy vs xywh confusion
2. invalid negative-width boxes
3. IoU division by zero
4. class-agnostic NMS unintentionally
5. suppressing boxes at wrong threshold direction
6. evaluation duplicates counted as multiple true positives
7. applying augmentations without transforming boxes/masks
8. resizing masks with inappropriate interpolation
9. evaluating mAP without stating IoU protocol
10. thresholding segmentation logits before training loss

## 35. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 36. Checklist

- [ ] box formats
- [ ] IoU
- [ ] NMS
- [ ] anchors / anchor-free
- [ ] one-stage / two-stage
- [ ] Faster R-CNN
- [ ] YOLO concepts
- [ ] semantic / instance / panoptic segmentation
- [ ] U-Net
- [ ] Mask R-CNN
- [ ] AP / mAP

## 37. What's Next

Chapter 45 studies embedding retrieval systems and vector databases: exact search, ANN, IVF, HNSW and product quantization.
