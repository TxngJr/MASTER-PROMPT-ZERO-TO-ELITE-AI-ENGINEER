# Mini Project — Full FT vs LoRA

Use the same tiny pretrained model and adaptation dataset.

Compare:
- full fine-tuning
- LoRA q/v-only
- LoRA all-linear

Track:
- total/trainable parameters
- optimizer-state estimate
- validation loss
- adaptation accuracy
- retained base-task score
- checkpoint size
- inference before/after adapter merge

Optional Hugging Face lab:

~~~bash
python -m pip install -r requirements-batch19-peft.txt
~~~
