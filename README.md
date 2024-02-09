# Concept Bottleneck Models

To encode images and concepts for a dataset (e.g. CUB):
```bash
python encode.py --device cuda --dataset-dir data/CUB_200_2011/ --concept-dir data/lm4cv/cub_attributes.txt
```

To train and test a model (e.g. CUB, 100 epochs):
```bash
python main.py --device cuda --dataset-dir data/CUB_200_2011/ --num-concepts 200 --stage-one-epochs 100 --stage-two-epochs 100
```
