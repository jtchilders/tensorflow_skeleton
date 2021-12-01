# tensorflow_skeleton
A customizable Tensorflow skeleton complete with dataset placeholders and model placeholders.

# ResNet example
There is a resnet example that trains on the ImageNet dataset (ILSVC). It can be run using:
```bash
python main.py -c configs/ilsvc.json
```
The config files is ready to run on ThetaGPU using
```bash
qsub submit_scripts/thetagpu_ilsvc_resnet.sh
```
