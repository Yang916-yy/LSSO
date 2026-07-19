_base_ = './mask-rcnn_vllama-b_common.py'

model = dict(backbone=dict(mixer='rrlsso', rank=32))
