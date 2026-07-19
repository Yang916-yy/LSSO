_base_ = './mask-rcnn_vllama-b_rrlsso-r32.py'

model = dict(backbone=dict(window_size=None))
