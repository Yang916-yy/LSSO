_base_ = './upernet_vllama-b_common.py'

model = dict(backbone=dict(mixer='lsso', rank=32))
