# Architecture

Dependencies point inward:

~~~
experiments / integrations / benchmarks
                    |
                 lsso.ball
                    |
          PyTorch reference or strict CUDA op
~~~

config.py owns the two supported ablation axes, reference.py owns the QR frame,
accretive generator, and equilibrium mathematics, and model.py owns parameters
and nn.Module behavior. Framework adapters may not implement operator math.
