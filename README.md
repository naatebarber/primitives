# Machine Learning Primitives

Whenever I go to use a new machine learning algorithm / architecture / technique, I normally write it out myself first using plain Rust and NDArray. 
Doing all the differentiation and grunt work helps me appreciate this stuff more, and I like that nothing is hidden. I figured I would collect these 
handwritten architectures as I go. This is not at all a library, but moreso a set of primitives / a toolbox I'll be revisiting from time to time.

### Sections
- `envs`: Various simulators used to test stuff
- `nn`: Core neural net components, stuff like basic FFN, Attention, Layernorm, CTRNN, etc. TBD some experimentals in the `nn/experimental` dir.
- `optim`: Gradient collection and training, AdamW, SGD
- `rl`: A couple RL algos, SAC is still a bit sketch, but PPO & TD3 are solid.
- `util`: Graphing and benchmarking
- `f`: Function set
