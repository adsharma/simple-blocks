The key idea is that we have a proliferation of tile based python
programming libraries (numpy, scipy, cupy, taichi, triton and now
cuTile).

They all do the same thing, but use different/incompatible syntax.
Leading to an "array unification" committee.

Here's an alternative that pushes a transpilation based approach.

```
uv pip install .
python3 example.py
Result is correct: True
Block size ((16, 16)): 0.2523s - Correct: True
Block size ((32, 32)): 0.0430s - Correct: True
Block size ((64, 64)): 0.0125s - Correct: True
Block size ((128, 128)): 0.0069s - Correct: True
```

Vendors compete by providing their own decorator that's faster/future-proof
etc, but fundamentally compatible with this simple/dumb numpy based
decorator included here.

If the array format needs to be munged, AST needs to be rewritten,
they can do so inside the decorator.

Someone who doesn't understand how GPUs work can still use python
breakpoints in simple_blocks/block.py to understand/debug on a CPU.

They can run unittests on CI with different block sizes and concurrency
to ensure that the code is fundamentally correct.
