# Third-party code and provenance

The files under `emo-r3/examples/reward_function/emor3.py` and
`emo-r3/examples/format_prompt/emor3.jinja` are copied from the EMO-R3
research code in the project environment. Their upstream project is
[SeerRay-Lab/emo-r3](https://github.com/SeerRay-Lab/emo-r3).
`emo-r3/LICENSE` is the upstream Apache License 2.0. Preserve the original
copyright header and attribution if redistributing these files.

The remainder of this source archive is project-specific experiment code;
this package does not assign a license to it. The repository owner should
choose a license before offering reuse rights to others.

The official EMO-R3 training framework is an external dependency if the
starting EMO-R3 checkpoint must be recreated. This archive includes the
reward function and format template used by the TokenDrop continuation,
but not the full upstream training framework or any upstream model weights.
