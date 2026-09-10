# Reproducible smoke experiment evidence

These are measured outputs, not target or invented scores. Environment and policy
logic run on CPU with no neural inference. Local validation: Python 3.12,
58 tests passed including all 14 pre-existing legacy/foundation tests; the fatal
Flake8 gate passed. No pre-existing Creature environment/test suite existed at the
starting commit, so single-creature compatibility tests were added in this milestone.

- `generations-seed42.jsonl`: 20 evaluated generations, 50 founders, 300 steps,
  evolution seed 42, selection worlds 11/22/33, heuristic genomes, two elites.
- `heldout-seed42.json`: next bred population after 20 generations evaluated on
  heldout 101/202/303. Those worlds never participate in selection. The initial
  heuristic population and random policy use matched world and policy seeds.
- `continuous-200-seed11.json`: 200 founders, cap 200, 120 ticks, genome seed 42,
  world seed 11. 41 births occurred as spaces opened; 101 creatures survived and
  the maximum biological generation was 2. Living population never exceeded 200.

Selection mean fitness rose from 59.308 (generation 0) to 70.300 (generation 19).
Heldout mean fitness: evolved 70.477, fixed initial heuristic 58.442, random 25.369.
Diversity fell from 0.0857 to 0.0122, showing convergence worth monitoring.
This demonstrates improvement on the defined artificial objective in one
training run, not universal intelligence or financial applicability. Independent
evolution seeds are needed for stronger conclusions; a seed standard deviation
across three worlds is not a statistical significance test.

The per-generation file was captured before final reporting-only fields were
added; it preserves the original measured records. Re-running the documented
command yields the same fitness trajectory and additional final diversity fields.
