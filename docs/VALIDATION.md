# Synthetic validation and field pilot

On the merged runtime, 36 targeted tests passed. The independent timing audit passed all 11 scenes. Three mixed-rate end-to-end CLI chains passed, including applying the saved rig to a separate recording.

The [synchronization benchmark](synchronization_benchmark.md), [raw results](synchronization_results.json) and [plot](synchronization.png) capture the final Phase 6 synthetic run. It used seed 0 at each of five imposed delays from 0 to 3 frames. All five cases passed, with a maximum offset error of 0.454 ms across those cases.

The preceding frozen snapshot passed 496 source tests and seven installed-wheel checks. The baseline geometry benchmark had 114 successful rows over 48 tasks and three seeds. Its executable geometry path remained byte-identical in the merge. The release wheel passed installation, metadata and targeted runtime tests on Windows AMD64 with Python 3.13 and 3.14. Hosted CI coverage remains to be measured.

These figures describe synthetic inputs. See the [pilot protocol](PILOT_PROTOCOL.md) for field measurements.
