# Demo and Presentation Notes

## Live run (≈ 2 min)

```bash
source ~/.venvs/ubi2/bin/activate
# teaching-team episode: folder of raw .pcap/.csv captures, optional support/<label>/...
python run_episode.py /path/to/episode --out predictions.csv
head predictions.csv            # window_id, room, label, confidence

# rehearsal episode built from our validation captures (neutral names q001...)
python scripts/make_demo_episode.py --per-class 3
python run_episode.py episodes/demo --out episodes/demo/predictions.csv
python scripts/make_demo_episode.py --score episodes/demo
```

Prerequisites on the demo machine: `raw/` data, then `scripts/build_dataset.py`,
`scripts/task1_room.py`, `scripts/task2_ssl.py` have been run (windows + `results/models/*.pt`).
If an episode capture has too few full-band observations, the runner warns and uses all
observations instead of failing.

## Talking points (≈ 8 min)

1. **The data problem.** Room A is raw Nexmon PCAP from a BCM4366c0; we wrote our own
   block-float CSI decoder. Key discovery: ~71% of Room-A packets are 20 MHz frames inside
   the 80 MHz layout (192 subcarriers are noise). Show `subband_energy_share.png`: bimodal,
   so QC is unambiguous. Leaves ~59 real observations per capture, never 150.
2. **Windowing decision.** One window per capture, resampled to 150 in both rooms (padding
   would hand the room classifier a trivial cue).
3. **Shortcut we found and removed.** After resampling, temporal bandwidth alone identified
   the room 100% even for the same activities and tracked packet count (r = 0.84). A shared
   5 Hz low-pass cut it to 75.7% / r = 0.18. Show the dynamics-only control line in
   `room_acc_vs_window_length.png`.
4. **Task 1.** Room is identified perfectly from 10 steps by the static multipath profile.
5. **Task 2.** SSL encoder: joint probe 0.943 vs 0.911 (random / handcrafted). UMAP: the
   representation is organized by room first — the domain gap.
6. **Task 3.** Primary: SSL embedding + event features, logistic regression with a group-level
   prior correction, routed. Macro-F1 at K=1/5/10/20: B→A 0.28/0.40/0.43/0.48,
   A→B 0.34/0.49/0.56/0.62. Exclusive activities learn well (0.74 and 0.85 at K=20).
   Story: the first primary failed in two ways — in target A the exclusive classes absorbed the
   shared ones (87% of shared errors at K=20), in target B STAND collapses into SIT. Cause:
   per-room alignment over different activity mixes (label shift). INLP, self-training, crops
   and EM did not help; a group-level prior fixed the target-A shortcut (+0.085 at K=20,
   bootstrap CI +0.04 to +0.12). The SIT↔STAND confusion remains — the central finding.
7. **Honest limitations.** Room = session confound; Room-A windows are upsampled; two Room-A
   validation captures unusable (148/150); capture-disjoint, not subject-disjoint.

## Likely questions

* *Why not pad?* Constant tails identify Room A trivially.
* *Is 100% room accuracy a shortcut?* Static-profile-only model is also 100%; the
  bandwidth shortcut was measured and removed; the session confound remains and is stated.
* *Why did you not use the sequence number to filter packets?* Bandwidth, not the sequence
  field, determines validity: seq-numbered frames are also often 20 MHz.
* *Leakage?* Validation never used for fitting, support or normalization; one window per
  capture; Room-B split frozen; tests check split isolation.
