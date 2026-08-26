
# Preliminary Dual-Leaderboard Evidence

The organizer published two preliminary leaderboards for the same 31 teams.

- Leaderboard A / Method 1 normalizes each task and metric before averaging.
  Team `cjunxiao` is rank `12/31`, with MRE score `0.23634`, MAE score
  `0.33091`, and final score `0.28363`. This is the original prize leaderboard.
- Leaderboard B / Method 2 first averages raw task metrics and then normalizes
  the aggregate values. Team `cjunxiao` is rank `15/31`, with normalized MRE
  `0.26128`, normalized MAE `0.29601`, and final score `0.27864`.
- In the organizer's best-team-per-task table, the PSAX MRE entry is
  `cjunxiao (0.000)`. The organizer-reported raw hidden PSAX MRE is `39.311`.

The exact hidden aggregates are MRE `24.9356305937` and MAE `26.8757741711`.
Leaderboard B displays averages `24.9333` and `26.8744` because it recomputes
them from organizer-published task metrics rounded to three decimals. The exact
aggregates remain the canonical performance values in the paper and main
results.

Status remains preliminary pending organizer eligibility review. This record is
a transcription of organizer notices received on 2026-08-21 and subsequently
updated with the dual-leaderboard clarification. It is not reconstructed from
validation data. Competition page:
https://www.codabench.org/competitions/15590/

Challenge record: https://doi.org/10.5281/zenodo.19736827
