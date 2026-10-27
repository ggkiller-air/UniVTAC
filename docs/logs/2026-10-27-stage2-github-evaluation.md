# Stage 2 publication and evaluation adapter

- Job 392551's `epoch0010.ckpt`, `epoch0020.ckpt` and a fixed `latest.ckpt`
  were uploaded by direct HTTPS API to ModelScope `ggkiller/UniViTac_ckpts`,
  under `20261027_Stage2Transformer_job392551/checkpoints`. No proxy or git
  transport was used. All three remote sizes and SHA-256 values matched.
- The uploaded latest snapshot is epoch 43, cursor 692, global step 92,000.
  It is pinned independently of the active training output. Download paths,
  hashes, language companions and evaluator commands are documented in
  `docs/MultitaskDPEvaluation.md`.
- The existing deployment wrapper supported only Stage 0 checkpoint fields.
  Added explicit Stage 2 kind/contract/config validation, the actual
  `Stage2Policy` constructor, strict raw/EMA restoration, recorded utility-RMS
  validation, and full-sensor enforcement. Existing Stage 0 behavior is retained.
  No changes to the training architecture or training snapshot were needed.
- Deployment CPU interface checks passed **10/10**, including Stage 0 behavior,
  Stage 2 raw/EMA selection and action dispatch, corrupted contract/config/RMS
  rejection, unsupported kind rejection and Stage 2 removal-condition rejection.
- The actual uploaded step-92,000 latest restored strictly on CPU with the
  original language bundle. Real full-size Transformer conditioning and one
  denoising-network forward produced finite outputs of the expected shape.
- The full Stage 2 CPU diagnostic passed model/gradient/phase checks,
  source-label alignment, task-balanced sampling, step-exact raw/EMA/optimizer/
  scheduler resume with 0/2 workers, preflight reset, immutable epoch-boundary
  retention/external archive and corrupt-label rejection. Report:
  `/public/home/wangzihao/projects/artifacts/job392551/publish_stage2_check_20261027/report.json`.
- Python and shell syntax and whitespace checks passed. Direct HTTP HEAD checks
  for the documented ModelScope language and latest-checkpoint URLs returned 200.
  No Isaac Lab simulation rollout or success-rate measurement was performed here.
