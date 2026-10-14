# BiCAT Molmo KI training schedule

The current Molmo KI policy keeps the Molmo2-ER ViT/connector and uses one
378 x 378 resized image per camera. The BTT tokenizer remains frozen.

- Completed optimizer updates 0–999: action-token CE only. The VLM,
  vision encoder, connector and continuous state projection learn from CE;
  the continuous action expert is not run or updated.
- At completed optimizer update 1000 and thereafter: CE + flow matching.
  The VLM's per-layer KV tensors are detached before expert conditioning.
  CE continues updating the VLM; flow updates the expert and its KV
  projections, without updating the VLM or state projection.

These settings are `policy.ce_warmstart_steps=1000` and
`policy.action_expert_detach_vlm=true`. Phase selection uses the trainer's
optimizer-step counter and therefore also applies after checkpoint resume.
Loading an old policy must explicitly override a saved false detach setting.

The 100000-step run uses all 100 box episodes without a validation split,
batch size 96 and gradient accumulation 1. It has 1000 CE-only updates and 99000 joint
updates. Existing ten-step smoke tests starting at step zero now measure
only the CE phase; earlier joint-loss throughput measurements do not apply
to that phase.

Warmup is linear for 500 updates for the VLM, connector and action expert,
and 1000 updates for the vision encoder. Each family then follows cosine
decay to `trainer.min_lr_ratio` (0.1) times its peak learning rate.

`test_ki_molmo2er_stages_bs96.pbs` runs real batch-96 optimizer updates
around the configured CE/flow boundary, plus a zero-CE probe of flow gradient
isolation. It jumps the counter rather than executing all preceding updates, and
does not test checkpoint save/resume. The test aligns learning rates to
the counter and checks the configured warmup and cosine endpoints.

Peak learning rates are VLM/vision/connector `1e-4` and expert `2e-4`.
At step 100000 they reach `1e-5` and `2e-5`, respectively.
