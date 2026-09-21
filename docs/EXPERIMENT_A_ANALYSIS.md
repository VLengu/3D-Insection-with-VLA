# Experiment A stop report

## Decision

Experiment A failed its predefined continuation gate. Experiments B-E must not
run until this conditioning-to-action disconnect is resolved.

## Paired-test result

| Metric | Full-3D baseline | Experiment A | A target |
|---|---:|---:|---:|
| success | 0.404 | 0.463 | diagnostic only |
| collision rate | 0.571 | 0.493 | diagnostic only |
| path mode accuracy | 0.533 | 0.541 | >= 0.750 |
| paired decisions differ | 0.164 | 0.179 | >= 0.600 |
| paired both modes correct | 0.115 | 0.131 | diagnostic only |
| route classifier accuracy | n/a | 1.000 | diagnostic only |

## Interpretation

The route head perfectly decodes the height-derived class, including every OOD
split, but the generated trajectory barely reacts. This is direct evidence of
a condition-to-action utilization failure: decoding a 3D label and adding its
embedding to a global modulation vector does not make the denoiser implement
the corresponding 3D route.

Likely causes, in priority order:

1. Route classification and ranking supervise only logits; neither loss
   requires the generated low/high trajectories to diverge.
2. The soft route embedding is globally added to the time/AdaLN condition, so
   the action network can learn to ignore it while the independent classifier
   remains perfect.
3. Reconstructed-trajectory supervision is masked to relatively clean
   diffusion steps, whereas route topology should be established at high
   noise.
4. A dense 24-step delta trajectory dilutes the few obstacle-adjacent steps
   that identify over versus around.

## Required analysis before B

- measure counterfactual action sensitivity to replacing/swapping only the
  route embedding while holding the full geometry fixed;
- compare predicted-route embedding with forced ground-truth route embedding;
- measure gradient magnitude from diffusion/trajectory loss into the route
  encoder and route embedding;
- decide between action-level paired supervision and a discrete hierarchical
  route-conditioned denoiser.

## Route-control probe result

The probe forced over, forced around, swapped, predicted, and zero route
embeddings while holding RGB, geometry, and initial diffusion noise fixed.

| Metric | Forced over | Forced around |
|---|---:|---:|
| footprint crossing rate | 0.785 | 0.683 |
| maximum Z | 0.226 | 0.040 |
| endpoint distance | 0.056 | 0.043 |

Forced over versus around produced a large trajectory L2 difference (1.039)
and maximum-Z difference (0.185), but changed the footprint-defined route mode
for only 17.9% of samples. Swapping pair route embeddings inverted the output
of an independent trajectory classifier while footprint-defined mode accuracy
fell below chance.

Therefore the route pathway is not ignored. It controls a vertical motion
primitive almost perfectly (over means lift; around means stay low), but does
not use obstacle-relative XY geometry to construct a lateral detour. The
independent classifier reaches 100% on expert trajectories and generated
predicted trajectories because it can recognize the vertical primitive; it is
not evidence that the generated path is collision-free or semantically around.

This rules out the original binary diagnosis. The failure is specifically a
global-route-to-local-geometry grounding failure. Collision loss alone may
push paths away from the obstacle but would not establish that the route token
is spatially grounded. The next candidate should expose object/obstacle tokens
to action tokens through geometry cross-attention, then repeat Experiment A's
causal probe before proceeding to Experiment B.

## Experiment A1: spatial-consistency test

A1 was split into two preregistered sub-experiments. A1a used inference-only
route interventions. A1b retained the Experiment A architecture and added
training-time spatial consistency: signed distance to the expanded obstacle
footprint, obstacle-local expert XY supervision, and paired low/high XY-delta
supervision. Paired samples shared diffusion timestep and noise.

The signed-distance metric treats the obstacle as a bounded rectangle rather
than a center point. A positive minimum distance means the entire sampled path
clears the expanded footprint; a negative value means the XY path enters it.

### A1a result

| Metric | Forced over | Forced around | Causal delta |
|---|---:|---:|---:|
| maximum Z | 0.226 | 0.040 | 0.186 |
| footprint clearance rate | 0.212 | 0.295 | 0.083 |
| minimum XY signed distance | -0.057 | -0.041 | 0.017 |
| footprint-defined mode flip | - | - | 0.169 |

A1a therefore confirms that inference intervention alone exposes a vertical
primitive but not a reliable lateral detour.

### A1b result

| Paired-test metric | Experiment A | A1b | Continuation gate |
|---|---:|---:|---:|
| success | 0.463 | 0.600 | diagnostic |
| collision rate | 0.493 | 0.365 | diagnostic |
| path mode accuracy | 0.541 | 0.747 | >= 0.750 |
| paired decisions differ | 0.179 | 0.567 | >= 0.600 |
| paired both modes correct | 0.131 | 0.531 | >= 0.500 |
| endpoint distance | 0.041 | 0.040 | no regression |

The matched-noise causal probe gives a slightly different sampling estimate
of predicted path-mode accuracy (0.763). More importantly, forced over versus
forced around changes the footprint-defined mode for 0.571 of samples, up from
0.169 in A1a. Forced-around clearance rises from 0.295 to 0.739 and its mean
minimum signed distance changes from -0.041 to +0.022. Maximum Z remains
route-sensitive (0.192 over versus 0.043 around).

### Stop decision

A1b demonstrates that the route-to-XY mapping is learnable with direct local
spatial supervision; the gain is not merely vertical route classification.
It nevertheless misses two preregistered continuation gates: paired decisions
differ (0.567 < 0.600) and forced mode flip (0.571 < 0.600). The paired-test
path-mode accuracy is also marginally below its gate (0.747 < 0.750), despite
the matched-noise probe estimate being above it.

Accordingly, stop after A1b. Do not start A-prime or Experiment B until the
borderline result is reviewed. The result supports A-prime's spatial binding
motivation, but does not yet justify claiming that the present global route
pathway has solved 3D spatial understanding.

## Experiment A-prime: geometry cross-attention

A-prime replaced global `z3d` AdaLN with five typed geometry tokens (start,
goal, expanded obstacle collision volume, start-goal/obstacle relations, and
camera frame). Every action token queried these tokens through a gated
cross-attention layer in every denoising block. Time and route topology stayed
as global AdaLN conditions. Data, paired noise, losses, loss weights, model
width/depth, training epochs, and seed matched A1b.

Checkpoint selection was also corrected: validation paired topology metrics,
not `paired_test`, now select the saved model. The held-out paired test is
reported only after the checkpoint is fixed.

### Main result

| Paired-test metric | A1b | A-prime | Gate |
|---|---:|---:|---:|
| success | 0.600 | 0.401 | >= 0.600 |
| collision rate | 0.365 | 0.327 | diagnostic |
| endpoint distance | 0.040 | 0.067 | <= 0.045 |
| path mode accuracy | 0.747 | 0.779 | >= 0.750 |
| paired decisions differ | 0.567 | 0.587 | >= 0.600 |
| paired both modes correct | 0.531 | 0.572 | >= 0.500 |

The architecture improves discrete geometric mode selection and slightly
reduces collision, but substantially degrades goal completion. Terminal
direction alignment in the matched-noise probe falls to -0.030, consistent
with obstacle geometry continuing to perturb action tokens after the local
avoidance decision should already be complete.

### Causal probe

| Metric | A1b | A-prime |
|---|---:|---:|
| predicted mode accuracy | 0.763 | 0.791 |
| forced mode flip | 0.571 | 0.364 |
| forced-around clearance | 0.739 | 0.581 |
| forced-over crossing | 0.789 | 0.737 |
| forced endpoint delta | 0.051 | 0.074 |

The higher predicted mode accuracy proves that geometry tokens are used. The
lower forced-route effect shows that the direct geometry stream competes with,
rather than cleanly binds to, the global route embedding: overriding the route
does not override the unchanged obstacle-height token. This probe therefore
measures route controllability as well as spatial grounding and must be
interpreted together with matched counterfactual-pair behavior.

### Stop decision

A-prime fails four preregistered gates: success, endpoint distance, paired
decisions differ, and forced mode flip. Do not run the pooled-token control,
additional training seeds, or Experiment B. The next revision should not add
more supervision indiscriminately. It must resolve two specific issues:

1. separate topology selection from continuous collision geometry so route
   and geometry do not provide conflicting commands;
2. localize or gate geometry cross-attention in trajectory/diffusion time so
   obstacle information does not corrupt terminal goal convergence.
