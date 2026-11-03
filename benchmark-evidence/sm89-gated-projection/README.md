# SM89 gated projection experiment evidence

Frozen validation data for the opt-in BF16 gated-projection PR.

- GPU-tested base: `b2a9480dc7aa2e8e6aacd09df1a375c34c55f5b5` plus the exact source files in the archive manifest.
- Published code was rebased onto `a8718969d2c3910625e381d503ecde28b4b50ab7`; the six proposed file contents are unchanged. The recorded GPU results are not a rerun of the newer framework.
- [Download the evidence ZIP](sm89-gated-projection-evidence.zip).
- ZIP SHA256: `301300822fe318d9b88c7a19d4f666b39235bb7ee35f85e166f71b73933f3b75`.

Includes the source snapshot, model/runtime hashes, frozen public-code timing input with Apache-2.0 attribution, serving reproduction scripts, paired latency data, per-question numeric GSM8K outcomes, and validation logs. No model weights or full model responses are included. The main code PR does not include this evidence branch.
