# causal_conv1d v26.6 adaptation

Source: local origin/v26.6.0, commit cf1947bff0e75522f9cc0bba5814e6705dc80aae.
Remote freshness could not be verified because GitHub was unreachable.
ABI checked against fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/docs/aclnnCausalConv1d.md and op_host/causal_conv1d_def.cpp.

Load executor_causal_conv1d_v26_6.py with atk_causal_conv1d_typical_ac_v26_6.json.
Registrations: npu_causal_conv1d_v26_6 and pyaclnn_causal_conv1d_v26_6.
Eight original shape/dtype combinations and accuracy thresholds are preserved.
Four *_cpu arrays become INT64 tensor metadata with fixed values; absent num_accepted_tokens is null.
activation becomes integer activation_mode; pad_slot_id=-1 is explicit.
No legacy CPU aclIntArray slots, nullBlockId or maxQueryLen are passed to ACLNN.
The reference still produces (y, conv_states); ACLNN mutates the original state buffer.
The JSON version field is the existing ATK schema version, not the FLA branch version.
GPU reference retains the existing dependency on gpu_triton_causal_conv1d.py.
Validation: Python compile, allocation-free ABI and reference-adapter checks for all eight cases. No ATK/NPU execution performed.
