    def _apply_noncontig_cache(self, input_data: InputDataset):
        """把 kv/kr cache 换成只有 dim0 非连续的视图。

        参考节点仍然拿到逻辑值，所以 golden 不受布局影响；这里改的只是被测算子
        看到的物理布局。ATK 建 aclTensor 时会带上 torch tensor 的真实 stride，
        算子 Tiling 侧再用 GetInputStride 取到 stride(0)。
        """
        kw = input_data.kwargs
        self.nc_caches: Dict[str, Optional[NonContigCache]] = {"kv_cache": None, "kr_cache": None}
        if self.nc_factor <= 1:
            return
        for name in ("kv_cache", "kr_cache"):
            tensor = kw.get(name)
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            cache = make_dim0_noncontig(tensor, self.nc_factor)
            self.nc_caches[name] = cache
            kw[name] = cache.view
        logging.info("[mla_prolog][nc] %s", describe(self.nc_caches))

    def init_by_input_data(self, input_data: InputDataset):
        self._acl_holders: List[torch.Tensor] = []
        seed = parse_case_seed(self._case_name())
        kw = input_data.kwargs
        _sanitize_optional_kwargs(kw)
        _rewrite_quant_gen_inputs(input_data, seed=seed)
        # gen_inputs 塞进 kwargs 的 _dequant_scale_*_fp 只给 golden 用，不能进 acl 原型
        for key in [k for k in list(kw) if k.startswith("_")]:
            kw.pop(key)

        _apply_deterministic_cache_index(input_data, seed=seed)
        self.enable_rope = _resolve_do_rope(kw)
        self.weight_quant_mode = int(kw.get("weightQuantMode", -1))
        tensor_slots, attr_slots, _, nz_names = self._layout()
        # inner infershape 缺省 doRope=true；doRope=false 的 bool 空槽会被 ATK 转成空
        # aclTensor。始终生成合法 (B,S,Dr)/(T,Dr) 表，super() 之后再写回 ACL 槽。
        _rewrite_rope_tables(input_data, seed=seed)
        rope_hold = {}
        for name in ("rope_sin", "rope_cos"):
            tensor = kw.get(name)
            if isinstance(tensor, torch.Tensor) and tensor.dtype != torch.bool and tensor.numel() > 0:
                rope_hold[name] = tensor.contiguous()
                if not self.enable_rope:
                    kw.pop(name, None)
        ordered = {}
        for name in tensor_slots:
            if name in kw and isinstance(kw[name], torch.Tensor):
                ordered[name] = kw[name]
        for key, value in kw.items():
            if key not in ordered:
                ordered[key] = value
        kw.clear()
        kw.update(ordered)
        self.nc_factor = parse_stride_factor(self._case_name())
        self.query_quant_mode = int(kw.get("queryQuantMode", 0))

        declared = [n for n, v in kw.items() if isinstance(v, torch.Tensor)]
        _move_to_npu(input_data, nz_names)
        self._apply_noncontig_cache(input_data)
        for name, held in list(rope_hold.items()):
            if held.device.type != "npu":
                rope_hold[name] = held.npu().contiguous()
        # kvCacheRef / krCacheRef are updated in place, so keep the device tensors around to
        # read the operator result back after the call. 非连续用例读的始终是逻辑视图。
        self.cache_refs = tuple(
            self.nc_caches[name].view if self.nc_caches.get(name) else input_data.kwargs[name]
            for name in ("kv_cache", "kr_cache")
        )

        if self.query_quant_mode == 1:
            t = _token_t(kw["token_x"])
            n = int(kw["weight_uk"].shape[0])
            self.dequant_q_nope = torch.zeros(
                (t, n, 1), dtype=torch.float32, device=kw["token_x"].device
            )
        else:
            self.dequant_q_nope = None

        # 对齐 mlapo：输出缓冲用 zeros 而不是 empty，避免未写满时读到脏数据。
        input_args, output_packages = self._init_with_zero_outputs(input_data)
        # optional 输出槽位自己拼：qq=1 时 dequantScaleQNopeOut 必须非空
        input_args = self._expand_to_prototype(
            list(input_args), declared, tensor_slots, attr_slots, len(output_packages), 0
        )
        for name, held in rope_hold.items():
            self._acl_holders.append(held)
            ptr = _acl_create_tensor(
                held, int(AclDataType.ACL_BF16), int(AclFormat.ACL_FORMAT_ND)
            )
            slot = tensor_slots.index(name)
            if slot < len(input_args):
                input_args[slot] = AclTensorStruct(
                    tensor=ptr,
                    addr=int(held.untyped_storage().data_ptr()),
                    pytensor=held,
                    data_size=int(held.numel()),
                )
            logging.info(
                "[mla_prolog][rope] DUT overwrite %s shape=%s numel=%s slot=%s",
                name, tuple(held.shape), int(held.numel()), slot,
            )
        if rope_hold and "doRope" in attr_slots:
            do_rope_slot = len(tensor_slots) + attr_slots.index("doRope")
            if do_rope_slot < len(input_args):
                input_args[do_rope_slot] = ctypes.c_bool(True)
                logging.info("[mla_prolog][rope] DUT force doRope=True to match injected tables")
        if self.dequant_q_nope is not None:
            dq_struct = self.torch_tensor_to_acl(
                self.dequant_q_nope, AclFormat.ACL_FORMAT_ND
            )
            input_args.append(dq_struct.tensor)
            self._acl_holders.append(self.dequant_q_nope)
        else:
            input_args.append(_null_tensor_ptr())
        input_args.append(_null_tensor_ptr())  # queryNormOutOptional
        input_args.append(_null_tensor_ptr())  # dequantScaleQNormOutOptional
        if hasattr(self, "backend") and self.backend is not None:
            self.backend.input_args = input_args
        return input_args, output_packages

    @staticmethod
    def _expand_to_prototype(input_args, declared, tensor_slots, attr_slots,
                             n_outputs, null_outputs):
        """ATK converts the declared inputs positionally; splice in a null aclTensor for every
        optional slot the case leaves out so the argument list matches the aclnn prototype."""
        unknown = [n for n in declared if n not in tensor_slots]
        if unknown:
            raise ValueError(f"case declares tensors that are not in the prototype: {unknown}")
        if list(declared) != [n for n in tensor_slots if n in declared]:
            raise ValueError("case tensors must be declared in aclnn prototype order")

        n_declared = len(declared)
        tail = input_args[n_declared:]
        if len(tail) != len(attr_slots) + n_outputs:
            raise ValueError(
                f"expected {len(attr_slots)} attrs + {n_outputs} outputs, got {len(tail)}"
            )

        it = iter(input_args[:n_declared])
        remaining = list(declared)
        expanded = []
        for slot in tensor_slots:
            if remaining and remaining[0] == slot:
                expanded.append(next(it))
                remaining.pop(0)
            else:
                expanded.append(_null_tensor_ptr())
        expanded.extend(tail)
        expanded.extend(_null_tensor_ptr() for _ in range(null_outputs))
        return expanded

    def _init_with_zero_outputs(self, input_data: InputDataset):
        """参考 mlapo_atk/exec.py：把 ATK 的 output 分配改成 torch.zeros。

        参考节点为绕开 ATK FP8 isinf 会以 float32 申报 query；这里按算子真实 dtype
        分配 queryOut：qq=1 时 wq=2→int8 / wq=3,4→fp8 / wq=5→HIFLOAT8。
        """
        backend = self.backend
        if not hasattr(backend, "convert_output_data"):
            return super().init_by_input_data(input_data)

        query_runtime_dtype = _query_out_runtime_dtype(input_data.kwargs)
        # ATK 对每个 output 回调时 index 从 0 递增；output_packages 顺序为 query, query_rope
        out_slot = {"i": 0}

        def my_convert_output_data(backend_instance, output_data, index):
            if isinstance(output_data, (list, tuple)):
                data_list = []
                for tmp in output_data:
                    data_list.extend(my_convert_output_data(backend_instance, tmp, index))
                return [nnopbase.create_x_list(data_list)]
            dtype = output_data.dtype
            torch_dtype = getattr(torch, dtype.replace("torch.", ""))
            slot = out_slot["i"]
            out_slot["i"] += 1
            if slot == 0:
                if query_runtime_dtype == torch.float8_e4m3fn:
                    torch_dtype = torch.float8_e4m3fn
                    dtype = "torch.float8_e4m3fn"
                elif query_runtime_dtype == torch.int8:
                    torch_dtype = torch.int8
                    dtype = "torch.int8"
                elif query_runtime_dtype == torch.uint8:
                    empty_tensor = torch.zeros(
                        tuple(output_data.shape), dtype=torch.uint8, device="npu"
                    )
                    backend_instance.output_cache.append(empty_tensor)
                    cur_index = index + len(backend_instance.input_args)
                    fmt = backend_instance.get_format(index=cur_index)
                    return [self.torch_tensor_to_acl(empty_tensor, fmt)]
            if dtype not in TORCH_TO_ACLTYPE:
                raise ValueError(f"TORCH_TO_ACLTYPE不支持的dtype：{dtype}")
            empty_tensor = torch.zeros(tuple(output_data.shape), dtype=torch_dtype, device="npu")
            backend_instance.output_cache.append(empty_tensor)
            cur_index = index + len(backend_instance.input_args)
            fmt = backend_instance.get_format(index=cur_index)
            storage_shape = backend_instance.get_storage_shape(index=cur_index)
            out_tensor = nnopbase.create_acl_tensor(empty_tensor, fmt, storage_shape)
            return [out_tensor]

        hif8 = (
            int(getattr(self, "weight_quant_mode", -1)) == 5
            and hasattr(AclDataType, "ACL_HIFLOAT8")
        )
        orig_out = backend.convert_output_data
        old_u8 = TORCH_TO_ACLTYPE.get("torch.uint8")
        try:
            backend.convert_output_data = types.MethodType(my_convert_output_data, backend)
            if hif8:
                # ATK 默认 uint8 → ACL_UINT8；wq=5 的 uint8 是 HiFloat8 存储。
                TORCH_TO_ACLTYPE["torch.uint8"] = int(AclDataType.ACL_HIFLOAT8)
            return super().init_by_input_data(input_data)
        finally:
            backend.convert_output_data = orig_out
            if old_u8 is not None:
                TORCH_TO_ACLTYPE["torch.uint8"] = old_u8

    def __call__(self):
        torch.npu.synchronize()
        self.backend.aclnn_x_get_workspace_size()
        torch.npu.synchronize()
        self.backend.aclnn_x()
        torch.npu.synchronize()

    def after_call(self, output_packages):
        torch.npu.synchronize()
        outs = list(super().after_call(output_packages))
        torch.npu.synchronize()
        if self.nc_factor > 1:
            kv_cache, kr_cache = self.cache_refs
            padding_ok, padding_detail = padding_report(self.nc_caches)
            logging.info(
                "[mla_prolog][nc] factor=%s kv_stride0=%s kr_stride0=%s padding=%s %s",
                self.nc_factor,
                int(kv_cache.stride(0)) if kv_cache.dim() else 0,
                int(kr_cache.stride(0)) if kr_cache.dim() else 0,
                "ok" if padding_ok else "POLLUTED", padding_detail,
            )
        # FP8 / HiFloat8 → FP32 再交给 ATK 双标杆（规避 isinf 不支持这些 dtype）
        qq = int(getattr(self, "query_quant_mode", 0))
        wq = int(getattr(self, "weight_quant_mode", -1))
        if (
            outs
            and qq == 1
            and wq in (2, 3, 4, 5)
            and isinstance(self.dequant_q_nope, torch.Tensor)
        ):
            q = outs[0].detach().cpu()
            scale = _reshape_dequant_for_query(
                self.dequant_q_nope.detach().cpu(), q
            )
            if wq == 5:
                q = _hif8_decode(q)
            else:
                q = q.to(torch.float32)
            outs[0] = q * scale
        return tuple(_to_compare_dtype(t) for t in outs)


# ---------------------------------------------------------------------------
# V4 ATK register entries
# ---------------------------------------------------------------------------

@register("function_mla_prolog_v4")
class FunctionMlaPrologV4(FunctionMlaPrologBaseApi):
    """Reference node for ATK default cv_fused_double_benchmark (CPU / NPU small-op)."""

    op_version = "v4"


@register("pyaclnn_function_mla_prolog_v4")
class PyAclnnMlaPrologV4(AclnnMlaPrologBaseApi):
    """DUT: aclnnMlaPrologV4WeightNz via ATK opp path."""

    op_version = "v4"

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnMlaPrologV4WeightNzGetWorkspaceSize("
            "const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, "
            "const aclTensor *weightUk, const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq, "
            "const aclTensor *rmsnormGammaCkv, const aclTensor *ropeSin, const aclTensor *ropeCos, "
            "aclTensor *kvCacheRef, aclTensor *krCacheRef, const aclTensor *cacheIndexOptional, "
            "const aclTensor *dequantScaleXOptional, const aclTensor *dequantScaleWDqOptional, "
            "const aclTensor *dequantScaleWUqQrOptional, const aclTensor *dequantScaleWDkvKrOptional, "
            "const aclTensor *quantScaleCkvOptional, const aclTensor *quantScaleCkrOptional, "
            "const aclTensor *smoothScalesCqOptional, const aclTensor *actualSeqLenOptional, "
            "const aclTensor *kNopeClipAlphaOptional, double rmsnormEpsilonCq, double rmsnormEpsilonCkv, "
            "char *cacheModeOptional, int64_t weightQuantMode, int64_t kvCacheQuantMode, "
            "int64_t queryQuantMode, int64_t ckvkrRepoMode, int64_t quantScaleRepoMode, int64_t tileSize, "
            "double qcQrScale, double kcScale, bool doRope, const aclTensor *queryOut, "
            "const aclTensor *queryRopeOut, "
            "const aclTensor *dequantScaleQNopeOutOptional, const aclTensor *queryNormOutOptional, "
            "const aclTensor *dequantScaleQNormOutOptional, "
            "uint64_t *workspaceSize, aclOpExecutor **executor)"
        )
