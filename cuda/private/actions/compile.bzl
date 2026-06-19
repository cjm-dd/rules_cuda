load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", CC_ACTION_NAMES = "ACTION_NAMES")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("//cuda/private:action_names.bzl", "ACTION_NAMES")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:rules/common.bzl", "ALLOW_CUDA_SRCS")
load("//cuda/private:toolchain.bzl", "find_cuda_toolkit")

_IDENTIFIER_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"

def _sanitize_identifier(value):
    result = []
    for char in value.elems():
        if char in _IDENTIFIER_CHARS:
            result.append(char)
        else:
            result.append("_")
    return "".join(result)

def _stable_module_id(ctx, src, basename_index, pic, rdc):
    raw = "{}_{}_{}_{}_{}_{}".format(
        ctx.label.package,
        ctx.label.name,
        src.short_path,
        basename_index,
        "pic" if pic else "nopic",
        "rdc" if rdc else "nordc",
    )
    return "_rules_cuda_{}".format(_sanitize_identifier(raw))

def _phase_inputs(base_inputs, direct = []):
    return depset(direct = direct, transitive = [base_inputs])

def _declare_nvcc_file(actions, intermediate_dir, filename):
    return actions.declare_file(paths.join(intermediate_dir, filename))

def _cuda_toolkit_root(cuda_toolkit):
    return paths.dirname(cuda_toolkit.cudafe.dirname)

def _cuda_phase_env(cuda_toolkit, env):
    root = _cuda_toolkit_root(cuda_toolkit)
    ret = dict(env)
    ret["LD_LIBRARY_PATH"] = "{}:{}".format(paths.join(root, "lib"), ret.get("LD_LIBRARY_PATH", ""))
    ret["NVVMIR_LIBRARY_DIR"] = paths.join(root, "nvvm/libdevice")
    ret["PATH"] = ":".join([
        paths.join(root, "nvvm/bin"),
        paths.join(root, "bin"),
        ret.get("PATH", ""),
    ])
    ret["TOP"] = root
    return ret

def _cuda_include(cuda_toolkit):
    return paths.join(_cuda_toolkit_root(cuda_toolkit), "include")

def _numeric_cuda_arch(arch):
    for char in arch.elems():
        if not char.isdigit():
            fail("split nvcc compile actions only support numeric CUDA archs, got '{}'".format(arch))
    return arch

def _cuda_arch_macro(arch):
    return "{}0".format(_numeric_cuda_arch(arch))

def _cuda_arch_list(cuda_archs_info):
    arch_macros = []
    seen = {}
    for arch_spec in cuda_archs_info.arch_specs:
        for stage2_arch in arch_spec.stage2_archs:
            arch_macro = _cuda_arch_macro(stage2_arch.arch)
            if arch_macro not in seen:
                seen[arch_macro] = True
                arch_macros.append(arch_macro)
    return ",".join(arch_macros)

def _cxx_standard_for_cuda_frontend(compile_flags):
    for flag in reversed(compile_flags):
        if flag.startswith("-std="):
            std = flag[len("-std="):]
            if std.startswith("gnu++"):
                std = "c++" + std[len("gnu++"):]
            if std.startswith("c++"):
                return "--{}".format(std)
            fail("unsupported CUDA C++ standard flag '{}'".format(flag))
    return None

def _cuda_frontend_flags(cuda_toolkit, cuda_feature_config, compile_flags):
    flags = []
    std = _cxx_standard_for_cuda_frontend(compile_flags)
    if std:
        flags.append(std)
    flags.extend([
        "--clang",
        "--clang_version={}".format(cuda_toolkit.cudafe_clang_version),
        "--display_error_number",
        "--unicode_source_kind=UTF-8",
        "--allow_managed",
    ])
    if cuda_helper.is_enabled(cuda_feature_config, "nvcc_extended_lambda"):
        flags.append("--extended-lambda")
    if cuda_helper.is_enabled(cuda_feature_config, "nvcc_relaxed_constexpr"):
        flags.append("--relaxed_constexpr")
    return flags

def _cuda_common_defines(cuda_feature_config, arch_list, cuda_arch = None, for_preprocess = True):
    defines = [
        "__CUDA_ARCH_LIST__={}".format(arch_list),
        "__NV_LEGACY_LAUNCH",
    ]
    if cuda_arch:
        defines.append("__CUDA_ARCH__={}".format(_cuda_arch_macro(cuda_arch)))
    if for_preprocess:
        defines.extend([
            "__CUDACC__",
            "__NVCC__",
        ])
        if cuda_helper.is_enabled(cuda_feature_config, "nvcc_extended_lambda"):
            defines.append("__CUDACC_EXTENDED_LAMBDA__")
        if cuda_helper.is_enabled(cuda_feature_config, "nvcc_relaxed_constexpr"):
            defines.append("__CUDACC_RELAXED_CONSTEXPR__")
    return defines

def _cuda_version_defines(cuda_toolkit):
    return [
        "__CUDACC_VER_MAJOR__={}".format(cuda_toolkit.version_major),
        "__CUDACC_VER_MINOR__={}".format(cuda_toolkit.version_minor),
        "__CUDACC_VER_BUILD__=0",
    ]

def _add_prefixed(args, prefix, values):
    for value in values:
        args.add(prefix)
        args.add(value)

def _add_defines(args, defines):
    for define in defines:
        args.add("-D{}".format(define))

def _add_include_flags(args, common, cuda_toolkit):
    args.add("-I")
    args.add(_cuda_include(cuda_toolkit))
    _add_prefixed(args, "-I", common.quote_includes)
    _add_prefixed(args, "-I", common.includes)
    _add_prefixed(args, "-isystem", common.system_includes)

def _add_cuda_preprocess_mode_flags(args, cuda_feature_config):
    if cuda_helper.is_enabled(cuda_feature_config, "dbg"):
        args.add_all(["-O0", "-g"])
    elif cuda_helper.is_enabled(cuda_feature_config, "fastbuild"):
        args.add_all(["-O0", "-g1"])
    elif cuda_helper.is_enabled(cuda_feature_config, "opt"):
        args.add_all(["-g1", "-ffunction-sections", "-fdata-sections", "-O3"])
        args.add("-DNDEBUG")

def _add_cuda_preprocess_common_args(
        args,
        common,
        cuda_toolkit,
        cuda_feature_config,
        arch_list,
        cuda_arch = None):
    args.add_all(common.compile_flags)
    _add_defines(args, common.local_defines + common.defines)
    _add_defines(args, common.host_local_defines + common.host_defines)
    _add_defines(args, _cuda_common_defines(cuda_feature_config, arch_list, cuda_arch = cuda_arch))
    _add_defines(args, _cuda_version_defines(cuda_toolkit))
    args.add("-DCUDA_DOUBLE_MATH_FUNCTIONS")
    _add_cuda_preprocess_mode_flags(args, cuda_feature_config)
    args.add("-m64")
    if common.sysroot:
        args.add("--sysroot={}".format(common.sysroot))
    _add_include_flags(args, common, cuda_toolkit)
    args.add_all(["-include", "cuda_runtime.h"])

def _run_host_preprocess(
        actions,
        host_compiler,
        cuda_toolkit,
        cuda_feature_config,
        common,
        env,
        inputs,
        output,
        src,
        arch_list):
    args = actions.args()
    _add_cuda_preprocess_common_args(args, common, cuda_toolkit, cuda_feature_config, arch_list)
    args.add_all(["-E", "-x", "c++"])
    args.add(src.path)
    args.add_all(["-o", output.path])
    actions.run(
        executable = host_compiler,
        arguments = [args],
        outputs = [output],
        inputs = inputs,
        env = env,
        mnemonic = "CudaHostPreprocess",
        progress_message = "CUDA host preprocess %s" % src.path,
    )

def _run_device_preprocess(
        actions,
        host_compiler,
        cuda_toolkit,
        cuda_feature_config,
        common,
        env,
        inputs,
        output,
        src,
        arch_list,
        arch):
    args = actions.args()
    _add_cuda_preprocess_common_args(args, common, cuda_toolkit, cuda_feature_config, arch_list, cuda_arch = arch)
    args.add_all(["-E", "-x", "c++"])
    args.add(src.path)
    args.add_all(["-o", output.path])
    actions.run(
        executable = host_compiler,
        arguments = [args],
        outputs = [output],
        inputs = inputs,
        env = env,
        mnemonic = "CudaDevicePreprocess",
        progress_message = "CUDA device preprocess %s for compute_%s" % (src.path, arch),
    )

def _run_cudafe(
        actions,
        cuda_toolkit,
        cuda_feature_config,
        common,
        env,
        inputs,
        output,
        module_id,
        stub,
        src,
        preprocessed_src):
    args = actions.args()
    args.add_all(_cuda_frontend_flags(cuda_toolkit, cuda_feature_config, common.compile_flags))
    args.add("--orig_src_file_name")
    args.add(src.path)
    args.add("--orig_src_path_name")
    args.add(src.path)
    args.add("--parse_templates")
    args.add("--m64")
    args.add("--gen_c_file_name")
    args.add(output.path)
    args.add("--stub_file_name")
    args.add(stub.path)
    args.add("--module_id_file_name")
    args.add(module_id.path)
    args.add(preprocessed_src.path)
    actions.run(
        executable = cuda_toolkit.cudafe,
        arguments = [args],
        outputs = [output],
        inputs = inputs,
        env = env,
        mnemonic = "CudaCudafe",
        progress_message = "CUDA cudafe %s" % src.path,
    )

def _run_cicc(
        actions,
        cuda_toolkit,
        cuda_feature_config,
        common,
        env,
        inputs,
        outputs,
        src,
        arch_files,
        module_id,
        fatbin_c_basename):
    args = actions.args()
    args.add_all(_cuda_frontend_flags(cuda_toolkit, cuda_feature_config, common.compile_flags))
    args.add("--orig_src_file_name")
    args.add(src.path)
    args.add("--orig_src_path_name")
    args.add(src.path)
    args.add("-arch")
    args.add("compute_{}".format(arch_files.arch))
    args.add_all([
        "-m64",
        "--no-version-ident",
        "-ftz=0",
        "-prec_div=1",
        "-prec_sqrt=1",
        "-fmad=1",
        "--include_file_name",
        fatbin_c_basename,
        "-tused",
        "--module_id_file_name",
        module_id.path,
        "--gen_c_file_name",
        arch_files.cudafe_c.path,
        "--stub_file_name",
        arch_files.stub.path,
        "--gen_device_file_name",
        arch_files.gpu.path,
        arch_files.cpp1_ii.path,
        "-o",
        arch_files.ptx.path,
    ])
    actions.run(
        executable = cuda_toolkit.cicc,
        arguments = [args],
        outputs = outputs,
        inputs = inputs,
        env = env,
        mnemonic = "CudaCicc",
        progress_message = "CUDA cicc %s for compute_%s" % (src.path, arch_files.arch),
    )

def _run_ptxas(
        actions,
        cuda_toolkit,
        common,
        env,
        inputs,
        output,
        src,
        arch,
        ptx):
    args = actions.args()
    args.add("-arch=sm_{}".format(arch))
    args.add("-m64")
    args.add_all(common.ptxas_flags)
    args.add(ptx.path)
    args.add_all(["-o", output.path])
    actions.run(
        executable = cuda_toolkit.ptxas,
        arguments = [args],
        outputs = [output],
        inputs = inputs,
        env = env,
        mnemonic = "CudaPtxas",
        progress_message = "CUDA ptxas %s for sm_%s" % (src.path, arch),
    )

def _run_fatbinary(
        actions,
        cuda_toolkit,
        env,
        inputs,
        outputs,
        src,
        split_files):
    args = actions.args()
    args.add("--create={}".format(split_files.fatbin.path))
    args.add("-64")
    args.add("--cicc-cmdline=-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 ")
    for image in split_files.fatbinary_inputs:
        args.add("--image3=kind=elf,sm={},file={}".format(image.arch, image.file.path))
    args.add("--embedded-fatbin={}".format(split_files.fatbin_c.path))
    actions.run(
        executable = cuda_toolkit.fatbinary,
        arguments = [args],
        outputs = outputs,
        inputs = inputs,
        env = env,
        mnemonic = "CudaFatbinary",
        progress_message = "CUDA fatbinary %s" % src.path,
    )

def _run_host_compile(
        ctx,
        actions,
        cc_toolchain,
        cc_feature_configuration,
        common,
        additional_inputs,
        name,
        src,
        arch_list,
        host_arch,
        pic):
    (_, compilation_outputs) = cc_common.compile(
        actions = actions,
        feature_configuration = cc_feature_configuration,
        cc_toolchain = cc_toolchain,
        name = name,
        srcs = [src],
        user_compile_flags = ctx.fragments.cpp.cxxopts + ctx.fragments.cpp.copts + common.compile_flags + common.host_compile_flags,
        includes = common.includes,
        quote_includes = common.quote_includes,
        system_includes = common.system_includes,
        defines = common.defines + common.host_defines,
        local_defines = common.local_defines + common.host_local_defines + _cuda_common_defines(None, arch_list, cuda_arch = host_arch, for_preprocess = False) + ["CUDA_DOUBLE_MATH_FUNCTIONS"],
        additional_inputs = additional_inputs + common.headers.to_list(),
        disallow_nopic_outputs = pic,
        disallow_pic_outputs = not pic,
    )
    objects = compilation_outputs.pic_objects if pic else compilation_outputs.objects
    if len(objects) != 1:
        fail("expected one generated CUDA host object for {}, got {}".format(src.path, len(objects)))
    return objects[0]

def _declare_nvcc_split_files(actions, intermediate_dir, basename, cuda_archs_info):
    arch_specs = cuda_archs_info.arch_specs
    if not arch_specs:
        fail("split nvcc compile actions require at least one configured CUDA arch")

    host_arch = arch_specs[len(arch_specs) - 1].stage1_arch
    host_stub = None
    per_arch = []
    fatbinary_inputs = []
    for arch_spec in arch_specs:
        arch = arch_spec.stage1_arch
        ptx = _declare_nvcc_file(actions, intermediate_dir, "{}.compute_{}.ptx".format(basename, arch))
        gpu_stage2s = []
        for stage2_arch in arch_spec.stage2_archs:
            if stage2_arch.lto:
                fail("split nvcc compile actions do not yet support lto_{} outputs".format(stage2_arch.arch))
            if stage2_arch.virtual:
                fail("split nvcc compile actions do not yet support virtual compute_{} fatbinary images".format(stage2_arch.arch))
            if stage2_arch.arch != arch:
                fail("split nvcc compile actions do not yet support sm_{} from compute_{}".format(stage2_arch.arch, arch))
            cubin = _declare_nvcc_file(actions, intermediate_dir, "{}.compute_{}.cubin".format(basename, arch))
            gpu_stage2s.append(struct(arch = stage2_arch.arch, cubin = cubin))
            fatbinary_inputs.append(struct(arch = stage2_arch.arch, file = cubin))

        stub = _declare_nvcc_file(actions, intermediate_dir, "{}.compute_{}.cudafe1.stub.c".format(basename, arch))
        if arch == host_arch:
            host_stub = stub
        per_arch.append(struct(
            arch = arch,
            cpp1_ii = _declare_nvcc_file(actions, intermediate_dir, "{}.compute_{}.cpp1.ii".format(basename, arch)),
            cudafe_c = _declare_nvcc_file(actions, intermediate_dir, "{}.compute_{}.cudafe1.c".format(basename, arch)),
            gpu = _declare_nvcc_file(actions, intermediate_dir, "{}.compute_{}.cudafe1.gpu".format(basename, arch)),
            ptx = ptx,
            stage2s = gpu_stage2s,
            stub = stub,
        ))

    if not host_stub:
        fail("could not determine nvcc host stub output")

    return struct(
        cpp4_ii = _declare_nvcc_file(actions, intermediate_dir, "{}.cpp4.ii".format(basename)),
        fatbin = _declare_nvcc_file(actions, intermediate_dir, "{}.fatbin".format(basename)),
        fatbin_c = _declare_nvcc_file(actions, intermediate_dir, "{}.fatbin.c".format(basename)),
        fatbinary_inputs = fatbinary_inputs,
        host_arch = host_arch,
        host_cudafe_cpp = _declare_nvcc_file(actions, intermediate_dir, "{}.compute_{}.cudafe1.cpp".format(basename, host_arch)),
        host_stub = host_stub,
        module_id = _declare_nvcc_file(actions, intermediate_dir, "{}.module_id".format(basename)),
        per_arch = per_arch,
    )

def _run_nvcc_split_compile(
        ctx,
        actions,
        cc_toolchain,
        cc_feature_configuration,
        cuda_toolkit,
        host_compiler,
        cuda_feature_config,
        env,
        base_inputs,
        src,
        basename,
        intermediate_dir,
        module_id,
        cuda_archs_info,
        common,
        pic):
    split_files = _declare_nvcc_split_files(actions, intermediate_dir, basename, cuda_archs_info)
    cuda_env = _cuda_phase_env(cuda_toolkit, env)
    arch_list = _cuda_arch_list(cuda_archs_info)

    actions.write(
        output = split_files.module_id,
        content = module_id,
    )

    _run_host_preprocess(
        actions = actions,
        host_compiler = host_compiler,
        cuda_toolkit = cuda_toolkit,
        cuda_feature_config = cuda_feature_config,
        common = common,
        env = cuda_env,
        inputs = base_inputs,
        output = split_files.cpp4_ii,
        src = src,
        arch_list = arch_list,
    )

    _run_cudafe(
        actions = actions,
        cuda_toolkit = cuda_toolkit,
        cuda_feature_config = cuda_feature_config,
        common = common,
        env = cuda_env,
        inputs = _phase_inputs(base_inputs, [split_files.cpp4_ii, split_files.module_id]),
        output = split_files.host_cudafe_cpp,
        module_id = split_files.module_id,
        stub = split_files.host_stub,
        src = src,
        preprocessed_src = split_files.cpp4_ii,
    )

    for arch_files in split_files.per_arch:
        _run_device_preprocess(
            actions = actions,
            host_compiler = host_compiler,
            cuda_toolkit = cuda_toolkit,
            cuda_feature_config = cuda_feature_config,
            common = common,
            env = cuda_env,
            inputs = base_inputs,
            output = arch_files.cpp1_ii,
            src = src,
            arch_list = arch_list,
            arch = arch_files.arch,
        )
        _run_cicc(
            actions = actions,
            cuda_toolkit = cuda_toolkit,
            cuda_feature_config = cuda_feature_config,
            common = common,
            env = cuda_env,
            inputs = _phase_inputs(base_inputs, [arch_files.cpp1_ii, split_files.module_id]),
            outputs = [arch_files.cudafe_c, arch_files.gpu, arch_files.ptx, arch_files.stub],
            src = src,
            arch_files = arch_files,
            module_id = split_files.module_id,
            fatbin_c_basename = split_files.fatbin_c.basename,
        )
        for stage2_files in arch_files.stage2s:
            _run_ptxas(
                actions = actions,
                cuda_toolkit = cuda_toolkit,
                common = common,
                env = cuda_env,
                inputs = _phase_inputs(base_inputs, [arch_files.ptx]),
                output = stage2_files.cubin,
                src = src,
                arch = stage2_files.arch,
                ptx = arch_files.ptx,
            )

    _run_fatbinary(
        actions = actions,
        cuda_toolkit = cuda_toolkit,
        env = cuda_env,
        inputs = _phase_inputs(base_inputs, [image.file for image in split_files.fatbinary_inputs]),
        outputs = [split_files.fatbin, split_files.fatbin_c],
        src = src,
        split_files = split_files,
    )

    return _run_host_compile(
        ctx = ctx,
        actions = actions,
        cc_toolchain = cc_toolchain,
        cc_feature_configuration = cc_feature_configuration,
        common = common,
        additional_inputs = [split_files.fatbin_c, split_files.host_stub],
        name = "{}_host_compile".format(module_id),
        src = split_files.host_cudafe_cpp,
        arch_list = arch_list,
        host_arch = split_files.host_arch,
        pic = pic,
    )

def compile(
        ctx,
        cuda_toolchain,
        cc_toolchain,
        srcs,
        common,
        pic = False,
        rdc = False,
        _prefix = "_objs"):
    """Perform CUDA compilation, return compiled object files.

    Notes:

    - If `rdc` is set to `True`, then an additional step of device link must be performed.
    - The rules should call this action only once in case srcs have non-unique basenames,
      say `foo/kernel.cu` and `bar/kernel.cu`.

    Args:
        ctx: A [context object](https://bazel.build/rules/lib/ctx).
        cuda_toolchain: A `platform_common.ToolchainInfo` of a cuda toolchain, Can be obtained with `find_cuda_toolchain(ctx)`.
        cc_toolchain: A `CcToolchainInfo`. Can be obtained with `find_cpp_toolchain(ctx)`.
        srcs: A list of `File`s to be compiled.
        common: A cuda common object. Can be obtained with `cuda_helper.create_common(ctx)`
        pic: Whether the `srcs` are compiled for position independent code.
        rdc: Whether the `srcs` are compiled for relocatable device code.
        _prefix: DON'T USE IT! Prefix of the output dir. Exposed for device link to redirect the output.

    Returns:
        An compiled object `File`.
    """
    actions = ctx.actions
    cc_feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    host_compiler = cc_common.get_tool_for_action(feature_configuration = cc_feature_configuration, action_name = CC_ACTION_NAMES.cpp_compile)
    cuda_toolkit = find_cuda_toolkit(ctx)

    cuda_feature_config = cuda_helper.configure_features(ctx, cuda_toolchain, requested_features = [ACTION_NAMES.cuda_compile])
    artifact_category_name = cuda_helper.get_artifact_category_from_action(ACTION_NAMES.cuda_compile, pic, rdc)

    basename_counter = {}
    src_and_indexed_basenames = []
    for src in srcs:
        # this also filter out all header files
        basename = cuda_helper.get_basename_without_ext(src.basename, ALLOW_CUDA_SRCS, fail_if_not_match = False)
        if not basename:
            continue
        basename_index = basename_counter.setdefault(basename, default = 0)
        basename_counter[basename] += 1
        src_and_indexed_basenames.append((src, basename, basename_index))

    ret = []
    for src, basename, basename_index in src_and_indexed_basenames:
        filename = None
        filename = cuda_helper.get_artifact_name(cuda_toolchain, artifact_category_name, basename)

        # Objects are placed in <_prefix>/<tgt_name>/<filename>.
        # For files with the same basename, say srcs = ["kernel.cu", "foo/kernel.cu", "bar/kernel.cu"], we get
        # <_prefix>/<tgt_name>/0/kernel.<ext>, <_prefix>/<tgt_name>/1/kernel.<ext>, <_prefix>/<tgt_name>/2/kernel.<ext>.
        # Otherwise, the index is not presented.
        if basename_counter[basename] > 1:
            filename = "{}/{}".format(basename_index, filename)
        output_file = "{}/{}/{}".format(_prefix, ctx.attr.name, filename)

        var = cuda_helper.create_compile_variables(
            cuda_toolchain,
            cuda_feature_config,
            common.cuda_archs_info,
            common.sysroot,
            source_file = src.path,
            output_file = output_file,
            host_compiler = host_compiler,
            compile_flags = common.compile_flags,
            host_compile_flags = common.host_compile_flags,
            include_paths = common.includes,
            quote_include_paths = common.quote_includes,
            system_include_paths = common.system_includes,
            defines = common.local_defines + common.defines,
            host_defines = common.host_local_defines + common.host_defines,
            ptxas_flags = common.ptxas_flags,
            use_pic = pic,
            use_rdc = rdc,
        )
        env = cuda_helper.get_environment_variables(cuda_feature_config, ACTION_NAMES.cuda_compile, var)

        cuda_subtools = [
            cuda_toolkit.cicc,
            cuda_toolkit.cudafe,
            cuda_toolkit.fatbinary,
            cuda_toolkit.ptxas,
        ]
        inputs = depset(
            direct = [src] + cuda_subtools,
            transitive = [common.headers, cc_toolchain.all_files, cuda_toolchain.all_files],
        )
        ret.append(_run_nvcc_split_compile(
            ctx = ctx,
            actions = actions,
            cc_toolchain = cc_toolchain,
            cc_feature_configuration = cc_feature_configuration,
            cuda_toolkit = cuda_toolkit,
            host_compiler = host_compiler,
            cuda_feature_config = cuda_feature_config,
            env = env,
            base_inputs = inputs,
            src = src,
            basename = basename,
            intermediate_dir = "{}/{}/{}.nvcc-keep".format(_prefix, ctx.attr.name, filename),
            module_id = _stable_module_id(ctx, src, basename_index, pic, rdc),
            cuda_archs_info = common.cuda_archs_info,
            common = common,
            pic = pic,
        ))
    return ret
