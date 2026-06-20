CUDA_COMPILE = "cuda-compile"  # cuda compile comprise of host and device compilation

CUDA_PREPROCESS = "cuda-preprocess"

CUDA_FRONTEND = "cuda-frontend"

CUDA_DEVICE_COMPILE = "cuda-device-compile"

CUDA_ASSEMBLE = "cuda-assemble"

CUDA_FATBINARY = "cuda-fatbinary"

CUDA_DEVICE_LINK = "cuda-dlink"

ACTION_NAMES = struct(
    cuda_assemble = CUDA_ASSEMBLE,
    cuda_compile = CUDA_COMPILE,
    cuda_device_compile = CUDA_DEVICE_COMPILE,
    cuda_fatbinary = CUDA_FATBINARY,
    cuda_frontend = CUDA_FRONTEND,
    cuda_preprocess = CUDA_PREPROCESS,
    device_link = CUDA_DEVICE_LINK,
)
