import torch
import psutil
import GPUtil

def get_vm_usage_metrics():
    """
    Prints the VM usage metrics.
    """
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')
    # CPU usage
    cpu_load = psutil.cpu_percent(interval=1, percpu=True)
    for id, load in enumerate(cpu_load):
        print(f"CPU {id} load: {load:.2f}")
    # RAM usage
    ram = psutil.virtual_memory()
    print(f"RAM Total: {ram.total/(1024**3):.2f} GB, Used: {(ram.used)/(1024**3):.2f} GB")
    # GPU
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id} ({gpu.name}) load: {gpu.load*100}%")
            print(f"GPU {gpu.id} ({gpu.name}) VRAM Total: {gpu.memoryTotal} MB, Used {gpu.memoryUsed} MB")
    # Disk 
    disk = psutil.disk_usage('/')
    print(f"Disk Total: {disk.total/(1024**3):.2f} GB, Used: {(disk.used)/(1024**3):.2f} GB")



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || All params: {all_param} || Trainable %: {100 * trainable_params / all_param:.2f}"
    )


def get_optimal_training_config():
    if not torch.cuda.is_available():
        config = {
            'bf16': False, 
            'fp16': False, 
            'tf32': False,
            'attn_implementation': 'sdpa',
            'packing': False,
            'liger_kernel': False,
        }
        print("No GPU detected. Using CPU with conservative training config:")
        print(config)
        return config

    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    cc_major = compute_capability[0]

    # Determine attention implementation based on compute capability
    # Flash Attention 3: Hopper (sm_90+) → H100, H200
    # Flash Attention 2: Ampere/Ada (sm_80+) → A100, A10, A30, RTX 3090/4090...
    # SDPA: fallback for older architectures (T4, V100, etc.)
    if cc_major >= 9:
        attn_implementation = 'flash_attention_3'
    elif cc_major >= 8:
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = 'sdpa'

    packing = attn_implementation in ('flash_attention_2', 'flash_attention_3')

    # Liger Kernel requires Ampere+ (sm_80+) — same boundary as bf16
    # Fused Triton kernels (RMSNorm, RoPE, SwiGLU, etc.) fail on Turing/T4
    liger_kernel = cc_major >= 8

    # Determine precision
    # H100 / Hopper (sm_90+)
    if cc_major >= 9 or 'H100' in gpu_name:
        precision = {'bf16': True, 'fp16': False, 'tf32': True}

    # A100, A10, other Ampere/Ada (sm_80+)
    elif cc_major >= 8 or any(x in gpu_name for x in ['A100', 'A10']):
        precision = {'bf16': True, 'fp16': False, 'tf32': True}

    # T4, V100 and older — no BF16, no Liger
    else:
        precision = {'bf16': False, 'fp16': True, 'tf32': False}

    config = {
        **precision,
        'attn_implementation': attn_implementation,
        'packing': packing,
        'liger_kernel': liger_kernel,
    }

    gpu_count = torch.cuda.device_count()
    gpu_info = f"{gpu_count}x {gpu_name} (sm_{compute_capability[0]}{compute_capability[1]})" if gpu_count > 1 else f"{gpu_name} (sm_{compute_capability[0]}{compute_capability[1]})"
    print(f"Detected GPU: {gpu_info}")
    print(f"Training config: {config}")

    return config

# precision_config = get_optimal_precision()
# # Enable TF32 for Ampere+ GPUs
# if precision_config.get('tf32', False):
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True