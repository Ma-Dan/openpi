#!/usr/bin/env python3
"""启动 Libero PyTorch 策略服务器。

此脚本用于加载 PyTorch 格式的 Libero 策略模型并提供 WebSocket 服务。
"""

import dataclasses
import logging
import os
import socket

import torch
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


# PyTorch 模型默认路径
DEFAULT_PYTORCH_CHECKPOINT_DIR = os.path.expanduser(
    "~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
)


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi05_libero").
    config: str = "pi05_libero"
    # Checkpoint directory.
    dir: str = DEFAULT_PYTORCH_CHECKPOINT_DIR


@dataclasses.dataclass
class Args:
    """Arguments for the serve_libero_policy_pytorch script."""

    # Checkpoint configuration.
    checkpoint: Checkpoint = dataclasses.field(default_factory=Checkpoint)

    # If provided, will be used in case the "prompt" key is not present in the data,
    # or if the model doesn't have a default prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000

    # Record the policy's behavior for debugging.
    record: bool = False

    # Device to use for PyTorch inference. If not specified, will auto-detect.
    # Options: "cuda", "cuda:0", "mps", "cpu"
    device: str | None = None


def get_best_device() -> str:
    """自动检测最佳设备。

    Returns:
        设备名称字符串
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def print_device_info(device: str):
    """打印设备信息。"""
    logging.info("=" * 50)
    logging.info("设备信息")
    logging.info("=" * 50)
    logging.info(f"  PyTorch 版本: {torch.__version__}")
    logging.info(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"  CUDA 版本: {torch.version.cuda}")
        logging.info(f"  CUDA 设备数量: {torch.cuda.device_count()}")
        logging.info(f"  CUDA 设备名称: {torch.cuda.get_device_name(0)}")

    logging.info(f"  MPS 可用: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        logging.info(f"  MPS 已构建: {torch.backends.mps.is_built()}")

    logging.info(f"  使用设备: {device}")
    logging.info("=" * 50)


def create_policy(args: Args) -> _policy.Policy:
    """Create a PyTorch policy from the given arguments."""
    # 确定设备
    device = args.device if args.device else get_best_device()
    print_device_info(device)

    # 检查 checkpoint 目录是否存在
    checkpoint_dir = os.path.expanduser(args.checkpoint.dir)
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint 目录不存在: {checkpoint_dir}")

    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"PyTorch 模型文件不存在: {safetensors_path}")

    logging.info(f"加载 PyTorch 模型...")
    logging.info(f"  配置: {args.checkpoint.config}")
    logging.info(f"  Checkpoint 目录: {checkpoint_dir}")
    logging.info(f"  模型文件: {safetensors_path}")

    # 获取配置
    config = _config.get_config(args.checkpoint.config)

    # 创建策略，指定 PyTorch 设备
    policy = _policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=device,
    )

    return policy


def main(args: Args) -> None:
    """主函数。"""
    logging.info("启动 Libero PyTorch 策略服务器...")

    # 创建策略
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("创建服务器 (主机: %s, IP: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )

    logging.info("服务器启动在 ws://0.0.0.0:%d", args.port)
    logging.info("等待客户端连接...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
