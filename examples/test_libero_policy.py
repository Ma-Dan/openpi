#!/usr/bin/env python3
"""测试 Libero 策略推理的完整示例代码。

此脚本演示如何：
1. 加载预训练的 Libero 策略模型 (pi05_libero)
2. 创建模拟输入数据
3. 运行推理并获取动作输出
"""

import time

import numpy as np

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download


def make_libero_example() -> dict:
    """创建 Libero 策略的随机输入示例。

    Returns:
        包含观测数据和提示的字典
    """
    return {
        # 状态数据 (8维: 末端位姿等)
        "observation/state": np.random.rand(8),
        # 外部相机图像 (224x224x3, uint8)
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # 腕部相机图像 (224x224x3, uint8)
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # 任务提示
        "prompt": "pick up the fork",
    }


def load_policy_once():
    """加载配置和策略模型（只加载一次）。

    Returns:
        tuple: (config, policy) 配置和策略对象
    """
    print("=" * 60)
    print("预加载模型 (pi05_libero)")
    print("=" * 60)

    # 1. 获取配置
    print("步骤 1: 加载配置 'pi05_libero'...")
    config = _config.get_config("pi05_libero")
    print(f"  配置名称: {config.name}")
    print(f"  模型类型: {config.model.model_type}")
    print(f"  动作维度: {config.model.action_dim}")
    print(f"  动作时域: {config.model.action_horizon}")

    # 2. 下载 checkpoint（如果尚未下载）
    print("\n步骤 2: 下载/定位模型 checkpoint...")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    print(f"  Checkpoint 目录: {checkpoint_dir}")

    # 3. 创建训练好的策略
    print("\n步骤 3: 创建训练好的策略模型...")
    start_time = time.time()
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    load_time = time.time() - start_time
    print(f"  策略创建完成，耗时: {load_time:.2f} 秒")
    print(f"  策略元数据: {policy.metadata}")

    # 显示后端和设备信息
    backend = "PyTorch" if policy._is_pytorch_model else "JAX"
    print(f"\n步骤 3.1: 后端和设备信息...")
    print(f"  后端类型: {backend}")
    if policy._is_pytorch_model:
        import torch
        print(f"  推理设备: {policy._pytorch_device}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        print(f"  MPS 可用: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print(f"  MPS 已构建: {torch.backends.mps.is_built()}")
    else:
        import jax
        print(f"  JAX 设备: {jax.devices()}")
        print(f"  JAX 后端: {jax.default_backend()}")

    print("\n模型预加载完成！")
    return config, policy


def test_libero_policy_inference(config=None, policy=None):
    """测试 Libero 策略推理的完整流程。

    Args:
        config: 可选的配置对象，如果为 None 则会加载
        policy: 可选的策略对象，如果为 None 则会加载
    """
    # 如果没有传入预加载的模型，则加载
    if config is None or policy is None:
        config, policy = load_policy_once()

    print("\n" + "=" * 60)
    print("开始测试 Libero 策略推理 (pi05_libero)")
    print("=" * 60)

    # 4. 创建测试输入数据
    print("\n步骤 4: 创建测试输入数据...")
    example = make_libero_example()
    print(f"  输入键: {list(example.keys())}")
    print(f"  外部图像形状: {example['observation/image'].shape}")
    print(f"  腕部图像形状: {example['observation/wrist_image'].shape}")
    print(f"  状态形状: {example['observation/state'].shape}")
    print(f"  提示: '{example['prompt']}'")

    # 5. 运行推理
    print("\n步骤 5: 运行推理...")

    # 运行几次预热推理
    print("  进行预热推理...")
    for i in range(2):
        _ = policy.infer(example)

    # 正式推理并计时
    inference_times = []
    num_inferences = 5

    print(f"  进行 {num_inferences} 次正式推理...")
    for i in range(num_inferences):
        start_time = time.time()
        result = policy.infer(example)
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        inference_times.append(inference_time)
        print(f"    推理 {i+1}/{num_inferences}: {inference_time:.1f} ms")

    # 6. 分析结果
    print("\n步骤 6: 分析推理结果...")
    action_chunk = result["actions"]
    print(f"  输出动作块形状: {action_chunk.shape}")
    print(f"  动作块内容预览:\n{action_chunk[:3]}")  # 显示前3个动作

    # 显示推理时间统计
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    print(f"\n推理时间统计:")
    print(f"  平均: {avg_time:.1f} ms")
    print(f"  标准差: {std_time:.1f} ms")
    print(f"  最小: {min(inference_times):.1f} ms")
    print(f"  最大: {max(inference_times):.1f} ms")

    # 显示策略计时信息
    if "policy_timing" in result:
        print(f"\n策略内部计时:")
        for key, value in result["policy_timing"].items():
            print(f"  {key}: {value:.1f} ms")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

    return result


def test_multiple_prompts(config=None, policy=None):
    """测试不同提示下的策略推理。

    Args:
        config: 可选的配置对象，如果为 None 则会加载
        policy: 可选的策略对象，如果为 None 则会加载
    """
    # 如果没有传入预加载的模型，则加载
    if config is None or policy is None:
        config, policy = load_policy_once()

    print("\n" + "=" * 60)
    print("测试不同提示的推理")
    print("=" * 60)

    # 不同的测试提示
    prompts = [
        "pick up the fork",
        "open the drawer",
        "place the cup on the table",
        "press the button",
    ]

    for prompt in prompts:
        example = make_libero_example()
        example["prompt"] = prompt

        result = policy.infer(example)
        action_chunk = result["actions"]

        print(f"\n提示: '{prompt}'")
        print(f"  动作块形状: {action_chunk.shape}")
        print(f"  第一个动作: {action_chunk[0]}")

    print("\n多提示测试完成！")


def test_with_real_images(config=None, policy=None):
    """使用模拟的真实图像数据测试（模拟从相机获取的图像）。

    Args:
        config: 可选的配置对象，如果为 None 则会加载
        policy: 可选的策略对象，如果为 None 则会加载
    """
    # 如果没有传入预加载的模型，则加载
    if config is None or policy is None:
        config, policy = load_policy_once()

    print("\n" + "=" * 60)
    print("测试模拟真实图像数据")
    print("=" * 60)

    # 创建更真实的图像数据（模拟相机捕获）
    # 使用渐变和噪声模拟真实场景
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    wrist_image = np.zeros((224, 224, 3), dtype=np.uint8)

    # 添加一些渐变效果模拟光照
    for i in range(224):
        image[i, :, 0] = int(100 + 50 * np.sin(i / 30))  # 红色通道
        image[i, :, 1] = int(80 + 40 * np.cos(i / 25))   # 绿色通道
        image[:, i, 2] = int(60 + 30 * np.sin(i / 20))   # 蓝色通道

    # 腕部相机图像（更近距离的视角）
    for i in range(224):
        wrist_image[i, :, 0] = int(120 + 30 * np.sin(i / 15))
        wrist_image[i, :, 1] = int(90 + 35 * np.cos(i / 20))
        wrist_image[:, i, 2] = int(70 + 25 * np.sin(i / 18))

    # 添加一些随机噪声
    image = np.clip(image.astype(np.int32) + np.random.randint(-20, 20, (224, 224, 3)), 0, 255).astype(np.uint8)
    wrist_image = np.clip(wrist_image.astype(np.int32) + np.random.randint(-20, 20, (224, 224, 3)), 0, 255).astype(np.uint8)

    # 模拟真实的状态数据（末端位姿等，8维）
    # Libero 状态: [x, y, z, rx, ry, rz, rw, gripper]
    state = np.array([0.3, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0, 0.04])

    example = {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": "pick up the object",
    }

    print("运行推理...")
    result = policy.infer(example)
    action_chunk = result["actions"]

    print(f"动作块形状: {action_chunk.shape}")
    print(f"完整动作序列:")
    for i, action in enumerate(action_chunk):
        print(f"  步骤 {i+1}: {action}")

    print("\n真实图像模拟测试完成！")


def main():
    """主函数，运行所有测试。"""
    try:
        # 预加载模型一次
        config, policy = load_policy_once()

        # 运行基本推理测试（使用预加载的模型）
        test_libero_policy_inference(config, policy)

        # 运行多提示测试（使用预加载的模型）
        test_multiple_prompts(config, policy)

        # 运行真实图像模拟测试（使用预加载的模型）
        test_with_real_images(config, policy)

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
