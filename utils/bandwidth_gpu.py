import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time


def test_p2p_bandwidth(size_in_mb=1024, iterations=10):
    size = size_in_mb * 1024 * 1024  # 数据大小（字节）

    # 获取可用的设备数量
    num_devices = cuda.Device.count()
    if num_devices < 2:
        print("需要至少两个 GPU 设备！")
        return

    # 创建设备列表
    devices = [cuda.Device(i) for i in range(num_devices)]

    # 创建上下文
    contexts = [d.make_context() for d in devices]

    # 检查 P2P 支持
    p2p_pairs = []
    for i in range(num_devices):
        for j in range(num_devices):
            if i != j:
                can_access = cuda.Device.can_access_peer(devices[i], devices[j])
                if can_access:
                    p2p_pairs.append((i, j))
    if not p2p_pairs:
        print("没有找到支持 P2P 的 GPU 对！")
        # 清理上下文
        for ctx in contexts:
            ctx.pop()
        return

    print(f"找到 {len(p2p_pairs)} 对支持 P2P 的 GPU 设备。")

    for src_idx, dst_idx in p2p_pairs:
        print("-" * 60)
        print(f"测试 GPU {src_idx} 到 GPU {dst_idx} 的 P2P 带宽")

        src_ctx = contexts[src_idx]
        dst_ctx = contexts[dst_idx]

        # 从源上下文切换到目标上下文，启用 P2P 访问
        src_ctx.push()
        try:
            cuda.Context.enable_peer_access(devices[dst_idx], 0)
        except cuda.LogicError as e:
            print(f"无法启用 GPU {src_idx} 对 GPU {dst_idx} 的 P2P 访问：{e}")
            src_ctx.pop()
            continue
        src_ctx.pop()

        # 在源 GPU 上分配内存
        src_ctx.push()
        src_mem = cuda.mem_alloc(size)
        src_ctx.pop()

        # 在目标 GPU 上分配内存
        dst_ctx.push()
        dst_mem = cuda.mem_alloc(size)
        dst_ctx.pop()

        # 生成随机数据并复制到源 GPU
        h_data = np.random.rand(size // 4).astype(np.float32)
        src_ctx.push()
        cuda.memcpy_htod(src_mem, h_data)
        src_ctx.pop()

        # 同步上下文
        cuda.Context.synchronize()

        # 开始测量
        start_time = time.time()
        for _ in range(iterations):
            # P2P 内存传输
            src_ctx.push()
            cuda.memcpy_peer(dst_mem, dst_ctx, src_mem, src_ctx, size)
            src_ctx.pop()
        cuda.Context.synchronize()
        end_time = time.time()

        # 计算带宽
        total_bytes = size * iterations
        duration = end_time - start_time
        bandwidth = (total_bytes / duration) / (1024**3)  # 转换为 GB/s

        print(f"数据大小：{size_in_mb} MB")
        print(f"传输次数：{iterations}")
        print(f"总耗时：{duration:.6f} 秒")
        print(f"GPU {src_idx} 到 GPU {dst_idx} 的 P2P 带宽：{bandwidth:.2f} GB/s")

        # 关闭 P2P 访问
        src_ctx.push()
        cuda.Context.disable_peer_access(devices[dst_idx])
        src_ctx.pop()

        # 释放显存
        src_ctx.push()
        src_mem.free()
        src_ctx.pop()

        dst_ctx.push()
        dst_mem.free()
        dst_ctx.pop()

    # 清理上下文
    for ctx in contexts:
        ctx.pop()


if __name__ == "__main__":
    test_p2p_bandwidth(size_in_mb=1024, iterations=10)
