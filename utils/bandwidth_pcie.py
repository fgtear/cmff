import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time


def bytes_to_gb(bytes_value):
    """将字节转换为GB"""
    return bytes_value / (1024**3)


def measure_bandwidth_h2d(size_in_gb, iterations, use_pinned=False):
    """测量主机到设备(H2D)方向的带宽"""
    size = size_in_gb * 1024**3  # 将 GB 转换为字节
    
    # 根据use_pinned参数决定是否使用pinned memory
    if use_pinned:
        h_data = cuda.pagelocked_empty(int(size // 4), np.float32)
        np.copyto(h_data, np.random.rand(int(size // 4)).astype(np.float32))
    else:
        h_data = np.random.rand(int(size // 4)).astype(np.float32)  # 创建主机上的随机数据，转换为整数
    
    d_data = cuda.mem_alloc(h_data.nbytes)  # 在设备上分配内存

    # 预热，确保缓存等因素影响最小化
    cuda.memcpy_htod(d_data, h_data)

    # 开始测量
    start_time = time.time()
    for _ in range(iterations):
        cuda.memcpy_htod(d_data, h_data)
    cuda.Context.synchronize()  # 确保所有操作完成
    end_time = time.time()

    # 计算带宽
    total_bytes = h_data.nbytes * iterations
    duration = end_time - start_time
    bandwidth = (total_bytes / duration) / (1024**3)  # 转换为 GB/s

    print("-" * 99)
    print(f"数据大小：{size_in_gb} GB ({h_data.nbytes} 字节)")
    print(f"传输次数：{iterations}")
    print(f"使用Pinned Memory: {'是' if use_pinned else '否'}")
    print(f"总耗时：{duration:.6f} 秒")
    print(f"主机到设备带宽(H2D)：{bandwidth:.2f} GB/s")

    return bandwidth


def measure_bandwidth_d2h(size_in_gb, iterations, use_pinned=False):
    """测量设备到主机(D2H)方向的带宽"""
    size = size_in_gb * 1024**3  # 将 GB 转换为字节
    
    # 根据use_pinned参数决定是否使用pinned memory
    if use_pinned:
        h_data = cuda.pagelocked_empty(int(size // 4), np.float32)
        np.copyto(h_data, np.random.rand(int(size // 4)).astype(np.float32))
    else:
        h_data = np.random.rand(int(size // 4)).astype(np.float32)  # 创建主机上的随机数据
    
    d_data = cuda.mem_alloc(h_data.nbytes)  # 在设备上分配内存

    # 首先将数据复制到设备
    cuda.memcpy_htod(d_data, h_data)

    # 预热
    cuda.memcpy_dtoh(h_data, d_data)

    # 开始测量
    start_time = time.time()
    for _ in range(iterations):
        cuda.memcpy_dtoh(h_data, d_data)
    cuda.Context.synchronize()  # 确保所有操作完成
    end_time = time.time()

    # 计算带宽
    total_bytes = h_data.nbytes * iterations
    duration = end_time - start_time
    bandwidth = (total_bytes / duration) / (1024**3)  # 转换为 GB/s

    print("-" * 99)
    print(f"数据大小：{size_in_gb} GB ({h_data.nbytes} 字节)")
    print(f"传输次数：{iterations}")
    print(f"使用Pinned Memory: {'是' if use_pinned else '否'}")
    print(f"总耗时：{duration:.6f} 秒")
    print(f"设备到主机带宽(D2H)：{bandwidth:.2f} GB/s")

    return bandwidth


def measure_bidirectional_bandwidth(size_in_gb, iterations, use_pinned=False):
    """测量双向带宽"""
    size = size_in_gb * 1024**3  # 将 GB 转换为字节
    
    # 根据use_pinned参数决定是否使用pinned memory
    if use_pinned:
        h_data_send = cuda.pagelocked_empty(int(size // 4), np.float32)
        h_data_recv = cuda.pagelocked_empty(int(size // 4), np.float32)
        np.copyto(h_data_send, np.random.rand(int(size // 4)).astype(np.float32))
    else:
        h_data_send = np.random.rand(int(size // 4)).astype(np.float32)  # 用于发送到设备的数据
        h_data_recv = np.zeros_like(h_data_send)  # 用于从设备接收的数据

    d_data_send = cuda.mem_alloc(h_data_send.nbytes)  # 在设备上为发送分配内存
    d_data_recv = cuda.mem_alloc(h_data_recv.nbytes)  # 在设备上为接收分配内存

    # 首先将数据复制到设备
    cuda.memcpy_htod(d_data_send, h_data_send)

    # 预热
    cuda.memcpy_htod(d_data_recv, h_data_send)
    cuda.memcpy_dtoh(h_data_recv, d_data_send)

    # 开始测量
    start_time = time.time()
    for _ in range(iterations):
        # 同时进行两个方向的传输
        # 注意：这里由于PyCUDA的限制，无法真正并行执行，但这是最接近的模拟
        cuda.memcpy_htod(d_data_recv, h_data_send)
        cuda.memcpy_dtoh(h_data_recv, d_data_send)

    cuda.Context.synchronize()  # 确保所有操作完成
    end_time = time.time()

    # 计算总带宽 (双向带宽 = 两个方向的总数据量/总时间)
    total_bytes = 2 * h_data_send.nbytes * iterations  # 两个方向的总数据量
    duration = end_time - start_time
    bandwidth = (total_bytes / duration) / (1024**3)  # 转换为 GB/s

    print("-" * 99)
    print(f"数据大小：{size_in_gb} GB x 2方向 ({h_data_send.nbytes * 2} 字节)")
    print(f"传输次数：{iterations}")
    print(f"使用Pinned Memory: {'是' if use_pinned else '否'}")
    print(f"总耗时：{duration:.6f} 秒")
    print(f"双向总带宽：{bandwidth:.2f} GB/s")

    return bandwidth


def measure_bandwidth(size_in_gb, iterations, direction="h2d", use_pinned=False):
    """
    测量PCIe带宽

    参数:
    size_in_gb -- 数据大小(GB)
    iterations -- 重复次数
    direction -- 传输方向: "h2d"(主机到设备), "d2h"(设备到主机), "both"(双向测试), "all"(分别测试所有方向)
    use_pinned -- 是否使用pinned memory，默认为False
    """
    if direction == "h2d":
        return measure_bandwidth_h2d(size_in_gb, iterations, use_pinned)
    elif direction == "d2h":
        return measure_bandwidth_d2h(size_in_gb, iterations, use_pinned)
    elif direction == "both":
        return measure_bidirectional_bandwidth(size_in_gb, iterations, use_pinned)
    elif direction == "all":
        print("\n===== 测试主机到设备(H2D)带宽 =====")
        h2d_bw = measure_bandwidth_h2d(size_in_gb, iterations, use_pinned)

        print("\n===== 测试设备到主机(D2H)带宽 =====")
        d2h_bw = measure_bandwidth_d2h(size_in_gb, iterations, use_pinned)

        print("\n===== 测试双向带宽 =====")
        bidir_bw = measure_bidirectional_bandwidth(size_in_gb, iterations, use_pinned)

        print("\n===== 带宽测试总结 =====")
        print(f"使用Pinned Memory: {'是' if use_pinned else '否'}")
        print(f"H2D带宽: {h2d_bw:.2f} GB/s")
        print(f"D2H带宽: {d2h_bw:.2f} GB/s")
        print(f"双向带宽: {bidir_bw:.2f} GB/s")
        return h2d_bw, d2h_bw, bidir_bw
    else:
        raise ValueError(f"不支持的方向: {direction}, 请使用 'h2d', 'd2h', 'both' 或 'all'")


if __name__ == "__main__":
    # 演示字节到GB的转换
    # example_bytes = 32000000
    # print(f"{example_bytes} 字节 = {bytes_to_gb(example_bytes):.4f} GB")

    # 测试所有方向的带宽 - 使用普通内存
    print("\n小数据量测试 (普通内存):")
    measure_bandwidth(size_in_gb=0.0298, iterations=100, direction="all", use_pinned=False)

    # 测试所有方向的带宽 - 使用pinned memory
    print("\n小数据量测试 (Pinned内存):")
    measure_bandwidth(size_in_gb=0.0298, iterations=100, direction="all", use_pinned=True)

    # # 测试所有方向的带宽 - 使用普通内存
    # print("\n大数据量测试 (普通内存):")
    # measure_bandwidth(size_in_gb=1, iterations=20, direction="all", use_pinned=False)
    
    # # 测试所有方向的带宽 - 使用pinned memory
    # print("\n大数据量测试 (Pinned内存):")
    # measure_bandwidth(size_in_gb=1, iterations=20, direction="all", use_pinned=True)
