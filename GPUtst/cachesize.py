import torch

# 获取第一个可用的 GPU 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 获取 GPU 设备的属性
props = torch.cuda.get_device_properties(device)

# 打印 GPU 的总内存大小（以更友好的形式）
print(f"Total GPU Memory: {props.total_memory / (1024**3):.2f} GB")  # 将字节转换为 GB
