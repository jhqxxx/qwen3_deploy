from modelscope import snapshot_download

model_save_path = ""  # 目标保存路径
model_dir = snapshot_download('Qwen/Qwen3-0.6B',
                              cache_dir=model_save_path, revision='master')