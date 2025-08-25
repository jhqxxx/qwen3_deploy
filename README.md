### qwen3_deploy
rust部署Qwen3，使用http协议，支持流式输出

### Get started
* git clone https://github.com/jhqxxx/qwen3_deploy.git
* cd qwen3_deploy
* cargo run -- --model-path  /your-qwen3-model-path

### windows
* use cuda
    * wget https://github.com/jhqxxx/qwen3_deploy/releases/download/0.1.3/deploy-x86_64-pc-windows-msvc-cuda.exe
    * deploy-x86_64-pc-windows-msvc-cuda.exe  --model-path /your-qwen3-model-path
* use cpu
    * wget https://github.com/jhqxxx/qwen3_deploy/releases/download/0.1.3/deploy-x86_64-pc-windows-msvc-cpu.exe
    * deploy-x86_64-pc-windows-msvc-cpu.exe  --model-path /your-qwen3-model-path