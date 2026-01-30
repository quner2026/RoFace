# RoFace - 人脸识别开放平台

基于 Rust + OpenVINO 的高性能人脸识别服务，提供 REST 和 gRPC API。

[English](#english-version) | [中文](#中文版本)

---

## 中文版本

### 功能特性

- ✅ **人脸检测** - SCRFD 模型 (准确率 82.80%)
- ✅ **人脸特征提取** - R100 + Glint360K (准确率 90.66%)
- ✅ **人脸比对** - 1:1 人脸验证
- ✅ **人脸识别** - 1:N 人脸搜索
- ✅ **属性分析** - 年龄、性别、情绪识别
- ✅ **模型池管理** - 懒加载 + 5分钟自动卸载
- ✅ **批量推理** - 可选的高吞吐量模式

### 快速开始

#### 1. 安装 OpenVINO

下载并安装 [OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)。

**Windows:**
```powershell
# 在构建/运行前执行 setupvars 脚本
"C:\Program Files (x86)\Intel\openvino_2024\setupvars.bat"
```

**Linux:**
```bash
source /opt/intel/openvino_2024/setupvars.sh
```

#### 2. 下载模型

使用提供的 Python 脚本下载模型：

```powershell
# 确保已安装 Python 3.7+
python download_models.py
```

脚本会自动下载 buffalo_l 模型包并重命名为所需格式。

**手动下载链接：**
- SCRFD 检测器: [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- 情绪模型: [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus)

#### 3. 配置服务

编辑 `config.toml` 配置文件：

```toml
[server]
rest_port = 3000    # REST API 端口
grpc_port = 50051   # gRPC API 端口

[inference]
device = "CPU"      # 设备: "CPU", "GPU", "AUTO"
num_threads = 4     # CPU 线程数
model_idle_timeout = 300  # 模型空闲卸载时间(秒)

[recognition]
similarity_threshold = 0.5  # 相似度阈值 (0.0-1.0)
embedding_dim = 512        # 特征向量维度

[storage]
type = "sqlite"            # 存储类型: "sqlite" 或 "postgres"
sqlite_path = "data/faces.db"
```

#### 4. 编译和运行

```powershell
# 编译项目 (Release 模式)
cargo build --release

# 运行服务
cargo run --release
```

服务启动后会显示：
```
Face Recognition Service is ready!
REST: http://localhost:3000/health
gRPC: localhost:50051
```

### API 使用示例

#### 健康检查

```powershell
curl http://localhost:3000/health
```

响应：
```json
{
  "healthy": true,
  "version": "0.1.0",
  "models_loaded": {
    "detector": false,
    "embedder": false,
    "gender_age": false,
    "emotion": false
  }
}
```

#### 1. 检测人脸

```powershell
curl -X POST http://localhost:3000/api/v1/detect `
  -F "image=@test_photo.jpg" `
  -F "confidence_threshold=0.5"
```

响应：
```json
{
  "faces": [
    {
      "x1": 120.5,
      "y1": 80.3,
      "x2": 220.8,
      "y2": 200.5,
      "confidence": 0.98,
      "landmarks": [
        {"x": 140.2, "y": 120.5},
        {"x": 180.3, "y": 118.7},
        {"x": 160.5, "y": 145.2},
        {"x": 145.6, "y": 170.8},
        {"x": 175.4, "y": 169.2}
      ]
    }
  ],
  "inference_time_ms": 45
}
```

#### 2. 注册人脸

```powershell
curl -X POST http://localhost:3000/api/v1/register `
  -F "image=@person_face.jpg" `
  -F "person_id=emp_001" `
  -F "person_name=张三" `
  -F "metadata={\"department\":\"技术部\"}"
```

响应：
```json
{
  "success": true,
  "face_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Face registered successfully"
}
```

#### 3. 比对两张人脸

```powershell
curl -X POST http://localhost:3000/api/v1/compare `
  -F "image1=@face1.jpg" `
  -F "image2=@face2.jpg"
```

响应：
```json
{
  "similarity": 0.87,
  "is_same_person": true,
  "inference_time_ms": 92
}
```

#### 4. 识别人脸 (1:N 搜索)

```powershell
curl -X POST http://localhost:3000/api/v1/identify `
  -F "image=@unknown_face.jpg" `
  -F "top_k=5" `
  -F "threshold=0.6"
```

响应：
```json
{
  "matches": [
    {
      "person_id": "emp_001",
      "person_name": "张三",
      "face_id": "550e8400-e29b-41d4-a716-446655440000",
      "similarity": 0.91
    },
    {
      "person_id": "emp_005",
      "person_name": "李四",
      "face_id": "660e8400-e29b-41d4-a716-446655440001",
      "similarity": 0.72
    }
  ],
  "inference_time_ms": 68
}
```

#### 5. 分析人脸属性

```powershell
curl -X POST http://localhost:3000/api/v1/analyze `
  -F "image=@portrait.jpg"
```

响应：
```json
{
  "faces": [
    {
      "x1": 100.0,
      "y1": 50.0,
      "x2": 300.0,
      "y2": 280.0,
      "age": 28,
      "gender": "male",
      "gender_confidence": 0.95,
      "emotion": "happy",
      "emotion_confidence": 0.82
    }
  ],
  "inference_time_ms": 123
}
```

#### 6. 删除人脸

```powershell
curl -X DELETE http://localhost:3000/api/v1/faces/550e8400-e29b-41d4-a716-446655440000
```

### Python 客户端示例

查看 `examples/` 目录了解完整示例。

**简单示例：**

```python
import requests

# 服务地址
BASE_URL = "http://localhost:3000"

# 注册人脸
with open("person.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/v1/register",
        files={"image": f},
        data={
            "person_id": "user_001",
            "person_name": "测试用户"
        }
    )
    print(response.json())

# 识别人脸
with open("unknown.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/v1/identify",
        files={"image": f},
        data={"top_k": "3"}
    )
    result = response.json()
    
    if result["matches"]:
        best_match = result["matches"][0]
        print(f"识别为: {best_match['person_name']}")
        print(f"相似度: {best_match['similarity']:.2%}")
```

### 使用场景

#### 考勤系统

```python
# 员工打卡
def clock_in(face_image_path):
    with open(face_image_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/v1/identify",
            files={"image": f},
            data={"threshold": "0.7"}
        )
        result = resp.json()
        
        if result["matches"]:
            employee = result["matches"][0]
            # 记录打卡时间
            print(f"{employee['person_name']} 打卡成功")
            return employee
        else:
            print("未识别到员工")
            return None
```

#### 访客管理

```python
# 访客登记
def register_visitor(photo_path, name, company):
    with open(photo_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/v1/register",
            files={"image": f},
            data={
                "person_id": f"visitor_{timestamp}",
                "person_name": name,
                "metadata": json.dumps({
                    "company": company,
                    "visit_date": str(date.today())
                })
            }
        )
        return resp.json()
```

### GPU 加速

使用 Intel 集成显卡加速：

1. 安装 Intel GPU 驱动
2. 安装 OpenCL Runtime
3. 修改配置: `device = "GPU"`

**无需修改代码！**

### API 参考

#### REST API 端点

| 端点 | 方法 | 说明 | 参数 |
|------|------|------|------|
| `/health` | GET | 健康检查 | 无 |
| `/metrics` | GET | 服务指标 | 无 |
| `/api/v1/detect` | POST | 检测人脸 | image, confidence_threshold |
| `/api/v1/register` | POST | 注册人脸 | image, person_id, person_name, metadata |
| `/api/v1/compare` | POST | 比对人脸 | image1, image2 |
| `/api/v1/identify | POST | 识别人脸 | image, top_k, threshold |
| `/api/v1/analyze` | POST | 属性分析 | image |
| `/api/v1/faces/{id}` | DELETE | 删除人脸 | face_id (路径参数) |

#### gRPC API

端口: 50051

查看 `proto/face.proto` 了解服务定义。

### 性能优化建议

1. **批量处理**: 启用 `batch_enabled = true` 可提升吞吐量
2. **模型预加载**: 降低 `model_idle_timeout` 值减少首次推理延迟
3. **GPU 加速**: 使用 GPU 可提升 2-5 倍性能
4. **线程配置**: 根据 CPU 核心数调整 `num_threads`

### 常见问题

**Q: 模型加载失败？**
A: 确保运行了 `download_models.py` 并检查 `config.toml` 中的模型路径。

**Q: 推理速度慢？**
A: 首次推理需要加载模型。后续推理会更快。可以调整 `model_idle_timeout` 保持模型常驻内存。

**Q: 识别准确率低？**
A: 调整 `similarity_threshold` 参数。建议值: 0.4-0.6

**Q: 内存占用高？**
A: 模型会在空闲后自动卸载。可以降低 `model_idle_timeout` 减少内存占用。

### 许可证

MIT License

---

## English Version

### Features

- ✅ **Face Detection** - SCRFD model (82.80% accuracy)
- ✅ **Face Embedding** - R100 + Glint360K (90.66% accuracy)
- ✅ **Face Comparison** - 1:1 verification
- ✅ **Face Identification** - 1:N search
- ✅ **Attribute Analysis** - Age, gender, emotion
- ✅ **Model Pool** - Lazy loading with 5-minute auto-unload
- ✅ **Batch Inference** - Optional high-throughput mode

### Quick Start

#### 1. Install OpenVINO

Download and install [OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html).

**Windows:**
```powershell
"C:\Program Files (x86)\Intel\openvino_2024\setupvars.bat"
```

**Linux:**
```bash
source /opt/intel/openvino_2024/setupvars.sh
```

#### 2. Download Models

```powershell
python download_models.py
```

#### 3. Configure

Edit `config.toml`:

```toml
[server]
rest_port = 3000
grpc_port = 50051

[inference]
device = "CPU"  # or "GPU", "AUTO"
num_threads = 4

[recognition]
similarity_threshold = 0.5
```

#### 4. Build and Run

```powershell
cargo build --release
cargo run --release
```

### API Examples

See the Chinese version above for detailed API usage examples.

### License

MIT License
