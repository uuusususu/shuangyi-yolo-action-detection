# 自己的时序模型怎么训

给双翼项目用。YOLO 和手部模型只推理，不重训。训练对象只有一个：用你们工序录像训出来的 LSTM。别人仓库里的 `best_lstm_fine.pt` 不要用。

YOLO 认出画面里有什么零件。LSTM 认出这一段时间在做哪一步。步骤名来自你们配置的类别，不是开源项目的 S1 到 S5。权重必须你们自己训。

## 1. 整条链路

训练时和运行时走同一套特征。差别只在：训练时有标签、会改权重；运行时加载权重、只做前向。

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| 录像 | 每个步骤多段连续视频 | mp4 或连续帧 |
| 逐帧感知 | 你们现有的 YOLO + MediaPipe Hands | 框 + 最多两只手 21 点 |
| 一帧特征 | 框和手的坐标 | 长度为 F 的一行数 |
| 一个样本 | 连续 48 帧特征 | 张量 `(48, F)` + 步骤编号 |
| 训练一步 | 一批 `(B, 48, F)` 和标签 `(B,)` | 交叉熵，然后更新 LSTM |
| 保存 | 验证集准确率最高的那一轮 | 你们的 `best_lstm.pt` 和 `config.json` |

## 2. 不要训什么，要训什么

**保持推理，不要重训**

- YOLO：现有的检测模型。只用来出框。
- Hands：项目里已经接上的 MediaPipe，21 点，最多两只手。
- 换工序不要重训这两样。它们只负责每帧交一行数。

**唯一要训的：你们的 LSTM**

- 输入：最近 48 帧的特征表。
- 输出：idle、步骤1、步骤2 的分数。
- 标签：这段录像属于哪一步。整段同一个名字。

## 3. 先定类别和特征长度 F

特征长度必须训练和推理完全一样。零件类数变了，F 就变，旧权重立刻不能用。

```text
F = 零件类数 × 5 + 2 × 21 × 3
F = 零件类数 × 5 + 126
```

每个零件类只留置信度最高的一个框：置信度、中心 x、中心 y、宽、高，都除以画面宽高。手部固定 126 维，没有手的位置填 0。

| 例子 | 零件类 | F | LSTM 输出类 |
| --- | --- | --- | --- |
| 镜子开源项目 | 4（base / frame / mirror / screw） | 146 | 6（背景 + 五步） |
| 你们只配了散热片 | 1 | 131 | 2（idle + 散热片） |
| 你们配了 4 个步骤物体 | 4 | 146 | 5（idle + 四步） |

## 4. 采集数据

时序模型吃的是连续片段，不是散图。一个文件夹里丢一堆单张照片，训不出来正在做哪一步。

```text
clips/
  idle/          空闲工位、没人做动作
  step1/         第 1 步从动手到做完
  step2/
  step3/
  step4/
```

文件夹名就是标签。每个文件夹里放多条 mp4，或按时间编号的连续帧。一条视频只做这一个步骤。

| 节点 | 拍什么 | 建议条数 |
| --- | --- | --- |
| idle | 工位空着、手不在做步骤 | 10～20 |
| 每个步骤 | 这一步从开始做到结束，整段 | 20～40 |
| 快慢、不同人 | 同一动作多拍几种节奏 | 算在上面的条数里 |

一条 clip 整段都标成这个文件夹的名字。不要在一条视频里混两个步骤。想标完整一轮，就按步骤剪开，再分别放进对应文件夹。

## 5. 一帧怎么收成一行数

和开源项目里的 `build_feature_row` 同一套算法。差别只有零件类名改成你们的。

```python
def build_feature_row(detections, hands, frame_w, frame_h, class_names):
    # detections: 每项有 label, conf, box(x1,y1,x2,y2)
    # hands: 最多 2 只，每只 21 个 (x,y,z)，坐标已是 0～1
    det_feat = {name: [0, 0, 0, 0, 0] for name in class_names}
    for d in detections:
        name = d["label"]
        if name not in det_feat:
            continue
        x1, y1, x2, y2 = d["box"]
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        cx, cy = x1 + w / 2, y1 + h / 2
        fw, fh = max(frame_w, 1), max(frame_h, 1)
        if d["conf"] > det_feat[name][0]:
            det_feat[name] = [d["conf"], cx / fw, cy / fh, w / fw, h / fh]
    det_vec = []
    for name in class_names:          # 顺序必须固定
        det_vec.extend(det_feat[name])

    hand_vec = [0.0] * 126
    for hi, hand in enumerate(hands[:2]):
        for li, lm in enumerate(hand[:21]):
            base = hi * 63 + li * 3
            hand_vec[base:base + 3] = lm[:3]
    return det_vec + hand_vec         # 长度 = F
```

类名对不上的框直接丢掉。OBB 可以先取外接水平框再算中心和宽高，LSTM 不吃旋转角。

## 6. 窗口怎么切，标签怎么贴

网络一次只看固定 48 帧。一条 200 帧的步骤录像，用滑动窗口切成很多个样本，每个样本标签都是这一步。

| 量 | 建议值 | 含义 |
| --- | --- | --- |
| 窗长 T | 48 | 一个样本覆盖多少帧，和推理时一致 |
| 步长 | 8 或 16 | 窗口每次往后滑几帧。步长小则样本多、更重叠 |
| 批大小 B | 8 或 16 | 一次送进网络多少个窗口 |
| 一个样本 x | `(48, F)` | 48 行特征 |
| 一批 x | `(B, 48, F)` | LSTM 的输入，`batch_first=True` |
| 一批 y | `(B,)` | 整数，idle=0，step1=1，以此类推 |

```python
# 一条 clip 已经抽成 feat，形状 (T_clip, F)
# 整段标签是同一个 step_id
windows, labels = [], []
T = 48
stride = 8
if len(feat) < T:
    pad = np.zeros((T - len(feat), F), np.float32)
    feat = np.concatenate([pad, feat], 0)
    windows.append(feat)
    labels.append(step_id)
else:
    for start in range(0, len(feat) - T + 1, stride):
        windows.append(feat[start:start + T])
        labels.append(step_id)
```

推理时开源项目把 48、72、96 帧都重采样成 48 再送进去。你们第一版训练和推理都固定 48 帧即可。以后要加多尺度，训练仍按 48，推理再把长窗抽稀成 48。

## 7. 网络长什么样

结构可以直接用开源项目那份 `ActionLSTM`。改的是 `input_size` 和 `num_classes`，不是算法。

```python
import torch
import torch.nn as nn

class ActionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128,
                 num_layers=2, num_classes=5, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, 48, F)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # (B, num_classes)
```

`out[:, -1, :]` 表示只用读完整段 48 帧之后的摘要。分数还不是概率。训练时交给交叉熵；运行时再做 softmax。

## 8. 训练循环

每一批做四件事：前向、算损失、反向、更新。全部样本过一遍叫一个 epoch。留下验证集准确率最高的权重。

```python
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# X: (N, 48, F)  y: (N,)
# 已经按视频划分，不要把同一条视频拆进训练和验证
train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=16, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val), batch_size=16, shuffle=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ActionLSTM(
    input_size=F, hidden_size=128, num_layers=2, num_classes=C
).to(device)
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

best_acc = -1.0
best_path = Path("lstm_runs/best_lstm.pt")
best_path.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(40):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb.to(device)).argmax(1).cpu()
            correct += (pred == yb).sum().item()
            total += len(yb)
    acc = correct / max(total, 1)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), best_path)

config = {
    "input_size": F,
    "hidden_size": 128,
    "num_layers": 2,
    "num_classes": C,
    "window": 48,
    "label_map": [
        {"id": 0, "label": "idle"},
        {"id": 1, "label": "step1"},
        {"id": 2, "label": "step2"},
    ],
}
Path("lstm_runs/config.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
)
```

- 损失：`CrossEntropyLoss`
- 优化器：Adam，学习率 `1e-3`
- epoch：先 30 到 50
- 保存：验证准确率最高的 `state_dict`

按视频划验证集，不要按窗口划。同一条录像切出来的窗口如果一边训练一边验证，准确率会虚高，现场会掉。

## 9. 抽特征时用你们自己的检测

对每一条 clip 逐帧跑你们现有的 YOLO 和 Hands，不要用开源项目的镜子权重。

```python
# 一条视频 -> (T, F)
rows = []
for frame in video_frames:
    dets = your_yolo(frame)          # 你们的检测，不要用镜子权重
    hands = mediapipe_hands(frame)   # 项目里已有的 21 点
    rows.append(build_feature_row(
        dets, hands, frame.shape[1], frame.shape[0], YOUR_CLASS_NAMES
    ))
feat = np.stack(rows, axis=0)        # (T, F)
```

`YOUR_CLASS_NAMES` 的顺序写进 `config.json`，推理时必须按同一顺序拼特征。顺序一换，等于换了一种输入语言。

## 10. 训完怎么用

运行时不要把 YOLO 的单帧类别直接当步骤。先拼 48 帧特征，LSTM 给出当前步骤名，再交给现有的顺序引擎做稳定帧和跳步 NG。

```python
config = json.loads(Path("lstm_runs/config.json").read_text("utf-8"))
model = ActionLSTM(
    config["input_size"], config["hidden_size"],
    config["num_layers"], config["num_classes"],
)
model.load_state_dict(torch.load("lstm_runs/best_lstm.pt", map_location=device))
model.eval()

# 每帧：queue.append(一行特征)，长度维持 48
if len(queue) == 48:
    xb = torch.from_numpy(np.stack(queue)[None]).float().to(device)
    with torch.no_grad():
        probs = torch.softmax(model(xb), dim=1)[0]
    pred_id = int(probs.argmax())
    pred_name = id2label[pred_id]
    expected_p = float(probs[expected_id])
    # 预期步骤概率连续 K 帧过阈值
    # -> 交给 StepSequenceEngine 记一次命中
```

| 量 | 第一版建议 |
| --- | --- |
| 窗长 | 48 帧 |
| 通过阈值 | 0.50，现场再调 |
| 连续命中 K | 2 到 3，不要再叠到 8，窗口已经平滑过一次 |
| 跳步 NG | 继续用现有顺序引擎 |

## 11. 按这个顺序做

1. 定零件类名顺序和步骤名。写出 `class_names` 和 `label_map`，算出 F 和 C。
2. 按步骤录 clip。每个步骤至少十几条完整动作，另加 idle。
3. 逐帧跑你们的 YOLO + Hands，存 `(T, F)`。每条视频一个 npy，标签等于文件夹名。
4. 按 48 帧、步长 8 切窗口。得到 X 和 y，按视频划分验证。
5. 用上面的训练循环训 `ActionLSTM`。生成你们的 `best_lstm.pt` 和 `config.json`。
6. 推理时滑窗 48 帧，输出步骤名。接到现有步骤引擎，不接别人的权重。

## 12. 常见翻车

数据：

- 散图当样本。没有时间，伸手就会过。
- 一条视频里混两个步骤，标签是错的。
- 验证集和训练集切自同一条录像。

特征：

- 类名顺序训练和推理不一致。
- F 变了还加载旧权重。
- 手部点不是 21×3，或坐标不是同一套归一化。

开源项目只公开了推理和别人的权重，没有 LSTM 训练脚本。上面第 5 到第 8 节就是那份缺失的训练过程，按你们的类名和录像来做。

感知用双翼现有检测和 Hands。时序头从零训。顺序和 NG 仍用现有步骤引擎。
