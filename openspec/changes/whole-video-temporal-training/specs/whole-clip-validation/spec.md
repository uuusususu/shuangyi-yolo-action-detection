## ADDED Requirements

### Requirement: New training requires at least one whole held-out clip
训练新模型 SHALL 至少有一条参加的片子设成「只做验证」，至少一条设成「参加训练」。验证片子的每一帧 SHALL 都不进训练、特征标准化和类别权重。每个动作类别 SHALL 至少出现在一条训练片子和一条验证片子里，否则训练 SHALL 被拦住，提示写明缺哪一类、缺在哪一边。新训练 MUST NOT 再提供「同一批片子里留几段」的划分。

#### Scenario: No validation clip chosen
- **WHEN** 所有参加的片子都设成「参加训练」
- **THEN** 开始训练被拦住，提示「先把至少一条完整片子设成「只做验证」」

#### Scenario: Validation clip lacks a class
- **WHEN** 唯一的验证片子里有封装、没有包装
- **THEN** 训练被拦住，提示验证这一边缺包装

#### Scenario: Train and validation are disjoint
- **WHEN** 6 号片子「只做验证」，1、2、3、7、8 号「参加训练」
- **THEN** `split.json` 的验证一边只有 6 号片子，契约记下 `same_source_validation: false`

### Requirement: Each epoch validates by running validation clips whole
每一轮训练结束后，训练台 SHALL 把每条验证片子从第一帧到最后一帧一次连着跑完，用共用的切段函数解码，再和这条片子的逐帧标签比对打分。被忽略的帧 MUST NOT 进任何分数。

#### Scenario: Validation uses the same decode as the results page
- **WHEN** 第 30 轮训练结束
- **THEN** 每条验证片子的切段结果，由和结果页同一个切段函数、同一组契约字段、同一个最短动作时长得出

### Requirement: Validation scores measure segmentation, not isolated classification
训练台 SHALL 每一轮报出：
- 动作段（不含空闲）的段级 F1@10、F1@25、F1@50：一个预测段和一个还没配上的同类标注段交并比达到阈值，就算配上；
- 编辑分：动作段类别序列的归一化编辑距离，0–100；
- 逐帧准确率：只算没被忽略的帧；
- 空闲误报：标注为空闲的帧里被判成动作的比例；
- 预测的动作段数和标注的动作段数；
- 每一类的段级 F1@50、逐帧精确率和召回，以及逐帧混淆矩阵；
- 验证损失：不加权的逐帧交叉熵。
最好一轮 SHALL 取 F1@50 最高的那一轮；相同时取编辑分高的，再相同取逐帧准确率高的。

#### Scenario: A collapsed prediction scores low
- **WHEN** 一条验证片子有 12 段动作，模型只切出一段封装，后面一直到结尾都是空闲
- **THEN** F1@50 不超过 2/13，段数显示切出 1 段、标注 12 段

#### Scenario: Over-segmentation is penalized
- **WHEN** 一段包装被切成三截，中间夹着没被最短时长并掉的封装
- **THEN** 编辑分和 F1@50 都比完整配上一段时低

#### Scenario: Best epoch follows segmentation quality
- **WHEN** 第 40 轮逐帧准确率 0.95、F1@50 0.70，第 60 轮逐帧准确率 0.93、F1@50 0.88
- **THEN** 保留第 60 轮作为最好的一轮

### Requirement: The training page shows held-out segmentation scores
训练页 SHALL 显示四张分数卡：段级 F1@50、编辑分、逐帧准确率、空闲误报；SHALL 和训练损失、验证损失一起画出每一轮的曲线；SHALL 显示当前这一轮「切出 N 段 / 标注 M 段」。分数卡标题 SHALL 写明这些分数来自没参与训练的整条片子。每一类的柱状图和混淆矩阵 SHALL 在训练结束后显示。

#### Scenario: During training
- **WHEN** 一轮新训练跑到 100 轮里的第 25 轮
- **THEN** 四张卡显示第 25 轮的数，曲线有 25 个点，「切出 10 段 / 标注 12 段」是第 25 轮的结果

#### Scenario: Old runs keep their old cards
- **WHEN** 用户打开一轮契约里没有 `train_unit: "timeline"` 的旧训练
- **THEN** 页面显示原来的宏平均 F1、最低召回、空闲误报三张卡，并注明「旧训练方式，分数不能和新模型比」
