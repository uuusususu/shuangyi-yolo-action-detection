## ADDED Requirements

### Requirement: The results page can check a model on a labeled project clip
结果页的验证卡 SHALL 允许为选中的模型选：上传一条视频，或者直接选工程里已有的一条片子。选工程片子时，特征没过期 SHALL 直接用缓存，过期 SHALL 先抽，再用共用的切段函数切段。工程片子的视频和标注 MUST NOT 被复制或修改。

#### Scenario: Check a held-out project clip
- **WHEN** 用户选了一个新训练的模型，选工程片子 6.mp4，点验证
- **THEN** 页面播放这条视频并画出切出来的段，切段方式和训练时验证一样，`media/` 和 `annotations.json` 都不变

#### Scenario: Uploaded video still works
- **WHEN** 用户改为上传一条新视频
- **THEN** 验证照原来一样跑，结果存在选中那一轮的 `check/` 下，没有标注轨、没有分数

### Requirement: Truth and prediction are shown side by side with the same scores
验证的是标过的工程片子时，结果页 SHALL 在同一条时间轴上画出标注轨和预测轨，并显示和训练页一样的分数：F1@50、编辑分、逐帧准确率、空闲误报，以及预测段数和标注段数。点一段没对上的标注段，视频 SHALL 跳到它的开头。

#### Scenario: Collapsed prediction is visible
- **WHEN** 6 号片子标了 12 段动作，模型只切出 2 段
- **THEN** 两条轨都画出来，页面显示「切出 2 段 / 标注 12 段」，F1@50 和这个结果对得上

#### Scenario: Scores agree with training page
- **WHEN** 6 号片子就是选中那一轮的验证片子
- **THEN** 结果页对 6 号片子算出的分数，等于那一轮最好一轮在 6 号片子上的验证分数

### Requirement: The page says whether the clip took part in training
结果页 SHALL 读这一轮的 `split.json`，对选中的工程片子写明以下三种之一：「这条片子没参与训练，分数可信」「这条片子是这一轮的验证片」「这条片子参与过训练，分数偏乐观」。

#### Scenario: Checking a training clip
- **WHEN** 用户拿一轮把 1 号片子放在训练一边的模型去验证 1 号片子
- **THEN** 分数旁边显示「这条片子参与过训练，分数偏乐观」
