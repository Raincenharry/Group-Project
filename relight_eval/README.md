# 重光照比较工具

对应 `handoff/评估部分_我自己做.md` 第 2 节要写的三个脚本。只处理图像，不运行 PR-ENDO，不需要 GPU，在本机就能跑。

| 脚本 | 做什么 | 什么时候用 |
| --- | --- | --- |
| `check_data.py` | 检查 VR-Caps 组员交来的数据 | 阶段 1、阶段 3 收到数据后 |
| `fit_spot_power.py` | 只用真值图像拟合聚光衰减指数 n | 阶段 1，结果发给 PR-ENDO 组员 |
| `compare.py` | 比较真值图像和 PR-ENDO 的模拟图像 | 阶段 1 试跑、阶段 3 正式比较 |
| `selftest.py` | 用合成数据把上面三步跑一遍 | 改代码之后 |

指标、掩膜和输出格式按《PR-ENDO 重光照评估方案》第 9–11 节实现。和方案不同的地方见最后一节。

## 环境

文件夹里已经建好虚拟环境 `.venv`，用的是本机 conda 的 Python 3.13，额外只装了 OpenEXR。直接用它运行：

```bash
cd ~/Documents/GroupProject/relight_eval
.venv/bin/python selftest.py
```

在别的电脑上：`pip install -r requirements.txt`。没有 OpenEXR 时会改用 opencv 读 EXR，但 opencv 读不了带图层名的通道（比如 `ViewLayer.Depth.Z`）。

LPIPS 默认不算。要算的话先 `.venv/bin/pip install lpips`，再给 `compare.py` 加 `--lpips vgg`。第一次运行会下载约 500 MB 的 VGG 权重。图像较大时可以加 `--lpips-max-side 512`，先缩小再算，速度快很多，但数值和全分辨率不完全一样，报告里要写明。

## 按阶段使用

下面假设数据放在 `~/data/VRCaps_colon1_v1`，模拟图像放在 `~/data/pred_VRCaps_colon1_v1_20261120-1530`。所有输出写到当前目录下的 `eval_relight/<数据文件夹名>/`。

### 阶段 1：试跑

```bash
.venv/bin/python check_data.py ~/data/VRCaps_colon1_v1
.venv/bin/python fit_spot_power.py ~/data/VRCaps_colon1_v1
```

- `check_data.py` 每项输出一行 PASS / FAIL，详情在 `check_data.json`。有任何一项失败，退出码为 1。失败的项连同现象写进给 VR-Caps 组员的回执。
- `fit_spot_power.py` 输出 n，以及 `calibration.json` 和拟合曲线图 `spot_fit.png`。先看曲线图：如果点和曲线差得很多，说明 Unity 聚光灯的形状和 PR-ENDO 的模型差别大，L2 的结果要谨慎解读。确认后把 n 发给 PR-ENDO 组员，之后不再重新拟合。

收到试跑的模拟图像后：

```bash
.venv/bin/python compare.py ~/data/VRCaps_colon1_v1 ~/data/pred_VRCaps_colon1_v1_20261120-1530
```

看屏幕上的检查结果，或者 `checks.json`。试跑阶段主要看这几项：

| 检查 | 失败说明什么 | 找谁 |
| --- | --- | --- |
| `predictions_complete` | 有条目没有生成模拟图像 | PR-ENDO 组员 |
| `record_dataset` | 模拟图像是按另一个数据版本生成的 | PR-ENDO 组员 |
| `spot_power` | 他用的 n 和你发的不一样 | PR-ENDO 组员 |
| `l0_vs_train` | 标定帧上基准光照的还原比他报告的训练平均值差 1 dB 以上，可能是颜色空间或编码不一致 | 两人一起查 |
| `scale_L1_*`，`scale_L4_*` | 改光强或颜色后不是整体乘一个倍数，可能是编码处理错了 | PR-ENDO 组员 |
| `direction_*` | 预测和真值变亮的方向不一致，通常是正负号或坐标转换错了 | 先对照 `check_data.py` 的光源位姿检查，再找对应的人 |

### 阶段 3：正式比较

```bash
.venv/bin/python check_data.py ~/data/VRCaps_colon1_v1
.venv/bin/python compare.py ~/data/VRCaps_colon1_v1 ~/data/pred_VRCaps_colon1_v1_20261205-1000 --lpips vgg
```

`compare.py` 会自动读取 `eval_relight/<数据文件夹名>/calibration.json`。判断阈值可以用 `--gain-db`、`--corr`、`--gap-db` 修改，默认是 1 dB、0.8、3 dB。阈值要在看结果之前定下来。

## 输出

`eval_relight/<数据文件夹名>/<运行编号>/` 下：

| 文件 | 内容 |
| --- | --- |
| `per_image.csv` | 每个（配置，帧，真值版本）一行 |
| `summary.csv` | 每个（配置，真值版本）一行，只统计 `split` 为 `test` 的帧；带 `effective` 和 `trusted` 两列判断 |
| `summary_levels.csv` | 每个（级别，真值版本）一行，是该级别下各配置均值的均值 |
| `checks.json` | 上面表格里的检查 |
| `config.json` | 这次运行的全部设置，包括阈值、掩膜参数、PR-ENDO 组员的 `record.json` |
| `calibration.json` | 用到的 n 的副本 |
| `curves/` | `L2_angle.png`、`L2_ratio.png`、`L3_offset.png`、`L3_ratio.png`、`L5_sequence.png` |
| `panels/<配置>/<真值版本>/<帧>.jpg` | 每张的对比图 |
| `panels/_highlights/` | 每个配置增益最好、中位、最差的各一张 |

对比图从左到右：基准光照真值 G0、测试光照真值 G、PR-ENDO 预测 P、误差 |P−G| 放大 4 倍、对数比值误差（红色预测偏亮，蓝色偏暗，灰色不参与计算）。

`per_image.csv` 的主要列：

| 列 | 含义 |
| --- | --- |
| `psnr`，`ssim`，`lpips` | P 对 G |
| `psnr_base` 等 | 不做重光照（用 B）对 G |
| `psnr_gain` 等 | 重光照带来的提升；LPIPS 的增益取基线减预测，正值表示变好 |
| `psnr_l0` | B 对 G0，同一帧基准光照下的还原质量，是 `psnr` 的参考上限 |
| `psnr_changed`，`changed_frac` | 只在光照确实变了的区域算的 PSNR，以及该区域的面积占比 |
| `ratio_rmse`，`ratio_corr`，`ratio_bias` | 预测的亮度变化和真实亮度变化的对数比较；L1、L4 不算相关系数 |
| `shadow_frac` | 阴影区域占比 |
| `psnr_aligned` | 只用 L0 拟合每个通道一个倍数后的 PSNR，诊断用 |
| `psnr_scaled` | L1、L4：把 B 直接乘以光强或颜色倍数的 PSNR，理论上应该和 `psnr` 相等 |
| `psnr_base_spotfwd` | L2：用朝正前方的聚光灯预测去比，用来把"换成聚光灯"和"转动光束"两种变化分开 |
| `dir_*` | 变亮区域偏向哪一侧，真值和预测各一组；`dir_agree` 为 1 表示方向一致 |

## 和评估方案不同的地方

1. **对数比值的下限是 0.01**，和"足够亮"掩膜同一个阈值。方案没有规定下限。Unity 聚光灯锥外的亮度正好是 0，下限太小会让这些像素的对数误差无限放大。
2. **新增了方向检查**（`dir_*` 列和 `direction_*` 检查）。方案第 8.3 节的正负号检查需要用 PR-ENDO 重新渲染，现在我们不运行 PR-ENDO，所以改成直接比较真值和预测里变亮区域的方向。
3. **方案第 8.3 节的颜色空间检查**改为 `l0_vs_train`：比较标定帧上的基准光照还原和 PR-ENDO 组员报告的训练平均值。
4. **新增 `psnr_scaled`、`psnr_base_spotfwd` 两列**，对应方案第 10 节的"纯图像缩放"和"未旋转的聚光灯"两个对照。
5. **`fit_spot_power.py` 额外报告光轴附近聚光灯和点光源的亮度比**，正常应为 1。画面中心常是很暗的肠腔，会被掩膜排除；只有能取到内锥以内的像素时才判断，否则只报告数值。
6. **LPIPS 默认关闭**，需要时手动打开。

## 自测

```bash
.venv/bin/python selftest.py
```

生成一个合成的管道场景，完整跑一遍三个脚本，并故意制造三种错误，确认检查能抓到：

- 真值导出不是线性的
- 模拟图像的 yaw 和偏移方向反了
- 模拟图像用错了 n

全部通过时输出 `12/12 passed`，临时文件自动删除。加 `--keep --workdir <目录>` 可以保留生成的数据和结果来查看。

自测不覆盖的部分：LPIPS（需要下载权重）、真实 Unity 导出的文件，以及有阴影的场景（合成场景是从管道内部看，不会产生阴影）。
