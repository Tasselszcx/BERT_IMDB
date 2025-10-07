项目概述
基于BERT的IMDB电影评论情感分类系统，能够自动识别评论的情感倾向（正面/负面）。

环境配置
1. 创建Conda虚拟环境
bash
conda create -n bert-imdb-gpu python=3.10
conda activate bert-imdb-gpu
2. 安装依赖包
bash
# PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 项目核心依赖
pip install transformers[torch] datasets pandas numpy matplotlib seaborn scikit-learn tqdm jupyter
3. 验证环境
bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
代码结构说明
单元格1: 环境检查和导入
导入所有必要的库

检查GPU可用性

设置随机种子确保结果可重现

单元格2: 数据加载
加载IMDB电影评论数据集

显示数据集基本信息

确保正负样本平衡

单元格3: 数据预处理
加载BERT分词器

将文本转换为BERT输入格式

设置序列长度和填充

单元格4: 模型加载
加载预训练的BERT模型

配置为二分类任务

将模型移动到GPU

单元格5: 训练参数设置
配置训练超参数

启用混合精度训练

设置评估和保存策略

单元格6: 训练器创建
定义评估指标函数

创建Trainer实例

准备开始训练

单元格7: 模型训练
开始BERT模型微调

显示训练进度和损失

记录训练时间

单元格8: 模型评估
在测试集上评估模型性能

生成详细分类报告

保存训练好的模型

单元格9: 结果可视化
绘制混淆矩阵

显示准确率分布

可视化训练过程

单元格10: 新评论测试
使用训练好的模型分析新评论

显示情感分类和置信度

演示模型实际应用

单元格11: 性能统计
计算详细性能指标

分析各类别准确率

项目总结
