import time
import logging
from method.utils.utils import save_model


class BertTrainer(object):
    def __init__(self, args):
        # 训练控制参数
        self.current_step = 1
        self.total_steps = args.total_steps  # 总训练步数
        self.report_steps = args.report_steps  # 日志报告步数间隔
        self.save_checkpoint_steps = args.save_checkpoint_steps  # 模型保存步数间隔
        self.batch_size = args.batch_size
        self.output_model_path = args.output_model_path # 训练完成的模型保存路径

        # 训练状态统计
        self.start_time = time.time()  # 训练开始时间
        self.total_loss = 0.0  # 累计总损失
        self.total_correct_mlm = 0.0  # MLM正确预测数
        self.total_denominator = 0.0  # MLM总掩码数

    def forward_propagation(self, batch, model):
        # batch即为dataloader给出的内容
        src, tgt_pos, tgt = batch
        # TODO 1: 前向传播model得到loss、correct和denominator
        # TODO 2: 更新统计信息，然后返回损失
        pass

    def report_and_reset_stats(self):
        # 使用logging.info向log打印训练状态
        # log会在训练主文件中定义
        # 打印的内容类似：
        # 当前step数 ｜ 平均多少秒一个step ｜ 这report_steps个step的平均loss ｜ 这report_steps个step的平均mlm的accuracy
        pass

    def train(self, args, dataloader, model, optimizer, scheduler):
        model.train()  # 设置模型为训练模式
        loader_iter = iter(dataloader)  # 获取数据加载器的迭代器

        while True:
            # 训练直到达到指定step数量
            if self.current_step == self.total_steps + 1:
                break

            # 获取一个批次数据

            # 计算loss

            # loss反向传播

            # 更新参数、scheduler

            # 定期报告训练状态
            if self.current_step % self.report_steps == 0:
                self.report_and_reset_stats()
                self.start_time = time.time()

            # 定期保存模型
            if self.current_step % self.save_checkpoint_steps == 0:
                save_model(model, self.output_model_path + "/" + str(self.current_step))

            self.current_step += 1
