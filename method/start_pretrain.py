from method.utils.utils import *
from method.model.dataloader.BertDataloader import BertDataloader
from method.model.embedding.embedding import WordPosEmbedding
from method.model.transformer.encoder import TransformerEncoder
from method.model.target.BertTarget import BertTarget
from method.model.model_combine.BertModel import BertModel
from method.model.trainer.BertTrainer import BertTrainer


def train_and_validate(args):
    set_seed(args.seed)
    embedding = WordPosEmbedding(args)
    encoder = TransformerEncoder(args)
    target = BertTarget(args)
    model = BertModel(embedding, encoder, target)
    
    print(f"#parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 对于模型的参数，先进行初始化
    # 对于非layernorm的参数，进行mean=0，std=0.02的正态分布初始化
    # 然后把model放到device上

    worker(args, model)

def worker(args, model):
    train_loader = BertDataloader(args, shuffle=True)

    # 构造optimizer
    # 使用AdamW，对于非bias或layernorm的参数，施加0.01的weight_decay
    # lr使用args.learning_rate
    
    # 配置学习率调度器，根据预设步数调整学习率
    # 使用LambdaLR， 对optimizer的lr进行调节：
    # 前args.warmup * args.total_steps的step，lr线性增长到args.learning_rate
    # 其余时间，lr线性下降到0
    optimizer = None
    scheduler = None
    print("Worker is training ...")  # 打印训练开始的提示信息

    trainer = BertTrainer(args)
    trainer.train(args, train_loader, model, optimizer, scheduler)  # 启动训练过程
