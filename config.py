import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SEI Training Config')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--seq_len', default=7000, type=int)
    parser.add_argument('--input_dim', default=2, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epochs', default=75, type=int)
    parser.add_argument('--epochs2', default=20, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--lr_decay_epochs', default=[30, 60, 90], type=float)
    parser.add_argument('--embedding_dim', default=1024, type=int)

    parser.add_argument('--threshold', default=10000, type=int)
    parser.add_argument('--open_threshold', type=float, default=0.95, help='Threshold for open-set decision')
    parser.add_argument('--open_detector_threshold', type=float, default=0.000005, help='Threshold for open-set decision')
    parser.add_argument('--con_weight', default=1, type=float)
    parser.add_argument('--proto_weight', default=0.5, type=float)

    # 损失函数类型
    parser.add_argument('--loss_type', default='contrastive', type=str, choices=['ce', 'proto', 'contrastive'])

    parser.add_argument('--prototype_momentum', default=0.9, type=float)
    parser.add_argument('--margin', default=0.09, type=float)

    # openset 训练集合定义
    parser.add_argument('--train_data_close', default='F:/seidata/26ft-exp/train2-close', type=str)
    parser.add_argument('--train_data_open', default='F:/seidata/26ft-exp/train2-open', type=str)


    # openset 验证集合定义
    parser.add_argument('--val_data', default='F:/seidata/26ft-exp/val2', type=str)


    # openset 测试集
    parser.add_argument('--test_mixed', default='F:/seidata/26ft-exp/test2-mixed', type=str)

    parser.add_argument('--save_dir', default='model/', type=str)

    return parser.parse_args()
