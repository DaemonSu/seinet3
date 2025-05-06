import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SEI Training Config')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--seq_len', default=7000, type=int)
    parser.add_argument('--input_dim', default=2, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--lr_decay_epochs', default=[30, 60, 90], type=float)
    parser.add_argument('--embedding_dim', default=1024, type=int)

    parser.add_argument('--threshold', default=10000, type=int)
    parser.add_argument('--open_threshold', type=float, default=0.79, help='Threshold for open-set decision')
    parser.add_argument('--open_detector_threshold', type=float, default=0.000005, help='Threshold for open-set decision')
    parser.add_argument('--con_weight', default=1, type=float)
    parser.add_argument('--proto_weight', default=0.5, type=float)

    # 损失函数类型
    parser.add_argument('--loss_type', default='contrastive', type=str, choices=['ce', 'proto', 'contrastive'])

    parser.add_argument('--prototype_momentum', default=0.9, type=float)
    parser.add_argument('--margin', default=0.09, type=float)

    # openset 训练接定义
    parser.add_argument('--train_data', default='F:/seidata/IQdata/train2-close', type=str)
    parser.add_argument('--val2', default='F:/seidata/IQdata/val2', type=str)

    # openset 测试集
    parser.add_argument('--test_closeData', default='F:/seidata/IQdata/test2-close', type=str)
    parser.add_argument('--test_openData', default='F:/seidata/IQdata/openset2-open', type=str)
    parser.add_argument('--test_mixed', default='F:/seidata/IQdata/test2-mixed', type=str)

    # 其他
    parser.add_argument('--val_data', default='F:/seidata/IQdata/val2-close', type=str)


    parser.add_argument('--test_data', default='F:/seidata/IQdata/openset2-open', type=str)

    parser.add_argument('--train_second', default='F:/seidata/IQdata/train2-open', type=str)
    parser.add_argument('--val_second', default='F:/seidata/IQdata/val2-open', type=str)

    parser.add_argument('--save_dir', default='model/', type=str)

    return parser.parse_args()
