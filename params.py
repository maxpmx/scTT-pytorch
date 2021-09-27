import argparse

def get_parser():
    # model parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=768,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=8,
                        help="Number of Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--n_val", type=int, default=11,
                        help="Number of values")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of sentences per batch")
    parser.add_argument("--dataset", type=str, default='hcl',
                        help="Reference dataset")
    parser.add_argument("--experiment", type=str, default='2',
                        help="Experiment name")
    parser.add_argument("--n_epochs", type=int, default=2,
                        help="Maximum epoch size")
    parser.add_argument("--log_step", type=int, default=1,
                        help="evaluation step")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="DataLoader workers")
    parser.add_argument("--test", type=bool, default=False,
                        help="test")
    parser.add_argument("--reload", type=bool, default=False,
                        help="reload")       
    parser.add_argument("--finetune", type=bool, default=False,
                        help="finetune")         
    parser.add_argument("--ad", type=bool, default=True,
                        help="adversarial embedding")  
    parser.add_argument("--pooling", type=str, default='sum',
                        help="embedding pooling")         
    return parser