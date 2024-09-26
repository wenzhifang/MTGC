import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # hierarchical arguments
    parser.add_argument('--num_cells', type=int, default=10, help="number of cells")
    parser.add_argument('--E', type=int, default=10, help="rounds of edge every per global aggregation")
    # federated arguments
    parser.add_argument('--com_amount', type=int, default=100, help="rounds of training")
    parser.add_argument('--epoch', type=int, default=5, help="epoch")
    parser.add_argument('--num_client_per', type=int, default=10, help="number of users")
    parser.add_argument('--rule_arg', type=float, default=0.1, help="Dirichlet parameter")
    parser.add_argument('--rule', type=str, default='noniid', help="data setting")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    

    # other arguments
    args = parser.parse_args()
    return args
