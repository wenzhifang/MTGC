import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from utils_general import *
from utils_methods import *
from utils_options import args_parser

if __name__ == '__main__':
    
    args = args_parser()

    num_cells = args.num_cells
    num_client_per = args.num_client_per
    n_client = num_cells * num_client_per

    ###
    com_amount = args.com_amount
    E = args.E
    epoch = args.epoch
    rule  = args.rule
    rule_arg = args.rule_arg
    learning_rate = args.lr
    '''
    num_cells = 1
    num_client_per = 1
    n_client = num_cells * num_client_per

    ###
    com_amount = 100
    E = 1
    epoch = 1
    rule  = 'iid'
    rule_arg = 0.1
    learning_rate = 0.01
    '''
    
    weight_decay = 1e-3
    batch_size = 50
    lr_decay_per_round = 1
    print_per = 5
    model_name = 'cifar10'

    data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, num_cells=num_cells, rule=rule, unbalanced_sgm=0, rule_arg=rule_arg)


    # Model function
    model_func = lambda: client_model(model_name)
    # Initalise the model for all methods or load it from a saved initial model
    init_model = model_func()
    if not os.path.exists('Output/%s/%s_init_mdl.pt' % (data_obj.name, model_name)):
        print("New directory!")
        os.mkdir('Output/%s/' % (data_obj.name))
        torch.save(init_model.state_dict(), 'Output/%s/%s_init_mdl.pt' % (data_obj.name, model_name))
    else:
        # Load model
        init_model.load_state_dict(torch.load('Output/%s/%s_init_mdl.pt' % (data_obj.name, model_name)))

    ####
    print('FedAvg')
    FedAvg = train_FedAvg(data_obj=data_obj, learning_rate=learning_rate, batch_size=batch_size,
                                         epoch=epoch, com_amount=com_amount, E=E, print_per=print_per, weight_decay=weight_decay,
                                         model_func=model_func, init_model=init_model,
                                         lr_decay_per_round=lr_decay_per_round)


    # Save the dictionary containing the data to a single file
    data_to_save = {
        'com_amount': com_amount,
        'FedAvg': FedAvg
    }
    index = int(rule_arg * 10)
    torch.save(data_to_save, 'results/test_HFL_noniid_{}_N{}_H{}_E{}.pt'.format(index, num_cells, epoch, E))
    
   
