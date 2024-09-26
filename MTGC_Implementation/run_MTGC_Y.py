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
    
    num_cells = 10
    num_client_per = 10
    n_client = num_cells * num_client_per

    ###
    com_amount = 100
    E = 30
    epoch = 2
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
    init_model = model_func()
    # Initalise the model for all methods or load it from a saved initial model
    init_model = model_func()
    if not os.path.exists('Output/%s/%s_init_mdl.pt' % (data_obj.name, model_name)):
        print("New directory!")
        os.mkdir('Output/%s/' % (data_obj.name))
        torch.save(init_model.state_dict(), 'Output/%s/%s_init_mdl.pt' % (data_obj.name, model_name))
    else:
        # Load model
        init_model.load_state_dict(torch.load('Output/%s/%s_init_mdl.pt' % (data_obj.name, model_name)))
    
    print('MTGC_Y')
    n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
    n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
    n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)
    print_per_ = print_per*n_iter_per_epoch

    MTGC_Y = train_MTGC_Y(data_obj=data_obj, learning_rate=learning_rate,
                                             batch_size=batch_size, n_minibatch=n_minibatch, com_amount=com_amount,E=E,
                                             print_per=print_per_, weight_decay=weight_decay, model_func=model_func,
                                             init_model=init_model, lr_decay_per_round=lr_decay_per_round)



    # Save the dictionary containing the data to a single file
    data_to_save = {
        'com_amount': com_amount,
        'MTGC_Y': MTGC_Y
    }
    index = int(rule_arg * 10)
    torch.save(data_to_save, 'results/test_MTGC_y_noniid_{}_N{}_H{}_E{}.pt'.format(index, num_cells, epoch, E))
    
   