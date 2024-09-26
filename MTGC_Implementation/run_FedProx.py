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


    print('FedProx')

    mu = 1e-4
    FedProx = train_FedProx(data_obj=data_obj, learning_rate=learning_rate, batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, E=E, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model,
                                         mu=mu, lr_decay_per_round=lr_decay_per_round)

    # Save the dictionary containing the data to a single file
    data_to_save = {
        'com_amount': com_amount,
        'FedProx': FedProx
    }
    index = int(rule_arg * 10)
    
    torch.save(data_to_save, 'results/FedProx_{}_{}_N{}_H{}_E{}.pt'.format(rule, index, num_cells, epoch, E))



