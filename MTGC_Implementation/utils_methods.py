from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_general import *

### Methods
def train_FedAvg(data_obj,learning_rate, batch_size, epoch, com_amount, E, print_per, weight_decay, model_func, init_model, lr_decay_per_round):

    n_clnt=data_obj.n_client
    num_cells = data_obj.num_cells
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par


    avg_model = model_func().to(device)
    # avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    avg_model.load_state_dict(init_model.state_dict())
    
    
    log_name = './training_log/training_log_HFL_{}_N{:1d}.txt'.format(data_obj.dataset, num_cells)
    with open(log_name, 'w') as log_file:
        # Optionally write an initial message or leave it empty to just clear the file
        log_file.write('Starting HFL: {}_N{:1d} \n'.format(data_obj.dataset, num_cells))
    
    

    for t in range(com_amount):
        ## Revised hierarchical FedAvg
        for j in range(num_cells):

            selected_clnts = [i for i in range((n_clnt//num_cells)*j, (n_clnt//num_cells)*(j+1), 1)]
            edge_model = model_func().to(device)
            # edge_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
            edge_model.load_state_dict(avg_model.state_dict())
            for e in range(E):
                for clnt in selected_clnts:
                    print('---- Global round {}, Cell {}, Edge round {}, Training client {}'.format(t, j, e, clnt))
                    trn_x = clnt_x[clnt]
                    trn_y = clnt_y[clnt]
                    clnt_model = model_func().to(device)
                    # clnt_model.load_state_dict(copy.deepcopy(dict(edge_model.named_parameters())))
                    clnt_model.load_state_dict(edge_model.state_dict())

                    for params in clnt_model.parameters():
                        params.requires_grad = True
                    clnt_model = train_model(clnt_model, trn_x, trn_y, learning_rate * (lr_decay_per_round ** t), batch_size, epoch, print_per, weight_decay, data_obj.dataset)

                    clnt_params_list[clnt] = get_mdl_params([clnt_model], n_par)[0]
                edge_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))

        avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))
        ## Revised hierarchical FedAvg

        ###
        loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
        tst_perf_all[t] = [loss_tst, acc_tst]
        print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(t+1, acc_tst, loss_tst))
        with open(log_name, 'a') as log_file:
            log_file.write('Round {:3d}, testing accuracy {:.2f}\n'.format(t, acc_tst))
        #if acc_tst>=0.8:
            #break
            

    return tst_perf_all


def train_MTGC(data_obj, learning_rate, batch_size, n_minibatch, com_amount, E, print_per, weight_decay, model_func, init_model, lr_decay_per_round):

    n_clnt=data_obj.n_client
    num_cells = data_obj.num_cells
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt # normalize it

    tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    Y_state_param_list = np.zeros((num_cells, n_par)).astype('float32')

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    avg_model = model_func().to(device)
    
    log_name = './training_log/training_log_MTGC_{}_N{:1d}.txt'.format(data_obj.dataset, num_cells)
    with open(log_name, 'w') as log_file:
        # Optionally write an initial message or leave it empty to just clear the file
        log_file.write('Starting client correction: {}_N{:1d} \n'.format(data_obj.dataset, num_cells))
    
    for t in range(com_amount):
        Z_state_param_list = np.zeros((n_clnt, n_par)).astype('float32')
        ## Revised hierarchical FedAvg
        for j in range(num_cells):
            selected_clnts = [i for i in range((n_clnt // num_cells) * j, (n_clnt // num_cells) * (j + 1), 1)]
            edge_model = model_func().to(device)
            # edge_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
            edge_model.load_state_dict(avg_model.state_dict())
            for e in range(E):
                for clnt in selected_clnts:
                    print('---- Global round {}, Cell {}, Edge round {}, Training client {}'.format(t, j, e, clnt))
                    trn_x = clnt_x[clnt]
                    trn_y = clnt_y[clnt]

                    clnt_model = model_func().to(device)
                    # clnt_model.load_state_dict(copy.deepcopy(dict(edge_model.named_parameters())))
                    clnt_model.load_state_dict(edge_model.state_dict())

                    for params in clnt_model.parameters():
                        params.requires_grad = True
                    # state_params_diff_curr = y + z
                    state_params_diff_curr = torch.tensor(Z_state_param_list[clnt]+Y_state_param_list[j], dtype=torch.float32, device=device)
                    clnt_model = train_MTGC_mdl(clnt_model, model_func, state_params_diff_curr, trn_x, trn_y, learning_rate * (lr_decay_per_round ** t),
                                                           batch_size, n_minibatch, print_per, weight_decay, data_obj.dataset)
                    curr_model_param = get_mdl_params([clnt_model], n_par)[0]
                    clnt_params_list[clnt] = curr_model_param

                edge_model_params = np.mean(clnt_params_list[selected_clnts], axis=0)
                for clnt in selected_clnts:
                    curr_model_param = clnt_params_list[clnt]
                    Z_state_param_list[clnt] = Z_state_param_list[clnt] + 1/n_minibatch/learning_rate * (curr_model_param - edge_model_params)
                edge_model = set_client_from_params(model_func().to(device), edge_model_params)


        avg_model_params = np.mean(clnt_params_list, axis = 0)

        for j in range(num_cells):

            selected_clnts = [i for i in range((n_clnt // num_cells) * j, (n_clnt // num_cells) * (j + 1), 1)]
            edge_model_params = np.mean(clnt_params_list[selected_clnts], axis=0)
            Y_state_param_list[j] = Y_state_param_list[j] + 1/n_minibatch/learning_rate/E * (edge_model_params - avg_model_params)

        avg_model = set_client_from_params(model_func().to(device), avg_model_params)

        ###
        loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
        tst_perf_all[t] = [loss_tst, acc_tst]
        print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(t+1, acc_tst, loss_tst))
        with open(log_name, 'a') as log_file:
            log_file.write('Round {:3d}, testing accuracy {:.2f}\n'.format(t, acc_tst))
       

    return tst_perf_all

def train_MTGC_Z(data_obj, learning_rate, batch_size, n_minibatch, com_amount, E, print_per, weight_decay, model_func, init_model, lr_decay_per_round):

    n_clnt=data_obj.n_client
    num_cells = data_obj.num_cells
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt # normalize it

    tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    Y_state_param_list = np.zeros((num_cells, n_par)).astype('float32')
    zero_y = np.zeros((num_cells, n_par)).astype('float32')

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par


    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    log_name = './training_log/training_log_Z_{}_N{:1d}.txt'.format(data_obj.dataset, num_cells)
    with open(log_name, 'w') as log_file:
        # Optionally write an initial message or leave it empty to just clear the file
        log_file.write('Starting client correction: {}_N{:1d} \n'.format(data_obj.dataset, num_cells))

    for t in range(com_amount):
        Z_state_param_list = np.zeros((n_clnt, n_par)).astype('float32')
        ## Revised hierarchical FedAvg
        for j in range(num_cells):
            selected_clnts = [i for i in range((n_clnt // num_cells) * j, (n_clnt // num_cells) * (j + 1), 1)]
            edge_model = model_func().to(device)
            edge_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
            for e in range(E):
                for clnt in selected_clnts:
                    print('---- Global round {}, Cell {}, Edge round {}, Training client {}'.format(t, j, e, clnt))
                    trn_x = clnt_x[clnt]
                    trn_y = clnt_y[clnt]

                    clnt_model = model_func().to(device)
                    clnt_model.load_state_dict(copy.deepcopy(dict(edge_model.named_parameters())))

                    for params in clnt_model.parameters():
                        params.requires_grad = True
                    # state_params_diff_curr = y + z
                    state_params_diff_curr = torch.tensor(Z_state_param_list[clnt]+zero_y[j], dtype=torch.float32, device=device)
                    clnt_model = train_MTGC_mdl(clnt_model, model_func, state_params_diff_curr, trn_x, trn_y, learning_rate * (lr_decay_per_round ** t),
                                                           batch_size, n_minibatch, print_per, weight_decay, data_obj.dataset)
                    curr_model_param = get_mdl_params([clnt_model], n_par)[0]
                    clnt_params_list[clnt] = curr_model_param

                edge_model_params = np.mean(clnt_params_list[selected_clnts], axis=0)
                for clnt in selected_clnts:
                    curr_model_param = clnt_params_list[clnt]
                    Z_state_param_list[clnt] = Z_state_param_list[clnt] + 1/n_minibatch/learning_rate * (curr_model_param - edge_model_params)
                edge_model = set_client_from_params(model_func().to(device), edge_model_params)


        avg_model_params = np.mean(clnt_params_list, axis = 0)

        for j in range(num_cells):

            selected_clnts = [i for i in range((n_clnt // num_cells) * j, (n_clnt // num_cells) * (j + 1), 1)]
            edge_model_params = np.mean(clnt_params_list[selected_clnts], axis=0)
            Y_state_param_list[j] = Y_state_param_list[j] + 1/n_minibatch/learning_rate/E * (edge_model_params - avg_model_params)

        avg_model = set_client_from_params(model_func().to(device), avg_model_params)

        ###
        loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
        tst_perf_all[t] = [loss_tst, acc_tst]
        print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(t+1, acc_tst, loss_tst))
        with open(log_name, 'a') as log_file:
            log_file.write('Round {:3d}, testing accuracy {:.2f}\n'.format(t, acc_tst))
        #if acc_tst>=0.8:
            #break

    return tst_perf_all

def train_MTGC_Y(data_obj, learning_rate, batch_size, n_minibatch, com_amount, E, print_per, weight_decay, model_func, init_model, lr_decay_per_round):

    n_clnt=data_obj.n_client
    num_cells = data_obj.num_cells
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt # normalize it

    tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    Y_state_param_list = np.zeros((num_cells, n_par)).astype('float32')

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par


    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    log_name = './training_log/training_log_Y_{}_N{:1d}.txt'.format(data_obj.dataset, num_cells)
    with open(log_name, 'w') as log_file:
        # Optionally write an initial message or leave it empty to just clear the file
        log_file.write('Starting cell correction: {}_N{:1d} \n'.format(data_obj.dataset, num_cells))

    for t in range(com_amount):
        Z_state_param_list = np.zeros((n_clnt, n_par)).astype('float32')
        zero_z = np.zeros((n_clnt, n_par)).astype('float32')
        ## Revised hierarchical FedAvg
        for j in range(num_cells):
            selected_clnts = [i for i in range((n_clnt // num_cells) * j, (n_clnt // num_cells) * (j + 1), 1)]
            edge_model = model_func().to(device)
            edge_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
            for e in range(E):
                for clnt in selected_clnts:
                    print('---- Global round {}, Cell {}, Edge round {}, Training client {}'.format(t, j, e, clnt))
                    trn_x = clnt_x[clnt]
                    trn_y = clnt_y[clnt]

                    clnt_model = model_func().to(device)
                    clnt_model.load_state_dict(copy.deepcopy(dict(edge_model.named_parameters())))

                    for params in clnt_model.parameters():
                        params.requires_grad = True
                    # state_params_diff_curr = y + z
                    state_params_diff_curr = torch.tensor(zero_z[clnt]+Y_state_param_list[j], dtype=torch.float32, device=device)
                    clnt_model = train_MTGC_mdl(clnt_model, model_func, state_params_diff_curr, trn_x, trn_y, learning_rate * (lr_decay_per_round ** t),
                                                           batch_size, n_minibatch, print_per, weight_decay, data_obj.dataset)
                    curr_model_param = get_mdl_params([clnt_model], n_par)[0]
                    clnt_params_list[clnt] = curr_model_param

                edge_model_params = np.mean(clnt_params_list[selected_clnts], axis=0)
                for clnt in selected_clnts:
                    curr_model_param = clnt_params_list[clnt]
                    Z_state_param_list[clnt] = Z_state_param_list[clnt] + 1/n_minibatch/learning_rate * (curr_model_param - edge_model_params)
                edge_model = set_client_from_params(model_func().to(device), edge_model_params)


        avg_model_params = np.mean(clnt_params_list, axis = 0)

        for j in range(num_cells):

            selected_clnts = [i for i in range((n_clnt // num_cells) * j, (n_clnt // num_cells) * (j + 1), 1)]
            edge_model_params = np.mean(clnt_params_list[selected_clnts], axis=0)
            Y_state_param_list[j] = Y_state_param_list[j] + 1/n_minibatch/learning_rate/E * (edge_model_params - avg_model_params)

        avg_model = set_client_from_params(model_func().to(device), avg_model_params)

        ###
        loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
        tst_perf_all[t] = [loss_tst, acc_tst]
        print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(t+1, acc_tst, loss_tst))
        with open(log_name, 'a') as log_file:
            log_file.write('Round {:3d}, testing accuracy {:.2f}\n'.format(t, acc_tst))
        #if acc_tst>=0.8:
            #break

    return tst_perf_all

def train_FedDyn(data_obj, learning_rate, batch_size, epoch, com_amount, E, weight_decay, model_func,
                 init_model, alpha_coef, lr_decay_per_round):

    n_clnt = data_obj.n_client
    num_cells = data_obj.num_cells
    clnt_x = data_obj.clnt_x; clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt

    tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    local_param_list = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par

    edge_params_list = np.ones(num_cells).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par

    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    cld_model = model_func().to(device)
    cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    log_name = './training_log/training_log_fedyn_{}_N{:1d}.txt'.format(data_obj.dataset, num_cells)
    with open(log_name, 'w') as log_file:
        # Optionally write an initial message or leave it empty to just clear the file
        log_file.write('Starting fedyn: {}_N{:1d} \n'.format(data_obj.dataset, num_cells))
    
    

    for t in range(com_amount):
        for j in range(num_cells):
            selected_clnts = [i for i in range((n_clnt // num_cells) * j, (n_clnt // num_cells) * (j + 1), 1)]

            edge_model = model_func().to(device)
            edge_model.load_state_dict(copy.deepcopy(dict(cld_model.named_parameters())))
            edge_mdl_param = get_mdl_params([edge_model], n_par)[0]
            edge_mdl_param_tensor = torch.tensor(edge_mdl_param, dtype=torch.float32, device=device)
            for e in range(E):
                for clnt in selected_clnts:
                    # Train locally
                    print('---- Training client %d' % clnt)
                    trn_x = clnt_x[clnt]
                    trn_y = clnt_y[clnt]

                    clnt_model = model_func().to(device)
                    # Warm start from current avg model
                    clnt_model.load_state_dict(copy.deepcopy(dict(edge_model.named_parameters())))
                    for params in clnt_model.parameters():
                        params.requires_grad = True

                    # Scale down
                    alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                    local_param_list_curr = torch.tensor(local_param_list[clnt], dtype=torch.float32, device=device)
                    clnt_model = train_feddyn_mdl(clnt_model, model_func, alpha_coef_adpt, edge_mdl_param_tensor,
                                                         local_param_list_curr, trn_x, trn_y,
                                                         learning_rate * (lr_decay_per_round ** t), batch_size, epoch,
                                                         weight_decay, dataset_name=data_obj.dataset)
                    curr_model_par = get_mdl_params([clnt_model], n_par)[0]

                    # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                    local_param_list[clnt] += curr_model_par - edge_mdl_param
                    clnt_params_list[clnt] = curr_model_par

                avg_mdl_param = np.mean(clnt_params_list[selected_clnts], axis=0)
                edge_mdl_param = avg_mdl_param + np.mean(local_param_list[selected_clnts], axis=0)
                edge_mdl_param_tensor = torch.tensor(edge_mdl_param, dtype=torch.float32, device=device)


                edge_model = set_client_from_params(model_func(), edge_mdl_param)
            edge_params_list[j] = edge_mdl_param

        cld_mdl_param = np.mean(edge_params_list, axis=0)
        cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)


        loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, cld_model, data_obj.dataset)
        tst_perf_all[t] = [loss_tst, acc_tst]
        print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" % (t + 1, acc_tst, loss_tst))
        
        with open(log_name, 'a') as log_file:
            log_file.write('Round {:3d}, testing accuracy {:.2f}\n'.format(t, acc_tst))
        
        
    return tst_perf_all


def train_FedProx(data_obj, learning_rate, batch_size, epoch, com_amount, E, weight_decay, model_func,
                  init_model, mu, lr_decay_per_round):

    n_clnt = data_obj.n_client
    num_cells = data_obj.num_cells

    clnt_x = data_obj.clnt_x; clnt_y = data_obj.clnt_y

    # Average them based on number of datapoints (The one implemented)
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par

    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    log_name = './training_log/training_log_fedprox_{}_N{:1d}.txt'.format(data_obj.dataset, num_cells)
    with open(log_name, 'w') as log_file:
        # Optionally write an initial message or leave it empty to just clear the file
        log_file.write('Starting fedyn: {}_N{:1d} \n'.format(data_obj.dataset, num_cells))
    

    for t in range(com_amount):
        for j in range(num_cells):

            selected_clnts = [i for i in range((n_clnt // num_cells) * j, (n_clnt // num_cells) * (j + 1), 1)]
            edge_model = model_func().to(device)
            edge_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
            edge_mdl_param = get_mdl_params([edge_model], n_par)[0]
            edge_mdl_param_tensor = torch.tensor(edge_mdl_param, dtype=torch.float32, device=device)
            for e in range(E):
                for clnt in selected_clnts:
                    print('---- Training client %d' % clnt)
                    trn_x = clnt_x[clnt]
                    trn_y = clnt_y[clnt]
                    clnt_model = model_func().to(device)
                    clnt_model.load_state_dict(copy.deepcopy(dict(edge_model.named_parameters())))
                    for params in clnt_model.parameters():
                        params.requires_grad = True
                    clnt_model = train_fedprox_mdl(clnt_model, edge_mdl_param_tensor, mu, trn_x, trn_y,
                                                          learning_rate * (lr_decay_per_round ** t), batch_size, epoch,
                                                          weight_decay, data_obj.dataset)
                    clnt_params_list[clnt] = get_mdl_params([clnt_model], n_par)[0]

                # Scale with weights
                edge_mdl_param = np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0)
                edge_model = set_client_from_params(model_func(), edge_mdl_param)


                
                edge_mdl_param_tensor = torch.tensor(edge_mdl_param, dtype=torch.float32, device=device)


        avg_model = set_client_from_params(model_func().to(device), np.mean(clnt_params_list, axis=0))
        loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
        tst_perf_all[t] = [loss_tst, acc_tst]
        print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" % (t + 1, acc_tst, loss_tst))
        
        with open(log_name, 'a') as log_file:
            log_file.write('Round {:3d}, testing accuracy {:.2f}\n'.format(t, acc_tst))
        

    return tst_perf_all

