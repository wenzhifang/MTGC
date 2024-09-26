from utils_libs import *

class DatasetObject:
    def __init__(self, dataset, n_client, num_cells, rule, unbalanced_sgm=0, rule_arg=''):
        self.dataset  = dataset
        self.n_client = n_client
        self.num_cells = num_cells
        self.rule     = rule
        self.rule_arg = rule_arg
        rule_arg_str  = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        self.name = "%s_%d_%s_%s" %(self.dataset, self.n_client, self.rule, rule_arg_str)
        self.name += '_%f' %unbalanced_sgm if unbalanced_sgm!=0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = 'Data'
        self.set_data()
        
    def set_data(self):
        # Prepare data if not ready
        
        if not os.path.exists('%s/%s' %(self.data_path, self.name)):
            # Get Raw data                
            if self.dataset == 'mnist':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trnset = torchvision.datasets.MNIST(root='%s/Raw' %self.data_path, 
                                                    train=True , download=True, transform=transform)
                tstset = torchvision.datasets.MNIST(root='%s/Raw' %self.data_path, 
                                                    train=False, download=True, transform=transform)
                
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            
            if self.dataset == 'fashion_mnist':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trnset = torchvision.datasets.FashionMNIST(root='%s/Raw' %self.data_path, 
                                                    train=True , download=True, transform=transform)
                tstset = torchvision.datasets.FashionMNIST(root='%s/Raw' %self.data_path, 
                                                    train=False, download=True, transform=transform)
                
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;    
                
            
            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

                trnset = torchvision.datasets.CIFAR10(root='%s/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR10(root='%s/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
                
            if self.dataset == 'CIFAR100':
                print(self.dataset)
                # mean and std are validated here: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                                                                     std=[0.2675, 0.2565, 0.2761])])
                trnset = torchvision.datasets.CIFAR100(root='%s/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR100(root='%s/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            
            if self.dataset == 'TinyImageNet':
                print(self.dataset)
                transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[ 0.485, 0.456, 0.406 ],
                                                                    std=[ 0.229, 0.224, 0.225 ])
                                                ])
                
                transform_trn = transforms.Compose([transforms.RandomCrop(64, padding=8),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                    ]) 
                transform_tst = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                    ])
                
                #trnset = torchvision.datasets.ImageFolder(root='%s/Raw/tiny-imagenet-200/train' %self.data_path,transform=transform_trn)
                trnset = TinyImageNet(root='%s/Raw/tiny-imagenet-200' %self.data_path, train=True, transform=transform_trn)
                #tstset = torchvision.datasets.ImageFolder(root='%s/Raw/tiny-imagenet-200/val' %self.data_path,transform=transform_tst)
                tstset = TinyImageNet(root='%s/Raw/tiny-imagenet-200' %self.data_path, train=False, transform=transform_tst)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=100000, shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 64; self.height = 64; self.n_cls = 200;
            
            if self.dataset != 'emnist':
                trn_itr = trn_load.__iter__(); tst_itr = tst_load.__iter__() 
                # labels are of shape (n_data,)
                trn_x, trn_y = trn_itr.__next__()
                tst_x, tst_y = tst_itr.__next__()

                trn_x = trn_x.numpy(); trn_y = trn_y.numpy().reshape(-1,1)
                tst_x = tst_x.numpy(); tst_y = tst_y.numpy().reshape(-1,1)
            
            
            if self.dataset == 'emnist':
                emnist = io.loadmat(self.data_path + "/Raw/matlab/emnist-letters.mat")
                # load training dataset
                x_train = emnist["dataset"][0][0][0][0][0][0]
                x_train = x_train.astype(np.float32)

                # load training labels
                y_train = emnist["dataset"][0][0][0][0][0][1] - 1 # make first class 0
                '''

                # take first 10 classes of letters
                trn_idx = np.where(y_train < 10)[0]

                y_train = y_train[trn_idx]
                x_train = x_train[trn_idx]
                '''

                mean_x = np.mean(x_train)
                std_x = np.std(x_train)

                # load test dataset
                x_test = emnist["dataset"][0][0][1][0][0][0]
                x_test = x_test.astype(np.float32)

                # load test labels
                y_test = emnist["dataset"][0][0][1][0][0][1] - 1 # make first class 0
                '''

                tst_idx = np.where(y_test < 10)[0]

                y_test = y_test[tst_idx]
                x_test = x_test[tst_idx]
                '''
                
                x_train = x_train.reshape((-1, 1, 28, 28))
                x_test  = x_test.reshape((-1, 1, 28, 28))
                
                # normalise train and test features

                trn_x = (x_train - mean_x) / std_x
                trn_y = y_train
                
                tst_x = (x_test  - mean_x) / std_x
                tst_y = y_test
                
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 26;
            
            # Shuffle Data
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]
            
            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y
            
            
            ###
            n_data_per_clnt = int((len(trn_y)) / self.n_client)
            if self.unbalanced_sgm != 0:
                # Draw from lognormal distribution
                clnt_data_list = (np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalanced_sgm, size=self.n_client))
                clnt_data_list = (clnt_data_list/np.sum(clnt_data_list)*len(trn_y)).astype(int)
                diff = np.sum(clnt_data_list) - len(trn_y)

                # Add/Subtract the excess number starting from first client
                if diff!= 0:
                    for clnt_i in range(self.n_client):
                        if clnt_data_list[clnt_i] > diff:
                            clnt_data_list[clnt_i] -= diff
                            break
            else:
                clnt_data_list = (np.ones(self.n_client) * n_data_per_clnt).astype(int)
            ###
            if self.rule == 'noniid':
                idx_list = [np.where(trn_y==i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]
                cell_x = [np.zeros((n_data_per_clnt * self.n_client//self.num_cells, self.channels, self.height, self.width)).astype(np.float32)
                                   for _ in range(self.num_cells)]
                cell_y = [np.zeros((n_data_per_clnt * self.n_client//self.num_cells, 1)).astype(np.int64)
                          for _ in range(self.num_cells)]
                cell_data_list = [n_data_per_clnt * self.n_client//self.num_cells for _ in range(self.num_cells)]
                cls_priors   = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.num_cells)
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                empty_list = []
                while(np.sum(cell_data_list)!=0):
                    curr_cell = np.random.randint(self.num_cells)
                    # If current node is full resample a cell
                    print('Remaining Data: %d' %np.sum(cell_data_list))
                    if cell_data_list[curr_cell] <= 0:
                        continue
                    cell_data_list[curr_cell] -= 1
                    #curr_prior = prior_cumsum[curr_cell]
                    curr_prior = np.copy(prior_cumsum[curr_cell])
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if trn_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            empty_list.append(cls_label)
                            empty_list = list(set(empty_list))
                            curr_prior[empty_list] = 0
                            if curr_prior.sum() > 0:
                                curr_prior /= curr_prior.sum()
                                prior_cumsum[curr_cell] = np.cumsum(curr_prior)
                            continue
                        cls_amount[cls_label] -= 1
                        cell_x[curr_cell][cell_data_list[curr_cell]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                        cell_y[curr_cell][cell_data_list[curr_cell]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]
                        break

                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
                for j in range(self.num_cells):
                    #idx_list = [np.where(cell_y[j]==i)[0] for i in range(self.n_cls)]
                    idx_list = [np.where(cell_y[j]==i)[0] for i in range(self.n_cls) if np.where(cell_y[j]==i)[0].size > 0]
                    cell_cls = len(idx_list)
                    cls_amount = [len(idx_list[i]) for i in range(cell_cls)]
                    print('num of classes as cell {}: {}'.format(j,cell_cls))

                    cls_priors   = np.random.dirichlet(alpha=[self.rule_arg]*cell_cls,size=self.n_client//self.num_cells)
                    prior_cumsum = np.cumsum(cls_priors, axis=1)
                    empty_list = []

                    while(np.sum(clnt_data_list[j*self.n_client//self.num_cells:(j+1)*self.n_client//self.num_cells])!=0):
                        curr_clnt = np.random.randint(j*self.n_client//self.num_cells, (j+1)*self.n_client//self.num_cells)
                        # If current node is full resample a client
                        print('Remaining Data: %d' %np.sum(clnt_data_list[j*self.n_client//self.num_cells:(j+1)*self.n_client//self.num_cells]))
                        if clnt_data_list[curr_clnt] <= 0:
                            continue
                        clnt_data_list[curr_clnt] -= 1
                        #curr_prior = prior_cumsum[curr_clnt - j*self.n_client//self.num_cells]
                        curr_prior = np.copy(prior_cumsum[curr_clnt - j * self.n_client // self.num_cells])
                        while True:
                            cls_label = np.argmax(np.random.uniform() <= curr_prior)
                            # Redraw class label if trn_y is out of that class
                            if cls_amount[cls_label] <= 0:
                                empty_list.append(cls_label)
                                empty_list = list(set(empty_list))
                                curr_prior[empty_list] = 0
                                if curr_prior.sum() > 0:
                                    curr_prior /= curr_prior.sum()
                                    prior_cumsum[curr_clnt - j * self.n_client // self.num_cells] = np.cumsum(curr_prior)
                                continue
                            cls_amount[cls_label] -= 1
                            clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = cell_x[j][idx_list[cls_label][cls_amount[cls_label]]]
                            clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = cell_y[j][idx_list[cls_label][cls_amount[cls_label]]]
                            break

                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)
                        ###
            if self.rule == 'mix2':
                idx_list = [np.where(trn_y==i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]
                cell_x = [np.zeros((n_data_per_clnt * self.n_client//self.num_cells, self.channels, self.height, self.width)).astype(np.float32)
                                   for _ in range(self.num_cells)]
                cell_y = [np.zeros((n_data_per_clnt * self.n_client//self.num_cells, 1)).astype(np.int64)
                          for _ in range(self.num_cells)]
                cell_data_list = [n_data_per_clnt * self.n_client//self.num_cells for _ in range(self.num_cells)]
                cls_priors   = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.num_cells)
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                empty_list = []
                while(np.sum(cell_data_list)!=0):
                    curr_cell = np.random.randint(self.num_cells)
                    # If current node is full resample a cell
                    print('Remaining Data: %d' %np.sum(cell_data_list))
                    if cell_data_list[curr_cell] <= 0:
                        continue
                    cell_data_list[curr_cell] -= 1
                    #curr_prior = prior_cumsum[curr_cell]
                    curr_prior = np.copy(prior_cumsum[curr_cell])
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if trn_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            empty_list.append(cls_label)
                            empty_list = list(set(empty_list))
                            curr_prior[empty_list] = 0
                            if curr_prior.sum() > 0:
                                curr_prior /= curr_prior.sum()
                                prior_cumsum[curr_cell] = np.cumsum(curr_prior)
                            continue
                        cls_amount[cls_label] -= 1
                        cell_x[curr_cell][cell_data_list[curr_cell]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                        cell_y[curr_cell][cell_data_list[curr_cell]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]
                        break

                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
                for j in range(self.num_cells):
                    #idx_list = [np.where(cell_y[j]==i)[0] for i in range(self.n_cls)]
                    idx_list = [np.where(cell_y[j]==i)[0] for i in range(self.n_cls) if np.where(cell_y[j]==i)[0].size > 0]
                    cell_cls = len(idx_list)
                    cls_amount = [len(idx_list[i]) for i in range(cell_cls)]
                    print('num of classes as cell {}: {}'.format(j,cell_cls))

                    cls_priors   = np.random.dirichlet(alpha=[10000]*cell_cls,size=self.n_client//self.num_cells)
                    prior_cumsum = np.cumsum(cls_priors, axis=1)
                    empty_list = []

                    while(np.sum(clnt_data_list[j*self.n_client//self.num_cells:(j+1)*self.n_client//self.num_cells])!=0):
                        curr_clnt = np.random.randint(j*self.n_client//self.num_cells, (j+1)*self.n_client//self.num_cells)
                        # If current node is full resample a client
                        print('Remaining Data: %d' %np.sum(clnt_data_list[j*self.n_client//self.num_cells:(j+1)*self.n_client//self.num_cells]))
                        if clnt_data_list[curr_clnt] <= 0:
                            continue
                        clnt_data_list[curr_clnt] -= 1
                        #curr_prior = prior_cumsum[curr_clnt - j*self.n_client//self.num_cells]
                        curr_prior = np.copy(prior_cumsum[curr_clnt - j * self.n_client // self.num_cells])
                        while True:
                            cls_label = np.argmax(np.random.uniform() <= curr_prior)
                            # Redraw class label if trn_y is out of that class
                            if cls_amount[cls_label] <= 0:
                                empty_list.append(cls_label)
                                empty_list = list(set(empty_list))
                                curr_prior[empty_list] = 0
                                if curr_prior.sum() > 0:
                                    curr_prior /= curr_prior.sum()
                                    prior_cumsum[curr_clnt - j * self.n_client // self.num_cells] = np.cumsum(curr_prior)
                                continue
                            cls_amount[cls_label] -= 1
                            clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = cell_x[j][idx_list[cls_label][cls_amount[cls_label]]]
                            clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = cell_y[j][idx_list[cls_label][cls_amount[cls_label]]]
                            break

                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)



            
            if self.rule == 'Dirichlet':
                cls_priors   = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.n_client)
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                idx_list = [np.where(trn_y==i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]

                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
    
                while(np.sum(clnt_data_list)!=0):
                    curr_clnt = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    print('Remaining Data: %d' %np.sum(clnt_data_list))
                    if clnt_data_list[curr_clnt] <= 0:
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if trn_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1
                        clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                        clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break
                
                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)
                
                cls_means = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means[clnt,cls] = np.mean(clnt_y[clnt]==cls)
                prior_real_diff = np.abs(cls_means-cls_priors)
                print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
                print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))
            
            elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm==0:
                assert len(trn_y)//100 % self.n_client == 0 
                # Only have the number clients if it divides 500
                # Perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(trn_y[:, 0])
                n_data_per_clnt = len(trn_y) // self.n_client
                # clnt_x dtype needs to be float32, the same as weights
                clnt_x = np.zeros((self.n_client, n_data_per_clnt, 3, 32, 32), dtype=np.float32)
                clnt_y = np.zeros((self.n_client, n_data_per_clnt, 1), dtype=np.float32)
                trn_x = trn_x[idx] # 50000*3*32*32
                trn_y = trn_y[idx]
                n_cls_sample_per_device = n_data_per_clnt // 100
                for i in range(self.n_client): # devices
                    for j in range(100): # class
                        clnt_x[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :, :, :] = trn_x[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :, :, :]
                        clnt_y[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :] = trn_y[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :] 
            
            
            elif self.rule == 'iid':
                                
                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
            
                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_client):
                    clnt_x[clnt_idx_] = trn_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                    clnt_y[clnt_idx_] = trn_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                
                
                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)

            
            self.clnt_x = clnt_x; self.clnt_y = clnt_y

            self.tst_x  = tst_x;  self.tst_y  = tst_y
            
            # Save data
            os.mkdir('%s/%s' %(self.data_path, self.name))
            
            np.save('%s/%s/clnt_x.npy' %(self.data_path, self.name), clnt_x)
            np.save('%s/%s/clnt_y.npy' %(self.data_path, self.name), clnt_y)

            np.save('%s/%s/tst_x.npy'  %(self.data_path, self.name),  tst_x)
            np.save('%s/%s/tst_y.npy'  %(self.data_path, self.name),  tst_y)

        else:
            print("Data is already downloaded in the folder.")
            self.clnt_x = np.load('%s/%s/clnt_x.npy' %(self.data_path, self.name), allow_pickle=True)
            self.clnt_y = np.load('%s/%s/clnt_y.npy' %(self.data_path, self.name), allow_pickle=True)
            self.n_client = len(self.clnt_x)

            self.tst_x  = np.load('%s/%s/tst_x.npy'  %(self.data_path, self.name), allow_pickle=True)
            self.tst_y  = np.load('%s/%s/tst_y.npy'  %(self.data_path, self.name), allow_pickle=True)
            
            if self.dataset == 'mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            if self.dataset == 'fashion_mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'emnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'TinyImageNet':
                self.channels = 3; self.width = 64; self.height = 64; self.n_cls = 200;
                
        print('Class frequencies:')
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " %clnt + 
                  ', '.join(["%.3f" %np.mean(self.clnt_y[clnt]==cls) for cls in range(self.n_cls)]) + 
                  ', Amount:%d' %self.clnt_y[clnt].shape[0])
            count += self.clnt_y[clnt].shape[0]
        
        
        print('Total Amount:%d' %count)
        print('--------')

        print("      Test: " + 
              ', '.join(["%.3f" %np.mean(self.tst_y==cls) for cls in range(self.n_cls)]) + 
              ', Amount:%d' %self.tst_y.shape[0])

# Original prepration is from LEAF paper...
# This loads Shakespeare dataset only.
# data_path/train and data_path/test are assumed to be processed
# To make the dataset smaller,
# We take 2000 datapoints for each client in the train_set

class ShakespeareObjectCrop:
    def __init__(self, data_path, dataset_prefix, crop_amount=2000, tst_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name    = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')
        
        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as 
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.
        # Ignore groups information, combine test cases for different clients into one test data
        # Change structure to DatasetObject structure
        
        self.users = users
        
        self.n_client = len(users)
        self.user_idx = np.asarray(list(range(self.n_client)))
        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))
        
        tst_data_count = 0
        
        for clnt in range(self.n_client):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[clnt]]['x'])-crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[clnt]]['x'])[start:start+crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[clnt]]['y'])[start:start+crop_amount]
            
        tst_data_count = (crop_amount//tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))
        
        tst_data_count = 0
        for clnt in range(self.n_client):
            curr_amount = (crop_amount//tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[clnt]]['x'])-curr_amount)
            self.tst_x[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[clnt]]['x'])[start:start+curr_amount]
            self.tst_y[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[clnt]]['y'])[start:start+curr_amount]
            
            tst_data_count += curr_amount
        
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)
        
        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)
        
        # Convert characters to numbers
        
        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)
        
        self.tst_x_char = np.copy(self.tst_x)
        self.tst_y_char = np.copy(self.tst_y)
        
        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))
        

        for clnt in range(len(self.clnt_x_char)):
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))
            
            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)
                
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)
        
        
        self.tst_x = list(range(len(self.tst_x_char)))
        self.tst_y = list(range(len(self.tst_x_char)))
                
        for idx in range(len(self.tst_x_char)):
            self.tst_x[idx] = np.asarray(word_to_indices(self.tst_x_char[idx]))
            self.tst_y[idx] = np.argmax(np.asarray(letter_to_vec(self.tst_y_char[idx]))).reshape(-1)
        
        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)
        
        
class ShakespeareObjectCrop_noniid:
    def __init__(self, data_path, dataset_prefix, n_client=100, crop_amount=2000, tst_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name    = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')

        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as 
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.        
        # Change structure to DatasetObject structure
        
        self.users = users 

        tst_data_count_per_clnt = (crop_amount//tst_ratio)
        # Group clients that have at least crop_amount datapoints
        arr = []
        for clnt in range(len(users)):
            if (len(np.asarray(train_data[users[clnt]]['y'])) > crop_amount 
                and len(np.asarray(test_data[users[clnt]]['y'])) > tst_data_count_per_clnt):
                arr.append(clnt)

        # choose n_client clients randomly
        self.n_client = n_client
        np.random.seed(rand_seed)
        np.random.shuffle(arr)
        self.user_idx = arr[:self.n_client]
          
        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))
        
        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[idx]]['x'])-crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[idx]]['x'])[start:start+crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[idx]]['y'])[start:start+crop_amount]
            
        tst_data_count = (crop_amount//tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))
        
        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):
            
            curr_amount = (crop_amount//tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[idx]]['x'])-curr_amount)
            self.tst_x[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['x'])[start:start+curr_amount]
            self.tst_y[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['y'])[start:start+curr_amount]
            tst_data_count += curr_amount
            
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)
        
        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)
        
        # Convert characters to numbers
        
        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)
        
        self.tst_x_char = np.copy(self.tst_x)
        self.tst_y_char = np.copy(self.tst_y)
        
        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))
        

        for clnt in range(len(self.clnt_x_char)):
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))
            
            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)
                
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)
        
        
        self.tst_x = list(range(len(self.tst_x_char)))
        self.tst_y = list(range(len(self.tst_x_char)))
                
        for idx in range(len(self.tst_x_char)):
            self.tst_x[idx] = np.asarray(word_to_indices(self.tst_x_char[idx]))
            self.tst_y[idx] = np.argmax(np.asarray(letter_to_vec(self.tst_y_char[idx]))).reshape(-1)
        
        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)
    
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        if self.name == 'mnist' or self.name == 'fashion_mnist' or self.name == 'emnist':
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()
                
        elif self.name == 'TinyImageNet':
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()
            
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
        
            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
                
        elif self.name == 'shakespeare':
            
            self.X_data = data_x
            self.y_data = data_y
                
            self.X_data = torch.tensor(self.X_data).long()
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(self.y_data).float()
            
    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'mnist' or self.name == 'fashion_mnist' or self.name == 'emnist':
            X = self.X_data[idx, :]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y
        
        elif self.name == 'TinyImageNet':
            X = self.X_data[idx]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y
        
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img # Horizontal flip
                if (np.random.rand() > .5):
                # Random cropping 
                    pad = 4
                    extended_img = np.zeros((3,32 + pad *2, 32 + pad *2)).astype(np.float32)
                    extended_img[:,pad:-pad,pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:,dim_1:dim_1+32,dim_2:dim_2+32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y
    
        elif self.name == 'shakespeare':
            x = self.X_data[idx]
            y = self.y_data[idx] 
            return x, y
