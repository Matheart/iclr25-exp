import logging
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, SVHN
from torch.utils.data import Dataset, DataLoader
import ipdb
from os.path import join as pjoin

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    '''
    This is basically the same as TensorDataset except that
    we store internally the '.data' and '.targets' attributes
    so that we have consistent operations and transformations
    on these variables when switching between torchvision
    Datasets and our own datasets.
    '''
    def __init__(self, X, Y, transform=None):
        if type(X) != torch.Tensor:
            X = torch.tensor(X)
        if type(Y) != torch.Tensor:
            Y = torch.tensor(Y)
        assert(X.size(0) == Y.size(0))
        self._data = X
        self._targets = Y
        self._transform = transform

    @property
    def data(self):
        # could transform here but for CF5m it takes
        # too much memory...maybe need to handle this
        # better, for now .data is untransformed
        return self._data

    @data.setter
    def data(self, X):
        if type(X) != torch.Tensor:
            X = torch.tensor(X)
        self._data = X

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, Y):
        if type(Y) != torch.Tensor:
            Y = torch.tensor(Y)
        self._targets = Y

    def __getitem__(self, index):
        if self._transform is not None:
            return (self._transform(self._data[index]), self._targets[index])
        return (self._data[index], self._targets[index])

    def __len__(self):
        return self._data.size(0)

class BinaryCIFAR10(CIFAR10):
    def __init__(self, *args, exclude_classes=[], **kwargs):
        super(BinaryCIFAR10, self).__init__(*args, **kwargs)

        if len(exclude_classes) == 0:
            return

        targets = np.array(self.targets)
        exclude = np.array(exclude_classes).reshape(1, -1)
        mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)

        self.data = self.data[mask]
        self.targets = targets[mask].tolist()

def load(cfg_data, cfg_model, worker_init_fn, pca_vectors=None):
    train_transforms = transforms.Compose([
        #transforms.RandomCrop(cfg_data.IMAGE_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg_data.NORM_SCALE['mean'], cfg_data.NORM_SCALE['std'])
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg_data.NORM_SCALE['mean'], cfg_data.NORM_SCALE['std'])
    ])

    if cfg_data.DATASET == 'CIFAR10':
        train = CIFAR10(root=cfg_data.ROOT, train=True, transform=train_transforms,
                        download=True)
        test =  CIFAR10(root=cfg_data.ROOT, train=False, transform=test_transforms,
                        download=True)
        if cfg_data.EVAL_NOISY_TEST:
            noisy_test = CIFAR10(root=cfg_data.ROOT, train=False, transform=test_transforms,
                               download=True)
    elif cfg_data.DATASET == 'SVHN':
        train = SVHN(root=cfg_data.ROOT, split='train', transform=test_transforms,
                     download=True)
        test = SVHN(root=cfg_data.ROOT, split='test', transform=test_transforms,
                     download=True)
        if cfg_data.EVAL_NOISY_TEST:
            noisy_test = SVHN(root=cfg_data.ROOT, split='test', transform=test_transforms,
                               download=True)
    elif cfg_data.DATASET == 'BinaryCIFAR10':
        def target_transform_fn(target):
            # class map for CIFAR10:
            # 0 - airplane, 1 - automobile, 2 - bird, 3 - cat, 4 - deer, 5 - dog
            # 6 - frog, 7 - horse, 8 - ship, 9 - truck
            if target in [0, 1, 8, 9]:
                return 0
            return 1
        train = BinaryCIFAR10(root=cfg_data.ROOT, train=True, transform=train_transforms,
                        download=True, exclude_classes=[6, 7], target_transform=target_transform_fn)
        test =  BinaryCIFAR10(root=cfg_data.ROOT, train=False, transform=test_transforms,
                        download=True, exclude_classes=[6, 7], target_transform=target_transform_fn)
        if cfg_data.EVAL_NOISY_TEST:
            noisy_test = BinaryCIFAR10(root=cfg_data.ROOT, train=False, transform=test_transforms,
                               download=True, exclude_classes=[6, 7], target_transform=target_transform_fn)
    elif cfg_data.DATASET == 'MNIST':
        train = MNIST(root=cfg_data.ROOT, train=True, transform=test_transforms,
                        download=True)
        test =  MNIST(root=cfg_data.ROOT, train=False, transform=test_transforms,
                        download=True)
    elif cfg_data.DATASET == 'BinaryMNIST':
        def target_transform_fn(target):
            if target % 2 == 0:
                return 0
            return 1
        train = MNIST(root=cfg_data.ROOT, train=True, transform=test_transforms,
                        download=True, target_transform=target_transform_fn)
        test =  MNIST(root=cfg_data.ROOT, train=False, transform=test_transforms,
                        download=True, target_transform=target_transform_fn)
        if cfg_data.EVAL_NOISY_TEST:
            noisy_test = MNIST(root=cfg_data.ROOT, train=False, transform=test_transforms,
                               download=True, target_transform=target_transform_fn)
    elif cfg_data.DATASET == 'synthetic1':
        # sampling 100-dim vectors ~ N(0,1)
        trainX = torch.randn([cfg_data.DATA_N] + cfg_data.DIM_Syntc)
        testX = torch.randn([cfg_data.TEST_N] + cfg_data.DIM_Syntc) # 10000 "held out" test sets, like in CIFAR10
        trainX = trainX.T
        testX = testX.T
        trainX /= torch.linalg.norm(trainX, dim=0)
        testX /= torch.linalg.norm(testX, dim=0)
        trainX = trainX.T
        testX = testX.T
        # trainY = torch.randint(0, 2, (cfg_data.DATA_N,))
        # testY = torch.randint(0, 2, (cfg_data.TEST_N,))
        trainY = torch.zeros(cfg_data.DATA_N)
        testY = torch.zeros(cfg_data.TEST_N)

        train = CustomDataset(trainX, trainY)
        test = CustomDataset(testX, testY)
        if cfg_data.EVAL_NOISY_TEST:
            #noisyY = torch.randint(0, 2, (cfg_data.TEST_N,))
            noisyY = torch.zeros(cfg_data.TEST_N)
            noisy_test = CustomDataset(testX, noisyY)
    elif cfg_data.DATASET == 'cifar5m-flat':
        trainX, trainY, testX, testY = load_cifar5m(flatten=True)
        transform=None
        if pca_vectors is not None:
            # cf5m stored as uint8 for memory efficiency
            # we will convert -> float and apply PCA on the fly as a data transform
            transform = PCATransform(pca_vectors, cfg_model.INP_DIM, div_255=True)

        train = CustomDataset(trainX, trainY, transform=transform)
        test = CustomDataset(testX, testY, transform=transform)
    else:
        raise Exception('Unsupported dataset')

    if cfg_data.DATASET == 'SVHN':
        train.labels = np.array(train.labels).reshape((-1, ))
        test.labels = np.array(test.labels).reshape((-1, ))
        if cfg_data.EVAL_NOISY_TEST:
            noisy_test.labels = np.array(noisy_test.labels).reshape((-1, ))
    else:
        train.targets = np.array(train.targets).reshape((-1, )) ##### Making sure that targets are formatted as (N,)
        test.targets = np.array(test.targets).reshape((-1, )) ##### Making sure that targets are formatted as (N,)
        if cfg_data.EVAL_NOISY_TEST:
            noisy_test.targets = np.array(noisy_test.targets).reshape((-1, ))

    if cfg_data.LABEL_NOISE > 0.0:
        if cfg_data.DATASET == 'SVHN':
            n = train.labels.shape[0]
            n_classes = set(list(np.unique(train.labels)))
        else:
            n = train.targets.shape[0]
            n_classes = set(list(np.unique(train.targets)))
        label_samples = np.random.choice(np.arange(n), size=int(n*cfg_data.LABEL_NOISE), replace=False)
        for idx in label_samples:
            if cfg_data.DATASET == 'BinaryMNIST':
                # binarized labels use target_transform so switch evens -> odds and vice versa for label noise
                if train.targets[idx] % 2 == 0:
                    train.targets[idx] = 1
                else:
                    train.targets[idx] = 0
            elif cfg_data.DATASET == 'BinaryCIFAR10':
                if train.targets[idx] in [0, 1, 8, 9]:
                    train.targets[idx] = 2
                else:
                    train.targets[idx] = 1
            elif cfg_data.DATASET == 'synthetic1':
                # temp: all labels 1, switch to -1
                train.targets[idx] = -1
            else:
                # multi class label noise switches to any other class label
                if cfg_data.DATASET == 'SVHN':
                    other_labels = list(n_classes - {train.labels[idx]})
                    train.labels[idx] = np.random.choice(other_labels, size=1)
                else:
                    other_labels = list(n_classes - {train.targets[idx]})
                    train.targets[idx] = np.random.choice(other_labels, size=1)
        if cfg_data.EVAL_NOISY_TEST:
            n = noisy_test.targets.shape[0]
            label_samples = np.random.choice(np.arange(n), size=int(n*cfg_data.LABEL_NOISE), replace=False)
            for idx in label_samples:
                if cfg_data.DATASET == 'BinaryMNIST':
                    if noisy_test.targets[idx] % 2 == 0:
                        noisy_test.targets[idx] = 1
                    else:
                        noisy_test.targets[idx] = 0
                elif cfg_data.DATASET == 'BinaryCIFAR10':
                    if noisy_test.targets[idx] in [0, 1, 8, 9]:
                        noisy_test.targets[idx] = 2
                    else:
                        noisy_test.targets[idx] = 1
                elif cfg_data.DATASET == 'synthetic1':
                    noisy_test.targets[idx] = -1
                else:
                    other_labels = list(n_classes - {noisy_test.targets[idx]})
                    noisy_test.targets[idx] = np.random.choice(other_labels, size=1)

    #if cfg_data.DATASET_SUBSAMPLE_RATIO < 1.0:
    if cfg_data.TRAIN_N < cfg_data.DATA_N:
        samples = np.random.choice(cfg_data.DATA_N, size=cfg_data.TRAIN_N, replace=False)
        train = torch.utils.data.Subset(train, samples)

        '''
        if len(train.data.size()) == 4:
            # images
            train.data = train.data[samples,:,:,:]
        elif len(train.data.size()) == 2:
            # flat vectors
            train.data = train.data[samples,:]

        train.targets = np.array(train.targets)
        train.targets = train.targets[samples,]
        '''

    if cfg_data.RANDOMIZE_LABELS:
        # probs_tr = np.ones_like(train.targets) * cfg_data.LABEL_PROB
        # probs_te = np.ones_like(test.targets) * cfg_data.LABEL_PROB
        # new_targets_tr = np.random.binomial(1, probs_tr) # generates 0/1 random vector
        # new_targets_te = np.random.binomial(1, probs_te)
        # if cfg_data.BINARY_LABELS[0] < 0:
        #     new_targets_tr = cfg_data.BINARY_LABELS[1]*(2*new_targets_tr - 1) # convert {0,1} labels -> {-1, 1}
        #     new_targets_te = cfg_data.BINARY_LABELS[1]*(2*new_targets_te - 1)
        # train.targets = new_targets_tr
        # test.targets = new_targets_te
        train.targets = np.random.normal(loc=0.0, scale=np.sqrt(cfg_data.NOISE_VARIANCE), size=(train.targets.shape[0],))
        #train.targets = np.random.randn(train.targets.shape[0])
        if cfg_data.EVAL_NOISY_TEST:
            #noisy_test.targets = np.random.randn(noisy_test.targets.shape[0])
            noisy_test.targets = np.random.normal(loc=0.0, scale=np.sqrt(cfg_data.NOISE_VARIANCE), size=(noisy_test.targets.shape[0],))
        #train.targets = np.ones((train.targets.shape[0],)) + np.random.randn(train.targets.shape[0])
        #if cfg_data.EVAL_NOISY_TEST:
        #    noisy_test.targets = np.ones((noisy_test.targets.shape[0],)) + np.random.randn(noisy_test.targets.shape[0])

    '''
    try:
        logger.info('train.data.shape: %s' % str(train.data.shape))
        logger.info('test.data.shape: %s' % str(test.data.shape))
    except:
        logger.info('train.data.shape: %s' % str(train.dataset.data.shape))
        logger.info('test.data.shape: %s' % str(test.dataset.data.shape))
    '''

    train_loader = DataLoader(train, batch_size=cfg_data.BATCH_SIZE, shuffle=True,
                              num_workers=cfg_data.NUM_WORKERS, drop_last=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test, batch_size=cfg_data.BATCH_SIZE, shuffle=False,
                             num_workers=1, drop_last=False)
    if cfg_data.EVAL_NOISY_TEST:
        noisy_test_loader = DataLoader(noisy_test, batch_size=cfg_data.BATCH_SIZE, drop_last=False, shuffle=False,
                                       num_workers=1)
        return train_loader, [('clean_', test_loader), ('noisy_', noisy_test_loader)]

    return train_loader, [('', test_loader)]

def load_cifar5m(flatten=True, n_parts_to_load=5):
    '''
        Returns 5million synthetic samples.
        warning: returns as numpy array of unit8s, not torch tensors.
    '''

    nte = 10000 # num. of test samples to use (max 1e6)
    print('Downloading CIFAR 5mil...')
    local_dir = 'ANON'
    npart = 1000448
    X_tr = np.empty((n_parts_to_load*npart, 32, 32, 3), dtype=np.uint8)
    Ys = []
    print('Loading CIFAR 5mil...')
    for i in range(n_parts_to_load):
        z = np.load(pjoin(local_dir, f'part{i}.npz'))
        X_tr[i*npart: (i+1)*npart] = z['X']
        Ys.append(torch.tensor(z['Y']).long())
        print(f'Loaded part {i+1}/6')
    Y_tr = torch.cat(Ys)

    z = np.load(pjoin(local_dir, 'part5.npz')) # use the 6th million for test.
    print(f'Loaded part 6/6')

    X_te = z['X'][:nte]
    Y_te = torch.tensor(z['Y'][:nte]).long()

    if flatten:
        print('Flattening vectors')
        # flatten to (batch, inp_dim)
        X_tr = np.reshape(X_tr, (X_tr.shape[0], -1))
        X_te = np.reshape(X_te, (X_te.shape[0], -1))

    return X_tr, Y_tr, X_te, Y_te

class PCATransform(object):
    def __init__(self, pca_vectors, inp_dim, div_255=True):
        self.components_ = torch.Tensor(pca_vectors['components_']).float()
        self.mean_ = torch.Tensor(pca_vectors['mean_']).float()
        self.div_255 = div_255
        self.inp_dim = inp_dim

    def __call__(self, sample):
        if self.div_255:
            sample = sample.float()/255.0

        if self.inp_dim < self.components_.size(1):
            sample.unsqueeze_(0)
            sample = torch.mm(sample-self.mean_, self.components_.T[:, :self.inp_dim])
            sample.squeeze_()
        return sample
