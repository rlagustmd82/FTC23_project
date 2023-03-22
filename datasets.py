"""
Data loader
"""

def _get_download_dir():
    downloads_path = str(Path.home() / "Downloads/dipDECK_data")
    if not os.path.isdir(downloads_path):
        os.makedirs(downloads_path)
    return downloads_path

def _download_file(download_path, filename):
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(download_path, filename)
    ssl._create_default_https_context = ssl._create_default_https_context


def _load_data_file(filename, download_path, delimiter=",", last_column_are_labels=True):
    if not os.path.isfile(filename):
        _download_file(download_path, filename)
    datafile = np.genfromtxt(filename, delimiter=delimiter)
    if last_column_are_labels:
        data = datafile[:, :-1]
        labels = datafile[:, -1]
    else:
        data = datafile[:, 1:]
        labels = datafile[:, 0]
    return data, labels

def _load_torch_image_data(data_source, add_testdata):
    # Get data from source
    ssl._create_default_https_context = ssl._create_unverified_context
    dataset = data_source(root=_get_download_dir(), train=True, download=True)
    data = dataset.data
    labels = dataset.targets
    if add_testdata:
        testset = data_source(root=_get_download_dir(), train=False, download=True)
        data = torch.cat([data, testset.data], dim=0)
        labels = torch.cat([labels, testset.targets], dim=0)
    ssl._create_default_https_context = ssl._create_default_https_context
    # Flatten shape
    if data.dim() == 3:
        data = data.reshape(-1, data.shape[1] * data.shape[2])
    else:
        data = data.reshape(-1, data.shape[1] * data.shape[2] * data.shape[3])
    # Move data to CPU
    data_cpu = data.detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()
    return data_cpu, labels_cpu

def _image_z_transformation(data):
    return (data - np.mean(data)) / np.std(data)


def load_mnist(add_testdata=True, normalize=True):
    data, labels = _load_torch_image_data(torchvision.datasets.MNIST, add_testdata)
    if normalize:
        data = _image_z_transformation(data)
    return data, labels

def load_kmnist(add_testdata=True, normalize=True):
    data, labels = _load_torch_image_data(torchvision.datasets.KMNIST, add_testdata)
    if normalize:
        data = _image_z_transformation(data)
    return data, labels

def load_fmnist(add_testdata=True, normalize=True):
    data, labels = _load_torch_image_data(torchvision.datasets.FashionMNIST, add_testdata)
    if normalize:
        data = _image_z_transformation(data)
    return data, labels

def load_usps(add_testdata=True, normalize=True):
    dataset = torchvision.datasets.USPS(root=_get_download_dir(), train=True, download=True)
    data = dataset.data
    labels = dataset.targets
    if add_testdata:
        test_dataset = torchvision.datasets.USPS(root=_get_download_dir(), train=False, download=True)
        data = np.r_[data, test_dataset.data]
        labels = np.r_[labels, test_dataset.targets]
    data = data.reshape(-1, 256)
    if normalize:
        data = _image_z_transformation(data)
    return data, labels

def load_optdigits(add_testdata=True, normalize=True):
    filename = _get_download_dir() + "/optdigits.tra"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra")
    if add_testdata:
        filename = _get_download_dir() + "/optdigits.tes"
        test_data, test_labels = _load_data_file(filename,
                                                 "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes")
        data = np.r_[data, test_data]
        labels = np.r_[labels, test_labels]
    if normalize:
        data = _image_z_transformation(data)
    return data, labels

def load_pendigits(add_testdata=True, normalize = True):
    filename = _get_download_dir() + "/pendigits.tra"
    data, labels = _load_data_file(filename,
                                   "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra")
    if add_testdata:
        filename = _get_download_dir() + "/pendigits.tes"
        test_data, test_labels = _load_data_file(filename,
                                                 "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes")
        data = np.r_[data, test_data]
        labels = np.r_[labels, test_labels]
    if normalize:
        data = scale(data, axis=0)
    return data, labels

def load_letterrecognition(normalize = True):
    filename = _get_download_dir() + "/letter-recognition.data"
    if not os.path.isfile(filename):
        _download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
            filename)
    # Transform letters to integers
    letter_mappings = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4", "F": "5", "G": "6", "H": "7", "I": "8",
                       "J": "9", "K": "10", "L": "11", "M": "12", "N": "13", "O": "14", "P": "15", "Q": "16",
                       "R": "17", "S": "18", "T": "19", "U": "20", "V": "21", "W": "22", "X": "23", "Y": "24",
                       "Z": "25"}
    with open(filename, "r") as f:
        file_text = f.read()
    file_text = file_text.replace("\n", ",")
    for k in letter_mappings.keys():
        file_text = file_text.replace(k, letter_mappings[k])
    # Create numpy array
    datafile = np.fromstring(file_text, sep=",").reshape(-1, 17)
    data = datafile[:, 1:]
    labels = datafile[:, 0]
    if normalize:
        data = scale(data, axis=0)
    return data, labels