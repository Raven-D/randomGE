import numpy as np
from struct import unpack

def __read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img

def __read_label(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab
    
def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab

def load_mnist(train_image_path, train_label_path, test_image_path, test_label_path, normalize=True, one_hot=True):
    image = {
        'train' : __read_image(train_image_path),
        'test'  : __read_image(test_image_path)
    }

    label = {
        'train' : __read_label(train_label_path),
        'test'  : __read_label(test_label_path)
    }
    
    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])

_tr, _te = load_mnist('./mnist/train-images.idx3-ubyte', './mnist/train-labels.idx1-ubyte', './mnist/t10k-images.idx3-ubyte', './mnist/t10k-labels.idx1-ubyte')
train_data, train_label = _tr
test_data, test_label = _te
TRAIN_DATA_SIZE = len(train_data)
TEST_DATA_SIZE = len(test_data)
print('data loaded, train-data-size:%d test-data-size:%d' % (TRAIN_DATA_SIZE, TEST_DATA_SIZE))

train_index = 0
test_index = 0

def shuffle(data, label):
    mix = zip(data, label)
    np.random.shuffle(mix)
    tdata = []
    tlabel = []
    for i in range(len(mix)):
        m = mix[i]
        tdata.append(m[0])
        tlabel.append(m[1])
    return np.array(tdata), np.array(tlabel)

train_data, train_label = shuffle(train_data, train_label)
print('shuffle train data over...')

def get_batch_for_train(batch_size=64):
    global train_data, train_label, train_index
    end = train_index + batch_size
    if (end > TRAIN_DATA_SIZE - 1):
        end = TRAIN_DATA_SIZE - 1
    start = train_index
    sub_data = train_data[start:end]
    sub_label = train_label[start:end]
    train_index += batch_size
    if (end == TRAIN_DATA_SIZE - 1):
        train_index = 0
    return sub_data, sub_label

def get_infer_batch_count(batch_size=64):
    if (TEST_DATA_SIZE % batch_size != 0):
        return TEST_DATA_SIZE / batch_size + 1
    else:
        return TEST_DATA_SIZE / batch_size

def get_batch_for_infer(batch_size=64):
    global test_data, test_label, test_index
    end = test_index + batch_size
    if (end > TEST_DATA_SIZE - 1):
        end = TEST_DATA_SIZE - 1
    start = test_index
    print end
    sub_data = test_data[start:end]
    sub_label = test_label[start:end]
    test_index += batch_size
    if (end == TEST_DATA_SIZE - 1):
        test_index = 0
    return sub_data, sub_label