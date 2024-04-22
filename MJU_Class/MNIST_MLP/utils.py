### MNIST 데이터셋을 불러오는 데에 사용하는 라이브러리
# os : 파일 경로를 확인하는데 사용하는 라이브러리.
# gzip : 압축된 파일을 읽는데 사용하는 라이브러리.
# urlretrieve : url로부터 데이터를 다운로드하는데 사용하는 라이브러리.
# numpy : 행렬 연산에 필요한 라이브러리. MNIST 데이터를 불러와서 행렬로 변환하는데 사용.

import os
import gzip
from urllib.request import urlretrieve
import numpy as np

# MNIST를 다운받을 경로
url = 'http://yann.lecun.com/exdb/mnist/'

# MNIST 데이터셋의 파일명 (딕셔너리)
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}


def _download_mnist(dataset_dir):
    # 해당 경로가 없을 시 디렉토리 새로 생성
    os.makedirs(dataset_dir, exist_ok=True)

    # 해당 경로에 존재하지 않는 파일을 모두 다운로드
    for filename in key_file.values():
        if filename not in os.listdir(dataset_dir):
            urlretrieve(url + filename, os.path.join(dataset_dir, filename))
            print("Downloaded %s to %s" % (filename, dataset_dir))


def _images(path):
    '''
    MNIST 데이터셋 이미지을 NumPy Array로 변환하여 불러오기
    '''
    # gzip 파일을 열고, 이미지를 읽어서 1차원 배열로 변환
    with gzip.open(path) as f:
        # 첫 16 바이트는 magic_number, n_imgs, n_rows, n_cols 의 정보이므로 무시
        pixels = np.frombuffer(f.read(), 'B', offset=16)

    # 28*28=784 이므로 784차원으로 reshape해준 뒤, 0~255의 값을 0~1로 정규화
    return pixels.reshape(-1, 28*28).astype('float32') / 255


def _onehot(integer_labels):
    '''
    라벨 데이터를 one-hot encoding 하기
    
    예시)
    [2, 7] -> [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]  # 7
    '''
    # 라벨 데이터의 길이와 최댓값을 구함
    n_rows = len(integer_labels)        # 라벨 데이터의 길이
    n_cols = integer_labels.max() + 1   # 라벨 데이터의 최댓값 + 1

    # 0으로 채워진 (n_rows, n_cols) 크기의 행렬 생성
    onehot = np.zeros((n_rows, n_cols), dtype='uint8')
    # one-hot 행렬의 각 행에 해당하는 라벨을 1로 변경
    onehot[np.arange(n_rows), integer_labels] = 1

    return onehot


def _labels(path):
    '''
    MNIST 데이터셋 라벨을 NumPy Array로 변환하여 불러오기
    '''
    # gzip 파일을 열고, 라벨 데이터를 불러온 뒤, integer로 변환
    with gzip.open(path) as f:
        # 첫 8 바이트는 magic_number, n_labels 의 정보이므로 무시
        integer_labels = np.frombuffer(f.read(), 'B', offset=8)

    # one-hot 인코딩한 결과를 반환
    return _onehot(integer_labels)


def load_mnist(path, kind='train'):
    '''
    MNIST 데이터셋을 불러오기
    '''
    # MNIST 데이터셋을 다운로드
    _download_mnist(path)

    # 입력 인수에 따라 train 또는 test 데이터셋을 불러옴
    if kind == 'train':
        # 이미지와 라벨을 불러옴
        images = _images(os.path.join(path, key_file['train_img']))
        labels = _labels(os.path.join(path, key_file['train_label']))
    elif kind == 'test':
        # 이미지와 라벨을 불러옴
        images = _images(os.path.join(path, key_file['test_img']))
        labels = _labels(os.path.join(path, key_file['test_label']))
    else:
        raise ValueError("dataset argument should be 'train' or 'test'")

    return images, labels
