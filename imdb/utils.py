import os
import struct
import oneflow.core.record.record_pb2 as ofrecord
import numpy as np
import six


def colored_string(string: str, color: str or int) -> str:
    """在终端中显示一串有颜色的文字 [This code is copied from fitlog]

    :param string: 在终端中显示的文字
    :param color: 文字的颜色
    :return:
    """
    if isinstance(color, str):
        color = {
            "black": 30,
            "red": 31,
            "green": 32,
            "yellow": 33,
            "blue": 34,
            "purple": 35,
            "cyan": 36,
            "white": 37
        }[color]
        return "\033[%dm%s\033[0m" % (color, string)


def load_imdb_data(path):
    train_data = []
    train_labels = []
    
    with open(os.path.join(path, "train-part-0"), "rb") as f:
        for loop in range(0, 25000):
            length = struct.unpack("q", f.read(8))
            serilizedBytes = f.read(length[0])
            ofrecord_features = ofrecord.OFRecord.FromString(serilizedBytes)
            
            data = ofrecord_features.feature["data"].int64_list.value
            label = ofrecord_features.feature["labels"].int64_list.value
            
            train_data.append(data)
            train_labels.append(*label)
    
    test_data = []
    test_labels = []
    
    with open(os.path.join(path, "test-part-0"), "rb") as f:
        for loop in range(0, 25000):
            length = struct.unpack("q", f.read(8))
            serilizedBytes = f.read(length[0])
            ofrecord_features = ofrecord.OFRecord.FromString(serilizedBytes)
            
            data = ofrecord_features.feature["data"].int64_list.value
            label = ofrecord_features.feature["labels"].int64_list.value
            
            test_data.append(data)
            test_labels.append(*label)
    
    return (np.array(train_data), np.array(train_labels)), (np.array(test_data), np.array(test_labels))


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)
    
    lengths = []
    sample_shape = ()
    flag = True
    
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    
    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
    
    if maxlen is None:
        maxlen = np.max(lengths)
    
    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))
    
    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)
        
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
