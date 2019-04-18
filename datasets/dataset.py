
import numpy
np = numpy
import random
import time
import os
import glob

__audio_train_mean_std = np.array([-2.7492260671334582e-05,
                                   0.056233098718291352],
                                   dtype='float64')
# TODO:
#__huck_train_mean_std = ...

__train = lambda s: s.format('train')
__valid = lambda s: s.format('valid')
__test = lambda s: s.format('test')

def find_dataset(filepath):
    if os.path.exists(filepath):
        return filepath
    raise Exception('{} NOT FOUND!'.format(filepath))

### Basic utils ###
def __round_to(x, y):
    """round x up to the nearest y"""
    return int(numpy.ceil(x / float(y))) * y

def __normalize(data):
    """To range [0., 1.]"""
    data -= data.min(axis=1)[:, None]
    data /= data.max(axis=1)[:, None]
    return data

def __linear_quantize(data, q_levels):
    """
    floats in (0, 1) to ints in [0, q_levels-1]
    scales normalized across axis 1
    """
    # Normalization is on mini-batch not whole file
    #eps = numpy.float64(1e-5)
    #data -= data.min(axis=1)[:, None]
    #data *= ((q_levels - eps) / data.max(axis=1)[:, None])
    #data += eps/2
    #data = data.astype('int32')

    eps = numpy.float64(1e-5)
    data *= (q_levels - eps)
    data += eps/2
    data = data.astype('int32')
    return data

def __a_law_quantize(data):
    """
    :todo:
    """
    raise NotImplementedError

def linear2mu(x, mu=255):
    """
    From Joao
    x should be normalized between -1 and 1
    Converts an array according to mu-law and discretizes it

    Note:
        mu2linear(linear2mu(x)) != x
        Because we are compressing to 8 bits here.
        They will sound pretty much the same, though.

    :usage:
        >>> bitrate, samples = scipy.io.wavfile.read('orig.wav')
        >>> norm = __normalize(samples)[None, :]  # It takes 2D as inp
        >>> mu_encoded = linear2mu(2.*norm-1.)  # From [0, 1] to [-1, 1]
        >>> print mu_encoded.min(), mu_encoded.max(), mu_encoded.dtype
        0, 255, dtype('int16')
        >>> mu_decoded = mu2linear(mu_encoded)  # Back to linear
        >>> print mu_decoded.min(), mu_decoded.max(), mu_decoded.dtype
        -1, 0.9574371, dtype('float32')
    """
    x_mu = np.sign(x) * np.log(1 + mu*np.abs(x))/np.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16')

def mu2linear(x, mu=255):
    """
    From Joao with modifications
    Converts an integer array from mu to linear

    For important notes and usage see: linear2mu
    """
    mu = float(mu)
    x = x.astype('float32')
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    return np.sign(y) * (1./mu) * ((1. + mu)**np.abs(y) - 1.)

def __mu_law_quantize(data):
    return linear2mu(data)

def __batch_quantize(data, q_levels, q_type):
    """
    One of 'linear', 'a-law', 'mu-law' for q_type.
    """
    data = data.astype('float64')
    data = __normalize(data)
    if q_type == 'linear':
        return __linear_quantize(data, q_levels)
    if q_type == 'a-law':
        return __a_law_quantize(data)
    if q_type == 'mu-law':
        # from [0, 1] to [-1, 1]
        data = 2.*data-1.
        # Automatically quantized to 256 bins.
        return __mu_law_quantize(data)
    raise NotImplementedError

__RAND_SEED = 123
def __fixed_shuffle(inp_list):
    if isinstance(inp_list, list):
        random.seed(__RAND_SEED)
        random.shuffle(inp_list)
        return ### it does not return anything, means that it return itself, change it's self
    #import collections
    #if isinstance(inp_list, (collections.Sequence)):
    if isinstance(inp_list, numpy.ndarray):
        numpy.random.seed(__RAND_SEED)
        numpy.random.shuffle(inp_list)
        return
    # destructive operations; in place; no need to return
    raise ValueError("inp_list is neither a list nor a numpy.ndarray but a "+type(inp_list))

#def __make_random_batches(inp_list, batch_size):
def __make_random_batches(inp_list, sids, batch_size):
    '''
    inp_list: files: all 8s wavs
    sids: all 8s wavs' speaker IDs
    batch_size: default is 128
    
    return:
    batches: should be 
    '''
    batches = []
    for i in xrange(len(inp_list) / batch_size):
        batches.append(inp_list[i*batch_size:(i+1)*batch_size])
    
    #print numpy.array(batches).shape # shape=(44, 128, 128000)
    #sys.exit()
    
    ### batches for sids#######################################
    batches_sid = []
    for i in xrange(len(inp_list) / batch_size):
        batches_sid.append(sids[i*batch_size:(i+1)*batch_size])
    #print numpy.array(batches_sid).shape # shape=(44, 128)
    #sys.exit()   
    ###########################################################
    
    __fixed_shuffle(batches)
    __fixed_shuffle(batches_sid) # because the rand_seed is the same, their shuffle should be the same!!! already test ---test_rand_shuffle_same_seed.py
    return batches, batches_sid

### audio DATASET LOADER ###
def __audio_feed_epoch(files,
                       sids, ######################################## add by yong xu for speaker IDs
                       batch_size,
                       seq_len,
                       overlap,
                       q_levels,
                       q_zero,
                       q_type,
                       real_valued=False):
    """
    Helper function to load audio dataset.
    Generator that yields training inputs (subbatch, reset). `subbatch` contains
    quantized audio data; `reset` is a boolean indicating the start of a new
    sequence (i.e. you should reset h0 whenever `reset` is True).

    Feeds subsequences which overlap by a specified amount, so that the model
    can always have target for every input in a given subsequence.

    Assumes all flac files have the same length.

    returns: (subbatch, reset)
    subbatch.shape: (BATCH_SIZE, SEQ_LEN + OVERLAP)
    reset: True or False
    """
    #batches = __make_random_batches(files, batch_size)
    batches, batches_sid = __make_random_batches(files, sids, batch_size) # well test
    ''' well test:
    batches.shape=[44,128,128000]
    batches_sid.shape=[44,128], in the same shuffle order with batches 
    '''
    
#    for bch in batches:
    for j, bch in enumerate(batches): #j is the index of batches (one of 44)
        # batch_seq_len = length of longest sequence in the batch, rounded up to
        # the nearest SEQ_LEN.
        batch_seq_len = len(bch[0])  # 8*16000=128000, namely 8 seconds, 0-th batch in all 128 batches, so its length is equal to 128000
        #print batch_seq_len #128000
        batch_seq_len = __round_to(batch_seq_len, seq_len) #seq_len=512
        #print batch_seq_len #128000
        #print numpy.array(bch).shape #(128, 128000)
        #sys.exit()
         
        batch = numpy.zeros(
            (batch_size, batch_seq_len), # 128*128000, same size with each bch
            dtype='float64'
        )
        
        batch_sid = numpy.zeros(
            (batch_size), # 128
            dtype='int16'
        )

        mask = numpy.ones(batch.shape, dtype='float32') # 128*128000, same size with each bch

        for i, data in enumerate(bch): #i is the index of the all 128, data is one [128000]
            #data, fs, enc = scikits.audiolab.flacread(path)
            # data is float16 from reading the npy file
            batch[i, :len(data)] = data ################## this one is for completing the all data into 8s (e.g., 6s, 7s, 9s, 10s), cut or completing with zeros
            # This shouldn't change anything. All the flac files for audio
            # are the same length and the mask should be 1 every where.
            # mask[i, len(data):] = numpy.float32(0) ################## did not use orignially, starting from "len(data):", as zeros, means use the front, while mask the latter part
            
            batch_sid[i]=batches_sid[j][i] #example:[j(i=1,2),j(i=3,4),j(i=5,6)] ##list indices must be integers, not tuple (j, i) did not work
        
        #print batch_sid #[ 3  1  4  8  6  4 14 13  2...
        #print numpy.array(batch_sid).shape #128
        #pause
        if not real_valued: # hope the input should be real_valued (True) or not (False), should be here
            batch = __batch_quantize(batch, q_levels, q_type)
            #print batch # values belong to [0, 255]
            #print numpy.array(batch).shape #(128, 128000)
            #pause

            batch = numpy.concatenate([
                numpy.full((batch_size, overlap), q_zero, dtype='int32'),#q_zero=128, why overlap is in the front ? with q_zero, not the real values???
                batch
            ], axis=1)
            #print batch
            #print numpy.array(batch).shape #(128, 128008)
            #pause
        else:
            batch -= __audio_train_mean_std[0]
            batch /= __audio_train_mean_std[1]
            batch = numpy.concatenate([
                numpy.full((batch_size, overlap), 0, dtype='float32'),
                batch
            ], axis=1).astype('float32')

        mask = numpy.concatenate([
            numpy.full((batch_size, overlap), 1, dtype='float32'),
            mask
        ], axis=1)

        for i in xrange(batch_seq_len // seq_len):
            reset = numpy.int32(i==0) # reset is for another 8s sequence, reset is for the update of RNNs ???
            subbatch = batch[:, i*seq_len : (i+1)*seq_len + overlap] # the first 8 samples are zero, and the current 512 samples overlap with the next 512 samples with 8 samples
            submask = mask[:, i*seq_len : (i+1)*seq_len + overlap]
            #print subbatch.shape # all int 0-255  , shape=[128,520]  520=512+8?
            #print submask.shape # all 1.0
            #sys.Exit()
            yield (subbatch, batch_sid, reset, submask) #subbatch means splitted along the axis=1, while the axis=0 is the same, so each time use the same batch_sid
            ### this is the last yield back to the main_train program
            
def audio_train_feed_epoch(audio_file, *args):
    # import crash
    # asdf
    """
    :parameters:
        batch_size: int
        seq_len:
        overlap:
        q_levels:
        q_zero:
        q_type: One the following 'linear', 'a-law', or 'mu-law'

    4,340 (9.65 hours) in total
    With batch_size = 128:
        4,224 (9.39 hours) in total
        3,712 (88%, 8.25 hours)for training set
        256 (6%, .57 hours) for validation set
        256 (6%, .57 hours) for test set

    Note:
        32 of Beethoven's piano sonatas available on archive.org (Public Domain)

    :returns:
        A generator yielding (subbatch, reset, submask)
    """
    # Just check if valid/test sets are also available. If not, raise.
    find_dataset(__valid(audio_file))
    find_dataset(__test(audio_file))
    # Load train set
    data_path = find_dataset(__train(audio_file))
    
    ### yong xu for speaker ID####################################################
    files_data = numpy.load(data_path)
    files=files_data['wav'] # for all 8 seconds wav data, none*(8*16000)
    sids=numpy.uint8(files_data['sid']) # for all 8s speaker IDs data, one-dim int vector;;; unit8 here supposes that there are less than 256 speakers (0 to 255)
    generator = __audio_feed_epoch(files, sids, *args)
    return generator
    ###############################################################################

def audio_valid_feed_epoch(audio_file, *args):
    """
    See:
        audio_train_feed_epoch
    """
    data_path = find_dataset(__valid(audio_file))
    
    ### yong xu for speaker ID####################################################
    files_data = numpy.load(data_path)
    files=files_data['wav'] # for all 8 seconds wav data, none*(8*16000)
    sids=numpy.uint8(files_data['sid']) # for all 8s speaker IDs data, one-dim int vector;;; unit8 here supposes that there are less than 256 speakers (0 to 255)
    generator = __audio_feed_epoch(files, sids, *args)
    return generator
    ###############################################################################

def audio_test_feed_epoch(audio_file, *args):
    """
    See:
        audio_train_feed_epoch
    """
    data_path = find_dataset(__test(audio_file))

    ### yong xu for speaker ID####################################################
    files_data = numpy.load(data_path)
    files=files_data['wav'] # for all 8 seconds wav data, none*(8*16000)
    sids=numpy.uint8(files_data['sid']) # for all 8s speaker IDs data, one-dim int vector;;; unit8 here supposes that there are less than 256 speakers (0 to 255)
    generator = __audio_feed_epoch(files, sids, *args)
    return generator
    ###############################################################################
