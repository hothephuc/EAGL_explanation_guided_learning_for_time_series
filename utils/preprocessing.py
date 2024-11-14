import wfdb
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# read data
def read_record(path, channels):
    record = wfdb.rdrecord(path, sampfrom=0, channels=channels)
    return record.p_signal, record.sig_name

def read_annotate(path, extension):
    ann = wfdb.rdann(path, extension=extension, sampfrom=0)
    
    return ann.sample, ann.symbol

def spliting_index(chunk_size,chunk_step, max_index, num_overlap, start_idx ):
    num_overlap= chunk_size-num_overlap
    chunk_size = chunk_size
    max_index = max_index -chunk_size
    index_array = np.arange(0, chunk_size,chunk_step)[None, :]+np.arange(start_idx,max_index, num_overlap)[:,None]
    return index_array


def minmax_scale(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

def discretize(array, bin_num):
    bin_edges = np.linspace(array.min(), array.max(), bin_num + 1)
    return np.digitize(array, bins=bin_edges, right=False)


def differencing_signal(arr):
    return np.array([arr[0] - arr[len(arr) - 1]] + [arr[i] - arr[i - 1] for i in range(1, len(arr))])


def avg_pooling(arr, size):
    # Ensure the input is a NumPy array
    arr = np.array(arr)
    
    # Calculate the number of windows
    num_windows = len(arr) // size
    
    # Initialize the output array
    pooled_arr = np.zeros(num_windows)
    
    # Perform average pooling
    for i in range(num_windows):
        start = i * size
        end = start + size
        pooled_arr[i] = np.mean(arr[start:end])
    
    return pooled_arr


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# encode input into waves

def smoothing(len_signal, pos, symbol, signal_to_num):
    whole = np.zeros(len_signal, dtype=int)
    current = -1
    last_open = -1
    last_closed = -1
    last_t = -1
    first_t = True
    dub_t = []
    for i in range(len(pos)):
        if symbol[i] == '(':
            last_open = i
        elif symbol[i] == ')' and current != -1:
            pos_start = pos[last_closed] + 2 if last_open == -1 else pos[last_open]
            whole[pos_start:(pos[i] + 1)] = current
            last_closed = i
            
            # restart
            last_open = -1
            current = -1
            first_t = True
        elif symbol[i] in signal_to_num:
            current = signal_to_num[symbol[i]]

            # double 't' handling: second t
            if symbol[i] == "t":
                if first_t:
                    # last_t = i
                    first_t = False
                else:
                    # last_open = last_t
                    current = signal_to_num['t']
                    dub_t.append(pos[i])
            

    return whole, dub_t


# for validation

def count_consecutive_segments(array, values):
    array = np.array(array)
    segments_count = {value: 0 for value in values}
    segments_details = {value: [] for value in values}
    
    for value in values:
        is_segment = array == value
        if np.any(is_segment):
            diff = np.diff(np.concatenate(([0], is_segment, [0])))
            segment_starts = np.where(diff == 1)[0]
            segment_ends = np.where(diff == -1)[0]
            segments_count[value] = len(segment_starts)
            segments_details[value] = list(zip(segment_starts, segment_ends))
    
    return segments_count, segments_details


def find_different_segments_details(segments_details1, segments_details2, values):
    
    different_segments = {value: {'only_in_og': [], 'only_in_encoded': []} for value in values}
    
    for value in values:
        segments1 = set(segments_details1[value])
        segments2 = set(segments_details2[value])
        
        only_in_og = segments1 - segments2
        only_in_encoded = segments2 - segments1
        
        different_segments[value]['only_in_og'] = list(only_in_og)
        different_segments[value]['only_in_encoded'] = list(only_in_encoded)
    
    return different_segments

# validate
# count, c_detail = count_consecutive_segments(symbol, NUM_VALUE.keys())
# count_encoded, ce_detail = count_consecutive_segments(encoded_channel, NUM_VALUE.values())
# for key, value in NUM_VALUE.items():
#     count_encoded[key] = count_encoded.pop(value)
#     ce_detail[key] = ce_detail.pop(value)
# assert count == count_encoded, f"Encode error (missing waves):\n\
#     og: {count}, encoded: {count_encoded}"