#|**********************************************************************;
# Project           : Explainable Deep Driving
#
# File name         : utils_nlp.py
#
# Author            : Jinkyu Kim
#
# Date created      : 20190108
#
# Purpose           : Helper functions for nlp part
#
# Revision History  :
#
# Date        Author      Ref    Revision
# 20190108    jinkyu      1      initiated
#
# Remark
# Some functions are implemented by https://github.com/yunjey/show-attend-and-tell
#|**********************************************************************;

import  json
import  pandas      as      pd
from    src.utils   import  *
import  os
import  collections
from    sklearn.cluster import KMeans
from    sklearn.feature_extraction.text import TfidfVectorizer
from    collections     import Counter
import  numpy           as np
import  h5py
import  cPickle         as     pickle
# import  pickle         as     cPickle

def process_caption_data(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    data = []
    for annotation in caption_data['annotations']:
        annotation['caption'] = annotation['action'] + ' <SEP> ' + annotation['justification'] # ADD separator
        data += [annotation]

    caption_data = pd.DataFrame.from_dict(data)

    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace("'s"," 's").replace("'ve", " 've").replace("n't", " n't").replace("'re", " 're").replace("'d", " 'd").replace("'ll", " 'll")
        caption = caption.replace('.','').replace(',','').replace('"','').replace("'","").replace("`","")
        caption = caption.replace('&','and').replace('(',' ').replace(')',' ').replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces
        
        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)
    
    # delete captions if size is larger than max_length
    print( bcolors.BLUE + "[_process_caption_data] The number of captions before deletion: %d" %len(caption_data) + bcolors.ENDC ) 
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print( bcolors.BLUE + "[_process_caption_data] The number of captions after deletion: %d" %len(caption_data) + bcolors.ENDC ) 

    return caption_data


def build_vocab(annotations, size_of_dict=10000):
    print(bcolors.GREEN + '[_build_vocab] Build a vocabulary' + bcolors.ENDC)

    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1
        
        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    # limit the size of dictionary
    counter = counter.most_common(size_of_dict)

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2, u'<UNK>':3}
    idx_to_word = {0: u'<NULL>', 1: u'<START>', 2: u'<END>', 3:u'<UNK>'}
    idx = 3+1

    for word in counter:
        word_to_idx[word[0]] = idx
        idx_to_word[idx]     = word[0]
        idx += 1

    print(bcolors.BLUE + '[_build_vocab] Max length of caption: {}'.format(max_len)   + bcolors.ENDC)
    print(bcolors.BLUE + '[_build_vocab] Size of dictionary: {}'.format(size_of_dict) + bcolors.ENDC)

    return word_to_idx, idx_to_word


def build_caption_vector(annotations, word_to_idx, max_length=15):
    print(bcolors.GREEN + '[_build_caption_vector] String caption -> Indexed caption' + bcolors.ENDC)

    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)   

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ") # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
            else:
                cap_vec.append(word_to_idx['<UNK>'])
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        
        captions[i, :] = np.asarray(cap_vec)

    print(bcolors.BLUE + '[_build_caption_vector] Building caption vectors' + bcolors.ENDC)

    return captions

def build_file_names(annotations):
    image_file_names    = []
    id_to_idx           = {}
    idx                 = 0
    image_ids           = annotations['video_id']
    file_names          = annotations['vidName']

    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx

def build_image_idxs(annotations, id_to_idx):
    image_idxs  = np.ndarray(len(annotations), dtype=np.int32)
    image_ids   = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs


#------------------------------------------------------|| For Video preprocessing
def pad_video(video_feature, dimension):
    '''
    Fill pad to video to have same length.
    Pad in Left.
    video = [pad,..., pad, frm1, frm2, ..., frmN]
    '''
    padded_feature  = np.zeros(dimension)
    max_length      = dimension[0]
    current_length  = video_feature.shape[0]
    num_padding     = max_length - current_length

    if num_padding == 0:
        padded_feature  = video_feature
    elif num_padding < 0:
        steps           = np.linspace(0, current_length, num=max_length, endpoint=False, dtype=np.int32)
        padded_feature  = video_feature[steps]
    else:
        padded_feature[num_padding:] = video_feature

    return padded_feature


def fill_mask(max_length, current_length, zero_location='LEFT'):
    num_padding = max_length - current_length
    if num_padding <= 0:
        mask = np.ones(max_length)
    elif zero_location == 'LEFT':
        mask = np.ones(max_length)
        for i in range(num_padding):
            mask[i] = 0
    elif zero_location == 'RIGHT':
        mask = np.zeros(max_length)
        for i in range(current_length):
            mask[i] = 1

    return mask



def build_feat_matrix(annotations, max_length, fpath, hz=10, sampleInterval=5, FINETUNING=False):
    print(bcolors.GREEN + '[_build_feat_matrix] Collect feats and masks' + bcolors.ENDC)

    n_examples    = len(annotations)
    max_length_vid= max_length*sampleInterval

    all_logs    = {}
    all_logs['speed']       = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['course']      = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['accelerator'] = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['curvature']   = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['goaldir']     = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['timestamp']   = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['pred_accel']  = np.ndarray([n_examples, max_length], dtype=np.float32)
    all_logs['pred_courses']= np.ndarray([n_examples, max_length], dtype=np.float32)


    all_masks4Cap = np.ndarray([n_examples, max_length], dtype=np.float32)  
    #all_feats4Cap = np.memmap('/data2/tmp.dat', dtype='float32', mode='w+', shape=(n_examples, max_length, 64, 12, 20))
    all_feats4Cap = np.ndarray([n_examples, max_length, 64, 12, 20], dtype=np.float32) 
    all_attns4Cap = np.ndarray([n_examples, max_length, 12*20], dtype=np.float32) 

    sTimes       = annotations['sTime']      # staring timestamp
    eTimes       = annotations['eTime']      # ending timestamp
    vidNames     = annotations['vidName']    # video clip name
    video_ids    = annotations['video_id']   # index of video

    idx          = 0
    for sTime, eTime, vidName, video_id in zip(sTimes, eTimes, vidNames, video_ids):

        print(bcolors.BLUE + '[_build_feat_matrix] vidName: {}'.format(vidName) + bcolors.ENDC)

        # load feats
        #if (os.path.isfile(fpath+"feats_%s/"%(dataset)+'{}_{}'.format(video_id, vidName)+".h5")) == False: continue 
        if (os.path.isfile(fpath+"log/"+'{}_{}'.format(video_id, vidName) +".h5")) == False: continue 

        feats = h5py.File(fpath+"feat/"+'{}_{}'.format(video_id, vidName) +".h5", "r")
        logs  = h5py.File(fpath+"log/" +'{}_{}'.format(video_id, vidName) +".h5", "r")
        cams  = h5py.File(fpath+"cam/" +'{}_{}'.format(video_id, vidName) +".h5", "r")
        attns = h5py.File(fpath+"attn/"+'{}_{}'.format(video_id, vidName) +".h5", "r")

        # (synced) control commands
        timestamp           = np.squeeze(attns["timestamp"][:])
        curvature_value     = np.squeeze(attns["curvature"][:])
        accelerator_value   = np.squeeze(attns["accel"][:])
        speed_value         = np.squeeze(attns["speed"][:])
        course_value        = np.squeeze(attns["course"][:])
        goaldir_value       = np.squeeze(attns["goaldir"][:])
        acc_pred_value      = np.squeeze(attns["pred_accel"][:])
        course_pred_value   = np.squeeze(attns["pred_courses"][:])

        # Will pad +/- 1 second; extract Frames of Interest
        startStamp = timestamp[0] + float((int(sTime)-1))*1000
        endStamp   = timestamp[0] + float((int(eTime)+1))*1000

        ind2interest = np.where(np.logical_and(np.array(timestamp)>=startStamp, np.array(timestamp)<=endStamp))
        print('sTime: {}, eTime: {}, sStamp: {}, eStamp: {}, index: {}'.format(sTime, eTime, startStamp, endStamp, len(ind2interest[0])))

        feat         = feats['X'][:]
        feat         = feat[ind2interest]
        attn         = attns['attn'][:]
        attn         = attn[ind2interest]

        speed_value         = speed_value[ind2interest]
        course_value        = course_value[ind2interest]
        accelerator_value   = accelerator_value[ind2interest]
        curvature_value     = curvature_value[ind2interest]
        goaldir_value       = goaldir_value[ind2interest]
        acc_pred_value      = acc_pred_value[ind2interest]
        course_pred_value   = course_pred_value[ind2interest]

        ## feat (for captioning)
        feat             = feat[::sampleInterval]
        attn             = attn[::sampleInterval]
        speed_value      = speed_value[::sampleInterval]
        course_value     = course_value[::sampleInterval]
        accelerator_value= accelerator_value[::sampleInterval]
        curvature_value  = curvature_value[::sampleInterval]
        goaldir_value    = goaldir_value[::sampleInterval]
        acc_pred_value   = acc_pred_value[::sampleInterval]
        course_pred_value= course_pred_value[::sampleInterval]

        ## padding
        speed_value       = pad_video(speed_value,       (max_length,))
        course_value      = pad_video(course_value,      (max_length,))
        accelerator_value = pad_video(accelerator_value, (max_length,))
        curvature_value   = pad_video(curvature_value,   (max_length,))
        goaldir_value     = pad_video(goaldir_value,     (max_length,))
        acc_pred_value    = pad_video(acc_pred_value,    (max_length,))
        course_pred_value = pad_video(course_pred_value, (max_length,))
        timestamp         = pad_video(timestamp,         (max_length,))
        mask4Cap          = fill_mask(max_length, feat.shape[0], zero_location='LEFT')
        feat4Cap          = pad_video(feat, (max_length, 64, 12, 20))
        attn4Cap          = pad_video(attn, (max_length, 12*20))

        # accumulate
        all_feats4Cap[idx]          = feat4Cap
        all_masks4Cap[idx]          = mask4Cap
        all_attns4Cap[idx]          = attn4Cap
        all_logs['timestamp'][idx]  = timestamp
        all_logs['speed'][idx]      = speed_value
        all_logs['course'][idx]     = course_value
        all_logs['accelerator'][idx]= accelerator_value
        all_logs['curvature'][idx]  = curvature_value
        all_logs['goaldir'][idx]    = goaldir_value
        all_logs['pred_accel'][idx] = acc_pred_value
        all_logs['pred_courses'][idx]= course_pred_value

        idx += 1

    print(bcolors.BLUE + '[_build_feat_matrix] max_video_length: {} (caption), {} (control)'.format(max_length, max_length) + bcolors.ENDC)
    print(bcolors.BLUE + '[_build_feat_matrix] Sample freq: {} Hz'.format(hz/sampleInterval) + bcolors.ENDC)
    print(bcolors.BLUE + '[_build_feat_matrix] max_log_length: {}'.format(max_length) + bcolors.ENDC)

    return all_feats4Cap, all_masks4Cap, all_logs, all_attns4Cap

def cluster_texts(texts, clusters=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)
 
    clustering = collections.defaultdict(list)

    print("Top terms per cluster:")
    order_centroids = km_model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(clusters):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
 
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
 
    return clustering

def cluster_annotations(annotations, k=2):
    print(annotations['caption'])

    clusters = cluster_texts(annotations['justification'], k)

    ind_cluster = np.ndarray([len(annotations)], dtype=np.float32)
    for key, index in clusters.iteritems():
        ind_cluster[index] = key
        #print(key, value)

    return clusters, ind_cluster

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file  

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)









