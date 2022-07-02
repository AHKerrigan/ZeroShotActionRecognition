import os, numpy as np
from time import time

import cv2, torch
from torch.utils.data import Dataset
from scipy.spatial.distance import cdist

import pytorchvideo.transforms as trs
from tqdm import tqdm
from textutils import classes2embedding

from einops import rearrange

from torchvision import transforms

import random

def get_ucf101(opt):
    folder = opt.ucffolder
    fnames, labels = [], []
    for label in sorted(os.listdir(str(folder))):
        for fname in os.listdir(os.path.join(str(folder), label)):
            fnames.append(os.path.join(str(folder), label, fname))
            labels.append(label)

    classes = np.unique(labels)
    return fnames, labels, classes


def get_hmdb(opt):
    folder = opt.hmdbfolder
    fnames, labels = [], []
    for label in sorted(os.listdir(str(folder))):
        dir = os.path.join(str(folder), label)
        if not os.path.isdir(dir): continue
        for fname in sorted(os.listdir(dir)):
            if fname[-4:] != '.avi':
                continue
            fnames.append(os.path.join(str(folder), label, fname))
            labels.append(label.replace('_', ' '))

    fnames, labels = np.array(fnames), np.array(labels)
    classes = np.unique(labels)
    return fnames, labels, classes

def get_kinetics(opt):
    sourcepath = opt.kineticssource
    n_classes = '700'# if '700' in dataset else '400'
    with open(sourcepath, 'r') as f:
        data = [r[:-1].split(',') for r in f.readlines()]

    fnames, labels = [], []
    for x in data:
        if len(x) < 2: continue
        fnames.append(x[0])
        labels.append(x[1][1:])
    classes = sorted(np.unique(labels).tolist())

    #print("Nerfing classes so GPU doesn't die")
    #random.shuffle(classes)
    #classes = classes[:450]
    
    new_fnames, new_labels = [], []
    for x in range(len(fnames)):
        if labels[x] in classes:
            new_fnames.append(fnames[x])
            new_labels.append(labels[x])

    fnames = new_fnames
    labels = new_labels 

    return fnames, labels, classes


"""========================================================="""

def get_kinetics_ucf_hmbd(opt):


    # TESTING ON UCF101
    test_fnames, test_labels, test_classes = get_ucf101(opt)
    test_class_embedding = classes2embedding('ucf101', test_classes, opt)
    print('UCF101: total number of videos {}, classes {}'.format(len(test_fnames), len(test_classes)))

    # TESTING ON HMDB51
    test_fnames2, test_labels2, test_classes2 = get_hmdb(opt)
    test_class_embedding2 = classes2embedding('hmdb51', test_classes2, opt)
    print('HMDB51: total number of videos {}, classes {}'.format(len(test_fnames2), len(test_classes2)))



    if not opt.evaluate:

        # TRAINING ON KINETICS
        train_fnames, train_labels, train_classes = get_kinetics(opt)
        train_fnames, train_labels, train_classes = filter_samples(opt, train_fnames, train_labels, train_classes)
        train_class_embedding = classes2embedding('kinetics', train_classes, opt)
        print('KINETICS: total number of videos {}, classes {}'.format(len(train_fnames), len(train_classes)))

        # Filter overlapping classes
        test_class_embed_util = classes2embedding('ucf101', test_classes, opt, w2v=True)
        test_class2_embed_util = classes2embedding('hmdb51', test_classes2, opt, w2v=True)
        train_class_embed_util = classes2embedding('kinetics', train_classes, opt, w2v=True)

        train_fnames, train_labels, train_classes, train_class_embedding = filter_overlapping_classes(
            train_fnames, train_labels, train_classes, train_class_embed_util,
            np.concatenate([test_class_embed_util, test_class2_embed_util]),
            opt.class_overlap, train_class_embedding)
        print('After filtering) KINETICS: total number of videos {}, classes {}'.format(
            len(train_fnames), len(train_classes)))

        #train_fnames, train_labels, train_classes, train_class_embedding = filter_classes(opt,
        #                            train_fnames, train_labels, train_classes, train_class_embedding)

        # Initialize datasets
        train_dataset = VideoDataset(train_fnames, train_labels, train_class_embedding, train_classes,
                                     'kinetics%d' % len(train_classes), clip_len=opt.clip_len, n_clips=opt.n_clips,
                                     crop_size=opt.size, is_validation=False)

    n_clips = opt.n_clips if not opt.evaluate else max(5*5, opt.n_clips)
    val_dataset   = VideoDataset(test_fnames, test_labels, test_class_embedding, test_classes, 'ucf101',
                                 clip_len=opt.clip_len, n_clips=n_clips, crop_size=opt.size, is_validation=True,
                                 evaluation_only=opt.evaluate)
    val_dataset2  = VideoDataset(test_fnames2, test_labels2, test_class_embedding2, test_classes2, 'hmdb51',
                                 clip_len=opt.clip_len, n_clips=n_clips, crop_size=opt.size, is_validation=True,
                                 evaluation_only=opt.evaluate)
    if opt.evaluate:
        return {'training': [], 'testing': [val_dataset, val_dataset2]}
    else:
        return {'training': [train_dataset], 'testing': [val_dataset, val_dataset2]}

"""========================================================="""


def filter_samples(opt, fnames, labels, classes):
    """
    Select a subset of classes. Mostly for faster debugging.
    """
    fnames, labels = np.array(fnames), np.array(labels)
    if opt.train_samples != -1:
        sel = np.linspace(0, len(fnames)-1, min(opt.train_samples, len(fnames))).astype(int)
        fnames, labels = fnames[sel], labels[sel]
    return np.array(fnames), np.array(labels), np.array(classes)


def filter_classes(opt, fnames, labels, classes, class_embedding):
    """
    Select a subset of classes. Mostly for faster debugging.
    """
    sel = np.ones(len(classes)) == 1
    if opt.class_total > 0:
        sel = np.linspace(0, len(classes)-1, opt.class_total).astype(int)

    classes = np.array(classes)[sel].tolist()
    class_embedding = class_embedding[sel]
    fnames = [f for i, f in enumerate(fnames) if labels[i] in classes]
    labels = [l for l in labels if l in classes]
    return np.array(fnames), np.array(labels), np.array(classes), class_embedding


def filter_overlapping_classes(fnames, labels, classes, class_embedding_util, test_class_embedding, class_overlap, class_embedding_true, class_mask = None, ):


    # Following Brattoli protocol
    #class_embedding = np.mean(class_embedding, axis=1)
    #test_class_embedding = np.mean(test_class_embedding, axis=1)
    #print(class_embedding.shape, test_class_embedding.shape)

    class_embedding_util = class_embedding_util.mean(1)
    test_class_embedding = test_class_embedding.mean(1)

    class_distances = cdist(class_embedding_util, test_class_embedding, 'cosine').min(1)
    #print(class_distances.shape)
    sel = class_distances >= class_overlap

    #print("Class shpae and sel shape is", class_embedding.shape, sel.shape)
    classes = np.array(classes)[sel].tolist()
    class_embedding_true = class_embedding_true[sel]

    #print("sel is ", sel)
    if class_mask != None:
        class_mask = class_mask[sel]

    fnames = [f for i, f in enumerate(fnames) if labels[i] in classes]
    labels = [l for l in labels if l in classes]

    return fnames, labels, classes, class_embedding_true


"""========================================================="""

def load_clips_tsn(fname, clip_len=16, n_clips=1, is_validation=False):
    
    if not os.path.exists(fname):
        print('Missing: '+fname)
        return []
    # initialize a VideoCapture object to read video data into a numpy array
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count == 0 or frame_width == 0 or frame_height == 0:
        print('loading error, switching video ...')
        print(fname)
        return []

    total_frames = frame_count #min(frame_count, 300)
    sampling_period = max(total_frames // n_clips, 1)
    n_snipets = min(n_clips, total_frames // sampling_period)
    if not is_validation:
        starts = np.random.randint(0, max(1, sampling_period - clip_len), n_snipets)
    else:
        starts = np.zeros(n_snipets)
    #starts = np.zeros(n_snipets)
    offsets = np.arange(0, total_frames, sampling_period)
    selection = np.concatenate([np.arange(of+s, of+s+clip_len) for of, s in zip(offsets, starts)])

    frames = []
    count = ret_count = 0

    while count < selection[-1]+clip_len:
        retained, frame = capture.read()
        if count not in selection:
            count += 1
            continue
        if not retained:
            if len(frames) > 0:
                frame = np.copy(frames[-1])
            else:
                frame = (255*np.random.rand(frame_height, frame_width, 3)).astype('uint8')
            frames.append(frame)
            ret_count += 1
            count += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    capture.release()
    frames = np.stack(frames)
    total = n_clips * clip_len
    while frames.shape[0] < total:
        frames = np.concatenate([frames, frames[:(total - frames.shape[0])]])
    frames = frames.reshape([n_clips, clip_len, frame_height, frame_width, 3])
    return frames

'''
class VideoDataset(Dataset):

    def __init__(self, fnames, labels, class_embed, classes, name, load_clips=load_clips_tsn,
                 clip_len=8, n_clips=1, crop_size=112, is_validation=False, evaluation_only=False, input_type='rgb', class_mask=None):
        if 'kinetics' in name:
            fnames, labels = self.clean_data(fnames, labels)
        self.data = fnames
        self.labels = labels
        self.class_embed = class_embed
        self.class_name = classes
        self.name = name
        self.input_type = input_type
        self.class_mask = class_mask

        self.clip_len = clip_len
        self.n_clips = n_clips

        self.crop_size = crop_size  # 112
        self.is_validation = is_validation

        # prepare a mapping between the label names (strings) and indices (ints)
        if name in ['ucf_train', 'ucf_test']:
            assert len(set(classes)) == len(classes)
            self.label2index = {label: index for index, label in enumerate(classes)}
            # convert the list of label names into an array of label indices
            self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
        else:
            self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
            # convert the list of label names into an array of label indices
            self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
        
        print("label array is ",self.label_array)

        self.transform = get_transform(self.is_validation, crop_size, input_type=self.input_type)
        self.loadvideo = load_clips

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label_array[idx]
        
        if 'rgb' in self.input_type:
            buffer = self.loadvideo(sample, self.clip_len, self.n_clips, self.is_validation)
        elif 'flow' in self.input_type:
            buffer = self.loadvideo(sample, self.clip_len+1, self.n_clips, self.is_validation)
        else:
            assert NotImplementedError
            
        if len(buffer) == 0:
            print('Error found in video loading!')
            buffer = np.random.rand(self.n_clips, 3, self.clip_len, self.crop_size, self.crop_size).astype('float32')
            buffer = torch.from_numpy(buffer)
            return buffer, -1, self.class_embed[0], idx
            
        s = buffer.shape
        
        if 'flow' in self.input_type:
            buffer_gray = np.zeros((s[0], self.clip_len+1, s[2], s[3]), dtype=np.uint8)
            buffer_new = np.zeros((s[0], self.clip_len, s[2], s[3], s[4]))
            for i in range(self.n_clips):
                for j in range(self.clip_len+1):
                    buffer_gray[i, j] = cv2.cvtColor(buffer[i, j], cv2.COLOR_RGB2GRAY)
            for i in range(self.n_clips):
                for j in range(self.clip_len):
                    buffer_new[i, j, ..., :2] = compute_optical_flow2(buffer_gray[i, j], buffer_gray[i, j+1])
            buffer = buffer_new
            s = buffer.shape
        
        buffer = buffer.reshape(s[0] * s[1], s[2], s[3], s[4])
        buffer = torch.stack([torch.from_numpy(im) for im in buffer], 0)
        buffer = self.transform(buffer)
        buffer = buffer.reshape(3, s[0], s[1], self.crop_size, self.crop_size).transpose(0, 1)

            
        try:
            return buffer, label, self.class_embed[label], idx
        except:
            return buffer, -1000, np.zeros_like(self.class_embed[0]), idx

    def __len__(self):
        return len(self.data)

    @staticmethod
    def clean_data(fnames, labels):
        if not isinstance(fnames[0], str):
            print('Cannot check for broken videos')
            return fnames, labels
        broken_videos_file = 'assets/kinetics_broken_videos.txt'
        if not os.path.exists(broken_videos_file):
            print('Broken video list does not exists')
            return fnames, labels

        t = time()
        with open(broken_videos_file, 'r') as f:
            broken_samples = [r[:-1] for r in f.readlines()]
        data = [x[75:] for x in fnames]
        keep_sample = np.in1d(data, broken_samples) == False
        fnames = np.array(fnames)[keep_sample]
        labels = np.array(labels)[keep_sample]
        print('Broken videos %.2f%% - removing took %.2f' % (100 * (1.0 - keep_sample.mean()), time() - t))
        return fnames, labels
'''

def zsar_transform(split='train'):

    zsar_transforms = trs.create_video_transform(
                            mode=split, 
                            num_samples=None, 
                            crop_size=112, 
                            min_size=128,
                            max_size=128,
                            video_mean = (0.43216, 0.394666, 0.37645),
                            video_std = (0.22803, 0.22145, 0.216989),
                            aug_type='default')
    return zsar_transforms


    
class VideoDataset(Dataset):

    
    def __init__(self, fnames, labels, class_embed, classes, name, load_clips=load_clips_tsn,
                clip_len=8, n_clips=1, crop_size=112, is_validation=False, evaluation_only=False, input_type='rgb', class_mask=None):
    
        self.data = fnames
        self.labels = labels

        self.clip_len = clip_len
        self.n_clips = n_clips
        self.is_validation = is_validation
        self.class_embeds = class_embed

        self.classes = classes
        print("classes are of len", len(classes))

        self.crop_size = crop_size  # 112

        assert len(set(classes)) == len(classes)
        self.label2index = {label: index for index, label in enumerate(classes)}
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if is_validation:
            self.transform = zsar_transform('train')
        else:
            self.transform = zsar_transform('val')
        self.loadvideo = load_clips

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label_array[idx]

        buffer = self.loadvideo(sample, self.clip_len, self.n_clips, is_validation=False)

        if len(buffer) == 0:
            print('Error found in video loading!')
            buffer = np.random.rand(self.n_clips, 3, self.clip_len, self.crop_size, self.crop_size).astype('float32')
            buffer = torch.from_numpy(buffer)
            return buffer, -1, self.class_embed[0], idx

        buffer = rearrange(buffer, 'cl f h w ch -> ch (cl f) h w')

        #print(buffer)
        buffer = self.transform(torch.from_numpy(buffer))
        buffer = rearrange(buffer, 'ch (cl f) h w -> cl ch f h w', f=self.clip_len)

        #print("filename", sample)
        #print("label", label)
        #print("Classes", self.labels[idx])

        return buffer, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    from config import getopt
    from fse import Vectors
    from textutils import classes2embedding

    from nltk.corpus import wordnet as wn
    from nltk.stem.wordnet import WordNetLemmatizer
    def verbs2basicform(words):
        ret = []
        for w in words:
            analysis = wn.synsets(w)
            if any([a.pos() == 'v' for a in analysis]):
                w = WordNetLemmatizer().lemmatize(w, 'v')
            ret.append(w)
        return ret

    opt = getopt()

    dataloaders = get_kinetics_ucf_hmbd(opt)
    train_dataset = dataloaders['training'][0]

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=0, shuffle=True, drop_last=False)

    class_embeds = torch.from_numpy(train_dataloader.dataset.class_embeds).to(opt.device).unsqueeze(0).repeat(opt.gpus, 1, 1, 1)
    for i, (vid, label, sample, classname) in enumerate(train_dataloader):

        print(sample[5])
        print(label[5])
        print(classname[5])
        #print(class_embeds[:,label[5]])

        vecs = Vectors.from_pretrained("word2vec-google-news-300")

        cn = classname[5].split(" ")
        cn = verbs2basicform(cn)

        print(cn)
        cn = vecs[cn]
        print(cn.mean(0) - class_embeds[:,label[5]].cpu().numpy())


        
        

    
        break

        
    #fnames, labels, classes = get_kinetics(opt) 

    #print(fnames[111])
    #print(labels[111])
    #print("worked")