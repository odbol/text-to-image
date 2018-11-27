
import os
import re
import time
import nltk
import re
import string
import tensorlayer as tl
from utils import *
import json
import subprocess

dataset = 'product-logos'#'material-icons' # or '102flowers' or 'instagram'
need_256 = True # set to True for stackGAN

nltk.download('punkt')

cwd = os.getcwd()
VOC_FIR = cwd + '/' + dataset + '_vocab.txt'

if os.path.isfile(VOC_FIR):
    print("WARNING: vocab.txt already exists")

img_dir = os.path.join(cwd, dataset)

maxCaptionsPerImage = 1 # this will change depending on dataset.

def processCaptionsFlowers():
    maxCaptionsPerImage = 10
    caption_dir = os.path.join(cwd, 'text_c10')

    ## load captions
    caption_sub_dir = load_folder_list( caption_dir )
    captions_dict = {}
    processed_capts = []
    for sub_dir in caption_sub_dir: # get caption file list
        with tl.ops.suppress_stdout():
            files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt')
            for i, f in enumerate(files):
                file_dir = os.path.join(sub_dir, f)
                key = int(re.findall('\d+', f)[0])
                t = open(file_dir,'r')
                lines = []
                for line in t:
                    line = preprocess_caption(line)
                    lines.append(line)
                    processed_capts.append(tl.nlp.process_sentence(line, start_word="<S>", end_word="</S>"))
                assert len(lines) == maxCaptionsPerImage, "Every flower image have 10 captions"
                captions_dict[key] = lines
    print(" * %d x %d captions found " % (len(captions_dict), len(lines)))

    ## build vocab
    if not os.path.isfile(VOC_FIR):
        _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)

    ## load images
    with tl.ops.suppress_stdout():  # get image files list
        imgs_title_list = sorted(tl.files.load_file_list(path=img_dir, regx='^image_[0-9]+\.jpg'))

    return captions_dict, imgs_title_list

def processCaptionsInstagram():
    maxCaptionsPerImage = 1
    caption_dir = os.path.join(cwd, dataset)

    ## load captions
    captions_dict = {}
    processed_capts = []
    key = 0 # this is the index of the image files in imgs_title_list, matched with the key of the captions_dict. make sure you sort so they match.
    with tl.ops.suppress_stdout():
        files = sorted(tl.files.load_file_list(path=caption_dir, regx='^.+\.json'))
        for i, f in enumerate(files):
            print f
            file_dir = os.path.join(caption_dir, f)
            t = open(file_dir,'r')
            metadata = json.load(t)

            lines = []
            for edge in metadata["edge_media_to_caption"]["edges"]:
                if len(lines) >= maxCaptionsPerImage:
                    break

                caption = edge["node"]["text"].encode('utf-8', 'xmlcharrefreplace')
                #print caption
                line = preprocess_caption(caption.lower())
                #print line
                lines.append(line)
                processed_capts.append(tl.nlp.process_sentence(line, start_word="<S>", end_word="</S>"))
            # TODO(tyler): does it have to have 10 lines???
            assert len(lines) == maxCaptionsPerImage, "Every image must have " + maxCaptionsPerImage + " captions"
            captions_dict[key] = lines
            key += 1
    print(" * %d x %d captions found " % (len(captions_dict), len(lines)))

    ## build vocab
    if not os.path.isfile(VOC_FIR):
        _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)


    ## load images. note that these indexes must match up with the keys of captions_dict: i.e. they should be sorted in the same order.
    with tl.ops.suppress_stdout():  # get image files list
        imgs_title_list = sorted(tl.files.load_file_list(path=img_dir, regx='^.+\.jpg'))

    for i in 0, 1, 7, 34, 60:
        print "Spot check: %s should match with %s" % (captions_dict[i], imgs_title_list[i])

    return captions_dict, imgs_title_list


def processCaptionsMaterialIcons():
    maxCaptionsPerImage = 2
    caption_dir = os.path.join(cwd, dataset)

    ## load captions
    catagories_sub_dirs = load_folder_list( caption_dir )
    captions_dict = {}
    processed_capts = []
    imgs_title_list = []
    key = 0 # this is the index of the image files in imgs_title_list, matched with the key of the captions_dict. make sure you sort so they match.
    for category_dir in catagories_sub_dirs: 
        # just get the largest density of the largest resolution renders
        sub_dir = os.path.join(category_dir, "drawable-xxxhdpi")
        if not os.path.exists(sub_dir):
            continue
        with tl.ops.suppress_stdout():
            files = sorted(tl.files.load_file_list(path=sub_dir, regx='^.+black_48dp\.png'))
            for i, f in enumerate(files):
                print f

                caption = f.replace("ic_", "").replace("_", " ")[:-15] # strip off extension, dp, and color
                #print caption
                caption_processed = preprocess_caption(caption.lower())
                lines = [caption_processed, category_dir]
                for line in lines:
                    processed_capts.append(tl.nlp.process_sentence(line, start_word="<S>", end_word="</S>"))
                    
                # TODO(tyler): does it have to have 10 lines???
                assert len(lines) == maxCaptionsPerImage, "Every image must have " + maxCaptionsPerImage + " captions"
                captions_dict[key] = lines
                imgs_title_list.append(os.path.join(sub_dir, f))
                
                key += 1
    print(" * %d x %d captions found " % (len(captions_dict), len(lines)))

    ## build vocab
    if not os.path.isfile(VOC_FIR):
        _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)


    for i in 0, 1, 7, 34, 60:
        print "Spot check: %s should match with %s" % (captions_dict[i], imgs_title_list[i])

    return captions_dict, imgs_title_list



def processCaptionsProductLogos():
    maxCaptionsPerImage = 1
    caption_dir = os.path.join(cwd, dataset)

    extRegExToStrip = r'_\d+px\.svg'

    ## load captions
    catagories_sub_dirs = load_folder_list( caption_dir )
    captions_dict = {}
    processed_capts = []
    imgs_title_list = []
    key = 0 # this is the index of the image files in imgs_title_list, matched with the key of the captions_dict. make sure you sort so they match.
    for sub_dir in catagories_sub_dirs:
        with tl.ops.suppress_stdout():
            files = sorted(tl.files.load_file_list(path=sub_dir, regx='^.+' + extRegExToStrip))
            for i, svg_file in enumerate(files):
                print svg_file

                f = svg_file.replace(".svg", ".png")
                svgPath = os.path.join(sub_dir, svg_file)
                filePath = os.path.join(sub_dir, f)

                # convert svg to png using imagemagick
                if not os.path.exists(filePath): 
                    subprocess.check_call(["convert", "-background", "none", svgPath, filePath])

                caption = svg_file.replace("_", " ")
                caption = re.sub(extRegExToStrip, '', caption)
                #print caption
                caption_processed = preprocess_caption(caption.lower())
                lines = [caption_processed]
                for line in lines:
                    processed_capts.append(tl.nlp.process_sentence(line, start_word="<S>", end_word="</S>"))
                    
                # TODO(tyler): does it have to have 10 lines???
                assert len(lines) == maxCaptionsPerImage, "Every image must have " + maxCaptionsPerImage + " captions"
                captions_dict[key] = lines
                imgs_title_list.append(filePath)

                key += 1
    print(" * %d x %d captions found " % (len(captions_dict), len(lines)))

    ## build vocab
    if not os.path.isfile(VOC_FIR):
        _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)


    for i in 0, 1, 7, 34, 60:
        print "Spot check: %s should match with %s" % (captions_dict[i], imgs_title_list[i])

    return captions_dict, imgs_title_list




imgs_title_list = False
captions_dict = False
if dataset == '102flowers':
    """
    images.shape = [8000, 64, 64, 3]
    captions_ids = [80000, any]
    """
    captions_dict, imgs_title_list = processCaptionsFlowers()
elif dataset == 'instagram':
    captions_dict, imgs_title_list = processCaptionsInstagram()
elif dataset == 'material-icons':
    captions_dict, imgs_title_list = processCaptionsMaterialIcons()
elif dataset == 'product-logos':
    captions_dict, imgs_title_list = processCaptionsProductLogos()



if not os.path.isfile(VOC_FIR) or not captions_dict: 
    print("ERROR: vocab not generated.")
    exit(1)
else:
    vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")

    ## store all captions ids in list
    captions_ids = []
    try: # python3
        tmp = captions_dict.items()
    except: # python3
        tmp = captions_dict.iteritems()
    for key, value in tmp:
        for v in value:
            captions_ids.append( [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] + [vocab.end_id])  # add END_ID
            # print(v)              # prominent purple stigma,petals are white inc olor
            # print(captions_ids)   # [[152, 19, 33, 15, 3, 8, 14, 719, 723]]
            # exit()
    captions_ids = np.asarray(captions_ids)
    print(" * tokenized %d captions" % len(captions_ids))

    ## check
    #print captions_ids
    img_capt = (captions_dict.items()[1])[1][0]
    print("img_capt: %s" % img_capt)
    print("nltk.tokenize.word_tokenize(img_capt): %s" % nltk.tokenize.word_tokenize(img_capt))
    img_capt_ids = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(img_capt)]#img_capt.split(' ')]
    print("img_capt_ids: %s" % img_capt_ids)
    print("id_to_word: %s" % [vocab.id_to_word(id) for id in img_capt_ids])

    print(" * %d images found, start loading and resizing ..." % len(imgs_title_list))
    s = time.time()

    # time.sleep(10)
    # def get_resize_image(name):   # fail
    #         img = scipy.misc.imread( os.path.join(img_dir, name) )
    #         img = tl.prepro.imresize(img, size=[64, 64])    # (64, 64, 3)
    #         img = img.astype(np.float32)
    #         return img
    # images = tl.prepro.threading_data(imgs_title_list, fn=get_resize_image)
    images = []
    images_256 = []
    for name in imgs_title_list:
        # print(name)
        img_raw = scipy.misc.imread( os.path.join(img_dir, name) )
        img = tl.prepro.imresize(img_raw, size=[64, 64])    # (64, 64, 3)
        img = img.astype(np.float32)
        images.append(img)
        if need_256:
            img = tl.prepro.imresize(img_raw, size=[256, 256]) # (256, 256, 3)
            img = img.astype(np.float32)

            images_256.append(img)
    # images = np.array(images)
    # images_256 = np.array(images_256)
    print(" * loading and resizing took %ss" % (time.time()-s))

    n_images = len(captions_dict)
    n_captions = len(captions_ids)
    n_captions_per_image = maxCaptionsPerImage #len(lines) # 10

    print("n_captions: %d n_images: %d n_captions_per_image: %d" % (n_captions, n_images, n_captions_per_image))

    captions_ids_train, captions_ids_test = captions_ids[: 8000*n_captions_per_image], captions_ids[8000*n_captions_per_image :]
    images_train, images_test = images[:8000], images[8000:]
    if need_256:
        images_train_256, images_test_256 = images_256[:8000], images_256[8000:]
    n_images_train = len(images_train)
    n_images_test = len(images_test)
    n_captions_train = len(captions_ids_train)
    n_captions_test = len(captions_ids_test)
    print("n_images_train:%d n_captions_train:%d" % (n_images_train, n_captions_train))
    print("n_images_test:%d  n_captions_test:%d" % (n_images_test, n_captions_test))

    ## check test image
    # idexs = get_random_int(min=0, max=n_captions_test-1, number=64)
    # temp_test_capt = captions_ids_test[idexs]
    # for idx, ids in enumerate(temp_test_capt):
    #     print("%d %s" % (idx, [vocab.id_to_word(id) for id in ids]))
    # temp_test_img = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
    # save_images(temp_test_img, [8, 8], 'temp_test_img.png')
    # exit()

    # ## check the first example
    # tl.visualize.frame(I=images[0], second=5, saveable=True, name='temp', cmap=None)
    # for cap in captions_dict[1]:
    #     print(cap)
    # print(captions_ids[0:10])
    # for ids in captions_ids[0:10]:
    #     print([vocab.id_to_word(id) for id in ids])
    # print_dict(captions_dict)

    # ## generate a random batch
    # batch_size = 64
    # idexs = get_random_int(0, n_captions_test, batch_size)
    # # idexs = [i for i in range(0,100)]
    # print(idexs)
    # b_seqs = captions_ids_test[idexs]
    # b_images = images_test[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
    # print("before padding %s" % b_seqs)
    # b_seqs = tl.prepro.pad_sequences(b_seqs, padding='post')
    # print("after padding %s" % b_seqs)
    # # print(input_images.shape)   # (64, 64, 64, 3)
    # for ids in b_seqs:
    #     print([vocab.id_to_word(id) for id in ids])
    # print(np.max(b_images), np.min(b_images), b_images.shape)
    # from utils import *
    # save_images(b_images, [8, 8], 'temp2.png')
    # # tl.visualize.images2d(b_images, second=5, saveable=True, name='temp2')
    # exit()

import pickle
def save_all(targets, file):
    with open(file, 'wb') as f:
        pickle.dump(targets, f)

save_all(vocab, '_vocab.pickle')
save_all((images_train_256, images_train), '_image_train.pickle')
save_all((images_test_256, images_test), '_image_test.pickle')
save_all((n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test), '_n.pickle')
save_all((captions_ids_train, captions_ids_test), '_caption.pickle')
