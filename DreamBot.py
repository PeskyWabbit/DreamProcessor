from __future__ import print_function
from datetime import datetime
from urllib.request import Request
from io import BytesIO
from IPython.display import clear_output, Image, display, HTML
from imgurpython import ImgurClient
from PIL import Image
from random import randint
import praw
import time
import re
import urllib.request as rlib
import io
import numpy as np
import PIL.Image
import tensorflow as tf
import logging
import os
logging.getLogger('tensorflow').setLevel(logging.DEBUG)

USERAGENT = 'web:DreamProcessor:v0.1 (by /u/ThePeskyWabbit)'
FOOTER = "^^i.redd.it ^^and ^^imgur ^^posts \n\n ^^Made ^^by ^^/u/ThePeskyWabbit ^^check ^^/r/DreamProcessor ^^for ^^all ^^of ^^my ^^creations! ^^Source: https://github.com/PeskyWabbit/DreamProcessor"
PATH = "C:\\Users\\JoshLaptop\\PycharmProjects\\DreamBot\\commented.txt"
stringList = ["!dreambot"]

_image_formats = ['bmp', 'dib', 'eps', 'ps', 'gif', 'im', 'jpg', 'jpe', 'jpeg',
                  'pcd', 'pcx', 'png', 'pbm', 'pgm', 'ppm', 'psd', 'tif', 'tiff',
                  'xbm', 'xpm', 'rgb', 'rast', 'svg']

model_fn = "tensorflow_inception_graph.pb"

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name = 'input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/mixed3b' in op.name]
print(layers)

feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]

print('Number of layers', len(layers))
print('Total number of feature channels:', sum(feature_nums))


# Helper functions for TF Graph visualization

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>" % size)
    return strip_def


def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add()
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0] != '^' else '^' + rename_func(s[1:])
    return res_def

# Visualizing the network graph. Be sure expand the "mixed" nodes to see their
# internal structure. We are going to visualize "Conv2D" nodes.
tmp_def = rename_nodes(graph_def, lambda s: "/".join(s.split('_', 1)))
#show_graph(tmp_def)

print("selecting Layer and channel")
layer = 'mixed4b'
channel = 139  # picking some feature channel to visualize

print("generating noise")
# start with a gray image with a little noise
img_noise = np.random.uniform(size=(224, 224, 3)) + 130.0


def showarray(a, fmt='jpeg'):
    print("Entered showArray")
    a = np.uint8(np.clip(a, 0, 1) * 255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))



def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5


def T(layer):
    print(graph.get_tensor_by_name("import/%s:0" % layer))
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0" % layer)

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

'''
step increases the intesity. iter_n increases how many times the filter runs
defaults: step = 1.5    iter_n = 10     octave_n = 4     octave_scale = 1.4
pretty good settings: iter_n=20, step=1.5 octave_n=4 octave_scale=1.4
'''

def render_deepdream(t_obj, args, img0=img_noise,
                     iter_n=27, step=1.7, octave_n=4, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!

    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    # generate details octave by octave
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g * (step / (np.abs(g).mean() + 1e-7))
        clear_output()

    a = img / 255.0
    a = np.uint8(np.clip(a, 0, 1) * 255)

    PIL.Image.fromarray(a).save("temp." + args[0])
    print("DeepDream image saved as temp." + args[0])


def get_config():
    ''' Create a config parser for reading INI files '''
    try:
        import ConfigParser
        return ConfigParser.ConfigParser()
    except:
        import configparser
        return configparser.ConfigParser()

def directDownload(url):
    request = rlib.Request(url)
    response = rlib.urlopen(request)
    data = response.read()
    args = []
    split = url.split('.')
    print(split)
    args.append(split[3])
    print("Image type: " + args[0])

    img = Image.open(io.BytesIO(data))
    size = img.size
    print("Image size: " + str(size))
    fname = "temp." + args[0]
    while(size[0] > 1400 or size[1] > 1400):
        print("resizing img")
        img = img.resize((size[0] // 2, size[1] // 2), Image.ANTIALIAS)
        size = img.size
    print("resized = " + str(img.size))
    print("saving image")
    try:
        img.save(fname)
        print("Saved input picure as temp." + args[0])
    except:
        print("Failed to save image")
        args[0] = "fail"
    args.append(img.size)
    return args


def uploadImgur(argsList):
    album = None
    image_path = 'C:\\Users\\JoshLaptop\\PycharmProjects\\DreamBot\\temp.' + argsList[0]
    config = {
        'album': album,
        'name': 'Deep Dream Pic!',
        'title': 'Deep Dream Pic!',
        'description': 'Image processed through Deepdream filter {0}'.format(datetime.now())
    }

    print("Uploading temp." + argsList[0] + "...")
    image = imgurClient.upload_from_path(image_path, config=config, anon=False)
    print("done")
    return image

def imgurAuth():
    config = get_config()
    config.read('auth.ini')
    client_id = config.get('credentials', 'client_id')
    client_secret = config.get('credentials', 'client_secret')
    client = ImgurClient(client_id, client_secret)
    print("Authenticated as " + client_id + " on imgur client.\n")
    return client

def sendToSub(data, image, directUrl):
    print("sendToSub url received: " + directUrl )
    user = data.author.name
    url = image['link']

    #if its a comment
    if hasattr(data, "is_root"):
        sub = "/r/" + str(data.submission.subreddit)
        link = "https://www.reddit.com" + str(data.permalink) + "\n\nDirect image link: " + str(directUrl)

    #if its a message
    else:

        sub = "a private message."
        link = directUrl

    title = "DreamBot requested by /u/" + user + " in " + sub

    post = reddit.subreddit('dreamprocessor').submit(title, url=url)
    post.reply("Link to the original picture: " + link)
'''
mixed4a = 620
mixed4b = 648
mixed4c = 664
mixed4d = 704
mixed3a = 368
mixed3b = 640

'''
def renderAndReply(data, args, url):
    flag = False
    response = None
    filter = {
        0: ("mixed4a", 570),
        1: ('mixed4b', 598),
        2: ("mixed4c", 614),
        3: ("mixed4d", 654),
        4: ("mixed3a", 318),
        5: ("mixed3b", 590)
    }
    if hasattr(data, "is_root"):
        flag = True
        try:
            response = data.reply("I am processing your request! This comment will be edited when it is complete! If this "
                                 "never changes, it is likely I have reached my comment rate limit for this subreddit. Try again later.")
        except:
            print("Could not tell user their request was being processed...")
            pass

    img0 = PIL.Image.open('temp.' + args[0])
    if ('png' in args[0]):
        img0 = np.float32(img0)[:,:,:3]
    else:
        img0 = np.float32(img0)

    layerData = filter[randint(0,5)]
    layer = layerData[0]
    upper = layerData[1]
    int1 = randint(50, upper)
    int2 = randint(int1, upper)
    print("Layer range: (" + str(int1-50) + ", " + str(int2+50) + ")")
    render_deepdream(tf.square(T(layer)[:,:,:,int1-50:int2+50]), args, img0)

    try:
        image = uploadImgur(args)
        if(flag):
            sendToSub(data, image, url)
            response.edit("[Here is your Deep Dream picture]({0})".format(image['link']) + " Processed using imageset: " + layer + " layers " + str(int1) + " - " + str(int2) + "\n\n" + FOOTER)
            print("sent response to user")
        else:
            sendToSub(data, image, url)
            data.reply("[Here is your Deep Dream picture]({0})".format(image['link']) + layer + " layers " + str(int1) + " - " + str(int2) + "\n\n" + FOOTER)
            print("replied to PM")

    except:
        print("Comment or upload failed...")

def authenticate():
    print("Authenticating...")
    reddit = praw.Reddit('bot1', user_agent=USERAGENT)
    print("Authenticated as {}\n".format(reddit.user.me()))
    return reddit


def writeCommentToFile(id):
    print("saving comment ID: " + id)
    commentFile = open(PATH, 'a')
    commentFile.write(id + "\n\n")
    commentFile.close()

def failReply(comment):
    comment.reply("There was an error saving this picture. It is likely that it was a .png and was converted to a .jpg by the host site.\n\n" + FOOTER)
    writeCommentToFile(comment.id)

def processCall(data):
    url = None
    #if its a comment that called !dreambot
    if hasattr(data, "is_root"):
        print("processing a call from a comment")
        dataWithLink = data.parent()
        print("https://www.reddit.com" + str(dataWithLink.permalink))
        if hasattr(dataWithLink, "is_root"):
            print("processing a link from a comment")
            url = re.findall(
                'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                dataWithLink.body)
            url = str(url[0])
            if (')' in url):
                url = url[0:len(url) - 1]
            print("https://www.reddit.com" + dataWithLink.permalink)
        else:
            url = dataWithLink.url
        print("Comment url: " + url)

    elif hasattr(data, "mark_read"):
        print("Processing a call from a PM")
        url = re.findall(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            data.body)
        url = str(url[0])
        print("message url: " + url)

    else:
        print("Data was found to be neither a comment nor a message.")
    if ('gif' in url):
        data.reply(
            "Sorry but I cannot process gifs. If I could, It would take like 2 hours anyways so yeah. Sorry!\n\n" + FOOTER)
    elif ("//i.imgur" in url or "i.redd.it" in url or "//m.imgur" in url):
        print("i.imgur.com / i.redd.it / m.imgur link being used")
        args = directDownload(url)
        print("directUrl sending to renderAndReply: " + url)
        renderAndReply(data, args, url)
    elif ("//imgur.com/a/" in url):
        print("/imgur.com/a/ link being used")
        split = url.split("/")
        images = imgurClient.get_album_images(split[3])
        pic = images[0]
        args = directDownload(pic.link)
        print("directUrl sending to renderAndReply: " + url)
        renderAndReply(data, args, url)
    elif ("/imgur.com" in url):
        print("/imgur.com link being used")
        split = url.split("/")
        link = "https://i.imgur.com/" + split[3] + ".jpg"
        args = directDownload(link)
        print("directUrl sending to renderAndReply: " + url)
        renderAndReply(data, args, url)
    else:
        try:
            print("did not recognize link. trying last resort download...")
            args = directDownload(url)
            print("directUrl sending to renderAndReply: " + url)
            renderAndReply(data, args, url)
        except:
            try:
                data.reply("I am not compatible with these links yet.\n\n" + FOOTER)
            except:
                print("comment reply failed...")
    if hasattr(data, "is_root"):
        writeCommentToFile(data.id)


auth = True
while (auth):
    try:
        reddit = authenticate()
        imgurClient = imgurAuth()
        auth = False
    except:
        print("Authentication Failed, retying in 30 seconds.")
        time.sleep(30)


def runBot():
    SUBREDDITS = 'all-suicidewatch-depression-anxiety-askreddit'
    while (True):
        try:
            print("checking inbox")
            for message in reddit.inbox.unread(limit = 1):
                message.mark_read()
                if('!dreambot' in message.body):
                    processCall(message)

            print("pulling 500 comments...")
            commentFile = open(PATH, 'r')
            commentList = commentFile.read().splitlines()
            commentFile.close()
            try:
                for comment in reddit.subreddit(SUBREDDITS).comments(limit=500):
                    if (comment.id in commentList):
                        continue
                    for word in stringList:
                        match = re.findall(word, comment.body.lower())
                        if (match):
                            try:
                                print("Processing comment call!!")
                                processCall(comment)
                            except:
                                try:
                                    comment.reply("I'm testing some new things and I guess a real deal error occurred.")
                                except:
                                    print("couldnt send error reply")
            except:
                print("something really went wrong...")
        except:
            print("This would have crashed the whole bot. likely a timeout")
runBot()