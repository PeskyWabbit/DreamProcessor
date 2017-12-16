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

USERAGENT = 'web:DreamProcessor:v0.1 (by /u/ThePeskyWabbit)'
FOOTER = "^^I ^^am ^^a ^^bot!! ^^I ^^work ^^on ^^i.redd.it ^^and " \
         "^^imgur ^^posts!(not ^^galleries ^^yet) ^^I ^^now ^^use ^^a ^^randomly ^^selected ^^subset ^^of ^^filters ^^from ^^the ^^original ^^set ^^to ^^add ^^diversity." \
         " \n\n ^^Made ^^by ^^/u/ThePeskyWabbit ^^check ^^/r/DreamProcessor ^^for ^^all ^^of ^^my ^^creations!"
PATH = "C:\\Users\\Josh\\PycharmProjects\\DreamBot\\commented.txt"
stringList = ["!dreambot"]


#Starting here, this is mostly borrowed code from tensorflow. I definitely do not know enough about these
#algorithms to creat my own yet. The borrowed code ends at the render_deepdream function which I have tweaked a bit.
model_fn = "tensorflow_inception_graph.pb"

#creates the graph which is used for image recognition
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

#mostly black magic
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name = 'input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

#find all of the layers that match the name contents i am looking for. this returns six but there are wayyyy more.
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/mixed4c' in op.name]
print(layers)

feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]

print('Number of layers', len(layers))
print('Total number of feature channels:', sum(feature_nums))


# Helper functions for TF Graph visualization
# black magic
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

#not sure what this is even used for but it is indeed used.
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

#this is only used when im testing the dream filter.
print("selecting Layer and channel")
layer = 'mixed4b'
channel = 139  # picking some feature channel to visualize

#add some static to the pic so that more patterns can be found
print("generating noise")
#start with a gray image with a little noise
img_noise = np.random.uniform(size=(224, 224, 3)) + 130.0

#this is used for actually displaying the picture on my screen. Only works in Jupyter notebook
def showarray(a, fmt='jpeg'):
    print("Entered showArray")
    a = np.uint8(np.clip(a, 0, 1) * 255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


#not too sure what this does either. gotta have it.
def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5

#gets the graph/layer for processing
def T(layer):
    print(graph.get_tensor_by_name("import/%s:0" % layer))
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0" % layer)

#magic
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

#this does the editing of the image im pretty sure
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
#THIS IS UTTER MAGIC. I do plan on reading more into this when I have time but for now, it works and I get the very general idea of it.
def render_deepdream(t_obj, args, img0=img_noise,
                     iter_n=27, step=2.0, octave_n=4, octave_scale=1.4):
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
    print("DeepDream image saved.")

#used for imgurAuth()
def get_config():
    ''' Create a config parser for reading INI files '''
    try:
        import ConfigParser
        return ConfigParser.ConfigParser()
    except:
        import configparser
        return configparser.ConfigParser()

#downloads the picture from the given url, resizes it so the max resolution perameter is lower than 1400
#this helps show the effects on larger picture without having to zoom in close.
#args is set to return the filetype(.jpg/.png/etc) as well as the size of the image.
def directDownload(url):
    request = rlib.Request(url)
    response = rlib.urlopen(request)
    data = response.read()

    #create return list. adds filetype as first element and creates filename.
    args = []
    split = url.split('.')
    print(split)
    args.append(split[3])
    print("Image type: " + args[0])
    fname = "temp." + args[0]

    #opens and captures size for resizing
    img = Image.open(io.BytesIO(data))
    size = img.size
    print("Image size: " + str(size))

    #resizing loop
    while(size[0] > 1400 or size[1] > 1400):
        print("resizing img")
        img = img.resize((size[0] // 2, size[1] // 2), Image.ANTIALIAS)
        size = img.size

    print("resized = " + str(img.size))
    print("saving image")
    try:
        img.save(fname)
    except:
        #this is that lame fix. If I could, i would have used continue here but it had to be used in the
        #runBot() function so the fail arg is used for that purpose.
        print("Failed to save image")
        args[0] = "fail"
    args.append(img.size)
    return args

#uploads the picture to imgur and returns an Image object of the picture. I have not had a single error with this so no
#try clause has been necessary.
def uploadImgur(argsList):
    album = None
    image_path = 'C:\\Users\\Josh\\PycharmProjects\\DreamBot\\temp.' + argsList[0]
    config = {
        'album': album,
        'name': 'Deep Dream Pic!',
        'title': 'Deep Dream Pic!',
        'description': 'Image processed through Deepdream filter {0}'.format(datetime.now())
    }

    print("Uploading...")
    image = imgurClient.upload_from_path(image_path, config=config, anon=False)
    print("done")
    return image

#auth my account with imgur for fetching and uploading data. uses .ini file for user info.
def imgurAuth():
    config = get_config()
    config.read('auth.ini')
    client_id = config.get('credentials', 'client_id')
    client_secret = config.get('credentials', 'client_secret')
    client = ImgurClient(client_id, client_secret)
    print("Authenticated as " + client_id + " on imgur client.\n")
    return client

#copies all of the bots creations to /r/DreamProcessor and gives a link to the original in the comment of said post.
def sendToSub(comment, image):
    user = comment.author.name
    sub = str(comment.submission.subreddit)
    commentLink = "https://www.reddit.com" + comment.permalink
    title = "DreamBot requested by /u/" + user + " in /r/" + sub
    url = image['link']
    post = reddit.subreddit('dreamprocessor').submit(title, url=url)
    post.reply("Link to the original picture: " + commentLink)

#What you guys actually see is pretty much all brought together here. This opens the image that was downloaded, converts it
#to a format that tensorflow can work with, and uploads it to imgur and gives you the link.
def renderAndReply(comment, args):
    try:
        response = comment.reply("I am processing your request! This comment will be edited when it is complete! If this "
                             "never changes, something went wrong :X")
    except:
        print("Could not tell user their request was being processed...")

    img0 = PIL.Image.open('temp.' + args[0])
    #changing the perams of the image if its a .png. not sure exactly why this needs to happen, but it makes things work...
    if ('png' in args[0]):
        img0 = np.float32(img0)[:,:,:3]
    else:
        img0 = np.float32(img0)

    #generate a random range of layers to process the image with. this is still being tweaked.
    int1 = randint(50, 550)
    int2 = randint(int1, 613)
    print("Layer range: (" + str(int1-50) + ", " + str(int2+50) + ")")
    render_deepdream(tf.square(T('mixed4c')[:,:,:,int1-50:int2+50]), args, img0)

    #Post the pics!!!!
    try:
        image = uploadImgur(args)
        sendToSub(comment, image)
        response.edit("[Here is your Deep Dream picture]({0})".format(image['link']) + "\n\n" + FOOTER)
        print("sent response to user")
    except:
        print("Comment or upload failed...")

#boring reddit authenticate method.
def authenticate():
    print("Authenticating...")
    reddit = praw.Reddit('bot1', user_agent=USERAGENT)
    print("Authenticated as {}\n".format(reddit.user.me()))
    return reddit

#adds ciomment.id to list so it doesnt duplicate respond.
def writeCommentToFile(id):
    print("saving comment ID: " + id)
    commentFile = open(PATH, 'a')
    commentFile.write(id + "\n\n")
    commentFile.close()

#what you see when the picture couldnt be downloaded from imgur.
def failReply(comment):
    comment.reply("There was an error saving this picture. It is likely that it was a .png and was converted to a .jpg by the host site.\n\n" + FOOTER)
    writeCommentToFile(comment.id)

#lets get things started.
auth = True
while (auth):
    try:
        reddit = authenticate()
        imgurClient = imgurAuth()
        auth = False
    except:
        print("Authentication Failed, retying in 30 seconds.")
        time.sleep(30)

#the magic
def runBot():
    #post to all except the other ones mentioned.
    SUBREDDITS = 'all-suicidewatch-depression-anxiety-askreddit'
    while(True):
        print("pulling 500 comments...")
        commentFile = open(PATH, 'r')
        commentList = commentFile.read().splitlines()
        commentFile.close()
        try:
            for comment in reddit.subreddit(SUBREDDITS).comments(limit=500):
                if (comment.id in commentList):
                    continue
                #I use a list so I can add commands if need be. good for testing it on the side along side running the bot.
                for word in stringList:
                    #finds the command in comments.
                    match = re.findall(word, comment.body.lower())
                    if (match):
                        post = comment.parent()
                        print("https://www.reddit.com" + comment.permalink)

                        try:
                            #checks if the variable "post" is a comment or not.
                            if hasattr(post, 'is_root'):
                                print("responding to a comment picture")
                                #finds the url in the comment. only the first match is used.
                                url = re.findall(
                                    'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                                    post.body)
                                url = str(url[0])
                                #cleans a ) off the end in the case of it being a formatted text link [like this](link.com)
                                if (')' in url):
                                    url = url[0:len(url) - 1]
                            else:
                                url = post.url
                            #no gifs because it would take ages to process.
                            if('gif' in url):
                                comment.reply("Sorry but I cannot process gifs. If I could, It would take like 2 hours anyways so yeah. Sorry!\n\n" + FOOTER)
                            #these are both direct image links so they can be downloaded the same way
                            elif("//i.imgur" in url or "i.redd.it" in url):
                                print("/i.imgur.com or i.redd.it link being used")
                                args = directDownload(url)
                                #if the image oouldnt be downloaded
                                if("fail" in args):
                                    failReply(comment)
                                    continue
                                renderAndReply(comment, args)
                            #if its an album, grab the first picture and use its direct link
                            elif("//imgur.com/a/" in url):
                                print("/imgur.com/a/ link being used")
                                url = url.split("/")
                                images = imgurClient.get_album_images(url)
                                pic = images[0]
                                args = directDownload(pic.link)
                                #if the image oouldnt be downloaded
                                if ("fail" in args):
                                    failReply(comment)
                                    continue
                                renderAndReply(comment, args)
                            #if its a link to the imgur page but not the direct link, create the direct link.
                            elif("/imgur.com" in url):
                                print("/imgur.com link being used")
                                url = post.url.split("/")
                                link = "https://i.imgur.com/" + url[3] + ".jpg"
                                args = directDownload(link)
                                #if the image oouldnt be downloaded
                                if ("fail" in args):
                                    failReply(comment)
                                    continue
                                renderAndReply(comment, args)
                            else:
                                #all other cases of links. I do plan on adding a try/except in here to just try and
                                #download the link anyways because who knows, it may be a direct link!
                                try:
                                    comment.reply("I am not compatible with these links yet.\n\n" + FOOTER)
                                except:
                                    print("comment reply failed...")
                        except:
                            comment.reply("This is giving me errors. I'll tell the dev and maybe he can have a look at it...\n\n" + FOOTER)
                        writeCommentToFile(comment.id)
        except:
            print("something really went wrong...")

runBot()