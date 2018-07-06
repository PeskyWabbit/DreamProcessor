from __future__ import print_function
from datetime import datetime
from IPython.display import clear_output, Image, display, HTML
from imgurpython import ImgurClient
from PIL import Image
from random import randint
import praw
import time
import re
import io
import numpy as np
import PIL.Image
import tensorflow as tf
import logging
import evaluate
import utils as utils
import sys
import os
if (sys.version_info > (3, 0)):
     import urllib.request as rlib
else:
    import urllib2 as rlib
    

####Setting the seed ensures that an image with the exact same paramters are rendered in exactly the same way
p.random.seed(1)
logging.getLogger('tensorflow').setLevel(logging.DEBUG)

USERAGENT = 'web:DreamProcessor:v0.1 (by /u/ThePeskyWabbit)'
FOOTER = "^^I ^^work ^^on ^^i.redd.it ^^and ^^imgur ^^posts ^^and ^^links. ^^See ^^all ^^my ^^new ^^options ^^[here](https://imgur.com/a/QWANb)" \
         "\n\n^^check ^^/r/DreamProcessor ^^for ^^my" \
         " ^^new ^^command ^^options ^^and ^^all ^^of ^^my ^^creations! ^^https://github.com/PeskyWabbit/DreamProcessor"
lastUserToCall = ""
userStreak = 0
commented = []
"""
#####
Important:
Set to false to test locally
"""
auth = True
run_bot = True

PATH = "/home/jpeel/PycharmProjects/DreamBot/commented.txt"
regexes = "!dreambot(?:1[0-8]|[1-9])?(?:x[2-3])?", "!dbhowto"
combined = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))

model_fn = "tensorflow_inception_graph.pb"
inception_download_url="http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
save_folder="./renderedImages/"

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

with tf.gfile.FastGFile(model_fn, 'rb') as f:
    ###
    utils.maybe_download_and_extract(inception_download_url, ".")
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name = 'input')
#default 117.0
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

##Not neeeded, but useful for getting available layer names#
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
print(layers)

#feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]

#print('Number of layers', len(layers))
#print('Total number of feature channels:', sum(feature_nums))

###The below code is from Google's Tensorflow DeepDream tutorial Notbook'''

#Not really needed, but ensures render_deepdream is never called with an empty image
img_noise = np.random.uniform(size=(224, 224, 3)) + 130.0



def T(layer):
    print("Processing...")
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
                     iter_n=27, step=1.6, octave_n=4, octave_scale=1.6):
    t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!
    print("iter_n = " + str(iter_n))
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
    PIL.Image.fromarray(a).save("/home/jpeel/PycharmProjects/DreamBot/temp." + args[0])
    print("DeepDream image saved as temp1." + args[0])


def get_config():
    ''' Create a config parser for reading INI files '''
    try:
        import ConfigParser
        return ConfigParser.ConfigParser()
    except:
        import configparser
        return configparser.ConfigParser()


'''The above code is from Google's DeepDream Tensorflow Tutorial Notebook'''

#Download and resize image from the given URL
def directDownload(url):
    request = rlib.Request(url)
    response = rlib.urlopen(request)
    data = response.read()
    args = []
    split = url.split('.')
    args.append(split[-1])
    print("Image type: " + args[0])

    img = Image.open(io.BytesIO(data))
    size = img.size
    print("Image size: " + str(size))
    fname = "temp." + args[0]

    #resize image if needed
    if(size[0] > 1500 or size[1] > 1500):
        print("resizing img")
        max = np.maximum(size[0], size[1])
        divisor = float(max // 1500)
        newWidth = int(size[0] / divisor)
        newHeight = int(size[1] / divisor)
        img = img.resize((newWidth, newHeight), Image.ANTIALIAS)
        print("resized")

    try:
        img.save(fname)
        print("Saved input picure as temp." + args[0])
    except:
        print("Failed to save image")
        args[0] = "fail"
    return args

#post image to imgur and return the image post object
def uploadImgur(argsList):
    album = None
    image_path = '/home/jpeel/PycharmProjects/DreamBot/temp.' + argsList[0]
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

#crosspost to /r/dreamprocessor unless it is NSFW
def sendToSub(data, image, directUrl):
    global userStreak
    if(userStreak > 3):
        return
    #if its a comment
    if hasattr(data, "is_root"):
        sub = "/r/" + str(data.submission.subreddit)

    #if its a message
    else:
        return

    #crosspost
    try:
        print("1st one " + str(data.permalink()))
        permalink = str(data.permalink())
    except:
        try:
            print("2nd one " + str(data.permalink))
            permalink = str(data.permalink)
        except:
            permalink = "Picture received in PM."

    #Build comment for post in sub
    link = "www.reddit.com" + permalink + "\n\nDirect image link: " + str(directUrl)
    user = data.author.name
    url = image['link']

    title = "DreamBot requested by /u/" + user + " in " + sub

    #make post
    if (data.submission.over_18):
        post = reddit.subreddit('NSFWDreamBot').submit(title, url=url)
    else:
        post = reddit.subreddit('dreamprocessor').submit(title, url=url)
    post.reply("Link to the original post: " + link)

#data to map fnum to the desired filter
filters = {
        #not included in random
        1: ("mixed4c", [0, 664]),
        2: ("mixed3b", [0, 640]),
        #included in random
        3: ("mixed4a", [151, 152]),
        4: ("mixed4c", [211, 212]),
        5: ("mixed4c", [382, 383]),
        6: ("mixed4d", [72, 73]),
        7: ("mixed3a", [0, 368]),
        8: ("mixed3a", [31, 32]),
        9: ("mixed3a", [230, 240]),
        10: ("mixed3a", [5, 9]),
        11:("Strarry Night", "/home/jpeel/PycharmProjects/DreamBot/checks/starrynight"),
        12:("Alex Gray", "/home/jpeel/PycharmProjects/DreamBot/checks/alexgray"),
        13:("Fractal", "/home/jpeel/PycharmProjects/DreamBot/checks/stain"),
        14:("MS Paint Oil", "/home/jpeel/PycharmProjects/DreamBot/checks/MSOil"),
        15:("Rain Princess", "/home/jpeel/PycharmProjects/DreamBot/checks/rain"),
        16:("Trippy Watercolor", "/home/jpeel/PycharmProjects/DreamBot/checks/trippywc"),
        17:("Great Wave", "/home/jpeel/PycharmProjects/DreamBot/checks/wave"),
        18:("Fun Color", "/home/jpeel/PycharmProjects/DreamBot/checks/funcolor")
    }

#create the deepdream/style transfer picture and send it to the user. Should likely be split into 2 separate functions. Do later
def renderAndReply(data, args, url):
    banned = ["pics", "elitedangerous", "perfecttiming", "strangerthings", "travel", "aww", "rabbits", "moviedetails", "choosingbeggars", "battlestations", "interestingasfuck", "itookapicture", "gamingcirclejerk",
              "realgirls", "seattle", "natureisfuckinglit", "dankmemes", "random_acts_of_amazon", "tgirls", "earthporn"]

    flag = False
    response = None

    if hasattr(data, "is_root"):
        flag = True
        try:
            response = data.reply("I am processing your request! This comment will be edited when it is complete! If this "
                                 "never changes, there is likely a bug which will be sorted out as soon as my dev has a minute to look into it!")
        except:
            print("Could not tell user their request was being processed...")
            pass

    ####possible different function
    fnum = int(args[1][0])
    fMult = int(args[1][1])

    #if no int given or int is out of bounds, generate one for the random set of images.
    if (fnum == 0 or fnum > 18):
        rand = randint(3, 18)
        fnum = rand
        layerData = filters[rand]

    #args[0] is the filetype
    img0 = PIL.Image.open('temp.' + args[0])
    in_path = "/home/jpeel/PycharmProjects/DreamBot/temp."+str(args[0])
    out = "/home/jpeel/PycharmProjects/DreamBot/temp."+str(args[0])

    #style transfer filters
    ####seperate function
    i = 0
    if(fnum > 10):
        while(i < fMult):
            i+=1
            evaluate.process(in_path, out, filters[fnum][1])

    #non style transfer filters
    ###seperate function
    
    else:
        if ('png' in args[0]):
            img0 = np.float32(img0)[:,:,:3]
        else:
            img0 = np.float32(img0)

        if fMult > 3 or fMult < 1:
            fMult = 1

        else:
            layerData = filters[fnum]

        layerSet = layerData[0]
        range = layerData[1]
        int1 = range[0]
        int2 = range[1]

        #with tf.device('/gpu:0'):
        render_deepdream(tf.square(T(layerSet)[:,:,:,int1:int2]), args, img0, 27*fMult)


    ##possible different funtion
    #updload and respond to user
    try:
        image = uploadImgur(args)
        time.sleep(1)
        wholeResponse = "[Here is your DreamBot picture]({0})".format(
            image['link']) + " Made using option " + str(fnum) + "\n\n" + FOOTER

        #is comment
        if(flag):
            if(str(data.submission.subreddit).lower() in banned):
                data.author.message("The Mods of " + str(data.submission.subreddit) + " have Banned me :( Here's your picture", wholeResponse,)
            else:
                response.edit(wholeResponse)
            sendToSub(data, image, url)
            print("SENT RESPONSE TO USER")

        #is message
        else:
            sendToSub(data, image, url)
            data.reply(wholeResponse)
            print("replied to PM")

    except:
         print("Comment or upload failed...")


def authenticate():
    print("Authenticating...")
    reddit = praw.Reddit('bot1', user_agent=USERAGENT)
    print("Authenticated as {}\n".format(reddit.user.me()))
    return reddit

#save comment ID to text document to avoid duplicate replies. Text document is cleared every 500 comment batches
#def writeCommentToFile(id):
#    print("saving comment ID: " + id)
#    commentFile = open(PATH, 'a')
#    commentFile.write(id + "\n\n")
#    commentFile.close()

def failReply(comment):
    global commented
    comment.reply("There was an error saving this picture. It is likely that it was a .png and was converted to a .jpg by the host site.\n\n" + FOOTER)
    commented.append(comment.id)

#once command has been found in comment or PM, it is passed in here
def processCall(data, fNum):
    url = None

    global lastUserToCall
    global userStreak
    #if its a comment that called !dreambot
    if hasattr(data, "is_root"):

        if (lastUserToCall == data.author.name):
            userStreak += 1
        else:
            lastUserToCall = data.author.name
            userStreak = 0
        print(str(userStreak) + " / " + lastUserToCall)

        print("processing a call from a comment")
        dataWithLink = data.parent()

        try:
            commentLink = dataWithLink.permalink()
            print("https://www.reddit.com" + commentLink)

        except:
            try:
                print("https://www.reddit.com" + str(dataWithLink.permalink))
            except:
                print("Couldnt find any link whatsoever ")
        #if the parent of the command is a comment, find the URL in the comment
        if hasattr(dataWithLink, "is_root"):
            url = re.findall(
                'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                dataWithLink.body)
            try:
                url = str(url[0])
            except:
                print("couldnt find url")
                try:
                    data.reply("I couldn't find the link for some reason. this is not very common\n\n" + FOOTER)
                except:
                    return
            if (')' in url):
                url = url[0:len(url) - 1]
        #if the parent of the command comment is a post, get the post url
        else:
            url = dataWithLink.url
        print("Comment url: " + str(url))

    #if Private message, find link in private message
    elif hasattr(data, "mark_read"):
        url = re.findall(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            data.body)
        url = str(url[0])
        print("message url: " + url)

    else:
        print("Data was found to be neither a comment nor a message.")

    #no gif's yet...
    if ('gif' in url):
        data.reply(
            "Sorry but I cannot process gifs. If I could, It would take like 2 hours anyways so yeah. Sorry!\n\n" + FOOTER)

    elif ("//i.imgur" in url or "i.redd.it" in url or "m.imgur" in url and "/a/" not in url):
        print("i.imgur.com / i.redd.it / m.imgur link being used")
        args = directDownload(url)
        args.append(fNum)
        renderAndReply(data, args, url)

    elif ("//imgur.com/a/" in url or "m.imgur.com/a/" in url or "gallery" in url):
        print("/imgur.com/a/ / gallery link being used")
        split = url.split("/")
        images = imgurClient.get_album_images(split[4])
        pic = images[0]
        args = directDownload(pic.link)
        args.append(fNum)
        renderAndReply(data, args, url)

    elif ("/imgur.com" in url):
        print("/imgur.com link being used")
        split = url.split("/")
        link = "https://i.imgur.com/" + split[3] + ".jpg"
        args = directDownload(link)
        args.append(fNum)
        renderAndReply(data, args, url)

    else:
        #if none of the above URL's are present, try downloading it anyways
        try:
            print("did not recognize link. trying last resort download...")
            args = directDownload(url)
            args.append(fNum)
            renderAndReply(data, args, url)
        except:
            try:
                data.reply("Please be sure you are responding to a submission or a comment with a link in it. If you are, I am not compatible with those links.\n\n" + FOOTER)
            except:
                print("comment reply failed...")
    global commented
    if hasattr(data, "is_root"):
        commented.append(data.id)

while (auth):
    try:
        reddit = authenticate()
        imgurClient = imgurAuth()
        auth = False
    except:
        print("Authentication Failed, retying in 30 seconds.")
        time.sleep(30)

def purgeCommentList():
    print("purging contents of comment list")
    open(PATH, 'w').close()

#pull comments and look for the command. Once found, parse fnum and multiplier ints and pass to processCall function and save comment ID to file.
def runBot():
    SUBREDDITS = 'all-suicidewatch-depression-anxiety-askreddit'
    global commented
    inboxCount = 0
    count = 0
    while (True):
        try:
            if (inboxCount == 5):
                fNum = []
                print("checking inbox")
                for message in reddit.inbox.unread(limit=1):
                    message.mark_read()
                    match = combined.findall(message.body.lower())

                    if(match):
                        print("Message match = " + str(match))
                        fNum = re.findall("[1][0-8]|[1-9]", match[0])
                        if not fNum:
                            fNum.append(0)
                        if len(fNum) < 2:
                            fNum.append(1)
                        print(fNum)
                        processCall(message, fNum)
                inboxCount = 0

            inboxCount += 1
            print("pulling 500 comments...")
            #commentFile = open(PATH, 'r')
            #commentList = commentFile.read().splitlines()
            #commentFile.close()
            try:
                count +=1
                if count == 50:
                    print("purging comment file")
                    commented = []
                    count = 0

                for comment in reddit.subreddit(SUBREDDITS).comments(limit=500):
                    if (comment.id in commented):
                        continue
                    match = combined.findall(comment.body.lower())
                    if(match):
                        print("Comment match = " + str(match))
                        fNum = re.findall("[1][0-8]|[1-9]", match[0])
                        if not fNum:
                            fNum.append(0)
                        if len(fNum) < 2:
                            fNum.append(1)
                        print(fNum)

                        try:
                            print("Processing comment call!!")
                            processCall(comment, fNum)

                        except:
                            try:
                               print("failure on match try catch")
                               comment.reply("Please be sure there is a link in the post you are reponding to. If this is not the case, tagging for notification /u/ThePeskyWabbit\n\n")
                               commented.append(comment.id)
                            except:
                                print("couldnt send error reply")
            except:
                print("something really went wrong...")
        except:
            print("This would have crashed the whole bot. likely a timeout")

#one function to rule them all
if run_bot:
    runBot()
            
####local Test functions

def render_deepdream_local(layer, int1, int2, img0,
                     iter_n=27, step=1.6, octave_n=4, octave_scale=1.6):
    
    t_obj=tf.square(T(layer)[:,:,:,int1:int2])
    t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!
    print("iter_n = " + str(iter_n))
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
    ##Create precise filename
    filename=layer+"_"+str(int1)+"_"+str(int2)+".jpg"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder,filename), 'wb') as file:
        PIL.Image.fromarray(a).save(file, 'jpeg')


def load_image(filename):
    image = PIL.Image.open(filename)
    return np.float32(image)



