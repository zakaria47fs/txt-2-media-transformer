

from PIL import Image
import numpy as np
import gradio as gr
import paddlehub as hub
import urllib
import cv2
import re
import os
import requests

from share_btn import community_icon_html, loading_icon_html, share_js

import torch

from spectro import wav_bytes_from_spectrogram_image
from diffusers import StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler
 
import io
from os import path
from pydub import AudioSegment
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import *
import mutagen
from mutagen.mp3 import MP3
from mutagen.wave import WAVE


import time
import base64
import gradio as gr
from sentence_transformers import SentenceTransformer

import httpx
import json
from fastapi import FastAPI

from utils import get_tags_for_prompts, get_mubert_tags_embeddings, get_pat


minilm = SentenceTransformer('all-MiniLM-L6-v2')
mubert_tags_embeddings = get_mubert_tags_embeddings(minilm)


def get_track_by_tags(tags, pat, duration, maxit=20, loop=False):
    if loop:
        mode = "loop"
    else:
        mode = "track"
    r = httpx.post('https://api-b2b.mubert.com/v2/RecordTrackTTM',
                   json={
                       "method": "RecordTrackTTM",
                       "params": {
                           "pat": pat,
                           "duration": duration,
                           "tags": tags,
                           "mode": mode
                       }
                   })

    rdata = json.loads(r.text)
    assert rdata['status'] == 1, rdata['error']['text']
    trackurl = rdata['data']['tasks'][0]['download_link']

    print('Generating track ', end='')
    for i in range(maxit):
        r = httpx.get(trackurl)
        if r.status_code == 200:
            return trackurl
        time.sleep(1)


def generate_track_by_prompt(prompt):
    try:
        pat = get_pat("mail@mail.com")
        _, tags = get_tags_for_prompts(minilm, mubert_tags_embeddings, [prompt, ])[0]
        result = get_track_by_tags(tags, pat, int(30), loop=False)
        print(result)
        return result 
    except Exception as e:
        return str(e)


img_to_text = gr.Blocks.load(name="spaces/fffiloni/CLIP-Interrogator-2")
#text_to_music = gr.Interface.load("spaces/fffiloni/text-2-music") 

language_translation_model = hub.Module(name='baidu_translate')
language_recognition_model = hub.Module(name='baidu_language_recognition')

# style_list = ['古风', '油画', '水彩', '卡通', '二次元', '浮世绘', '蒸汽波艺术', 'low poly', '像素风格', '概念艺术', '未来主义', '赛博朋克', '写实风格', '洛丽塔风格', '巴洛克风格', '超现实主义', '默认']
style_list_EN = ['Chinese Ancient Style', 'Oil painting', 'Watercolor', 'Cartoon', 'Anime', 'Ukiyoe', 'Vaporwave', 'low poly', 'Pixel Style', 'Conceptual Art', 'Futurism', 'Cyberpunk', 'Realistic style', 'Lolita style', 'Baroque style', 'Surrealism', 'Detailed']

tips = {"en": "Tips: The input text will be translated into English for generation", 
        "jp": "ヒント: 入力テキストは生成のために中国語に翻訳されます", 
        "kor": "힌트: 입력 텍스트는 생성을 위해 중국어로 번역됩니다"}

count = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running app on "{}" device'.format(device))

model_id = "runwayml/stable-diffusion-v1-5"
eulera = EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
#pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, scheduler=eulera)
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=eulera)
pipe = pipe.to(device)

model_id2 = "riffusion/riffusion-model-v1"
#pipe2 = StableDiffusionPipeline.from_pretrained(model_id2, torch_dtype=torch.float16)
pipe2 = StableDiffusionPipeline.from_pretrained(model_id2)
pipe2 = pipe2.to(device)


def translate_language_example(text_prompts, style_indx):
    return translate_language(text_prompts)

def translate_language(text_prompts):
    global count
    try:
        count += 1
        tips_text = None
        language_code = language_recognition_model.recognize(text_prompts)
        if language_code != 'en':
            text_prompts = language_translation_model.translate(text_prompts, language_code, 'en')
    except Exception as e:
        error_text = str(e)
        return {status_text:error_text, language_tips_text:gr.update(visible=False), translated_language:text_prompts, trigger_component: gr.update(value=count, visible=False)}
    if language_code in tips:
        tips_text = tips[language_code]
    else:
        tips_text = tips['en']
    if language_code == 'en':
        return {language_tips_text:gr.update(visible=False), translated_language:text_prompts, trigger_component: gr.update(value=count, visible=False)}
    else:
        return {language_tips_text:gr.update(visible=True, value=tips_text), translated_language:text_prompts, trigger_component:  gr.update(value=count, visible=False)}



def get_result(text_prompts, style_indx, musicAI_indx, duration):
    style = style_list_EN[style_indx]
    prompt = style + "," + text_prompts
        
    sdresult = pipe(prompt, negative_prompt = "out of focus, scary, creepy, evil, disfigured, missing limbs, ugly, gross, missing fingers", num_inference_steps=50, guidance_scale=7, width=376, height=376)
    image_output = sdresult.images[0] if not sdresult.nsfw_content_detected[0] else Image.open("nsfw_placeholder.jpg")
    
    print("Generated image with prompt " + prompt)
    
    # Encode your PIL Image as a JPEG without writing to disk
    imagefile = "imageoutput.png"
    #img_np = np.array(image_output[0])
    #img_nparray= cv2.cvtColor(img_np, cv2.COLOR_BGR2RGBA)
    #img_blue_correction = Image.fromarray(img_nparray)
    #img_blue_correction.save(imagefile, img_blue_correction.format)
    image_output.save(imagefile, image_output.format)
    
    interrogate_prompt = prompt
    #interrogate_prompt = img_to_text(imagefile, 'fast', 4, fn_index=1)[0]
    print(interrogate_prompt)
    spec_image, music_output = get_music(interrogate_prompt + ", " + style_list_EN[style_indx], musicAI_indx, duration)
    
    video_merged = merge_video(music_output, image_output)
    return {spec_result:spec_image, imgfile_result:image_output, musicfile_result:"audio.wav", video_result:video_merged, status_text:'Success', share_button:gr.update(visible=True), community_icon:gr.update(visible=True), loading_icon:gr.update(visible=True)}

def get_music(prompt, musicAI_indx, duration):
    mp3file_name = "audio.mp3"
    wavfile_name = "audio.wav"
    if musicAI_indx == 0: 
        if duration == 5:
            width_duration=312
        else :
            width_duration = 312 + ((int(duration)-5) * 128)
        spec = pipe2(prompt, height=312, width=width_duration).images[0]
        print(spec)
        wav = wav_bytes_from_spectrogram_image(spec)
        with open(wavfile_name, "wb") as f:
            f.write(wav[0].getbuffer())
        
          
        #Convert to mp3, for video merging function
        wavfile = AudioSegment.from_wav(wavfile_name)
        wavfile.export(mp3file_name, format="mp3")
        return spec, mp3file_name
    else: 
        #result = text_to_music(prompt, fn_index=0)
        result = generate_track_by_prompt(prompt)
        
        print(f"""—————
        NEW RESULTS
        prompt : {prompt}
        music : {result}
        ———————
        """)
        
        url = result
        
        data = urllib.request.urlopen(url)
        
        f = open(mp3file_name,'wb')
        f.write(data.read())
        f.close()
        
        #Convert to wav, for sharing function only supports wav file
        mp3file = AudioSegment.from_mp3(mp3file_name)
        mp3file.export(wavfile_name, format="wav")
        
        return None, mp3file_name 

  
def merge_video(mp3file_name, image):
    print('wav audio converted to mp3 audio' )
    print('now getting duration of this mp3 audio' )
    #getting audio clip's duration
    audio_length = int(MP3(mp3file_name).info.length)
    print('Audio length is :',audio_length)
    
    file_name = 'video_no_audio.mp4'
    fps = 12
    slide_time = audio_length
    fourcc = cv2.VideoWriter.fourcc(*'MJPG')

    #W, H should be the same as input image
    out = cv2.VideoWriter(file_name, fourcc, fps, (576, 576))
    
    # for image in img_list:
    #     cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #     for _ in range(slide_time * fps):
    #         #cv_img = cv2.resize(np.array(cv_img), (1024, 1024))
    #         out.write(cv_img)

    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for _ in range(slide_time * fps):
        #cv_img = cv2.resize(np.array(cv_img),
        out.write(cv_img)
        
    out.release()
    
    
    #String a list of images into a video and write to memory
    print('video clip created successfully from images') 
        
    # loading video file
    print('Starting video and audio merge')
    videoclip = VideoFileClip(file_name) #("/content/gdrive/My Drive/AI/my_video1.mp4")
    print('loading video-clip')
       
    # loading audio file
    audioclip = AudioFileClip(mp3file_name) #.subclip(0, 15)
    print('loading mp3-format audio')  
    # adding audio to the video clip
    mergedclip = videoclip.set_audio(audioclip)
    print('video and audio merged successfully')  
    
    #Getting size and frame count of merged video file
    print('Getting size and frame count of merged video file')
    duration = mergedclip.duration
    frame_count = mergedclip.fps 
    print('duration is:',duration)
    print('frame count :', frame_count)
    mergedclip.to_videofile('mergedvideo.mp4')
    return 'mergedvideo.mp4'

def change_music_generator(current_choice):
    if current_choice == 0:
        return gr.update(visible=True)
    return gr.update(visible=False)

title="文生图生音乐视频 Text to Image to Music to Video with Riffusion"

description="An AI art generation pipeline, which supports text-to-image-to-music task."

css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .prompt h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
        }
        #share-btn * {
            all: unset;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
"""

block = gr.Blocks(css=css)

examples = [
    [
        '蒙娜丽莎，赛博朋克，宝丽来，33毫米',
        '蒸汽波艺术(Vaporwave)'
    ],
    [
        '一条由闪电制成的令人敬畏的龙',
        '概念艺术(Conceptual Art)'
    ],
    [
        'An awesome dragon made of lightning',
        '概念艺术(Conceptual Art)'
    ],
    [
        '少女在时代广场，舞蹈',
        '写实风格(Realistic style)'
    ],
    [
        'Peking Opera at New York',
        '默认(Default)'
    ],
    [
        '古风少女',
        '水彩(Watercolor)'
    ],
    [
        '辐射游戏角色',
         '默认(Default)'
    ],
    [
        'Fallout game character',
         '默认(Default)'
    ],
    [
        'Traditional Chinese Painting',
        '古风(Ancient Style)'
    ],
    [
        '原神游戏截图，pixiv, 二次元绘画作品',
        '二次元(Anime)'
    ],
    [
        'Genshin Impact Game Screenshot, pixiv, Anime Painting Artworks',
        '二次元(Anime)'
    ],
    [
        '原神角色设定, 哪吒, pixiv, 二次元绘画',
        '二次元(Anime)'
    ],
    [
        'Genshin Impact Character Design, Harry Potter, pixiv, Anime Painting',
        '二次元(Anime)'
    ],
    [
        '巨狼，飘雪，蓝色大片烟雾，毛发细致，烟雾缭绕，高清，3d，cg感，侧面照',
         '默认(Default)'
    ],
    [
        '汉服少女，中国山水画，青山绿水，溪水长流，古风，科技都市，丹青水墨，中国风',
         '赛博朋克(Cyberpunk)'
    ],
    [
        '戴着墨镜的赛博朋克女孩肖像，在夕阳下的城市中, 油画风格',
        '赛博朋克(Cyberpunk)'
    ],
    [
        'Portrait of a cyberpunk girl with sunglasses, in the city sunset, oil painting',
        '赛博朋克(Cyberpunk)'
    ],
    [
        '暗黑破坏神',
         '默认(Default)'
    ],
    [
        '火焰，凤凰，少女，未来感，高清，3d，精致面容，cg感，古风，唯美，毛发细致，上半身立绘',
         '默认(Default)'
    ],
    [
        '浮世绘日本科幻哑光绘画，概念艺术，动漫风格神道寺禅园英雄动作序列，包豪斯',
         '默认(Default)'
    ],
    [
        '一只猫坐在椅子上，戴着一副墨镜,海盗风格',
        '默认(Default)'
    ],
    [
        '稲妻で作られた畏敬の念を抱かせる竜、コンセプトアート',
        '油画(Oil painting)'
    ],
    [
        '번개로 만든 경외스러운 용, 개념 예술',
        '油画(Oil painting)'
    ],
    [
        '梵高猫头鹰',
        '蒸汽波艺术(Vaporwave)'
    ],
    [
        '萨尔瓦多·达利描绘古代文明的超现实主义梦幻油画',
        '写实风格(Realistic style)'
    ],
    [
        '夕阳日落时，阳光落在云层上，海面波涛汹涌，风景，胶片感',
        '默认(Default)'
    ],
    [
        'Sunset, the sun falls on the clouds, the sea is rough, the scenery is filmy',
        '油画(Oil painting)'
    ],
    [
        '夕日が沈むと、雲の上に太陽の光が落ち、海面は波が荒く、風景、フィルム感',
        '油画(Oil painting)'
    ],
    [
        '석양이 질 때 햇빛이 구름 위에 떨어지고, 해수면의 파도가 용솟음치며, 풍경, 필름감',
        '油画(Oil painting)'
    ],
]

with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                  margin-bottom: 10px;
                  margin-left: 220px;
                  justify-content: center;
                "
              >
              </div> 
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                  margin-bottom: 10px;
                  justify-content: center;
                ">
              <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 15px;">文生图生音乐视频</h1>
              </div> 
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                  margin-bottom: 10px;
                  justify-content: center;
                ">
               <h1 style="font-weight: 900; margin-bottom: 7px;">Text to Image to Music to Video</h1>
              </div> 
              <p style="margin-bottom: 10px; font-size: 94%">
                Powered by <a href="https://huggingface.co/riffusion/riffusion-model-v1" target="_blank">Riffusion Model V1</a>, <a href="https://huggingface.co/spaces/Mubert/Text-to-Music" target="_blank">Mubert AI</a>, <a href="https://huggingface.co/spaces/runwayml/stable-diffusion-v1-5" target="_blank">Stable Diffusion V1.5</a>, <a href="https://huggingface.co/spaces/pharma/CLIP-Interrogator" target="_blank">CLIP Interrogator</a>, fffiloni's <a href="https://huggingface.co/spaces/fffiloni/spectrogram-to-music" target="_blank">Riffusion Text-to-Music</a> and Baidu Language Translation projects
              </p>
            </div>
        """
    )
    
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt, multiple languages are supported now.",
                    elem_id="input-prompt",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )

                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )
        language_tips_text = gr.Textbox(label="language tips", show_label=False, visible=False, max_lines=1)
        styles = gr.Dropdown(label="风格(style)", choices=['古风(Ancient Style)', '油画(Oil painting)', '水彩(Watercolor)', 
        '卡通(Cartoon)', '二次元(Anime)', '浮世绘(Ukiyoe)', '蒸汽波艺术(Vaporwave)', 'low poly', 
        '像素风格(Pixel Style)', '概念艺术(Conceptual Art)', '未来主义(Futurism)', '赛博朋克(Cyberpunk)', '写实风格(Realistic style)', 
        '洛丽塔风格(Lolita style)', '巴洛克风格(Baroque style)', '超现实主义(Surrealism)', '默认(Default)'], value='默认(Default)', type="index")
        musicAI = gr.Dropdown(label="音乐生成技术(AI Music Generator)", choices=['Riffusion', 'Mubert AI'], value='Riffusion', type="index")
        duration_input = gr.Slider(label="Duration in seconds", minimum=5, maximum=10, step=1, value=5, elem_id="duration-slider", visible=True)
        status_text = gr.Textbox(
            label="处理状态(Process status)",
            show_label=True,
            max_lines=1,
            interactive=False
        )
        

    with gr.Column(elem_id="col-container"):
        with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=False)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
            share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)

    share_button.click(None, [], [], _js=share_js)

    video_result = gr.Video(type=None, label='Final Merged video', elem_id="output-video")
    imgfile_result = gr.Image(label="Art Cover", elem_id="output-img")
    musicfile_result = gr.Audio(type='filepath', label="Generated Music Track", elem_id="output-music")
    spec_result = gr.Image(label="Spectrogram Image")
    
    trigger_component = gr.Textbox(vaule="", visible=False) # This component is used for triggering inference funtion.
    translated_language = gr.Textbox(vaule="", visible=False)
    
    
    ex = gr.Examples(examples=examples, fn=translate_language_example, inputs=[text, styles], outputs=[language_tips_text, status_text, trigger_component, translated_language], cache_examples=False)
    ex.dataset.headers = [""]

    
    musicAI.change(fn=change_music_generator, inputs=[musicAI], outputs=[duration_input])
    text.submit(translate_language, inputs=[text], outputs=[language_tips_text, status_text, trigger_component, translated_language])
    btn.click(translate_language, inputs=[text], outputs=[language_tips_text, status_text, trigger_component, translated_language])
    trigger_component.change(fn=get_result, inputs=[translated_language, styles, musicAI, duration_input], outputs=[spec_result, imgfile_result, musicfile_result, video_result, status_text, share_button, community_icon, loading_icon])


    gr.Markdown(
        """  
  Space by [@DGSpitzer](https://www.youtube.com/channel/UCzzsYBF4qwtMwJaPJZ5SuPg)❤️ [@大谷的游戏创作小屋](https://space.bilibili.com/176003)
  [![Twitter Follow](https://img.shields.io/twitter/follow/DGSpitzer?label=%40DGSpitzer&style=social)](https://twitter.com/DGSpitzer)
  ![visitors](https://visitor-badge.glitch.me/badge?page_id=dgspitzer_txt2img2video)
        """
    )
    gr.HTML('''
    <div class="footer">
                <p>Model：<a href="https://huggingface.co/riffusion/riffusion-model-v1" style="text-decoration: underline;" target="_blank">Riffusion</a>
                </p>
    </div>
    ''')
    
#block.queue().launch(server_port=8000)


app = FastAPI()
@app.get("/")
def read_main():
    return {"message": "This is your main app"}

app = gr.mount_gradio_app(app, block, path="/gradio")


