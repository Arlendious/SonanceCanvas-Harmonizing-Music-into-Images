import gradio as gr
import os 
hf_token = os.environ.get('*****')

lpmc_client = gr.load("seungheondoh/LP-Music-Caps-demo", src="spaces")

from gradio_client import Client

client = Client("https://fffiloni-test-llama-api-debug.hf.space/", hf_token=hf_token)
lyrics_client = Client("https://fffiloni-music-to-lyrics.hf.space/")
visualizer_client = Client("https://fffiloni-animated-audio-visualizer.hf.space/")

from share_btn import community_icon_html, loading_icon_html, share_js

from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         torch_dtype=torch.float16, 
                                         use_safetensors=True, 
                                         variant="fp16")
pipe.to("cuda")

compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True]
)

#pipe.enable_model_cpu_offload()

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

from pydub import AudioSegment

def cut_audio(input_path, output_path, max_duration):
    audio = AudioSegment.from_file(input_path)

    if len(audio) > max_duration:
        audio = audio[:max_duration]

    audio.export(output_path, format="mp3")

    return output_path

def get_text_after_colon(input_text):
    # Find the first occurrence of ":"
    colon_index = input_text.find(":")
    
    # Check if ":" exists in the input_text
    if colon_index != -1:
        # Extract the text after the colon
        result_text = input_text[colon_index + 1:].strip()
        return result_text
    else:
        # Return the original text if ":" is not found
        return input_text


def solo_xd(prompt):
    images = pipe(prompt=prompt).images[0]
    return images

def get_visualizer_video(audio_in, image_in, song_title):

    title = f"""{song_title.upper()}\nMusic-to-Image demo by @fffiloni | HuggingFace
    """

    visualizer_video = visualizer_client.predict(
    				title,	# str in 'title' Textbox component
    				audio_in,	# str (filepath or URL to file) in 'audio_in' Audio component
    				image_in,	# str (filepath or URL to image) in 'image_in' Image component
                    "my_music_to_image_awesome_video.mp4",
    				api_name="/predict"
    )
    
    return visualizer_video[0]

def infer(audio_file, has_lyrics):
    print("NEW INFERENCE ...")
    gr.Info('Truncating your audio to the first 30 seconds')
    truncated_audio = cut_audio(audio_file, "trunc_audio.mp3", 30000)
    processed_audio = truncated_audio

    print("Calling LP Music Caps...")
    gr.Info('Calling LP Music Caps...')
    cap_result = lpmc_client(
    				truncated_audio,	# str (filepath or URL to file) in 'audio_path' Audio component
    				api_name="predict"
    )
    print(f"MUSIC DESC: {cap_result}")

    if has_lyrics == "Yes" : 
        print("""———
        Getting Lyrics ...
        Note: We only take the first minute of the song
        """)
        truncated_lyrics = cut_audio(audio_file, "trunc_lyrics.mp3", 60000)
        gr.Info("Getting Lyrics ...")
        lyrics_result = lyrics_client.predict(
        				truncated_lyrics,	# str (filepath or URL to file) in 'Song input' Audio component
        				fn_index=0
        )
        print(f"LYRICS: {lyrics_result}")
    
        llama_q = f"""
        I'll give you a music description + the lyrics of the song. 
        Give me an image description that would fit well with the music description, reflecting the lyrics too. 
        Be creative, do not do list, just an image description as required. Try to think about human characters first.
        Your image description must fit well for a stable diffusion prompt.
    
        Here's the music description :
    
        « {cap_result} »
    
        And here are the lyrics : 
    
        « {lyrics_result} »
        
        """
    elif has_lyrics == "No" : 
        
        llama_q = f"""
        I'll give you a music description. 
        Give me an image description that would fit well with the music description. 
        Be creative, do not do list, just an image description as required. Try to think about human characters first.
        Your image description must fit well for a stable diffusion prompt.
    
        Here's the music description :
    
        « {cap_result} »
        """
    print("""———
    Calling Llama2 ...
    """)
    gr.Info("Calling Llama2 ...")
    result = client.predict(
    				llama_q,	# str in 'Message' Textbox component
                    "M2I",
    				api_name="/predict"
    )    
    
    result = get_text_after_colon(result)

    print(f"Llama2 result: {result}")
    print("""———
    Calling SD-XL ...
    """)
    gr.Info('Calling SD-XL ...')
    prompt = result
    conditioning, pooled = compel(prompt)
    images = pipe(prompt_embeds=conditioning, pooled_prompt_embeds=pooled).images[0]

    print("Finished")
    
    return processed_audio, images, result, gr.update(visible=True), gr.Group.update(visible=True)

css = """
#col-container {max-width: 780px; margin-left: auto; margin-right: auto;}
a {text-decoration-line: underline; font-weight: 600;}
.animate-spin {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  from {
      transform: rotate(0deg);
  }
  to {
      transform: rotate(360deg);
  }
}
#share-btn-container {
  display: flex; 
  padding-left: 0.5rem !important; 
  padding-right: 0.5rem !important; 
  background-color: #000000; 
  justify-content: center; 
  align-items: center; 
  border-radius: 9999px !important; 
  max-width: 13rem;
}
div#share-btn-container > div {
    flex-direction: row;
    background: black;
    align-items: center;
}
#share-btn-container:hover {
  background-color: #060606;
}
#share-btn {
  all: initial; 
  color: #ffffff;
  font-weight: 600; 
  cursor:pointer; 
  font-family: 'IBM Plex Sans', sans-serif; 
  margin-left: 0.5rem !important; 
  padding-top: 0.5rem !important; 
  padding-bottom: 0.5rem !important;
  right:0;
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
#share-btn-container.hidden {
  display: none!important;
}
.footer {
    margin-bottom: 45px;
    margin-top: 10px;
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
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""<div style="text-align: center; max-width: 700px; margin: 0 auto;">
                <div
                style="
                    display: inline-flex;
                    align-items: center;
                    gap: 0.8rem;
                    font-size: 1.75rem;
                "
                >
                <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
                    Music To Image
                </h1>
                </div>
                <p style="margin-bottom: 10px; font-size: 94%">
                Sends an audio into <a href="https://huggingface.co/spaces/seungheondoh/LP-Music-Caps-demo" target="_blank">LP-Music-Caps</a>
                to generate a audio caption which is then translated to an illustrative image description with Llama2, and finally run through 
                Stable Diffusion XL to generate an image from the audio ! <br /><br />
                Note: Only the first 30 seconds of your audio will be used for inference.
                </p>
            </div>""")
        
        audio_input = gr.Audio(label="Music input", type="filepath", source="upload")
        
        with gr.Row():
            has_lyrics = gr.Radio(label="Does your audio has lyrics ?", choices=["Yes", "No"], value="No", info="If yes, the image should reflect the lyrics, but be aware that because we add a step (getting lyrics), inference will take more time.")
            song_title = gr.Textbox(label="Song Title", placeholder="Title: ", interactive=True, info="If you want to share your result, please provide the title of your audio sample :)", elem_id="song-title")
        
        infer_btn = gr.Button("Generate Image from Music")
        #lpmc_cap = gr.Textbox(label="Lp Music Caps caption")
        
        with gr.Group():
           
            with gr.Row():
                
                llama_trans_cap = gr.Textbox(label="Llama Image Suggestion", placeholder="Llama2 image prompt suggestion will be displayed here ;)", visible=True, lines=12, max_lines=18, elem_id="llama-prompt")
                
                with gr.Tab("Image Result"):
                    img_result = gr.Image(label="Image Result", elem_id="image-out", interactive=False, type="filepath")
                
                with gr.Tab("Video visualizer"):
                    
                    with gr.Column():
                        processed_audio = gr.Audio(type="filepath", visible=False)
                        visualizer_video = gr.Video(label="Video visualizer output")
                        get_visualizer_vid = gr.Button("Export as video !")
        
        with gr.Row():
            
            tryagain_btn = gr.Button("Try another image ?", visible=False)
            
            with gr.Group(elem_id="share-btn-container", visible=False) as share_group:
                    community_icon = gr.HTML(community_icon_html)
                    loading_icon = gr.HTML(loading_icon_html)
                    share_button = gr.Button("Share to community", elem_id="share-btn")
        
        gr.Examples(examples=[["./examples/electronic.mp3", "No"],["./examples/folk.wav", "No"], ["./examples/orchestra.wav", "No"]],
                    fn=infer,
                    inputs=[audio_input, has_lyrics],
                    outputs=[processed_audio, img_result, llama_trans_cap, tryagain_btn, share_group],
                    cache_examples=True
                   )

    #infer_btn.click(fn=infer, inputs=[audio_input], outputs=[lpmc_cap, llama_trans_cap, img_result])
    infer_btn.click(fn=infer, inputs=[audio_input, has_lyrics], outputs=[processed_audio, img_result, llama_trans_cap, tryagain_btn, share_group])
    share_button.click(None, [], [], _js=share_js)
    tryagain_btn.click(fn=solo_xd, inputs=[llama_trans_cap], outputs=[img_result])
    get_visualizer_vid.click(fn=get_visualizer_video, inputs=[processed_audio, img_result, song_title], outputs=[visualizer_video], queue=False)

demo.queue(max_size=20).launch()
