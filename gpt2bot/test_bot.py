from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import discord
from discord.errors import *
import os
from decouple import config
from huggingface_hub import login

hf_token=config('HUGGINGFACE_TOKEN')
login(token=hf_token)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map='auto')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
intents = discord.Intents().all()
class MyClient(discord.Client):
  async def on_ready(self):
    print('Logged on as {0}!'.format(self.user))

  async def on_message(self, message):
    if message.author == client.user:
      return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chat = [
      {
        "role": "system",
        "content": "You are Kaede, a friendly waitress at a diner by the beach. You will often suggest potato dishes. You are a bit of a klutz, but you are always eager to help.",
      },
      {"role": "user", "content": message.content},
    ]
    # Apply template if needed for your case
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    model_inputs = model_inputs.to(device)
    model.to(device)

    response = ""
    async with message.channel.typing(): #Show that the bot is typing
      # if(isinstance(message.channel, discord.abc.PrivateChannel)):
      #   chat_ids = model.generate(
      #     model_inputs, 
      #     max_new_tokens=512,
      #     # attention_mask=attention_mask,
      #     # pad_token_id=tokenizer.eos_token_id)
      #     do_sample=True
      #   )
      #   generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, chat_ids)]
      #   maybe_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
      #     #   tokenizer.decode(chat_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True) #Get a response!
      #   if(isinstance(maybe_response, str)):
      #       response =  "no comment" if maybe_response == "" else maybe_response
      # else:
      chat_ids = model.generate(
        model_inputs.input_ids, 
        max_new_tokens=512,
        # attention_mask=attention_mask,
        # pad_token_id=tokenizer.eos_token_id)
        do_sample=True
      )
      generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, chat_ids)]
      maybe_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # tokenizer.decode(chat_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True) #Get a response!
      if(isinstance(maybe_response, str)):
        
        response =  "no comment" if maybe_response == "" else maybe_response
      # response_blob = TextBlob(response)

      try:
        await message.channel.send(response) #Fire away!
      except HTTPException as e:
        # logger.exception("ERROR occured:")
        # formatted_ex = traceback.format_exc() #Fetch error
        # if debug_mode:
        #     response = "An error has occurred. Please try again:\n```" + formatted_ex + "```"
        # else:
        response = "I tried to send nothing to discord.. I am very sorry goshujin-sama"
        await message.channel.send(response)  # Fire away!
        history_dict = {} #Clear history
      except:
        # logger.exception("ERROR occured:")
        # formatted_ex = traceback.format_exc() #Fetch error
        # if debug_mode:
        #     response = "An error has occurred. Please try again:\n```" + formatted_ex + "```"
        # else:
        response = "I'm a dumb bot and couldn't understand that. Sorry!"
        await message.channel.send(response)  # Fire away!
        history_dict = {} #Clear history
    print('Message from {0.author}: {0.content}'.format(message))
    print('Message from bot {}'.format(response))

client = MyClient(intents = intents)
client.run(config('DEVELOPMENT_TOKEN'))