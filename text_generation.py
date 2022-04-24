# Thank you to COMP474 Winter 2022 team for the tutorial!
from transformers import TFAutoModelWithLMHead, AutoTokenizer, pipeline


# Give some text to a pretrained model in an NLP pipeline and ask it to generate more text similar to it.

text_generator = pipeline("text-generation")
model = TFAutoModelWithLMHead.from_pretrained("xlnet-base-cased")
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

text = """In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria) are discovered. The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the remainder of the story. 1883 Western Siberia, a young Grigori Rasputin is asked by his father and a group of men to perform magic. Rasputin has a vision and denounces one of the men as a horse thief. Although his father initially slaps him for making such an accusation, Rasputin watches as the man is chased outside and beaten. Twenty years later, Rasputin sees a vision of the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous, with people, even a bishop, begging for his blessing."""
prompt = "Today the weather is really nice and I am planning on "

# turn text into sequences of integers (ids)
inputs = tokenizer.encode(text + prompt, add_special_tokens=False, return_tensors="tf")

prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))

# generate the expanded text
outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
generated = prompt + tokenizer.decode(outputs[0])[prompt_length:]
print(generated)