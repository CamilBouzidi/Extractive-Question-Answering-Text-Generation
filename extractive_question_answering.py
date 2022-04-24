# Thank you to COMP474 Winter 2022 team!
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering, pipeline
import tensorflow as tf

nlp = pipeline("question-answering")

# use a BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# generate text and questions
text = """In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria) are discovered. The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the remainder of the story. 1883 Western Siberia, a young Grigori Rasputin is asked by his father and a group of men to perform magic. Rasputin has a vision and denounces one of the men as a horse thief. Although his father initially slaps him for making such an accusation, Rasputin watches as the man is chased outside and beaten. Twenty years later, Rasputin sees a vision of the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous, with people, even a bishop, begging for his blessing."""
questions = ["Who became famous?","What was discovered in 1991?"]


for question in questions:
    #tokenize the question
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="tf")
    #Get the ids array details
    input_ids = inputs["input_ids"].numpy()[0]
    #Instantiates the model classes of the library (with a question answering head) from a configuration.
    outputs = model(inputs)
    #Get the start scores
    answer_start_scores = outputs.start_logits
    #Get the end scores
    answer_end_scores = outputs.end_logits
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]  
    # Get the most likely end of answer with the argmax of the score
    answer_end = (tf.argmax(answer_end_scores, axis=1) + 1).numpy()[0]  
    #Select the ids based on the scores and convert them into strings.
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    #Print the question and result
    print(f"Question: {question}")
    print(f"Answer: {answer}")
