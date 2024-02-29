#!/usr/bin/env python
# -*- coding: utf-8 -*- 

from flask import Flask, render_template, request, redirect, url_for
from langdetect import detect

from transformers import T5ForConditionalGeneration, T5Tokenizer


app = Flask(__name__)

@app.route('/', methods=['GET'])
#trying to detect the language but its not necessary like the 
# facebook model
def detect_language():
    with open('input.txt', 'r') as file:
        file_contents = file.read()
        src_lang_detected = detect(str(file_contents))
    lang_code_src = src_lang_detected
    print(f"input.txt contents:\n{file_contents} \n language detected {src_lang_detected}")
    

    return f""" <p>  input.txt contents:\n{file_contents}$$$
                    \n --> language detected {src_lang_detected} </p>
                    
                    <p>(not necessary) write the following address plus and language code
                        for manual translation if detected language is wrong.
                        Example:</p>
                    
                    <a href="http://127.0.0.1:5000/{src_lang_detected}/fa">
                    http://127.0.0.1:5000/{src_lang_detected}/fa </a>
                    
                    <p>or Just:</p>
                    <a href="http://127.0.0.1:5000/fa"> http://127.0.0.1:5000/fa </a>
 
                    """ 


@app.route('/<string:lang_code_dest>', methods=['GET'])
def language_code_single_route( lang_code_dest):
    with open('input.txt', 'r') as file:
        lines = file.readlines()
        file_contents = ''.join(lines)

    model_name = 'jbochi/madlad400-3b-mt'
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # there is Problem with file_contents which is related
    # to the EndFile charachter that should be Deleted 
    print(file_contents)
    text = f'<2{lang_code_dest}> {file_contents[:-1]}' 

    print(text)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    print(model.device)
    outputs = model.generate(input_ids=input_ids,max_length= 500)

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print(type(output))
    return  output

if __name__=="__main__":
    app.run(port=5000,debug=False)
