from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import os
CUDA_VISIBLE_DEVICES=0


def summary(src_text,min_len,max_len):

    Model='google/pegasus-xsum'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    p_tokenizer=PegasusTokenizer.from_pretrained(Model)
    p_model=PegasusForConditionalGeneration.from_pretrained(Model).to(device)
    batch = p_tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
    s_encoded = p_model.generate(**batch,min_length=min_len,max_length=max_len, num_beams=5)
    s_decoded = p_tokenizer.batch_decode(s_encoded, skip_special_tokens=True)
    print(s_decoded)

cond='y'
while(cond=='y'):

    src_text=input('Enter text for summarization:')  

    while(1):
        spec_len=input('Do you want specific summary length (default minimum lenght = 30 & maximum length = 50) (y/n):')
        if spec_len=='y':
            min_len=input('please enter the minimum length:')
            max_len=input('please enter the maximum length:')
            if min_len.isdigit() and max_len.isdigit():            
                min_len=int(min_len)
                max_len=int(max_len)
                break
            else:print('Both lengths should be integers')
        elif spec_len=='n':
            min_len=30
            max_len=50
            break
        else:print('please tell me')
    
    summary(src_text,min_len,max_len)

    while(1):
        cond=input('do you want to summarize another text (y/n):')
        if cond=='y' or cond=='n':
            break
        else:print('please tell me')