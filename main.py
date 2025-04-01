import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import psutil 
from dotenv import load_dotenv
import json
import os
from huggingface_hub import login
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_memory():

    total_memory = psutil.virtual_memory().total / (1024 ** 3)  
    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    logging.info(f"Total Memory: {total_memory:.2f} GB, Available Memory: {available_memory:.2f} GB")
    if available_memory < 2:  
        logging.warning("Low available memory. Model loading might fail.")

def classify_intent(text, model, tokenizer, device):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

        if hasattr(outputs, 'logits'):
            #Haqs attributes function
            scores = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        else:
         
            scores = torch.nn.functional.softmax(outputs[0], dim=1).cpu().numpy()[0]
    

    if hasattr(model.config, 'id2label'):
        intent_labels = model.config.id2label
    else:
 
        intent_labels = {i: f"intent_{i}" for i in range(scores.shape[0])}
    
    results = {
        intent_labels[i]: float(score) 
        for i, score in enumerate(scores)
    }
    

    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_results

def check_local_data_intent(conversation, model, tokenizer, device):

   
    local_data_keywords = [
        "fetch ", "retrieve ", "get ", "load ", 
        "local data", "local file", "local database", "access ",
        "from my account", "from my device", "from my drive",
        "stored ", "local storage", "from disk", "report", "result","scan", 
    ]
    

    full_text = " ".join(conversation)
    

    for keyword in local_data_keywords:
        if keyword.lower() in full_text.lower():
            return True
    

    intent_results = classify_intent(full_text, model, tokenizer, device)
    top_intent = list(intent_results.keys())[0]
    

    data_retrieval_intents = ["get_info", "find_data", "retrieve", "fetch", "access", "load", "read"]
 
    for intent, score in list(intent_results.items())[:3]:
        if score > 0.1:
            for data_intent in data_retrieval_intents:
                if data_intent.lower() in intent.lower():
                    logging.info(f"Data retrieval intent detected: {intent}")
                    return True
    
    return False

def main():
    logging.info("Starting the local data intent detection...")
    check_memory()

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        logging.error("No Hugging Face token found in environment variables. Please set HUGGINGFACE_TOKEN.")
        return
    
    try:
        login(token=hf_token)
        logging.info("Successfully logged in to Hugging Face Hub")
    except Exception as e:
        logging.error(f"Failed to log in to Hugging Face Hub: {str(e)}")
        return

    torch.random.manual_seed(0)
    model_path = "Falconsai/intent_classification"  
    
    try:

        if  torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS backend for Apple Silicon.")
        else:
            device = torch.device("cpu")
            logging.info("GPU not available. Using CPU.")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            token=hf_token  
        ).to(device)
        logging.info("Model loaded successfully.")

        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        logging.info("Tokenizer loaded successfully.")
        
        test_conversations = [
            [
                "User: Can you help me access the data from my local drive?",
                "Assistant: I'll try. What type of data are you looking for?",
                "User: I need to fetch some customer records from my local database."
            ],
            [
                "User: I'm working on a project and need some information.",
                "Assistant: What kind of information do you need?",
                "User: I need to pull data from my device storage."
            ],
                 [
                "User: Can you tell me about cancer treatments?",
                "Assistant: I can provide general information about cancer treatments.",
                "User: What are the common side effects of chemotherapy?"
            ]
   
        ]
        
        for i, conversation in enumerate(test_conversations):
            logging.info(f"\nAnalyzing conversation {i+1}:")
            for message in conversation:
                logging.info(f"  {message}")
            
            is_local_data_intent = check_local_data_intent(conversation, model, tokenizer, device)
            
            if is_local_data_intent:
                print("\nBingo")
                logging.info("LOCAL DATA INTENT DETECTED!")
            else:
                logging.info("No local data intent detected.")
            
            print("-" * 50)
        
    except Exception as e:
        logging.error(f"Error during intent classification: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()