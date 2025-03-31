import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
import psutil  # For memory checks

responseForInternalCall = {
    "userId":"" ,
    "documentId": "",
    "response":""

}


responseForChatCalls = {  # Corrected variable name
    "messageId": "",
    "userId": "",
    "response": ""
}




prompt = (
        f"You are a functional helper for a cancer qna bot. Remember you only give responses in json format nothing else. "
        f"Whatever response you have fill the keys in the sample json provided {responseForChatCalls} {responseForInternalCall}\n\n"
        "User: Can you find what are the stages of breast cancer?"
    )
def check_memory():
    """Check available memory and log warnings if insufficient."""
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  
    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    logging.info(f"Total Memory: {total_memory:.2f} GB, Available Memory: {available_memory:.2f} GB")
    if available_memory < 2:  
        logging.warning("Low available memory. Model loading might fail.")

def main():
    logging.basicConfig(level=logging.INFO)  # Enable logging
    logging.info("Starting the program...")
    check_memory()  # Check memory before loading the model

    torch.random.manual_seed(0)
    model_path = "microsoft/Phi-4-mini-instruct"
    try:

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS backend for Apple Silicon.")
        else:
            device = torch.device("cpu")
            logging.info("MPS not available. Falling back to CPU.")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=None, 
            torch_dtype=torch.float32, 
            trust_remote_code=True
        ).to(device)  
        logging.info("Model loaded successfully.")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {str(e)}")
        return

    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device.index if device.type != "cpu" else -1  # Align pipeline device with model
        )
        logging.info("Pipeline initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing pipeline: {str(e)}")
        return

    generation_args = {
        "max_new_tokens": 1000,
        "return_full_text": False,
        "do_sample": False, 
    }
    try:
        logging.info("Generating response...")
        output = pipe(prompt, **generation_args)
        print(f"response generated here {output}")  
        if output and "generated_text" in output[0]:
            logging.info("Response generated successfully.")
            print(output[0]['generated_text'])
        else:
            logging.warning("Unexpected output format.")
            print("Unexpected output format:", output)
    except Exception as e:
        logging.error(f"Error during text generation: {str(e)}")

if __name__ == "__main__":
    main()