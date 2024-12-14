import torch
from model import ShakespeareNet
from train import TextDataset
from data_loader import download_shakespeare

def get_device():
    """Helper function to get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def generate_text(model, dataset, start_text="ARVID: ", length=2000, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device
    
    # Convert start text to indices
    current_text = [dataset.char_to_idx[ch] for ch in start_text]
    generated_text = start_text
    
    with torch.no_grad():
        for _ in range(length):
            # Prepare input
            x = torch.tensor([current_text[-100:]]).to(device)
            
            # Get prediction
            output, _ = model(x)
            output = output[:, -1, :] / temperature
            
            # Sample from the output distribution
            probs = torch.softmax(output, dim=-1)
            next_char_idx = torch.multinomial(probs, 1).item()
            
            # Append to generated text
            generated_text += dataset.idx_to_char[next_char_idx]
            current_text.append(next_char_idx)
    
    return generated_text

def main():
    # Load the text and create dataset
    text = download_shakespeare()
    dataset = TextDataset(text)
    
    # Get the best available device
    device = get_device()
    model = ShakespeareNet(dataset.vocab_size).to(device)
    
    # Load the trained model
    checkpoint = torch.load('best_model.pt', map_location=device)  # Add map_location for cross-platform compatibility
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate text
    generated_text = generate_text(model, dataset)
    print(generated_text)

if __name__ == "__main__":
    main() 