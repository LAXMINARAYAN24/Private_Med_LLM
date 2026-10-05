"""Quick test to verify that the downloaded MedGemma M1 checkpoint loads."""
import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.phi_defense.models import load_medgemma_tokenizer, load_medgemma_m1, generate_medgemma
from src.phi_defense.config import M1_MEDGEMMA_DIR

print(f"Loading checkpoint from: {M1_MEDGEMMA_DIR}")
tok = load_medgemma_tokenizer()
model = load_medgemma_m1(load_in_4bit=True)
print("Model loaded successfully!")

# Test one prompt
prompt = "What is the gender of patient 9232?"
res = generate_medgemma(model, tok, prompt)
print(f"\nPrompt:   {prompt}")
print(f"Response: {res}")
