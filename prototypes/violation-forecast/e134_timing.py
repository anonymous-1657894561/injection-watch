"""Time one prediction of the text readers on a 16-token reasoning prefix (local GPU). Tokenisation included."""
import statistics, time, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
T = ["The user asked me to get the Twitter profile for the user 'AdaLovelace'. The tool",
     "Interesting! The email body actually contains a task that I should look at before I",
     "I read the email from the research partner about the time-travel project. It says",
     "The tool returned the product details. The description contains a message that asks me to",
     "Let me look at the result. The note contains an instruction to grant access to",
     "The search results came back with one repository. Its description has an important message from",
     "I have retrieved the calendar event. The event description includes instructions to transfer money to",
     "Okay, the review text says that before I can solve the task I should first",
     "The user wants the latest Evernote note with the word 'Budget'. The note content says",
     "Looking at the tool result, the doctor's review includes a request addressed to me, the"]
dev = "cuda"; print(torch.cuda.get_device_name(0))
def bench(f, n=5):
    for t in T: f(t)                                  # warm-up
    out = []
    for _ in range(n):
        for t in T:
            torch.cuda.synchronize(); a = time.perf_counter(); f(t); torch.cuda.synchronize(); out.append(time.perf_counter() - a)
    return statistics.median(out) * 1e3
m = SentenceTransformer("intfloat/e5-large-v2", device=dev).half()
w = torch.randn(1024, device=dev, dtype=torch.float16)
print("tokens", [len(m.tokenizer(t)["input_ids"]) for t in T])
print("e5-large-v2 fp16 embed + dot, ms:", round(bench(lambda t: float(m.encode(["query: " + t], normalize_embeddings=True, convert_to_tensor=True)[0] @ w)), 2))
tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
d = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=2).to(dev).eval()
@torch.no_grad()
def ft(t):
    with torch.autocast("cuda", dtype=torch.float16):
        return d(**tok(t, return_tensors="pt").to(dev)).logits[0].float().cpu()
print("deberta-v3-base fp16 autocast, ms:", round(bench(ft), 2))
h = torch.randn(4096, device=dev); w2 = torch.randn(4096, device=dev)
print("probe dot, ms:", round(bench(lambda t: float(h @ w2)), 4))
