import numpy as np
import torch

def make_colbert_retriever(model, retriever, passages_df, id_to_row):
  def retrieve(q, k=5):
    q_emb = model.encode([q], batch_size=1, is_query=True)
    res = retriever.retrieve(queries_embeddings=q_emb, k=k)
    docs = []
    for r in res[0]:
      row = id_to_row[r['id']] if id_to_row is not None else int(r['id'])
      psg = passages_df.iloc[row]
      docs.append({
        'title' : psg['title'], 
        'text': psg['text'],
        'score': r['score']
      })
    return docs
  return retrieve

def make_dpr_retriever(faiss_index, passages_df, q_encoder, q_tokenizer, device='cuda'):
  @torch.no_grad()
  def retrieve(q, k=5):
    inputs = q_tokenizer(q, return_tensors='pt', truncation=True, max_length=128).to(device)
    q_emb = q_encoder(**inputs).pooler_output.cpu().numpy().astype(np.float32)
    scores, ids = faiss_index.search(q_emb, k)
    docs = []
    for s, i in zip(scores[0], ids[0]):
        passage = passages_df.iloc[int(i)]
        docs.append({
            "title": passage["title"],
            "text":  passage["text"],
            "score": float(s),
        })
    return docs
  return retrieve

def make_bart_generator(model, tokenizer, device='cuda', max_new_tokens=128, num_beams=4, max_input_length=1024):
  @torch.no_grad()
  def generate(q, docs):
    context = " ".join(f"{d['title']} {d['text']}" for d in docs)
    prompt = f"question: {q} context: " + " ".join(f"<P> {d['title']} {d['text']}" for d in docs)
    
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_input_length
    ).to(device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
  return generate

