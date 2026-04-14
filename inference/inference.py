import torch

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    if max_length is None:
        max_length = supported_context_length

    input_ids = input_ids[: min(max_length, supported_context_length)]

    if len(input_ids) < max_length:
        input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    non_pad_mask = (input_tensor != pad_token_id)
    last_idx = non_pad_mask.sum(dim=1) - 1
    last_idx_item = last_idx.item()

    with torch.no_grad():
        all_logits = model(input_tensor)
        logits = all_logits[0, last_idx_item, :]

    predicted_label = torch.argmax(logits).item()
    return "spam" if predicted_label == 1 else "not spam"
