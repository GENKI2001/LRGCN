import torch
import torch.nn.functional as F


def fit(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(
        data.x,
        data.edge_index,
        edge_weight=None,                 # GCN 向けに明示
        adj=getattr(data, "adj", None),   # 使うモデルだけが受け取って使う
        adj2=getattr(data, "adj2", None),
        x_label=getattr(data, "labelx", None),
        train_mask=getattr(data, "train_mask", None),
    )
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    
    if hasattr(model, 'kl_loss') and model.kl_loss is not None:
        loss = loss + model.kl_loss
    
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def test(model, data):
    model.eval()
    logits = model(
        data.x,
        data.edge_index,
        adj=data.adj,
        adj2=data.adj2,
        edge_weight=None,
        x_label=getattr(data, "labelx", None),
        train_mask=getattr(data, "train_mask", None),
    )
    preds = logits.argmax(dim=1)

    def acc(mask):
        mask = mask.bool()
        return float((preds[mask] == data.y[mask]).sum().item() / int(mask.sum()))

    return {
        "train": acc(data.train_mask),
        "val": acc(data.val_mask),
        "test": acc(data.test_mask),
    }

