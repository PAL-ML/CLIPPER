import torch 
import numpy as np
import clip

from tqdm import tqdm

def zeroshot_classifier(model, classnames):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [classname] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def get_topk_labels(output, target, id2label_dict, topk=5):
    pred = np.argpartition(output.cpu().numpy()[0], -topk)[-topk:]
    predicted_labels = list(map(lambda x: id2label_dict[str(x)], pred))
    ground_truth = id2label_dict[str(target.cpu().numpy().item())]
    return predicted_labels, ground_truth

def is_wrongly_labelled(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    is_correct = correct.cpu().numpy().item()
    return not is_correct

def logits_to_pred(output, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    return pred.cpu().numpy()[0].item()