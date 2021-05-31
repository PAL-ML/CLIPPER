from sklearn.metrics import jaccard_score, hamming_loss, accuracy_score, f1_score, average_precision_score, precision_score, recall_score, top_k_accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

import clip
import torch
import numpy as np

class CLIPFewShotClassifier:
    def __init__(self, model_parameters):
        self.num_classes = model_parameters["num_classes"]
        self.input_dims = model_parameters["input_dims"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('ViT-B/32', self.device)

    def zeroshot_classifier(self, classnames):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [classname] #format with class
                texts = clip.tokenize(texts).cuda() #tokenize
                class_embeddings = self.model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    def predict_label(self, img_feats, zs_weights):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img_feats = torch.from_numpy(img_feats)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        img_feats = img_feats.to(device)

        similarity = (100.0 * img_feats @ zs_weights).softmax(dim=-1)

        values, preds = similarity.topk(1)
        preds = preds.cpu().squeeze().numpy()
        return preds


    def predict_scores(self, img_feats, zs_weights):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img_feats = torch.from_numpy(img_feats)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        img_feats = img_feats.to(device)

        similarity = (100.0 * img_feats @ zs_weights).softmax(dim=-1)
        return similarity.cpu().numpy()

    def evaluate_single_label_metrics(
        self, 
        x, 
        y, 
        label_mapping,
        zs_weights,
        metrics=['accuracy', 'ap', 'map', 'c_f1', 'o_f1', 'c_precision', 'o_precision', 'c_recall', 'o_recall', 'top1_accuracy', 'top5_accuracy', 'classwise_accuracy', 'c_accuracy']
    ):

        pred = self.predict_label(x, zs_weights)
        pred_mapped = [label_mapping[p] for p in pred]
        
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y)
        pred_bin = lb.transform(pred_mapped)

        ## For MAP
        # can be optimised by using a common process for all metrics,
        # instead of calculating y_map and y_bin separately.
        # Not doing that now to avoid unforseen changes in the other metrics.
        # TODO: Why pred scores has more classes than y
        pred_scores = self.predict_scores(x, zs_weights) # logits after sigmoid
        reverse_label_mapping = {label_mapping[v]:v for v in label_mapping}
        y_scores = np.zeros((len(y), len(label_mapping))) # one hot encoding of y
        for idx_l, label in enumerate(y):
            y_scores[idx_l, reverse_label_mapping[label]] = 1
        
        metric_values = {}

        if 'accuracy' in metrics:
            accuracy = accuracy_score(y, pred_mapped)
            metric_values['accuracy'] = accuracy
        if 'ap' in metrics:
            ap_value = average_precision_score(y_scores, pred_scores)
            metric_values['ap'] = ap_value
        if 'map' in metrics:
            map_value = average_precision_score(
                y_scores, pred_scores, average='macro'
                )
            metric_values['map'] = map_value
        if 'c_f1' in metrics:
            c_f1_value = f1_score(y, pred_mapped, average='macro')
            metric_values['c_f1'] = c_f1_value
        if 'o_f1' in metrics:
            o_f1_value = f1_score(y, pred_mapped, average='micro')
            metric_values['o_f1'] = o_f1_value
        if 'c_precision' in metrics:
            c_precision_value = precision_score(y, pred_mapped, average='macro', zero_division=0)
            metric_values['c_precision'] = c_precision_value
        if 'o_precision' in metrics:
            o_precision_value = precision_score(y, pred_mapped, average='micro', zero_division=0)
            metric_values['o_precision'] = o_precision_value
        if 'c_recall' in metrics:
            c_recall_value = recall_score(y, pred_mapped, average='macro', zero_division=0)
            metric_values['c_recall'] = c_recall_value
        if 'o_recall' in metrics:
            o_recall_value = recall_score(y, pred_mapped, average='micro', zero_division=0)
            metric_values['o_recall'] = o_recall_value
        if 'top1_accuracy' in metrics:
            # y_single = [reverse_label_mapping[label_list[0]] for label_list in y]
            top1_accuracy_value = top_k_accuracy_score(
                y, pred_scores, k=1
                )
            metric_values['top1_accuracy'] = top1_accuracy_value
        if 'top5_accuracy' in metrics:
            # y_single = [reverse_label_mapping[label_list[0]] for label_list in y]
            top5_accuracy_value = top_k_accuracy_score(
                y, pred_scores, k=5
                )
            metric_values['top5_accuracy'] = top5_accuracy_value
        if 'classwise_accuracy' in metrics:
            classwise_accuracy_value = self._get_classwise_accuracy(
                y_bin, pred_bin
                )
            metric_values['classwise_accuracy'] = classwise_accuracy_value
        if 'c_accuracy' in metrics:
            c_accuracy_value = self._get_c_accuracy(y_bin, pred_bin)
            metric_values['c_accuracy'] = c_accuracy_value

        return metric_values

    def _get_classwise_accuracy(self, y_bin, pred_bin):
        classwise_accuracy = []

        for col in range(y_bin.shape[1]):
            col_values = y_bin[:, col] == pred_bin[:, col]
            col_acc = np.sum(col_values)/len(col_values)
            classwise_accuracy.append(col_acc)

        return classwise_accuracy

    def _get_c_accuracy(self, y_bin, pred_bin):
        classwise_accuracy = self._get_classwise_accuracy(y_bin, pred_bin)
        c_accuracy = np.mean(classwise_accuracy)

        return c_accuracy