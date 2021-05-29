import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
from sklearn.metrics import jaccard_score, hamming_loss, accuracy_score, f1_score, average_precision_score, precision_score, recall_score, top_k_accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

class WeightImprintingClassifier:
    def __init__(self, model_parameters):
        self.num_classes = model_parameters["num_classes"]
        self.input_dims = model_parameters["input_dims"]
        self.scale = model_parameters["scale"]
        if self.scale:
            self.s = tf.Variable(1., shape=tf.TensorShape(None))
        if "multi_label" in model_parameters.keys():
            self.multi_label = model_parameters["multi_label"]
        else:
            # defaults to single label classifier
            self.multi_label = False
        self.model = self.build_model(model_parameters)
        self._normalize_dense_layer()

    def build_model(self, parameters):
        """
        classifier model definition
        """

        input_dims = parameters["input_dims"]
        num_classes = parameters["num_classes"]
        scale = parameters["scale"]

        if "dense_layer_weights" in parameters:
            dense_layer_weights = parameters["dense_layer_weights"]
            # print(conv_layer_weights[:, :, :10, :])
            dense_layer_initializer = keras.initializers.Constant(dense_layer_weights)
        else:
            dense_layer_initializer = keras.initializers.glorot_uniform()

        _input = keras.layers.Input(shape = (input_dims,))
        output = keras.layers.Dense(
            num_classes,
            use_bias=False, 
            kernel_initializer=dense_layer_initializer,
            name="dense_layer",
            )(_input)
        if scale:
            output = tf.math.multiply(self.s, output)

        if self.multi_label:
            output = keras.activations.sigmoid(output)
        else:
            output = keras.activations.softmax(output)

        model = keras.Model(_input, output)

        if self.multi_label:
            model.compile(
                optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=["accuracy"]
                )
        else:            
            model.compile(
                optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=["accuracy"]
                )

        return model

    def train(self, x, y, epochs, batch_size=32, plot=False):
        """
        Train classifier model
        """
        history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
        self._normalize_dense_layer()

        if plot:
            self.plot_acc(history, "weight_imprinting_acc.png")
            self.plot_loss(history, "weight_imprinting_loss.png")

    def _plot_acc(self, history, filename):
        plt.plot(history.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig(filename)

    def _plot_loss(self, history, filename):
        plt.plot(history.history['loss'])
        plt.title('model accuracy')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig(filename)

    def predict(self, inp_embedding, threshold=None):
        if self.multi_label:
            if threshold is None:
                threshold = 0.7
            classes = self.predict_multi_label(inp_embedding, threshold)
        else:
            classes = self.predict_single_label(inp_embedding)

        return classes

    def predict_scores(self, inp_embedding):
        """
        Predict final layer logits for embeddings.
        """
        scores = self.model.predict(inp_embedding)

        return scores

    def predict_multi_label(self, inp_embedding, threshold=0.7):
        """
        Predict multi label output for embeddings.
        """
        pred = self.model.predict(inp_embedding)

        # For multi label classification with threshold
        pred_class_labels = np.where(pred >= threshold)
        classes = []
        for i in range(len(inp_embedding)):
            classes.append(pred_class_labels[1][pred_class_labels[0] == i])

        return classes

    def predict_single_label(self, inp_embedding):
        """
        Run inference on classifier model
        """

        pred = self.model.predict(inp_embedding)

        classes = np.argmax(pred, axis=1)

        return classes

    def evaluate(self, x, y, batch_size=32):
        raise NotImplementedError

    def evaluate_single_label_metrics(
        self, 
        x, 
        y, 
        label_mapping,
        metrics=['accuracy', 'ap', 'map', 'c_f1', 'o_f1', 'c_precision', 'o_precision', 'c_recall', 'o_recall', 'top1_accuracy', 'top5_accuracy', 'classwise_accuracy', 'c_accuracy']
    ):
        """
        Evaluate single label metrics for single label experiment setup.
        """
        pred = self.predict_single_label(x)
        pred_mapped = [label_mapping[p] for p in pred]
        
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y)
        pred_bin = lb.transform(pred_mapped)

        ## For MAP
        pred_scores = self.predict_scores(x) # logits after sigmoid
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
            top1_accuracy_value = top_k_accuracy_score(
                y, pred_scores, k=1
                )
            metric_values['top1_accuracy'] = top1_accuracy_value
        if 'top5_accuracy' in metrics:
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

    def evaluate_multi_label_metrics(
        self, 
        x, 
        y, 
        label_mapping, 
        threshold=0.7,
        metrics=['hamming', 'jaccard', 'subset_accuracy', 'ap', 'map', 'c_f1', 'o_f1', 'c_precision', 'o_precision', 'c_recall', 'o_recall', 'top1_accuracy', 'top5_accuracy', 'classwise_accuracy', 'c_accuracy']
    ):
        """
        Evaluate multi label metrics for multi label experiment setup.
        """
        pred = self.predict_multi_label(x, threshold) # list of list of multi-label predictions
        pred_mapped = [[label_mapping[val] for val in p] for p in pred] # multi-label predictions mapped to original class names

        mlb = MultiLabelBinarizer()
        y_bin = mlb.fit_transform(y)
        pred_bin = mlb.transform(pred_mapped)

        ## For MAP
        pred_scores = self.predict_scores(x) # logits after sigmoid
        reverse_label_mapping = {label_mapping[v]:v for v in label_mapping}
        y_scores = np.zeros((len(y), len(label_mapping))) # one hot encoding of y
        for idx_l, label_list in enumerate(y):
            for l in label_list:
                y_scores[idx_l, reverse_label_mapping[l]] = 1

        metric_values = {}
        if 'hamming' in metrics:
            hamming_score = hamming_loss(y_bin, pred_bin)
            metric_values['hamming'] = hamming_score
        if 'jaccard' in metrics:
            jaccard_index = jaccard_score(y_bin, pred_bin, average='samples')
            metric_values['jaccard'] = jaccard_index
        if 'subset_accuracy' in metrics:
            subset_accuracy = accuracy_score(y_bin, pred_bin)
            metric_values['subset_accuracy'] = subset_accuracy
        if 'ap' in metrics:
            ap_value = average_precision_score(y_scores, pred_scores)
            metric_values['ap'] = ap_value
        if 'map' in metrics:
            map_value = average_precision_score(
                y_scores, pred_scores, average='macro'
                )
            metric_values['map'] = map_value
        if 'c_f1' in metrics:
            c_f1_value = f1_score(y_bin, pred_bin, average='macro')
            metric_values['c_f1'] = c_f1_value
        if 'o_f1' in metrics:
            o_f1_value = f1_score(y_bin, pred_bin, average='micro')
            metric_values['o_f1'] = o_f1_value
        if 'c_precision' in metrics:
            c_precision_value = precision_score(y_bin, pred_bin, average='macro', zero_division=0)
            metric_values['c_precision'] = c_precision_value
        if 'o_precision' in metrics:
            o_precision_value = precision_score(y_bin, pred_bin, average='micro', zero_division=0)
            metric_values['o_precision'] = o_precision_value
        if 'c_recall' in metrics:
            c_recall_value = recall_score(y_bin, pred_bin, average='macro', zero_division=0)
            metric_values['c_recall'] = c_recall_value
        if 'o_recall' in metrics:
            o_recall_value = recall_score(y_bin, pred_bin, average='micro', zero_division=0)
            metric_values['o_recall'] = o_recall_value
        if 'top1_accuracy' in metrics:
            y_single = [reverse_label_mapping[label_list[0]] for label_list in y]
            top1_accuracy_value = top_k_accuracy_score(
                y_single, pred_scores, k=1
                )
            metric_values['top1_accuracy'] = top1_accuracy_value
        if 'top5_accuracy' in metrics:
            y_single = [reverse_label_mapping[label_list[0]] for label_list in y]
            top5_accuracy_value = top_k_accuracy_score(
                y_single, pred_scores, k=5
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
        """
        Evaluate per class accuracy.
        """
        classwise_accuracy = []

        for col in range(y_bin.shape[1]):
            col_values = y_bin[:, col] == pred_bin[:, col]
            col_acc = np.sum(col_values)/len(col_values)
            classwise_accuracy.append(col_acc)

        return classwise_accuracy

    def _get_c_accuracy(self, y_bin, pred_bin):
        """
        Evaluate mean per class accuracy.
        """
        classwise_accuracy = self._get_classwise_accuracy(y_bin, pred_bin)
        c_accuracy = np.mean(classwise_accuracy)

        return c_accuracy

    def add_new_classes(self, x, y, is_one_hot=True):
        """
        Add new classes to classifier i.e. Weight Imprinting 
        """

        new_class_embedding, new_label_mapping = self.get_imprinting_weights(
            x, y, is_one_hot, self.multi_label
            )

        old_weights = self.model.get_layer(name="dense_layer").get_weights()[0]
        new_weights = np.concatenate([old_weights, new_class_embedding], axis=1)

        self.num_classes += new_class_embedding.shape[1]

        new_model_parameters = {
            "input_dims": self.input_dims,
            "num_classes": self.num_classes,
            "scale": self.scale,
            "dense_layer_weights": new_weights,
            "multi_label": self.multi_label
        }
        self.model = self.build_model(new_model_parameters)
        self._normalize_dense_layer()

        return new_label_mapping

    @staticmethod
    def get_imprinting_weights(x, y, is_one_hot=True, multi_label=False):
        """
        x and y gives only the newly added data
        """
        dense_layer_weights_arr = []
        label_mapping = {}

        if is_one_hot:
            if multi_label:
                _labels = [[] for _ in len(y)]
                indices = np.where(y == 1)
                for idx in range(len(indices[0])):
                    _labels[indices[0][idx]].append(indices[1][idx])
            else:
                _labels = np.argmax(y, axis=1)
        else:
            _labels = y

        la_idx = -1

        if multi_label:
            _unique_labels = np.unique(
                np.array(list(itertools.chain.from_iterable(_labels)))
                )
        else:
            _unique_labels = np.unique(_labels)

        for la in _unique_labels:
            if multi_label:
                la_indices = [la in labels_list for labels_list in _labels]
                la_emb = x[la_indices].mean(axis=0)
            else:
                la_emb = x[_labels == la].mean(axis=0)

            dense_layer_weights_arr.append(la_emb)
            la_idx += 1
            label_mapping[la_idx] = la

        dense_layer_weights = np.stack(dense_layer_weights_arr).T
        dense_layer_weights = dense_layer_weights / np.linalg.norm(dense_layer_weights, axis=0)

        return dense_layer_weights, label_mapping

    def _normalize_dense_layer(self):
        """
        Normalise the dense layer weights
        """
        old_weights = self.model.get_layer(name="dense_layer").get_weights()[0]
        norm_weights = old_weights / np.linalg.norm(old_weights, axis=0)

        self.model.get_layer(name="dense_layer").set_weights([norm_weights])

    @staticmethod
    def preprocess_input(x):
        """
        The embeddings have to be normalised before passing to weight 
        imprinting.
        """
        epsilon = 1e-6
        
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)

        x_norm = np.linalg.norm(x, axis=1)
        zero_indices = np.where(x_norm == 0)[0]
        x_norm[zero_indices] = epsilon

        processed_x = (x.T/x_norm).T

        return processed_x

    def evaluate_continual_metrics(
        self, 
        x, 
        y, 
        label_mapping, 
        threshold=0.7,
        metrics=['hamming', 'jaccard', 'subset_accuracy', 'f1_score']
    ):
        """
        Evaluate multi label metrics for continual learning experiment setup.
        """

        for i in range(len(y)):
            for j in range(len(y[i])):
                if y[i][j] not in label_mapping.values() and len(y[i]) == 1:
                    y[i][j] = -1
                elif y[i][j] not in label_mapping.values() and len(y[i]) > 1:
                    del y[i][j]

        pred = self.predict_multi_label(x, threshold)
        pred_mapped = [[label_mapping[val] for val in p] for p in pred]

        for i in range(len(pred_mapped)):
            if len(pred_mapped[i]) == 0:
                pred_mapped[i] = [-1]
        

        mlb = MultiLabelBinarizer()
        y_bin = mlb.fit_transform(y)
        pred_bin = mlb.transform(pred_mapped)


        hamming_score = 0
        jaccard_index = 0metric_values = {}
        if 'hamming' in metrics:
            hamming_score = hamming_loss(y_bin, pred_bin)
            metric_values['hamming'] = hamming_score
        if 'jaccard' in metrics:
            jaccard_index = jaccard_score(y_bin, pred_bin, average='samples')
            metric_values['jaccard'] = jaccard_index
        if 'subset_accuracy' in metrics:
            subset_accuracy = accuracy_score(y_bin, pred_bin)
            metric_values['subset_accuracy'] = subset_accuracy
        if 'f1_score' in metrics:
            f1_score_value = f1_score(y_bin, pred_bin, average='samples')
            metric_values['f1_score'] = f1_score_value

        return metric_values

