import gc
import os, argparse

import paddle
import paddlenlp
import pandas as pd
from paddle.vision import BaseTransform, Transpose
from paddle.vision.datasets import Cifar10, Cifar100, FashionMNIST

from paddle.vision.transforms import Compose, Normalize, ToTensor, Resize
import librosa
from paddlenlp.transformers import RobertaTokenizer

from paddle.io import Dataset, DataLoader, TensorDataset
from datasets import load_dataset, config, load_from_disk
import numpy as np
import paddle.nn as nn
import paddle.metric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from PIL import Image
# Example usage:
import sys
from jiwer import wer, cer

vision_dataset = {
    "CIFAR10": {
        "num_class": 10,
        "mean_std": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    },
    "CIFAR100": {
        "num_class": 100,
        "mean_std": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    },
    "FashionMNIST": {
        "num_class": 10,
        "mean_std": ([0.5], [0.5])
    },
    "kinetics400": {
        "num_class": 27,
        "mean_std": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    }
}

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "CMC", "targetModels"))


class GrayScaleToRGB(BaseTransform):
    def __init__(self):
        super(GrayScaleToRGB, self).__init__()

    def _apply_image(self, img):

        img_array = np.array(img)

        if img_array.ndim == 2:
            # Expand a single-channel image into three channels
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            img_array = np.concatenate([img_array] * 3, axis=2)

        # Convert the numpy array back into a PIL image
        return Image.fromarray(img_array)


class PaddleAudioTrainer:
    class CharTokenizer:
        def __init__(self):
            self.char_map = self.build_char_map()
            self.index_map = {v: k for k, v in self.char_map.items()}  # Create a mapping from numerical indices to characters

        @staticmethod
        def build_char_map():
            chars = "abcdefghijklmnopqrstuvwxyz' "
            char_map = {char: i + 1 for i, char in enumerate(chars)}  # The index starts from 1
            return char_map

        def text_to_sequence(self, text):
            """
            # Convert the text string into a sequence of numbers.
            """
            return [self.char_map[char] for char in text.lower() if char in self.char_map]

        def sequence_to_text(self, sequence):
            """
            # Convert the sequence of numbers back into a text string.
            """
            return ''.join(self.index_map.get(index, '') for index in sequence)

        def decode_logits_to_text(self, logits):
            """
            Use greedy decoding to convert the model's logits output into text sequences.

            Parameters:
                logits: Logits output from the model, shaped [batch_size, time_steps, num_classes]

            Returns:
                A list of decoded text sequences.
            """
            # Use argmax to select the most probable character index at each time step


            pred_indices = logits.argmax(-1)

            decoded_texts = []
            for index_seq in pred_indices:
                # Convert the sequence of numbers into text
                text = self.sequence_to_text(index_seq)
                decoded_texts.append(text)

            return decoded_texts

    def __init__(self, model_name, datasets, sample_rate=16000, duration=3, batch_size=32, learning_rate=2e-4,
                 num_epochs=50, save_model_after_training=False, device="cpu"):
        self.tokenizer = self.CharTokenizer()
        self.model_name = model_name
        self.device = device
        self.datasets = datasets  # A list of dataset names
        self.sample_rate = sample_rate
        self.duration = duration
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_model_after_training = save_model_after_training
        self.transform = Compose([
            Resize(sample_rate * duration),
            ToTensor()
        ])
        self.model = None
        self.optimizer = None

    def select_model(self):
        # (n_batch, 1, (n_time - kernel_size + 1) * hop_length)
        self.max_waveform_length = (384 - 5 + 1) * 64
        if self.model_name == "wavernn_small":
            from EVALUATION.datasets.models.audio.targetModels.wavernn.wavernn import WaveRNN
            self.model = WaveRNN(upsample_scales=[4, 4, 4], n_classes=10, hop_length=64, n_res_block=6,
                                 n_rnn=256, n_fc=256, kernel_size=5, n_freq=128, n_hidden=64, n_output=128)
        elif self.model_name == "wavernn_large":
            from EVALUATION.datasets.models.audio.targetModels.wavernn.wavernn import WaveRNN
            self.model = WaveRNN(upsample_scales=[4, 4, 4], n_classes=10, hop_length=64, n_res_block=10,
                                 n_rnn=512, n_fc=512, kernel_size=5, n_freq=128, n_hidden=128, n_output=128)
        elif self.model_name == "wav2letter_waveform":
            from EVALUATION.datasets.models.audio.targetModels.wav2letter.wav2letter import Wav2Letter
            self.model = Wav2Letter(num_classes=10, input_type="mfcc", num_features=1)
        elif self.model_name == "wav2letter_mfcc":
            from EVALUATION.datasets.models.audio.targetModels.wav2letter.wav2letter import Wav2Letter
            self.model = Wav2Letter(num_classes=10, input_type="mfcc", num_features=128)
        elif self.model_name == "deepspeech_small":
            from EVALUATION.datasets.models.audio.targetModels.deepspeech import DeepSpeech
            self.model = DeepSpeech(n_feature=128, n_hidden=512, n_class=10, dropout=0.1)
        elif self.model_name == "deepspeech_large":
            from EVALUATION.datasets.models.audio.targetModels.deepspeech import DeepSpeech
            self.model = DeepSpeech(n_feature=128, n_hidden=2048, n_class=10, dropout=0.2)
        else:
            raise ValueError("Unsupported model")
        self.optimizer = paddle.optimizer.Adam(parameters=self.model.parameters(), learning_rate=self.learning_rate)

    # Load and preprocess audio data
    def load_and_process_audio(self, dataset_name):
        def collate_fn(batch):
            """
            Custom collate_fn for handling sequences of different lengths.
            """

            waveforms, mel_specs, labels = zip(*batch)


            max_mel_length = 384

            # Initialize the list of padded MFCC features and labels
            padded_waveforms = []
            padded_mel_specs = []

            for waveform, mfcc in zip(waveforms, mel_specs):
                waveform = paddle.to_tensor(waveform, dtype='float32')
                mfcc = paddle.to_tensor(mfcc, dtype='float32')
                # Pad the MFCC features
                padded_mel_spec = []
                # Pad the waveform
                padding_length_waveform = self.max_waveform_length - waveform.shape[0]
                if padding_length_waveform > 0:
                    waveform = paddle.nn.functional.pad(waveform, pad=[0, padding_length_waveform],
                                                        mode='constant', value=0)
                else:
                    waveform = waveform[:self.max_waveform_length]
                padded_waveforms.append(waveform)

                for mel_spec in mfcc:
                    # Pad each mel spectrogram
                    padding_length_mel = max_mel_length - mel_spec.shape[-1]
                    if padding_length_mel > 0:
                        padded_mel_spec.append(
                            paddle.nn.functional.pad(mel_spec, pad=[0, padding_length_mel],
                                                     mode='constant', value=0))
                    else:
                        padded_mel_spec.append(mel_spec[:max_mel_length])
                padded_mel_specs.append(padded_mel_spec)
            # Convert the padded list into a tensor
            waveforms_tensor = paddle.to_tensor(np.array(padded_waveforms), dtype='float32')
            mel_specs_tensor = paddle.to_tensor(np.array(padded_mel_specs), dtype='float32')
            labels_tensor = paddle.to_tensor(np.array(labels), dtype='int32')

            return waveforms_tensor, mel_specs_tensor, labels_tensor

        if dataset_name == "urbansound8k":
            class urbansoundDataset(Dataset):
                def __init__(self, dataset_split, tokenizer, split_ratio=0.9):
                    super().__init__()

                    dataset = load_dataset("danavery/urbansound8K")


                    half_dataset_size = int(len(dataset['train']) * 0.4)
                    dataset['train'] = dataset['train'].select(range(half_dataset_size))


                    num_train_samples = int(len(dataset['train']) * split_ratio)
                    if dataset_split == 'train':
                        self.dataset = dataset['train'].select(range(num_train_samples))
                    elif dataset_split == 'test':
                        self.dataset = dataset['train'].select(range(num_train_samples, len(dataset['train'])))
                    else:
                        raise ValueError("dataset_split must be 'train' or 'test'")

                    self.tokenizer = tokenizer

                def __getitem__(self, idx):
                    data = self.dataset[idx]


                    audio = data['audio']['array']
                    sr = data['audio']['sampling_rate']


                    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
                    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)


                    label = data['classID']


                    return paddle.to_tensor(audio, dtype='float32'), paddle.to_tensor(mel_spec,
                                                                                      dtype='float32'), paddle.to_tensor(
                        label, dtype='int32')

                def __len__(self):
                    return len(self.dataset)

            train_dataset = urbansoundDataset(dataset_split="train", tokenizer=self.tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
            test_dataset = urbansoundDataset(dataset_split="test", tokenizer=self.tokenizer)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        else:
            raise ValueError("dataset unsupported")
        return train_loader, test_loader

    def train(self, data_name, train_loader):
        paddle.device.set_device(self.device)
        self.model.train()
        criterion = paddle.nn.CrossEntropyLoss()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_idx, (waveform, mel_spec, labels) in enumerate(train_loader):
                self.optimizer.clear_grad()
                waveform = waveform.unsqueeze(1)
                if "mfcc" in self.model_name:
                    outputs = self.model(mel_spec)
                elif "waveform" in self.model_name:
                    outputs = self.model(waveform)
                else:
                    mel_spec = mel_spec.unsqueeze(1)
                    outputs = self.model(waveform, mel_spec)
                outputs_avg = paddle.mean(outputs.squeeze(1), axis=1)
                loss = criterion(outputs_avg, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                # print(f'batch {batch_idx}, all batches: {len(train_loader)}')
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}')  # Print average loss for the epoch

    def evaluate(self, dataset_name, test_loader):
        self.model.eval()
        if dataset_name == "librispeech" or dataset_name == "urbansound8k":
            true_labels = []
            pred_labels = []
            for batch_idx, (waveform, mel_spec, labels) in enumerate(test_loader):
                waveform = waveform.unsqueeze(1)
                if "mfcc" in self.model_name:
                    outputs = self.model(mel_spec)
                    # Calculate the average probability for each class across all time steps
                    outputs = paddle.mean(outputs, axis=-1)
                elif "waveform" in self.model_name:
                    outputs = self.model(waveform)
                    # Calculate the average probability for each class across all time steps
                    outputs = paddle.mean(outputs, axis=-1)
                else:
                    mel_spec = mel_spec.unsqueeze(1)
                    outputs = self.model(waveform, mel_spec)

                predicted_classes = paddle.argmax(outputs, axis=1).numpy()


                true_labels.extend(labels.numpy())
                pred_labels.extend(predicted_classes)


            # accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
            precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
            recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
            f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

            # print(
            #     f'Accuracy: {accuracy:.4f}, Precision (Macro Avg.): {precision_macro:.4f},'
            #     f' Recall (Macro Avg.): {recall_macro:.4f}')
            print(
                f' Precision (Macro Avg.): {precision:.4f},'
                f' Recall (Macro Avg.): {recall:.4f},'
                f' F1 (Macro Avg.): {f1:.4f},'
            )

    def train_and_evaluate(self):
        for dataset_name in self.datasets:
            print(f"Training on {dataset_name}")
            train_loader, test_loader = self.load_and_process_audio(dataset_name)
            self.select_model()
            self.train(dataset_name, train_loader)
            self.evaluate(dataset_name, test_loader)
            if self.save_model_after_training:

                if not os.path.exists(os.path.join("CMC", "paddle_pipline_model_result")):
                    os.makedirs(os.path.join("CMC", "paddle_pipline_model_result"))
                self.save_model(os.path.join("CMC", "paddle_pipline_model_result",
                                             "{}_{}_{}".format(self.model_name, dataset_name, self.num_epochs)))
            del self.model
            paddle.device.cuda.empty_cache()
            gc.collect()

    def save_model(self, path):
        """
        Save the model parameters to the specified path.

        Parameters:
        - path (str): The file path where the model parameters will be saved.
        """
        paddle.save(self.model.state_dict(), f"{path}.pdparams")
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(path)


class PaddleTextTrainer:
    def __init__(self, model_name, tokenizer, datasets, max_seq_length=128, batch_size=256, learning_rate=2e-5,
                 num_epochs=50, save_model_after_training=False, device="cpu"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = tokenizer
        self.device = device
        self.datasets = datasets  # A list of dataset names
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = None
        self.save_model_after_training = save_model_after_training

    def load_and_process_data(self, data_name):
        if data_name == "IMDB":
            train_dataset = load_dataset('imdb', split='train')
            test_dataset = load_dataset('imdb', split='test')
        elif data_name == "SST2":
            train_dataset = load_dataset('glue', 'sst2', split='train')
            test_dataset = load_dataset('glue', 'sst2', split='validation')
        else:
            raise ValueError("Unsupported dataset")
        # Check if the returned value is a DatasetTuple
        if isinstance(train_dataset, paddlenlp.datasets.dataset.DatasetTuple):
            train_dataset = train_dataset['train']
        if isinstance(test_dataset, paddlenlp.datasets.dataset.DatasetTuple):
            test_dataset = test_dataset['test']

        def transform_function(example):
            if data_name == "IMDB":
                # Adjust the tokenizer's preprocessing according to the task type
                encoded_inputs = self.tokenizer(text=example['text'],
                                                max_seq_len=self.max_seq_length,
                                                truncation=True,
                                                padding='max_length',
                                                return_attention_mask=False)
                return {'input_ids': encoded_inputs['input_ids'], 'labels': example['label']}
            elif data_name == "SST2":
                # Adjust the tokenizer's preprocessing according to the task type
                encoded_inputs = self.tokenizer(text=example['sentence'],
                                                max_seq_len=self.max_seq_length,
                                                truncation=True,
                                                padding='max_length',
                                                return_attention_mask=False)
                return {'input_ids': encoded_inputs['input_ids'], 'labels': example['label']}

        def collate_fn(batch):

            input_ids = []
            labels = []
            for sample in batch:

                if 'input_ids' in sample and 'labels' in sample:
                    input_ids.append(sample['input_ids'])
                    labels.append(sample['labels'])
                else:

                    raise KeyError("'labels' key not found in the sample.")

            input_ids = paddle.to_tensor(input_ids, dtype='int64')
            labels = paddle.to_tensor(labels, dtype='int64')
            return input_ids, labels

        column = None

        if data_name == "IMDB":
            column = 'text'
        elif data_name == "SST2":
            column = 'sentence'
        train_dataset = train_dataset.map(transform_function, batched=False, remove_columns=[column])
        test_dataset = test_dataset.map(transform_function, batched=False, remove_columns=[column])

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=collate_fn)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False,
                                     collate_fn=collate_fn)
        return train_dataloader, test_dataloader

    def select_model(self):
        if self.model_name == "roberta_model_6":
            from EVALUATION.datasets.models.text.targetModels.roberta.model import RobertaModel, RobertaEncoderConf, \
                RobertaClassificationHead
            encoder_conf = RobertaEncoderConf(
                num_encoder_layers=6,
                # Keep other parameters as default
            )
            self.model = RobertaModel(encoder_conf=encoder_conf,
                                      head=RobertaClassificationHead(num_classes=2, input_dim=768))
        elif self.model_name == "roberta_model_12":
            from EVALUATION.datasets.models.text.targetModels.roberta.model import RobertaModel, RobertaEncoderConf, \
                RobertaClassificationHead
            encoder_conf = RobertaEncoderConf(
                num_encoder_layers=12,
                # Keep other parameters as default
            )
            self.model = RobertaModel(encoder_conf=encoder_conf,
                                      head=RobertaClassificationHead(num_classes=2, input_dim=768))
        elif self.model_name == "roberta_model_24":
            from EVALUATION.datasets.models.text.targetModels.roberta.model import RobertaModel, RobertaEncoderConf, \
                RobertaClassificationHead
            encoder_conf = RobertaEncoderConf(
                num_encoder_layers=24,
                # Keep other parameters as default
            )
            self.model = RobertaModel(encoder_conf=encoder_conf,
                                      head=RobertaClassificationHead(num_classes=2, input_dim=768))
        self.optimizer = paddle.optimizer.AdamW(parameters=self.model.parameters(), learning_rate=self.learning_rate)

    def train(self, train_loader):
        paddle.device.set_device(self.device)
        self.model.train()  # Set the model to training mode
        for epoch in range(self.num_epochs):
            total_loss = 0  # Initialize total loss for the epoch
            for batch_id, (input_ids, labels) in enumerate(train_loader()):
                logits = self.model(input_ids)  # Forward pass
                loss = paddle.nn.functional.cross_entropy(logits, labels)  # Compute loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update parameters
                self.optimizer.clear_grad()  # Clear gradients
                total_loss += loss.item()  # Accumulate the loss

            avg_loss = total_loss / len(train_loader)  # Calculate average loss for the epoch
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}')  # Print average loss for the epoch

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds, all_labels = [], []
        with paddle.no_grad():
            for input_ids, labels in test_loader:
                logits = self.model(input_ids)
                preds = paddle.argmax(logits, axis=1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())


        # acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro',
                                                                   zero_division=0)

        print(
            f' Precision (Macro Avg.): {precision:.4f},'
            f' Recall (Macro Avg.): {recall:.4f},'
            f' F1 (Macro Avg.): {f1:.4f},'
        )

    def train_and_evaluate(self):
        for dataset_name in self.datasets:
            print(f"Training on {dataset_name}")
            train_loader, test_loader = self.load_and_process_data(dataset_name)
            self.select_model()
            self.train(train_loader)
            self.evaluate(test_loader)
            if self.save_model_after_training:

                if not os.path.exists(os.path.join("CMC", "paddle_pipline_model_result")):
                    os.makedirs(os.path.join("CMC", "paddle_pipline_model_result"))
                self.save_model(os.path.join("CMC", "paddle_pipline_model_result",
                                             "{}_{}_{}".format(self.model_name, dataset_name, self.num_epochs)))
            del self.model
            gc.collect()

    def save_model(self, path):
        """
        Save the model parameters to the specified path.

        Parameters:
        - path (str): The file path where the model parameters will be saved.
        """
        paddle.save(self.model.state_dict(), f"{path}.pdparams")
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(path)


class PaddleVisionTrainer:
    def __init__(self, model_name, datasets, batch_size=256, learning_rate=0.001, num_epochs=100,
                 save_model_after_training=False, device="cpu"):
        self.model = None
        self.model_name = model_name.lower()
        self.datasets = datasets  # A list of dataset names
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.save_model_after_training = save_model_after_training
        self.optimizer = None

    def load_dataset(self, dataset_name):
        global num_classes
        assert dataset_name in self.datasets, "The datasets are unsupported yet"
        if dataset_name in vision_dataset:
            num_classes = vision_dataset[dataset_name]["num_class"]
        if self.model_name == "alexnet":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.alexnet.alexnet import AlexNet
            self.model = AlexNet(num_classes=num_classes)
        elif self.model_name == "resnet18":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.resnet.resnet import ResNet, BasicBlock, Bottleneck
            self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        elif self.model_name == "resnet34":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.resnet.resnet import ResNet, BasicBlock, Bottleneck
            self.model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
        elif self.model_name == "resnet50":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.resnet.resnet import ResNet, BasicBlock, Bottleneck
            self.model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        elif self.model_name == "resnet101":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.resnet.resnet import ResNet, BasicBlock, Bottleneck
            self.model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
        elif self.model_name == "resnet152":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.resnet.resnet import ResNet, BasicBlock, Bottleneck
            self.model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
        elif self.model_name == "densenet121":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.densenet.densenet import DenseNet
            self.model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_classes)
        elif self.model_name == "densenet161":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.densenet.densenet import DenseNet
            self.model = DenseNet(growth_rate=48, block_config=(6, 12, 36, 24), num_classes=num_classes)
        elif self.model_name == "densenet169":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.densenet.densenet import DenseNet
            self.model = DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), num_classes=num_classes)
        elif self.model_name == "densenet201":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.densenet.densenet import DenseNet
            self.model = DenseNet(growth_rate=32, block_config=(6, 12, 48, 32), num_classes=num_classes)
        elif self.model_name == "densenet264":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.densenet.densenet import DenseNet
            self.model = DenseNet(growth_rate=32, block_config=(6, 12, 64, 48), num_classes=num_classes)
        elif self.model_name == "squeezenet1_0":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.squeezenet.squeezenet import SqueezeNet
            self.model = SqueezeNet(version="1_0", num_classes=num_classes)
        elif self.model_name == "squeezenet1_1":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.squeezenet.squeezenet import SqueezeNet
            self.model = SqueezeNet(version="1_1", num_classes=num_classes)
        elif self.model_name == "vgg11":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.vgg.vgg import VGG, make_layers, cfgs
            self.model = VGG(make_layers(cfgs["A"], batch_norm=True), num_classes=num_classes)
        elif self.model_name == "vgg13":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.vgg.vgg import VGG, make_layers, cfgs
            self.model = VGG(make_layers(cfgs["B"], batch_norm=True), num_classes=num_classes)
        elif self.model_name == "vgg16":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.vgg.vgg import VGG, make_layers, cfgs
            self.model = VGG(make_layers(cfgs["D"], batch_norm=True), num_classes=num_classes)
        elif self.model_name == "vgg19":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.vgg.vgg import VGG, make_layers, cfgs
            self.model = VGG(make_layers(cfgs["E"], batch_norm=True), num_classes=num_classes)
        elif self.model_name == "mnasnet0_5":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.mnasnet.mnasnet import MNASNet
            self.model = MNASNet(alpha=0.5, num_classes=num_classes)
        elif self.model_name == "mnasnet0_75":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.mnasnet.mnasnet import MNASNet
            self.model = MNASNet(alpha=0.75, num_classes=num_classes)
        elif self.model_name == "mnasnet1_0":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.mnasnet.mnasnet import MNASNet
            self.model = MNASNet(alpha=1.0, num_classes=num_classes)
        elif self.model_name == "mnasnet1_3":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.mnasnet.mnasnet import MNASNet
            self.model = MNASNet(alpha=1.3, num_classes=num_classes)
        elif self.model_name == "mnasnet1_5":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.mnasnet.mnasnet import MNASNet
            self.model = MNASNet(alpha=1.5, num_classes=num_classes)
        elif self.model_name == "mobilenetv2":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.mobilenet import mobilenetv2, mobilenetv3, _utils
            self.model = mobilenetv2.MobileNetV2(num_classes=num_classes)
        elif self.model_name == "mobilenetv3_small":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.mobilenet import mobilenetv2, mobilenetv3, _utils
            width_mult = 1.0  # You can adjust the width multiplier as needed
            inverted_residual_setting = [
                mobilenetv3.InvertedResidualConfig(input_channels=16, kernel=3, expanded_channels=16, out_channels=16,
                                                   use_se=False,
                                                   activation='RE', stride=2, dilation=1, width_mult=width_mult),

            ]
            self.model = mobilenetv3.MobileNetV3(inverted_residual_setting, last_channel=1280, num_classes=num_classes)
        elif self.model_name == "mobilenetv3_large":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.mobilenet import mobilenetv2, mobilenetv3, _utils
            width_mult = 1.0  # You can adjust the width multiplier as needed
            inverted_residual_setting = [
                mobilenetv3.InvertedResidualConfig(input_channels=16, kernel=3, expanded_channels=64, out_channels=24,
                                                   use_se=False,
                                                   activation='RE', stride=2, dilation=1, width_mult=width_mult),
            ]
            self.model = mobilenetv3.MobileNetV3(inverted_residual_setting,
                                                 last_channel=_utils._make_divisible(1280 * width_mult, 8),
                                                 num_classes=num_classes)
        elif self.model_name == "shufflenet_v2_0_5":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.shufflenetv2.shufflenetv2 import ShuffleNetV2
            self.model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                                      stages_out_channels=[24, 48, 96, 192, 1024],
                                      num_classes=num_classes)
        elif self.model_name == "shufflenet_v2_1_0":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.shufflenetv2.shufflenetv2 import ShuffleNetV2
            self.model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                                      stages_out_channels=[24, 116, 232, 464, 1024],
                                      num_classes=num_classes)
        elif self.model_name == "shufflenet_v2_1_5":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.shufflenetv2.shufflenetv2 import ShuffleNetV2
            self.model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                                      stages_out_channels=[24, 176, 352, 704, 1024],
                                      num_classes=num_classes)
        elif self.model_name == "shufflenet_v2_2_0":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.shufflenetv2.shufflenetv2 import ShuffleNetV2
            self.model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                                      stages_out_channels=[24, 244, 488, 976, 2048],
                                      num_classes=num_classes)
        elif self.model_name == "inception3highres":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.inception import Inception3
            self.model = Inception3(num_classes=num_classes, aux_logits=True, transform_input=True)
        elif self.model_name == "inception3lite":
            from EVALUATION.datasets.models.pytorch.vision.targetModels.inception import Inception3
            self.model = Inception3(num_classes=num_classes, aux_logits=False, transform_input=False)
        self.optimizer = paddle.optimizer.Adam(parameters=self.model.parameters(), learning_rate=self.learning_rate)

        class ConvertTo3Channels(BaseTransform):
            def __init__(self):
                super(ConvertTo3Channels, self).__init__()

            def _apply_image(self, img):
                # Assume img is a numpy array or a PIL image
                if isinstance(img, Image.Image):
                    img = np.array(img)
                return np.repeat(img[:, :, np.newaxis], 3, axis=2)

        if dataset_name == "FashionMNIST":
            transform = Compose([
                ConvertTo3Channels(),
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=vision_dataset[dataset_name]["mean_std"][0],
                          std=vision_dataset[dataset_name]["mean_std"][1])
            ])
        else:
            if dataset_name != "google_stock_prices":
                transform = Compose([
                    Resize((224, 224)),
                    ToTensor(),
                    Normalize(mean=vision_dataset[dataset_name]["mean_std"][0],
                              std=vision_dataset[dataset_name]["mean_std"][1])
                ])
        if dataset_name == 'CIFAR10':
            train_dataset = Cifar10(mode='train', transform=transform)
            test_dataset = Cifar10(mode='test', transform=transform)
        elif dataset_name == 'CIFAR100':
            train_dataset = Cifar100(mode='train', transform=transform)
            test_dataset = Cifar100(mode='test', transform=transform)
        elif dataset_name == 'FashionMNIST':
            train_dataset = FashionMNIST(mode='train', transform=transform)
            test_dataset = FashionMNIST(mode='test', transform=transform)
        else:
            raise ValueError("Unsupported dataset in {}".format(self.model_name))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
        return train_loader, test_loader

    def train_and_evaluate(self):
        for dataset_name in self.datasets:
            print(f"Training on {dataset_name}")
            train_loader, test_loader = self.load_dataset(dataset_name)
            self.train(train_loader)
            self.evaluate(test_loader)
            if self.save_model_after_training:

                if not os.path.exists(os.path.join("CMC", "paddle_pipline_model_result")):
                    os.makedirs(os.path.join("CMC", "paddle_pipline_model_result"))
                self.save_model(os.path.join("CMC", "paddle_pipline_model_result",
                                             "{}_{}_{}".format(self.model_name, dataset_name, self.num_epochs)))
            del self.model
            gc.collect()

    def train(self, train_loader):
        self.model.train()
        paddle.device.set_device(self.device)
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_id, data in enumerate(train_loader()):
                images, labels = data
                outputs = self.model(images)
                if "inception3" not in self.model_name:
                    loss = self.criterion(outputs, labels)
                else:
                    # For Inception model, handle auxiliary outputs
                    main_output = outputs.logit
                    loss = self.criterion(main_output, labels)
                    if outputs.aux_logits is not None:
                        aux_output = outputs.aux_logits
                        aux_loss = self.criterion(aux_output, labels)
                        loss += 0.4 * aux_loss  # Combining main loss and auxiliary loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
                total_loss += loss.item()
            loss_value = total_loss / len(train_loader())
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss_value:.4f}')  # Print average loss for the epoch

        # Omit the previous load_dataset, train_and_evaluate, train, evaluate methods

    def predict(self, image):
        """
        Predict the class of a single image.

        Parameters:
        - image: Paddle tensor of shape (C, H, W) after applying the same transformations as during training.

        Returns:
        - int: The predicted class index.
        """
        self.model.eval()
        with paddle.no_grad():
            logits = self.model(image.unsqueeze(0))  # Add batch dimension
            predicted_class = logits.argmax(axis=1).item()
        return predicted_class

    def evaluate(self, test_loader):
        self.model.eval()
        y_true_list = []
        y_pred_list = []
        for batch_id, data in enumerate(test_loader()):
            images, labels = data

            logits = self.model(images)
            predicted = paddle.argmax(logits, axis=1)


            y_true_list.append(labels.numpy())
            y_pred_list.append(predicted.numpy())


        y_true_all = np.concatenate(y_true_list, axis=0)
        y_pred_all = np.concatenate(y_pred_list, axis=0)



        # accuracy = 100 * np.sum(np.array(y_true_all) == np.array(y_pred_all)) / len(y_true_all)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro')

        print(
            f' Precision (Macro Avg.): {precision:.4f},'
            f' Recall (Macro Avg.): {recall:.4f},'
            f' F1 (Macro Avg.): {f1:.4f},'
        )

    def save_model(self, path):
        """
        Save the model parameters to the specified path.

        Parameters:
        - path (str): The file path where the model parameters will be saved.
        """
        paddle.save(self.model.state_dict(), f"{path}.pdparams")
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(path)


def main(args):
    paddle.device.set_device(args.device)
    if args.model_type == 'vision' or args.model_type == 'ts':
        model_names = parse_list_arg(args.model_names)
        for model_name in model_names:
            print("=========================================================")
            print("===================={}=========================".format(model_name))
            trainer = PaddleVisionTrainer(model_name=model_name, datasets=parse_list_arg(args.datasets),
                                          learning_rate=args.learning_rate, batch_size=args.batch_size,
                                          num_epochs=args.num_epochs, save_model_after_training=args.save_model,
                                          device=args.device)
            trainer.train_and_evaluate()

    elif args.model_type == 'text':
        model_names = parse_list_arg(args.model_names)
        tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base")  # Assuming the tokenizer is consistent across frameworks
        for model_name in model_names:
            print("=========================================================")
            print("===================={}=========================".format(model_name))
            trainer = PaddleTextTrainer(model_name=model_name, datasets=parse_list_arg(args.datasets),
                                        learning_rate=args.learning_rate,
                                        tokenizer=tokenizer, batch_size=args.batch_size, num_epochs=args.num_epochs,
                                        save_model_after_training=args.save_model, device=args.device)
            trainer.train_and_evaluate()
    elif args.model_type == 'audio':
        model_names = parse_list_arg(args.model_names)
        for model_name in model_names:
            print(fr"=========================================================")
            print(fr"===================={model_name}=========================")
            trainer = PaddleAudioTrainer(model_name=model_name, datasets=parse_list_arg(args.datasets),
                                         batch_size=args.batch_size, learning_rate=args.learning_rate,
                                         num_epochs=args.num_epochs, save_model_after_training=args.save_model,
                                         device=args.device)
            trainer.train_and_evaluate()


def parse_list_arg(arg_value):
    # Replace commas with spaces and split by spaces, filtering out empty strings
    return [item for item in arg_value.replace(',', ' ').split() if item]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate models")
    parser.add_argument("--device", type=str, default='gpu:2',
                        help="Device to use for training: 'cpu' or 'gpu:x' where x is the GPU device number")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Type of the model: 'vision'、 'text'、 ’audio‘、 'ts'")
    parser.add_argument("--model_names", type=str, required=True,
                        help="Comma-separated list of model names to train and evaluate")
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated list of datasets to use")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--save_model", default=False, action='store_true',
                        help="Flag to save the model after training")

    args = parser.parse_args()
    main(args)