import gc, os, sys
import argparse
from typing import Optional

import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from paddlenlp.transformers import RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import librosa
from datasets import load_dataset, config, load_from_disk
from CMC.special_mapping import vision_dataset

sys.path.append(os.path.join(os.getcwd(), "CMC", "sourceModels"))
# train and evaluate vision models
from EVALUATION.datasets.models.vision.sourceModels.resnet import ResNet, BasicBlock, Bottleneck
from EVALUATION.datasets.models.vision.sourceModels.alexnet import AlexNet
from EVALUATION.datasets.models.vision.sourceModels.densenet import DenseNet
from EVALUATION.datasets.models.vision.sourceModels.squeezenet import SqueezeNet
from EVALUATION.datasets.models.vision.sourceModels.vgg import VGG, make_layers, cfgs
from EVALUATION.datasets.models.vision.sourceModels.mnasnet import MNASNet
from EVALUATION.datasets.models.vision.sourceModels.mobilenet import mobilenetv2, mobilenetv3, _utils
from EVALUATION.datasets.models.vision.sourceModels.shufflenetv2 import ShuffleNetV2
from EVALUATION.datasets.models.vision.sourceModels.inception import Inception3, InceptionOutputs

# train and evaluate text models
from EVALUATION.datasets.models.text.sourceModels.roberta.model import RobertaModel, RobertaEncoderConf, RobertaClassificationHead

# train and evaluate audio models
from EVALUATION.datasets.models.audio.sourceModels.wav2letter import Wav2Letter
from EVALUATION.datasets.models.audio.sourceModels.wavernn import WaveRNN
from EVALUATION.datasets.models.audio.sourceModels.deepspeech import DeepSpeech


def get_transform(is_grayscale):
    """
    Return the appropriate preprocessing transform sequence based on whether the image is grayscale.
    """
    if is_grayscale:
        # Grayscale images need to be converted to three channels
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        # Apply preprocessing directly to color images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform


class PyTorchAudioTrainer:
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
            Convert a text string into a sequence of numbers.
            """
            return [self.char_map[char] for char in text.lower() if char in self.char_map]

        def sequence_to_text(self, sequence):
            """
            Convert a sequence of numbers back into a text string.
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
        self.model = None
        self.optimizer = None

    def select_model(self):
        self.max_waveform_length = (384 - 5 + 1) * 6
        # (n_batch, 1, (n_time - kernel_size + 1) * hop_length)
        if self.model_name == "wavernn_small":
            self.model = WaveRNN(upsample_scales=[4, 4, 4], n_classes=10, hop_length=64, n_res_block=4,
                                 n_rnn=128, n_fc=128, kernel_size=5, n_freq=128, n_hidden=32, n_output=32).to(self.device)
        elif self.model_name == "wavernn_medium":
            self.model = WaveRNN(upsample_scales=[4, 4, 4], n_classes=10, hop_length=64, n_res_block=6,
                                 n_rnn=256, n_fc=256, kernel_size=5, n_freq=128, n_hidden=64, n_output=64).to(self.device)
        elif self.model_name == "wavernn_large":
            self.model = WaveRNN(upsample_scales=[4, 4, 4], n_classes=10, hop_length=64, n_res_block=8,
                                 n_rnn=512, n_fc=512, kernel_size=5, n_freq=128, n_hidden=128, n_output=128).to(self.device)
        elif self.model_name == "wav2letter_waveform":
            self.model = Wav2Letter(num_classes=10, input_type="mfcc", num_features=1).to(self.device)
        elif self.model_name == "wav2letter_mfcc":
            self.model = Wav2Letter(num_classes=10, input_type="mfcc", num_features=128).to(self.device)
        elif self.model_name == "deepspeech_small":
            self.model = DeepSpeech(n_feature=128, n_hidden=512, n_class=10, dropout=0.1).to(self.device)
        elif self.model_name == "deepspeech_large":
            self.model = DeepSpeech(n_feature=128, n_hidden=2048, n_class=10, dropout=0.2).to(self.device)
        else:
            raise ValueError("Unsupported model")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # Load and preprocess audio data
    def load_and_process_audio(self, dataset_name):
        def collate_fn(batch):
            """
            Custom collate_fn for handling sequences of different lengths.
            """
            # Unpack batch data
            waveforms, mel_specs, labels = zip(*batch)
            labels = torch.tensor(labels, dtype=torch.int32)

            max_mel_length = 384

            # Initialize the list of padded MFCC features and labels
            padded_waveforms = []
            padded_mel_specs = []

            for waveform, mfcc in zip(waveforms, mel_specs):
                waveform = torch.tensor(waveform, dtype=torch.float32)
                mfcc = torch.tensor(mfcc, dtype=torch.float32)

                padded_mel_spec = []

                padding_length_waveform = self.max_waveform_length - waveform.shape[0]
                if padding_length_waveform > 0:
                    waveform = torch.nn.functional.pad(waveform, pad=[0, padding_length_waveform],
                                                        mode='constant', value=0)
                else:
                    waveform = waveform[:self.max_waveform_length]
                padded_waveforms.append(waveform)

                for mel_spec in mfcc:

                    padding_length_mel = max_mel_length - mel_spec.shape[-1]
                    if padding_length_mel > 0:
                        padded_mel_spec.append(
                            torch.nn.functional.pad(mel_spec, pad=[0, padding_length_mel],
                                                     mode='constant', value=0))
                    else:
                        padded_mel_spec.append(mel_spec[:max_mel_length])
                padded_mel_spec = torch.stack(padded_mel_spec)
                padded_mel_specs.append(padded_mel_spec)
            # Convert the padded list into a tensor
            waveforms_tensor = torch.stack(padded_waveforms)
            mel_specs_tensor = torch.stack(padded_mel_specs)
            labels_tensor = labels

            return waveforms_tensor, mel_specs_tensor, labels_tensor

        if dataset_name == "urbansound8k":
            class urbansoundDataset(Dataset):
                def __init__(self, dataset_split, tokenizer, split_ratio=0.9):
                    super().__init__()

                    dataset = load_dataset("danavery/urbansound8K")


                    num_train_samples = int(len(dataset['train']) * split_ratio)
                    if dataset_split == 'train':
                        self.dataset = dataset['train'].select(range(num_train_samples))
                    elif dataset_split == 'test':
                        self.dataset = dataset['train'].select(range(num_train_samples, len(dataset['train'])))
                    else:
                        raise ValueError("dataset_split must be 'train' or 'test'")
                    print(dataset_split, "dataset length: ", len(self.dataset))
                    self.tokenizer = tokenizer

                def __getitem__(self, idx):
                    data = self.dataset[idx]


                    audio = data['audio']['array']
                    sr = data['audio']['sampling_rate']

                    # Initialize the list of padded MFCC features and labels
                    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
                    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)


                    label = data['classID']


                    return (torch.tensor(audio, dtype=torch.float32),
                            torch.tensor(mel_spec,dtype=torch.float32),
                            torch.tensor(label, dtype=torch.int32))

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
        if data_name == "librispeech" or data_name == "urbansound8k":
            self.model.train()
            criterion = torch.nn.CrossEntropyLoss()
            for epoch in range(self.num_epochs):
                total_loss = 0
                for batch_idx, (waveform, mel_spec, labels) in enumerate(train_loader):
                    waveform, mel_spec, labels = waveform.to(self.device), mel_spec.to(self.device), labels.to(self.device)
                    waveform = waveform.unsqueeze(1)
                    if "mfcc" in self.model_name:
                        outputs = self.model(mel_spec)
                    elif "waveform" in self.model_name:
                        outputs = self.model(waveform)
                    else:
                        mel_spec = mel_spec.unsqueeze(1)
                        outputs = self.model(waveform, mel_spec)
                    outputs_avg = torch.mean(outputs.squeeze(1), dim=1)
                    loss = criterion(outputs_avg, labels.long())
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_loss += loss.item()
                    # print(f'batch {batch_idx}, Loss: {loss.item()}')
                avg_loss = total_loss / len(train_loader)
                print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')

    def evaluate(self, dataset_name, test_loader):
        self.model.eval()
        if dataset_name == "librispeech" or dataset_name == "urbansound8k":
            all_preds = []
            all_labels = []
            for batch_idx, (waveform, mel_spec, labels) in enumerate(test_loader):
                waveform, mel_spec, labels = waveform.to(self.device), mel_spec.to(self.device), labels.to(self.device)
                waveform = waveform.unsqueeze(1)
                if "mfcc" in self.model_name:
                    outputs = self.model(mel_spec)
                    # Calculate the average probability for each class across all time steps
                    outputs = torch.mean(outputs, dim=-1)  # Assume the last dimension is the time dimension
                elif "waveform" in self.model_name:
                    outputs = self.model(waveform)
                    # Calculate the average probability for each class across all time steps
                    outputs = torch.mean(outputs, dim=-1)  # Assume the last dimension is the time dimension
                else:
                    mel_spec = mel_spec.unsqueeze(1)
                    outputs = self.model(waveform, mel_spec)
                    outputs_squeezed = torch.squeeze(outputs, dim=1)
                    outputs = torch.mean(outputs_squeezed, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1).cpu().numpy()


                all_preds.extend(predicted_classes.tolist())
                all_labels.extend(labels.cpu().tolist())


            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

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
            gc.collect()

    def save_model(self, path):
        """
        Save the model parameters to the specified path.

        Parameters:
        - path (str): The file path where the model parameters will be saved.
        """
        torch.save(self.model.state_dict(), f"{path}.pth")
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(path)


class PyTorchTextTrainer:
    def __init__(self, model_name, tokenizer, datasets, device="cuda:1", max_seq_length=128, batch_size=256,
                 learning_rate=2e-5, num_epochs=50, save_model_after_training=False):
        self.save_model_after_training = save_model_after_training
        self.model_name = model_name
        self.model = None
        self.datasets = datasets
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = None

    def load_and_process_data(self, data_name):
        if data_name == "IMDB":
            train_dataset = load_dataset('imdb', split='train')
            test_dataset = load_dataset('imdb', split='test')
        elif data_name == "SST2":
            train_dataset = load_dataset('glue', 'sst2', split='train')
            test_dataset = load_dataset('glue', 'sst2', split='validation')
        else:
            raise ValueError("Unsupported dataset")

        def transform_function(example):
            if data_name == "IMDB":
                # Apply your tokenizer preprocessing here
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
            # Initialize the batch of input_ids and labels
            input_ids = []
            labels = []
            for sample in batch:
                # Check if the sample contains 'input_ids' and 'labels'
                if 'input_ids' in sample and 'labels' in sample:
                    input_ids.append(sample['input_ids'])
                    labels.append(sample['labels'])
                else:
                    # If 'labels' are not found, an error should be raised or the sample should be skipped
                    raise KeyError("'labels' key not found in the sample.")

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            return input_ids, labels

        column = None

        if data_name == "IMDB":
            column = 'text'
        elif data_name == "SST2":
            column = 'sentence'
        else:

            raise KeyError("'data_name' key not found in the sample.")
        train_dataset = train_dataset.map(transform_function, batched=False, remove_columns=[column])
        test_dataset = test_dataset.map(transform_function, batched=False, remove_columns=[column])

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=collate_fn)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False,
                                     collate_fn=collate_fn)
        return train_dataloader, test_dataloader

    def select_model(self):
        if self.model_name == "roberta_model_6":
            encoder_conf = RobertaEncoderConf(num_encoder_layers=6)
        elif self.model_name == "roberta_model_12":
            encoder_conf = RobertaEncoderConf(num_encoder_layers=12)
        elif self.model_name == "roberta_model_24":
            encoder_conf = RobertaEncoderConf(num_encoder_layers=24)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        self.model = RobertaModel(encoder_conf=encoder_conf,
                                  head=RobertaClassificationHead(num_classes=2, input_dim=768)).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for input_ids, labels in train_loader:
                input_ids, labels = input_ids.to(self.device), labels.to(self.device)

                outputs = self.model(input_ids)
                logits = outputs if not isinstance(outputs, tuple) else outputs[0]
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}')

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for input_ids, labels in test_loader:
                input_ids, labels = input_ids.to(self.device), labels.to(self.device)

                outputs = self.model(tokens=input_ids)

                logits = outputs if not isinstance(outputs, tuple) else outputs[0]
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    def save_model(self, path):
        """
        Save the model parameters to the specified path.

        Parameters:
        - path (str): The file path where the model parameters will be saved.
        """
        torch.save(self.model.state_dict(), f"{path}.pth")
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(path)

    def train_and_evaluate(self):
        for dataset_name in self.datasets:
            print(f"Training on {dataset_name}")
            train_loader, test_loader = self.load_and_process_data(dataset_name)
            self.select_model()
            self.train(train_loader)
            self.evaluate(test_loader)
            if self.save_model_after_training:

                if not os.path.exists(os.path.join("CMC", "pytorch_pipline_model_result")):
                    os.makedirs(os.path.join("CMC", "pytorch_pipline_model_result"))
                self.save_model(os.path.join("CMC", "pytorch_pipline_model_result",
                                             "{}_{}_{}".format(self.model_name, dataset_name, self.num_epochs)))
            del self.model
            gc.collect()


class PyTorchVisionTrainer:
    def __init__(self, model_name, datasets, batch_size=128, learning_rate=0.001, num_epochs=100, device="cuda:0",
                 save_model_after_training=False):
        self.optimizer = None
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None  # The model needs to be accompanied by changes in the dataset
        self.model_name = model_name.lower()
        self.datasets = datasets
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.save_model_after_training = save_model_after_training

    def load_dataset(self, dataset_name):
        assert dataset_name in self.datasets, "The datasets are unsupported yet"
        num_classes = vision_dataset[dataset_name]["num_class"]
        if self.model_name == "alexnet":
            self.model = AlexNet(num_classes=num_classes).to(self.device)
        elif self.model_name == "resnet18":
            self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(self.device)
        elif self.model_name == "resnet34":
            self.model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes).to(self.device)
        elif self.model_name == "resnet50":
            self.model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes).to(self.device)
        elif self.model_name == "resnet101":
            self.model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes).to(self.device)
        elif self.model_name == "resnet152":
            self.model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes).to(self.device)
        elif self.model_name == "densenet121":
            self.model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_classes).to(self.device)
        elif self.model_name == "densenet161":
            self.model = DenseNet(growth_rate=48, block_config=(6, 12, 36, 24), num_classes=num_classes).to(self.device)
        elif self.model_name == "densenet169":
            self.model = DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), num_classes=num_classes).to(self.device)
        elif self.model_name == "densenet201":
            self.model = DenseNet(growth_rate=32, block_config=(6, 12, 48, 32), num_classes=num_classes).to(self.device)
        elif self.model_name == "densenet264":
            self.model = DenseNet(growth_rate=32, block_config=(6, 12, 64, 48), num_classes=num_classes).to(self.device)
        elif self.model_name == "squeezenet1_0":
            self.model = SqueezeNet(version="1_0", num_classes=num_classes).to(self.device)
        elif self.model_name == "squeezenet1_1":
            self.model = SqueezeNet(version="1_1", num_classes=num_classes).to(self.device)
        elif self.model_name == "vgg11":
            self.model = VGG(make_layers(cfgs["A"], batch_norm=True), num_classes=num_classes).to(self.device)
        elif self.model_name == "vgg13":
            self.model = VGG(make_layers(cfgs["B"], batch_norm=True), num_classes=num_classes).to(self.device)
        elif self.model_name == "vgg16":
            self.model = VGG(make_layers(cfgs["D"], batch_norm=True), num_classes=num_classes).to(self.device)
        elif self.model_name == "vgg19":
            self.model = VGG(make_layers(cfgs["E"], batch_norm=True), num_classes=num_classes).to(self.device)
        elif self.model_name == "mnasnet0_5":
            self.model = MNASNet(alpha=0.5, num_classes=num_classes).to(self.device)
        elif self.model_name == "mnasnet0_75":
            self.model = MNASNet(alpha=0.75, num_classes=num_classes).to(self.device)
        elif self.model_name == "mnasnet1_0":
            self.model = MNASNet(alpha=1.0, num_classes=num_classes).to(self.device)
        elif self.model_name == "mnasnet1_3":
            self.model = MNASNet(alpha=1.3, num_classes=num_classes).to(self.device)
        elif self.model_name == "mnasnet1_5":
            self.model = MNASNet(alpha=1.5, num_classes=num_classes).to(self.device)
        elif self.model_name == "mobilenetv2":
            self.model = mobilenetv2.MobileNetV2(num_classes=num_classes).to(self.device)
        elif self.model_name == "mobilenetv3_small":
            width_mult = 1.0  # You can adjust the width multiplier as needed
            inverted_residual_setting = [
                mobilenetv3.InvertedResidualConfig(input_channels=16, kernel=3, expanded_channels=16, out_channels=16,
                                                   use_se=False,
                                                   activation='RE', stride=2, dilation=1, width_mult=width_mult),

            ]
            self.model = mobilenetv3.MobileNetV3(inverted_residual_setting, last_channel=1280,
                                                 num_classes=num_classes).to(self.device)
        elif self.model_name == "mobilenetv3_large":
            width_mult = 1.0  # You can adjust the width multiplier as needed
            inverted_residual_setting = [
                mobilenetv3.InvertedResidualConfig(input_channels=16, kernel=3, expanded_channels=64, out_channels=24,
                                                   use_se=False,
                                                   activation='RE', stride=2, dilation=1, width_mult=width_mult),
            ]
            self.model = mobilenetv3.MobileNetV3(inverted_residual_setting,
                                                 last_channel=_utils._make_divisible(1280 * width_mult, 8),
                                                 num_classes=num_classes).to(self.device)
        elif self.model_name == "shufflenet_v2_0_5":
            self.model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                                      stages_out_channels=[24, 48, 96, 192, 1024],
                                      num_classes=num_classes).to(self.device)
        elif self.model_name == "shufflenet_v2_1_0":
            self.model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                                      stages_out_channels=[24, 116, 232, 464, 1024],
                                      num_classes=num_classes).to(self.device)
        elif self.model_name == "shufflenet_v2_1_5":
            self.model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                                      stages_out_channels=[24, 176, 352, 704, 1024],
                                      num_classes=num_classes).to(self.device)
        elif self.model_name == "shufflenet_v2_2_0":
            self.model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                                      stages_out_channels=[24, 244, 488, 976, 2048],
                                      num_classes=num_classes).to(self.device)
        elif self.model_name == "inception3highres":
            self.model = Inception3(num_classes=num_classes, aux_logits=True, transform_input=True).to(self.device)
        elif self.model_name == "inception3lite":
            self.model = Inception3(num_classes=num_classes, aux_logits=False, transform_input=False).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if dataset_name == "FashionMNIST":
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=vision_dataset[dataset_name]["mean_std"][0],
                                     std=vision_dataset[dataset_name]["mean_std"][1])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=vision_dataset[dataset_name]["mean_std"][0],
                                     std=vision_dataset[dataset_name]["mean_std"][1])
            ])
        if dataset_name == 'CIFAR10':
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'CIFAR100':
            train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
            test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'FashionMNIST':
            train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        else:
            raise ValueError("Unsupported dataset")

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.num_epochs):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                if not isinstance(outputs, InceptionOutputs):
                    outputs = outputs.to(self.device)
                    loss = self.criterion(outputs, labels)
                else:
                    # For Inception
                    main_output = outputs.logits.to(self.device)
                    loss = self.criterion(main_output, labels)
                    if outputs.aux_logits is not None:
                        aux_output = outputs.aux_logits.to(self.device)
                        aux_loss = self.criterion(aux_output, labels)
                        # main_loss and aux_loss
                        loss += 0.4 * aux_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    def evaluate(self, test_loader):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Calculate the various metrics
        accuracy = 100 * np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        print(f'Accuracy: {accuracy:.2f}%')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

    def predict(self, image):
        self.model.eval()
        image = image.to(self.device)
        with torch.no_grad():
            image = image.unsqueeze(0)  # Add a batch dimension
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()

    def save_model(self, path):
        """
        Save the model's state dictionary to the specified path.

        Parameters:
        - path (str): The file path where the model's state dictionary will be saved.
        """
        torch.save(self.model.state_dict(), f"{path}.pth")
        print(f"Model's state dict saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(path)

    def train_and_evaluate(self):
        for dataset_name in self.datasets:
            print(f"Processing dataset: {dataset_name}")
            train_loader, test_loader = self.load_dataset(dataset_name)
            self.train(train_loader)
            if self.save_model_after_training:

                if not os.path.exists(os.path.join("CMC", "pytorch_pipline_model_result")):
                    os.makedirs(os.path.join("CMC", "pytorch_pipline_model_result"))
                self.save_model(os.path.join("CMC", "pytorch_pipline_model_result",
                                             "{}_{}_{}".format(self.model_name, dataset_name, self.num_epochs)))
            self.evaluate(test_loader)
            del self.model
            gc.collect()


def parse_list_arg(arg_value):
    # Replace commas with spaces and split by spaces, filtering out empty strings
    return [item for item in arg_value.replace(',', ' ').split() if item]


def main(args):
    if args.model_type == 'vision':
        model_names = parse_list_arg(args.model_names)
        for model_name in model_names:
            print("=========================================================")
            print("===================={}=========================".format(model_name))
            trainer = PyTorchVisionTrainer(model_name=model_name, datasets=parse_list_arg(args.datasets),
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
            trainer = PyTorchTextTrainer(model_name=model_name, datasets=parse_list_arg(args.datasets),
                                         tokenizer=tokenizer, learning_rate=args.learning_rate,
                                         batch_size=args.batch_size, num_epochs=args.num_epochs,
                                         save_model_after_training=args.save_model, device=args.device)
            print("Training and evaluating RoBERTa Base model...")
            trainer.train_and_evaluate()

    elif args.model_type == 'audio':
        model_names = parse_list_arg(args.model_names)
        for model_name in model_names:
            print(fr"=========================================================")
            print(fr"===================={model_name}=========================")
            trainer = PyTorchAudioTrainer(model_name=model_name, datasets=parse_list_arg(args.datasets),
                                         batch_size=args.batch_size, learning_rate=args.learning_rate,
                                         num_epochs=args.num_epochs, save_model_after_training=args.save_model,
                                         device=args.device)
            trainer.train_and_evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate PyTorch models")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for training: 'cpu', 'cuda:0', 'cuda:1', etc.")
    parser.add_argument("--model_type", type=str, required=True, help="Type of the model: 'vision' or 'text'")
    parser.add_argument("--model_names", type=str, required=True,
                        help="List of model names to train and evaluate, separated by commas or spaces")
    parser.add_argument("--datasets", type=str, required=True,
                        help="List of datasets to use, separated by commas or spaces")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--save_model", default=False, action='store_true',
                        help="Flag to save the model after training")

    args = parser.parse_args()
    main(args)