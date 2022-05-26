import numpy as np
import sentencepiece as spm
import torch
from PIL import Image, ImageFile
from torchvision.transforms import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataReader:
    def __init__(self, dir, image0, image1, vocab_file):
        self.vocab = spm.SentencePieceProcessor(model_file='/content/drive/MyDrive/XRay_WebApp/Backend/input/vocab.model')
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.images = [image0, image1]
        self.max_len = 1000

    def get_data(self, history):
        imgs, vpos = [], []
        for i in range(len(self.images)):
            img = Image.open(self.images[i]).convert('RGB')
            imgs.append(self.transform(img).unsqueeze(0))  # (1,C,W,H)
            vpos.append(1)  # W
        cur_len = len(vpos)
        for i in range(cur_len, 2):
            imgs.append(torch.zeros_like(imgs[0]))
            vpos.append(-1)  # Empty mask

        sources = []
        imgs = torch.cat(imgs, dim=0)  # (V,C,W,H)
        imgs = imgs[None, :]
        vpos = np.array(vpos, dtype=np.int64)  # (V)
        vpos = vpos[None, :]
        vpos = torch.from_numpy(vpos)
        sources.append((imgs, vpos))
        encoded_source_info = [self.vocab.bos_id()] + self.vocab.encode(history) + [self.vocab.eos_id()]
        source_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        source_info[:min(len(encoded_source_info), self.max_len)] = encoded_source_info[
                                                                    :min(len(encoded_source_info), self.max_len)]
        sources.append(torch.from_numpy(source_info))
        return sources
