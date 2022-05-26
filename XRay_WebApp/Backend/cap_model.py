import torch
from torch import nn

from Backend.models import CNN, MVCNN, TNN, Classifier, ClsGen, ClsGenInt
from Backend.base_cmn import BaseCMN
from Backend.data import DataReader


class Model:
    def __init__(self, directory, image0, image1):
        self.dataset = DataReader(directory, image0, image1, '/vocab.model')
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        args = {'fc_features': 1024, 'num_embed': 1024, 'num_heads': 8, 'fd_dim': 256, 'dropout': 0.1,
                'decease_related_topics': 114, 'num_classes': 2, 'num_layers': 3, 'd_model': 512, 'd_ff': 512,
                'topk': 32, 'max_seq_length': 300, 'd_vf': 1024, 'cmm_size': 2048, 'cmm_dim': 512, 'use_bn': 0,
                'drop_prob_lm': 0.5}

        vocab_size = len(self.dataset.vocab)
        posit_size = 1000

        visual_extractor = CNN(backbone)
        visual_extractor = MVCNN(visual_extractor)
        text_feat_extractor = TNN(embed_dim=args['num_embed'], num_heads=args['num_heads'], fwd_dim=args['fd_dim'],
                                  dropout=args['dropout'], num_layers=1, num_tokens=vocab_size, num_posits=posit_size)
        cls_model = Classifier(num_topics=args['decease_related_topics'], num_states=args['num_classes'],
                               cnn=visual_extractor, tnn=text_feat_extractor, fc_features=args['fc_features'],
                               embed_dim=args['num_embed'], num_heads=args['num_heads'], dropout=args['dropout'])
        gen_model = BaseCMN(args, self.dataset.vocab)

        cls_gen_model = ClsGen(cls_model, gen_model, args['decease_related_topics'], args['num_embed'])
        cls_gen_model = nn.DataParallel(cls_gen_model).cuda()
        int_text_feat_extractor = TNN(embed_dim=args['num_embed'], num_heads=args['num_heads'], fwd_dim=args['fd_dim'],
                                      dropout=args['dropout'], num_layers=1,
                                      num_tokens=vocab_size, num_posits=posit_size)
        int_model = Classifier(num_topics=args['decease_related_topics'], num_states=args['num_classes'], cnn=None,
                               tnn=int_text_feat_extractor, embed_dim=args['num_embed'],
                               num_heads=args['num_heads'], dropout=args['dropout'])
        int_model = nn.DataParallel(int_model).cuda()
        self.model = ClsGenInt(cls_gen_model.module.cpu(), int_model.module.cpu(), freeze_evaluator=True)
        self.model = nn.DataParallel(self.model).cuda()
        checkpoint = torch.load('/content/X_GEN/XRay_WebApp/Backend/input/model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_report(self, history, device='cpu', threshold=None):
        self.model.eval()

        with torch.no_grad():
            source = self.dataset.get_data(history=history)
            source = self.data_to_device(source, device)

            # Use single input if there is no clinical history
            if threshold is not None:
                output, _ = self.model(image=source[0], history=source[3], threshold=threshold)

            else:
                output, _ = self.model(source[0])

            candidate = self.decode_report(output[0])
        return candidate

    def data_to_device(self, data, device='cpu'):
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, tuple):
            data = tuple(self.data_to_device(item, device) for item in data)
        elif isinstance(data, list):
            data = list(self.data_to_device(item, device) for item in data)
        elif isinstance(data, dict):
            data = dict((k, self.data_to_device(v, device)) for k, v in data.items())
        else:
            raise TypeError('Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.')
        return data

    def decode_report(self, caption):
        decoded = ''
        for j in range(len(caption)):
            tok = self.dataset.vocab.id_to_piece(int(caption[j]))
            if tok == '</s>':
                break  # Manually stop generating token after </s> is reached
            elif tok == '<s>':
                continue
            elif tok == '▁':  # space
                if len(decoded) and decoded[-1] != ' ':
                    decoded += ' '
            elif tok in [',', '.', '-', ':']:  # or not tok.isalpha():
                if len(decoded) and decoded[-1] != ' ':
                    decoded += ' ' + tok + ' '
                else:
                    decoded += tok + ' '
            else:  # letter
                decoded += tok
        return decoded

