import torch

from config import config
from src.data.tokenizer import Tokenizer

class InferenceModel:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model = self.load_model(config['inference']['path'])
        self.max_len = config['inference']['max_len']
        self.temperature = config['inference']['temperature']

    def load_model(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        model_name = config['inference']['model']
        if model_name.lower() == 'rnn':
            from src.archs.rnn import RNN as nn
        else:
            raise NotImplementedError('chel, takoy modeli net')
        model = nn(self.tokenizer.pad_idx, self.tokenizer.vocab_size)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model
    
    def next_token(self, x, temperature):
        logits = self.model(x)[-1, :]
        out = self.pick_by_distribution(logits/temperature)
        return out


    @staticmethod
    def pick_by_distribution(logits):
        probs = torch.exp(logits)
        sample = torch.distributions.Categorical(probs).sample()
        return sample
    
    @torch.inference_mode()
    def gen_continuation(self, prefix=''):
        out = None
        t_count = 0

        x = self.tokenizer.encode(prefix)[:-1]
        out = self.next_token(x, temperature=1.5)
        x = torch.cat((x, out.unsqueeze(0)))
        t_count += 1

        while out != self.tokenizer.eos_idx and t_count < self.max_len:
            out  = self.next_token(x, temperature=self.temperature)
            x = torch.cat((x, out.unsqueeze(0)))
            t_count += 1
        x = self.tokenizer.decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(' ##', '')
        return x


