
import torch
import torch.nn as nn
        
class JoinText(nn.Module):
  def __init__(
      self,
      signal_model,

      signal_in_chanels = 1280,
      text_in_chanels = 768,
      out_chanels = 1280,
    ):
    
      super().__init__()
      self.signal_model = signal_model

      self.signal_in_chanels = signal_in_chanels
      self.text_in_chanels = text_in_chanels
      self.out_chanels = out_chanels
      
      self.W_s = nn.Linear(self.signal_in_chanels, self.out_chanels)
      self.W_t = nn.Linear(self.text_in_chanels, self.out_chanels)

  def forward(
      self,
      signal,
      text_features,
    ):
    
      output = self.signal_model(signal)
      signal_embedding = output['signal_embedding']
      logits = output['logits']

      signal_embedding = self.W_s(signal_embedding)
      text_embedding = self.W_t(text_features)

      return {'logits': logits, 
              'signal_embedding': signal_embedding, 
              'text_embedding': text_embedding}
  