# from https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# pylint: disable=invalid-name,arguments-differ,redefined-outer-name
from typing import Dict

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy


@Model.register('name-classifier')
class NameClassifier(Model):
    def __init__(self,
                 name_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.name_embedder = name_embedder
        self.encoder = encoder
        # This will be hidden_dim to output_dim
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))

        self.accuracy = CategoricalAccuracy()

    def forward(self,
                name: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> torch.Tensor:
        embedded = self.name_embedder(name)
        masked = get_text_field_mask(name)
        encoded = self.encoder(embedded, masked)
        tag_logits = self.hidden2tag(encoded)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels, masked)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, masked)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}