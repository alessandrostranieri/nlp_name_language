import logging
from typing import List, Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, SequenceLabelField
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, TokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("name-reader")
class NameLanguageDatasetReader(DatasetReader):

    def __init__(self,
                 lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        # This is the default
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                name, language = line.rsplit(maxsplit=1)
                yield self.text_to_instance([Token(character) for character in name], [language] * len(name))

    @overrides
    def text_to_instance(self, name_character: List[Token], language: List[str] = None) -> Instance:  # type: ignore
        name_field = TextField(name_character, self.token_indexers)
        fields = {"name": name_field}

        if language:
            label_field = SequenceLabelField(labels=language, sequence_field=name_field)
            fields["labels"] = label_field

        return Instance(fields)
