import numpy as np
import pandas as pd
import torch
from transformers import *


class TransformerRSM(object):

    def __init__(self, stimulus_name, model_name='bert-base-uncased', file_path=None, verbose=False):

        self.stimulus_name = stimulus_name
        self.model_name = model_name
        self.verbose = verbose
        self.stimulus_df = self._load_stimulus(file_path=file_path)

        self.attention_array = None

        # A list of lists: array[TR][layer] will be a tensor of shape (n_tokens, d_model) which contains the
        # model embeddings for that layer.
        self.tr_activations_array = None

        if 'bert' in self.model_name:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

            # Models can return full list of hidden-states & attentions weights at each layer
            self.transformer = BertModel.from_pretrained(self.model_name,
                                                         output_hidden_states=True,
                                                         output_attentions=True)

            # Use special tokens with BERT, but then slice them off when returning activations
            self.use_special_tokens = True
            self.last_token_index = -1

        elif 'gpt' in self.model_name:

            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

            # Models can return full list of hidden-states & attentions weights at each layer
            self.transformer = GPT2Model.from_pretrained(self.model_name,
                                                         output_hidden_states=True,
                                                         output_attentions=True)

            # Don't use special tokens with GPT, so return whole list of embeddings
            self.use_special_tokens = False
            self.last_token_index = None

        self.transformer.eval()

    def _load_stimulus(self, file_path=None):

        if file_path is None:
            file_path = 'data/stimuli/{}/tr_tokens.csv'.format(self.stimulus_name)

        print("Looking for TR-aligned tokens in {}".format(file_path))
        stimulus_df = pd.read_csv(file_path)

        stimulus_df.n_tokens = stimulus_df.n_tokens.fillna(0)
        stimulus_df.tokens = stimulus_df.tokens.fillna("")

        print("Loaded {} TRs.".format(len(stimulus_df)))

        return stimulus_df

    def process_stimulus_activations(self, num_context_trs=20):

        tr_chunked_tokens = self.stimulus_df.tokens.values
        self.tr_activations_array = []

        # Enumerate over the TR-aligned tokens
        for i, tr in enumerate(tr_chunked_tokens):

            # Add a new empty array to our BERT outputs
            self.tr_activations_array.append([])

            # Get the full context window for this TR, e.g. the appropriate preceding number of TRs.
            context_window_index_start = max(0, i - num_context_trs)
            window_stimulus = " ".join(tr_chunked_tokens[context_window_index_start:i + 1])
            window_token_ids = torch.tensor([self.tokenizer.encode(window_stimulus,
                                                                   add_special_tokens=self.use_special_tokens)])

            tr_token_ids = torch.tensor([self.tokenizer.encode(tr, add_special_tokens=False)])[0]

            with torch.no_grad():
                bert_states, _ = self.transformer(window_token_ids)[-2:]

            if self.verbose:
                if len(tr_token_ids) > 0:
                    tr_tokens = self.tokenizer.convert_ids_to_tokens(tr_token_ids.numpy())
                    print("TR {}: {} --> {}".format(i, tr, tr_tokens))
                else:
                    if self.last_token_index is None:
                        tr_tokens = self.tokenizer.convert_ids_to_tokens(
                            window_token_ids[0][-1:].numpy())
                    else:
                        tr_tokens = self.tokenizer.convert_ids_to_tokens(
                            window_token_ids[0][-2:self.last_token_index].numpy())
                    print("Empty TR. Using token: {}".format(tr_tokens))

            for layer in bert_states:

                if len(tr_token_ids) > 0:
                    # last_token_index is either -1 or None
                    # If -1, we slice off the last token and don't include (e.g. SEP token for BERT)
                    # If None, we include the last token (e.g. for GPT)
                    tr_activations = layer[0][-(len(tr_token_ids) + 1):self.last_token_index]

                else:

                    # ASSUMPTION: if we don't have any tokens in this TR, use the last "content" token's representation.
                    if self.last_token_index is None:
                        tr_activations = layer[0][-1:]
                    else:
                        tr_activations = layer[0][-2:self.last_token_index]

                # Append this set of activations onto our list of stimuli
                self.tr_activations_array[-1].append(tr_activations)

        print("Processed {} TRs for activations.".format(len(self.tr_activations_array)))

    @classmethod
    def mean_tr_response_across_tokens(cls, tr_layer_tokens_tensor):
        """tr_tensor is tr x layer x tokens; squash it to tr x layer x mean and return that."""

        tr_layer_mean_tensor = []
        for tr in tr_layer_tokens_tensor:
            tr_layer_mean_tensor.append([])
            for layer in tr:
                tr_layer_mean_tensor[-1].append(torch.mean(layer, 0))

        return tr_layer_mean_tensor

    @classmethod
    def layerwise_rsms_from_vectors(cls, tr_layer_tensor):
        """tr_layer_tensor is tr x layer x vector. Produce a layer-wise RSM of it. """

        # We now have the layers in a TR-based index: tr_activation_array[tr_index][layer_index]
        # To get a layer-wise RSM, we need to reverse this indexing.
        layerwise_rsms = []
        for layer_index in range(0, len(tr_layer_tensor[0])):
            # Loop over our TR array and grab the target layer for each
            layer_activations = [tr_layer_tensor[tr_index][layer_index] for tr_index, _ in enumerate(tr_layer_tensor)]

            stacked_layer = np.stack([layer_vector.numpy() for layer_vector in layer_activations], axis=0)
            rsm_dataframe = pd.DataFrame(np.corrcoef(stacked_layer))
            layerwise_rsms.append(rsm_dataframe)

        return layerwise_rsms
