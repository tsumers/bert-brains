import copy

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

        # A list of lists: array[TR][layer] will be a tensor of shape (n_tokens, d_model) which contains the
        # model embeddings for that layer.


        # A list of lists for attentions. array[tr][layer] will be a tensor of shape (n_tokens x n_tokens) for
        # however many attention tokens we want to consider.
        self.tr_attentions_array = None

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

    # BASIC processing: generate embeddings and attention outputs for stimulus.

    def process_stimulus_activations(self, num_context_trs=20):

        tr_chunked_tokens = self.stimulus_df.tokens.values
        tr_activations_array = []
        tr_tokens_array = []

        # Enumerate over the TR-aligned tokens
        for i, tr in enumerate(tr_chunked_tokens):

            # Add a new empty array to our BERT outputs
            tr_activations_array.append([])

            # Get the full context window for this TR, e.g. the appropriate preceding number of TRs.
            context_window_index_start = max(0, i - num_context_trs)
            window_stimulus = " ".join(tr_chunked_tokens[context_window_index_start:i + 1])
            window_token_ids = torch.tensor([self.tokenizer.encode(window_stimulus,
                                                                   add_special_tokens=self.use_special_tokens)])

            tr_token_ids = torch.tensor([self.tokenizer.encode(tr, add_special_tokens=False)])[0]

            with torch.no_grad():
                embeddings, _ = self.transformer(window_token_ids)[-2:]

            if len(tr_token_ids) > 0:
                tr_tokens = self.tokenizer.convert_ids_to_tokens(tr_token_ids.numpy())
                if self.verbose:
                    print("TR {}: {} --> {}".format(i, tr, tr_tokens))
            else:
                if self.last_token_index is None:
                    tr_tokens = self.tokenizer.convert_ids_to_tokens(
                        window_token_ids[0][-1:].numpy())
                else:
                    tr_tokens = self.tokenizer.convert_ids_to_tokens(
                        window_token_ids[0][-2:self.last_token_index].numpy())
                if self.verbose:
                    print("Empty TR. Using token: {}".format(tr_tokens))

            tr_tokens_array.append(tr_tokens)

            for layer in embeddings:

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
                tr_activations_array[-1].append(tr_activations)

        self.stimulus_df["activations"] = tr_activations_array
        self.stimulus_df["transformer_activations_tr_tokens"] = tr_tokens_array
        print("Processed {} TRs for activations.".format(len(tr_activations_array)))

    def process_stimulus_attentions(self, num_window_tokens=10):
        """Return window_tokens x tr_tokens x num_heads attention matrix"""

        tr_chunked_tokens = self.stimulus_df.tokens.values
        self.tr_attentions_array = []

        # Enumerate over the TR-aligned tokens
        for i, tr in enumerate(tr_chunked_tokens):

            # Get all of the stimulus up to this point (since we'll chop it down later)
            window_stimulus = " ".join(tr_chunked_tokens[max(0, i-60):i + 1])

            # Get the list of BERT tokens involved in that window
            window_tokens = self.tokenizer.encode_plus(window_stimulus, return_tensors='pt',
                                                       add_special_tokens=self.use_special_tokens)

            # window_token_ids is now a tensor containing *all* of the tokens in this TR stimulus window.
            # We need to filter it down to our fixed dimensionality, num_window_tokens
            window_token_ids = window_tokens['input_ids']

            if len(window_token_ids[0]) < num_window_tokens:
                # We don't have enough words yet. This is typical for the first few TRs-- we need to build up context.
                if self.verbose:
                    print("TR {}: not enough tokens ({}/{})".format(i, len(window_token_ids[0]), num_window_tokens))
                continue

            # N.B.: we don't currently account for CLS / SEP token here, so we'll have 18-19 "usable" tokens in the
            # case that we have them in input.
            truncated_window_token_ids = window_token_ids[0][-num_window_tokens:]
            if self.verbose:
                print("TR {}: more than enough tokens ({}/{}), trimmed to {}.".format(i,
                                                                                      len(window_token_ids[0]),
                                                                                      num_window_tokens,
                                                                                      len(truncated_window_token_ids)))

            with torch.no_grad():

                attentions = self.transformer(truncated_window_token_ids.reshape(1, -1))[-1]

                # attentions is now a tuple of length n_layers
                # Each element of the tuple contains the outputs of each attention head in that layer.
                # So, for example, attentions[0] will be of
                #       torch.Size([1, n_heads, num_window_tokens, num_window_tokens])
                # Running this on bert-base with num_window_tokens = 40 will yield:
                # len(attentions) = 12 --> 12 layers in the model
                # attentions[0].shape = torch.Size([1, 12, 40, 40]) --> 12 attention heads, 40x40 attention weights
                if self.verbose:
                    print("Extracting heads of shape {}.".format(attentions[0].shape))

                # for layer in attentions:
                #     # Need to flatten our attentions to 1D vector.
                #     # This is complicated but basically collapses all attention heads down to a single vector
                #     # which will have length = n_heads * n_window_tokens * n_window_tokens
                #     tr_attentions = layer[0].reshape(1, -1)[0]

                self.tr_attentions_array.append(attentions)

    # ADVANCED processing: use the stimulus_df entries to generate more interesting representations.

    def _mask_head_attention(self, head_matrix, window, mask_token_self_attention=True):
        """Mask out (e.g. set to zero) all token-token attention weights that are not of interest."""

        masked_head_matrix = head_matrix.detach().clone()

        if self.use_special_tokens:
            # Mask attention both to and from last token (SEP)
            masked_head_matrix[:, -1] = 0
            masked_head_matrix[-1, :] = 0

            # and walk the window back 1 to account for that
            window += 1

            # Mask attention both to and from first token (possibly CLS)
            masked_head_matrix[:, 0] = 0
            masked_head_matrix[0, :] = 0

        # Remove tokens' attentions to themselves
        if mask_token_self_attention:
            masked_head_matrix.fill_diagonal_(0)

        # Finally, mask attention *from* preceding (non-window) tokens
        masked_head_matrix[:-window, :] = 0

        return masked_head_matrix

    def mask_non_tr_attentions(self, num_tokens_per_tr):
        """Iterate over the attentions array and mask out attentions that (probably) don't contribute meaningfully.

        num_tokens_per_tr is an estimate for how many BERT/GPT tokens appear per TR. This function will mask out
        all attention weights except those coming *from* the tr tokens, going *to* the window tokens.
        So each matrix is originally [num_window_tokens x num_window_tokens]; after masking it will have
        [num_tokens_per_tr x num_window_tokens] nonzero entries."""

        # self.tr_attentions_array = [num_trs][num_layers][0][heads][from_token][to_token]
        # so for bert-base-uncased, 10 TRs: [10][12][0][12][num_window_tokens][num_window_tokens]

        masked = copy.deepcopy(self.tr_attentions_array)

        for tr in range(0, len(masked)):
            for layer in range(0, len(masked[tr])):
                for head in range(0, len(masked[tr][layer][0])):
                    masked[tr][layer][0][head] = self._mask_head_attention(masked[tr][layer][0][head], num_tokens_per_tr)

        return masked

    @classmethod
    def attention_head_magnitudes(cls, attention_array, p='inf'):

        tr_attention_vector_array = []
        for tr in range(0, len(attention_array)):
            tr_attention_vector_array.append([])
            for layer in range(0, len(attention_array[tr])):
                for head in range(0, len(attention_array[tr][layer][0])):
                    norm = torch.norm(attention_array[tr][layer][0][head], p=float(p))
                    tr_attention_vector_array[-1].append(norm)

        return tr_attention_vector_array

    def mean_tr_response_across_tokens(self, input_tensor_name="activations"):
        """tr_tensor is tr x layer x tokens; squash it to tr x layer x mean and return that."""

        tr_layer_mean_tensor = []
        for tr in self.stimulus_df[input_tensor_name]:
            tr_layer_mean_tensor.append([])
            for layer in tr:
                tr_layer_mean_tensor[-1].append(torch.mean(layer, 0))

        return tr_layer_mean_tensor

    @classmethod
    def max_l2_move_per_layer(cls, tr_layer_tokens_tensor):
        """tr_tensor is tr x layer x tokens; look at l2 distance for each token / layer."""

        tr_layer_l2_distance_tensor = []
        for tr in tr_layer_tokens_tensor:
            tr_layer_l2_distance_tensor.append([])
            for layer_num, (layer_one, layer_two) in enumerate(zip(tr, tr[1:])):

                # Get the number of tokens to compare
                n_tokens = layer_one.shape[0]
                l2_differences = [torch.dist(layer_one[i], layer_two[i], 2) for i in range(0, n_tokens)]
                max_l2_distance = max(l2_differences)
                # print("Layer {}: distance {}".format(layer_num, max_l2_distance))

                tr_layer_l2_distance_tensor[-1].append(max_l2_distance.item())

        return tr_layer_l2_distance_tensor

    @classmethod
    def layer_activations_from_tensor(cls, tr_layer_tensor, layer_index):
        """Return all entries for a given layer across all TRs."""

        return [tr_layer_tensor[tr_index][layer_index] for tr_index, _ in enumerate(tr_layer_tensor)]

    @classmethod
    def layerwise_rsms_from_vectors(cls, tr_layer_tensor):
        """tr_layer_tensor is tr x layer x vector. Produce a layer-wise RSM of it. """

        # We now have the layers in a TR-based index: tr_activation_array[tr_index][layer_index]
        # To get a layer-wise RSM, we need to reverse this indexing.
        layerwise_rsms = []
        for layer_index in range(0, len(tr_layer_tensor[0])):
            # Loop over our TR array and grab the target layer for each
            layer_activations = cls.layer_activations_from_tensor(tr_layer_tensor, layer_index)

            stacked_layer = np.stack([layer_vector.numpy() for layer_vector in layer_activations], axis=0)
            rsm_dataframe = pd.DataFrame(np.corrcoef(stacked_layer))
            layerwise_rsms.append(rsm_dataframe)

        return layerwise_rsms

    @classmethod
    def full_rsm_from_scalars(cls, tr_layer_tensor):
        """tr_layer_tensor is tr x layer x scalar: each TR-Layer is represented by a single scalar value.

        Return an RSM that looks at the whole model's state. This can be used with outputs of max_l2_move_per_layer."""

        rsm_dataframe = pd.DataFrame(np.corrcoef(tr_layer_tensor))
        return rsm_dataframe
