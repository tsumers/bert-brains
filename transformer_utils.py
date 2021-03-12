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
        tr_z_reps_array = []

        # Enumerate over the TR-aligned tokens
        for i, tr in enumerate(tr_chunked_tokens):

            # Get the full context window for this TR, e.g. the appropriate preceding number of TRs.
            context_window_index_start = max(0, i - num_context_trs)
            window_stimulus = " ".join(tr_chunked_tokens[context_window_index_start:i + 1])
            window_token_ids = torch.tensor([self.tokenizer.encode(window_stimulus,
                                                                   add_special_tokens=self.use_special_tokens)])

            tr_token_ids = torch.tensor([self.tokenizer.encode(tr, add_special_tokens=False)])[0]

            # For empty TRs, append None and continue on.
            # We'll forward-fill these later.
            if len(tr_token_ids) == 0:
                tr_tokens_array.append(None)
                tr_activations_array.append(None)
                tr_z_reps_array.append(None)
                continue
        
            z_reps = []
            with torch.no_grad():
                embeddings, _ = self.transformer(window_token_ids)[-2:]
                for layer_num, layer in enumerate(embeddings[:-1]):
                    z_reps.append(self.transformer.encoder.layer[layer_num].attention.self(layer)[0])


            tr_tokens = self.tokenizer.convert_ids_to_tokens(tr_token_ids.numpy())
            tr_tokens_array.append(tr_tokens)

            if self.verbose:
                print("\nTR {}: Window Stimulus: {}".format(i, window_stimulus))
                print("\t TR stimulus: {}".format(tr_tokens))

            # Add a new empty array to our BERT outputs
            tr_activations_array.append([])
            for layer in embeddings:

                # last_token_index is either -1 or None
                # If -1, we slice off the last token and don't include (e.g. SEP token for BERT)
                # If None, we include the last token (e.g. for GPT)
                tr_activations = layer[0][-(len(tr_token_ids) + 1):self.last_token_index]
                tr_activations_array[-1].append(tr_activations)
                    
            tr_z_reps_array.append([])
            for z in z_reps:

                # last_token_index is either -1 or None
                # If -1, we slice off the last token and don't include (e.g. SEP token for BERT)
                # If None, we include the last token (e.g. for GPT)
                tr_z_reps = z[0][-(len(tr_token_ids) + 1):self.last_token_index]
                tr_z_reps_array[-1].append(tr_z_reps)

        self.stimulus_df["activations"] = tr_activations_array
        self.stimulus_df["z_reps"] = tr_z_reps_array
        self.stimulus_df["transformer_tokens_in_tr"] = tr_tokens_array

        # Forward-fill our activations, but *not* the tokens-in-TR
        self.stimulus_df["activations"].ffill(inplace=True)
        self.stimulus_df["z_reps"].ffill(inplace=True)

        self.stimulus_df["n_transformer_tokens_in_tr"] = list(map(lambda x: len(x) if x else 0, tr_tokens_array))

        print("Processed {} TRs for activations.".format(len(tr_activations_array)))

    def process_stimulus_attentions(self, num_window_tokens=20):
        """Return window_tokens x tr_tokens x num_heads attention matrix"""

        tr_chunked_tokens = self.stimulus_df.tokens.values
        tr_attentions_array = []
        tr_tokens_array = []
        first_successful_window = None

        # Enumerate over the TR-aligned tokens
        for i, tr in enumerate(tr_chunked_tokens):

            # Get all of the stimulus up to this point (since we'll chop it down later)
            # N.B. if there are *missing* attention windows in the middle of the sequence, this is likely the culprit.
            # Increase it or remove it.
            PRECEEDING_TR_CONTEXT_CAP = 200
            window_stimulus = " ".join(tr_chunked_tokens[max(0, i - PRECEEDING_TR_CONTEXT_CAP):i + 1])

            if i % 100 == 0:
                print("Processing TR {}".format(i))

            # Dangerous, but suppresses an annoying error message that comes up if we don't do this.
            # https://github.com/huggingface/transformers/issues/3050#issuecomment-682167272
            if self.verbose is False:
                import logging
                set_global_logging_level(logging.ERROR)
            # Get the list of BERT tokens involved in that window
            window_tokens = self.tokenizer.encode_plus(window_stimulus, return_tensors='pt',
                                                       add_special_tokens=self.use_special_tokens)

            # window_token_ids is now a tensor containing *all* of the tokens in this TR stimulus window.
            # We need to filter it down to our fixed dimensionality, num_window_tokens
            window_token_ids = window_tokens['input_ids']

            if len(window_token_ids[0]) < num_window_tokens:
                if first_successful_window is not None:
                    raise ValueError("ERROR: failed attention context window in middle of sequence. \
                    Consider adjusting PRECEEDING_TR_CONTEXT_CAP upwards.")

                # We don't have enough words yet. This is typical for the first few TRs-- we need to build up context.
                tr_attentions_array.append(None)
                tr_tokens_array.append(self.tokenizer.convert_ids_to_tokens(window_token_ids[0].tolist()))

                if self.verbose:
                    print("TR {}: not enough tokens ({}/{})".format(i, len(window_token_ids[0]), num_window_tokens))
                continue

            else:
                first_successful_window = i

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
                squeezed = [layer.squeeze() for layer in attentions]
                tr_attentions_array.append(squeezed)
                tr_tokens_array.append(self.tokenizer.convert_ids_to_tokens(truncated_window_token_ids.tolist()))

                # attentions is now a tuple of length n_layers
                # Each element of the tuple contains the outputs of each attention head in that layer.
                # So, for example, attentions[0] will be of
                #       torch.Size([1, n_heads, num_window_tokens, num_window_tokens])
                # Running this on bert-base with num_window_tokens = 40 will yield:
                # len(attentions) = 12 --> 12 layers in the model
                # attentions[0].shape = torch.Size([12, 40, 40]) --> 12 attention heads, 40x40 attention weights
                if self.verbose:
                    print("Extracting heads of shape {}.".format(squeezed[0].shape))

                ## OLD LAYER LOGIC: need to delete this after integrating it elsewhere
                # for layer in attentions:
                #     # Need to flatten our attentions to 1D vector.
                #     # This is complicated but basically collapses all attention heads down to a single vector
                #     # which will have length = n_heads * n_window_tokens * n_window_tokens
                #     tr_attentions = layer[0].reshape(1, -1)[0]

        self.stimulus_df["attentions"] = tr_attentions_array
        self.stimulus_df["attentions_transformer_tokens"] = tr_tokens_array

    # ADVANCED processing: use the stimulus_df entries to generate more interesting representations.

    def _mask_head_attention(self, head_matrix, window, include_backwards=True, include_forwards=True):
        """Mask out (e.g. set to zero) all token-token attention weights that are not of interest."""

        masked_head_matrix = head_matrix.detach().clone()

        if self.use_special_tokens:
            # Mask attention both to and from last token (SEP)
            masked_head_matrix[:, -1] = 0
            masked_head_matrix[-1, :] = 0

            # and walk the window back 1 to account for that
            window += 1

        if self.use_special_tokens or "gpt" in self.model_name:
            # GPT-2 attends to the first token for `null` attention: https://www.aclweb.org/anthology/W19-4808.pdf
            # and BERT generally uses special tokens, which puts SEP there.
            # So for both, we mask out the attention.
            masked_head_matrix[:, 0] = 0
            masked_head_matrix[0, :] = 0

        # Mask out tokens' attentions to themselves
        masked_head_matrix.fill_diagonal_(0)

        # Mask out attention between non-TR tokens (e.g. upper-left chunk)
        masked_head_matrix[:-window, :-window] = 0

        # If we don't want forward attentions (e.g. previous tokens looking to this TR)
        if include_forwards is False:
            # Then only keep below diagonal
            masked_head_matrix.triu_()
        # If we don't want backwards attentions (e.g. tokens in this TR looking to previous tokens)
        if include_backwards is False:
            # Then only keep above the diagonal
            masked_head_matrix.tril_()

        return masked_head_matrix

    def mask_non_tr_attentions(self, num_tokens_per_tr=None, include_backwards=True, include_forwards=True,
                               masked_col_name="masked_attentions"):
        """Iterate over the attentions array and mask out attentions that (probably) don't contribute meaningfully.

        num_tokens_per_tr is an estimate for how many BERT/GPT tokens appear per TR. This function will mask out
        all attention weights except those coming *from* the tr tokens, going *to* the window tokens.
        So each matrix is originally [num_window_tokens x num_window_tokens]; after masking it will have
        [num_tokens_per_tr x num_window_tokens] nonzero entries."""

        # Attention structure: = stimulus_df[num_trs][num_layers][heads][from_token][to_token]
        # so for bert-base-uncased, 10 TRs: [10][12][12][num_window_tokens][num_window_tokens]

        masked = copy.deepcopy(self.stimulus_df.attentions.values)

        if num_tokens_per_tr is None:
            n_tokens = self.stimulus_df.n_transformer_tokens_in_tr
        else:
            n_tokens = np.full(len(masked), num_tokens_per_tr)

        for tr in range(0, len(masked)):
            # We didn't process attentions for this TR, likely because there weren't enough tokens.
            if masked[tr] is None:
                continue

            for layer in range(0, len(masked[tr])):
                for head in range(0, len(masked[tr][layer])):
                    masked[tr][layer][head] = self._mask_head_attention(masked[tr][layer][head], n_tokens[tr],
                                                                        include_backwards=include_backwards,
                                                                        include_forwards=include_forwards)

        self.stimulus_df[masked_col_name] = masked

    def compute_attention_head_magnitudes(self, p='inf', attention_col="masked_attentions"):

        tr_attention_vector_array = []
        for tr in range(0, len(self.stimulus_df[attention_col])):

            if self.stimulus_df[attention_col][tr] is None:
                tr_attention_vector_array.append(None)
                continue
            else:
                # One entry per TR
                tr_attention_vector_array.append([])

            for layer in range(0, len(self.stimulus_df[attention_col][tr])):
                # One entry per layer
                tr_attention_vector_array[-1].append([])
                for head in range(0, len(self.stimulus_df[attention_col][tr][layer])):
                    norm = torch.norm(self.stimulus_df[attention_col][tr][layer][head], p=float(p))
                    tr_attention_vector_array[-1][-1].append(norm.item())

        self.stimulus_df["attention_heads_L{}".format(p)] = tr_attention_vector_array

    def compute_attention_head_distances(self, attention_col="masked_attentions"):
        """ Implements Attention Distance metric as in:
        https://www.aclweb.org/anthology/W19-4808.pdf
        """

        # First, build the appropriate dimension distance-weighting mask.
        # Choose TR magic number 100 because by this point we've (almost certainly) built up enough
        # context to actually have an attention matrix (this is about 2.5 min into story).
        # We just need to get the dimensions so we can build the right shaped mask.
        example_head_matrix = self.stimulus_df[attention_col][200][0][0]
        n_tokens = example_head_matrix.shape[0]

        # Construct our distance mask
        index_array = np.array(range(1, n_tokens + 1))
        columns = np.tile(index_array, (n_tokens, 1))
        rows = np.tile(index_array, (n_tokens, 1)).T
        distance_mask = abs((rows - columns))

        tr_attention_vector_array = []
        for tr in range(0, len(self.stimulus_df[attention_col])):

            if tr % 100 == 0:
                print("Processing TR {}.".format(tr))

            if self.stimulus_df[attention_col][tr] is None:
                tr_attention_vector_array.append(None)
                continue
            else:
                # One entry per TR
                tr_attention_vector_array.append([])

            for layer in range(0, len(self.stimulus_df[attention_col][tr])):
                # One entry per layer
                tr_attention_vector_array[-1].append([])
                for head in range(0, len(self.stimulus_df[attention_col][tr][layer])):
                    head_matrix = self.stimulus_df[attention_col][tr][layer][head]
                    distance_weighted_attention = np.multiply(distance_mask, head_matrix)
                    tr_attention_vector_array[-1][-1].append(distance_weighted_attention.sum().item())

        self.stimulus_df["attention_distances"] = tr_attention_vector_array

    def mean_tr_response_across_tokens(self, input_tensor_name="activations"):
        """tr_tensor is tr x layer x tokens; squash it to tr x layer x mean and return that."""

        tr_layer_mean_tensor = []
        for tr in self.stimulus_df[input_tensor_name]:
            tr_layer_mean_tensor.append([])
            for layer in tr:
                tr_layer_mean_tensor[-1].append(torch.mean(layer, 0))

        return tr_layer_mean_tensor

    def layerwise_token_movement(self):
        """tr_tensor is tr x layer x tokens; look at l2 distance for each token / layer."""

        activations_layerwise_l2_difference = []
        for tr in self.stimulus_df.activations:

            activations_layerwise_l2_difference.append([])

            for layer_num, (layer_one, layer_two) in enumerate(zip(tr, tr[1:])):

                # Get the number of tokens to compare
                n_tokens = layer_one.shape[0]
                l2_differences = [torch.dist(layer_one[i], layer_two[i], 2).item() for i in range(0, n_tokens)]
                activations_layerwise_l2_difference[-1].append(l2_differences)

        # will be [n_layers][n_tokens], where [0][0] gives the L2 distance for the first token across the first layer.
        self.stimulus_df["activation_layerwise_l2_distances"] = activations_layerwise_l2_difference

    def end_to_end_token_movement(self):
        """Set activation_end_to_end_l2_distances to L2 displacement from initial to final embeddings."""

        activations_end_to_end_l2_difference = []
        for tr in self.stimulus_df.activations:

            # Get the number of tokens to compare
            n_tokens = tr[0].shape[0]
            end_to_end_l2_differences = [torch.dist(tr[0][i], tr[-1][i], 2).item() for i in range(0, n_tokens)]
            activations_end_to_end_l2_difference.append(end_to_end_l2_differences)

        # will be [n_tokens], where [0] gives the L2 distance for the first token across the whole model.
        self.stimulus_df["activation_end_to_end_l2_distances"] = activations_end_to_end_l2_difference

    def semantic_composition_from_activations(self):

        end_to_end_mean_l2 = self.stimulus_df["activation_end_to_end_l2_distances"].apply(lambda x: np.mean(x))
        end_to_end_mean_l2_normed = self.normalize_col(end_to_end_mean_l2)

        end_to_end_max_l2 = self.stimulus_df["activation_end_to_end_l2_distances"].apply(lambda x: np.max(x))
        end_to_end_max_l2_normed = self.normalize_col(end_to_end_max_l2)

        layerwise_mean_l2 = self.stimulus_df["activation_layerwise_l2_distances"].apply(
            lambda x: [np.mean(layer) for layer in x])
        df = pd.DataFrame.from_records(layerwise_mean_l2)
        normalized = (df - df.mean()) / df.std()
        layerwise_mean_l2_normed = [list(r) for r in normalized.to_records(index=False)]

        layerwise_max_l2 = self.stimulus_df["activation_layerwise_l2_distances"].apply(
            lambda x: [np.max(layer) for layer in x])
        df = pd.DataFrame.from_records(layerwise_max_l2)
        normalized = (df - df.mean()) / df.std()
        layerwise_max_l2_normed = [list(r) for r in normalized.to_records(index=False)]

        all_semantic_composition = []
        for layerwise_mean, e2e_mean, layerwise_max, e2e_max in zip(layerwise_mean_l2_normed, end_to_end_mean_l2_normed,
                                                                    layerwise_max_l2_normed, end_to_end_max_l2_normed):
            all_semantic_composition.append(layerwise_max + [e2e_max] + layerwise_mean + [e2e_mean])

        self.stimulus_df["semantic_composition"] = all_semantic_composition

    @classmethod
    def normalize_col(cls, col):
        de_meaned = col - col.mean()
        return de_meaned / de_meaned.std()

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


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    import re
    prefix_re = re.compile(fr'^(?:{"|".join(prefices)})')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
