import copy

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, logging
import spacy


class TransformerRSM(object):

    def __init__(self, stimulus_name, model_name, file_path=None, verbose=False):

        self.stimulus_name = stimulus_name
        self.model_name = model_name
        self.verbose = verbose
        self.stimulus_df = self._load_stimulus(file_path=file_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Models can return full list of hidden-states & attentions weights at each layer
        self.transformer = AutoModel.from_pretrained(self.model_name,
                                                     output_hidden_states=True,
                                                     output_attentions=True)
        # bert-base-uncased or bert-large-uncased
        if 'bert' in self.model_name:

            # Use special tokens with BERT, but then slice them off when returning activations
            self.use_special_tokens = True
            self.last_token_index = -1

        # gpt2, gpt2-xl, gpt-neo-2.7B
        elif 'gpt' in self.model_name:

            # Don't use special tokens with GPT, so return whole list of embeddings
            self.use_special_tokens = False
            self.last_token_index = None

        self.transformer.eval()

    def _load_stimulus(self, file_path=None):

        if file_path is None:
            file_path = 'data/stimuli/{}/tr_tokens.csv'.format(self.stimulus_name)

        print("Looking for TR-aligned tokens in {}".format(file_path))
        try:
            stimulus_df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError("Could not find target file. "
                                    "You may need to use `Gentle Transcript Processing` notebook to get tokens.")

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
        tr_glove_array = []
        
        nlp = spacy.load('en_core_web_lg')

        # Enumerate over the TR-aligned tokens
        for i, tr in enumerate(tr_chunked_tokens):

            if self.verbose and i % 100 == 0:
                print("Processing TR {}.".format(i))

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
                tr_glove_array.append(None)
                continue

            z_reps = []
            with torch.no_grad():
                embeddings, _ = self.transformer(window_token_ids)[-2:]
                for layer_num, layer in enumerate(embeddings[:-1]):
                    if 'bert' in self.model_name:
                        z_reps.append(self.transformer.encoder.layer[layer_num].attention.self(layer)[0])

                    elif 'gpt' in self.model_name:
                        attn_layer = self.transformer.h[layer_num].attn
                        query, key, value = attn_layer.c_attn(self.transformer.h[layer_num].ln_1(layer)).split(attn_layer.split_size,dim=2)
                        try:
                            query = attn_layer._split_heads(query,attn_layer.num_heads,attn_layer.head_dim)
                            key = attn_layer._split_heads(key,attn_layer.num_heads,attn_layer.head_dim)
                            value = attn_layer._split_heads(value,attn_layer.num_heads,attn_layer.head_dim)
                            if layer_num==0:
                                print('This part of implementation is for newer version of transformers and has not been tested.')
                        except AttributeError:
                            query = attn_layer.split_heads(query)
                            key = attn_layer.split_heads(key, k=True)
                            value = attn_layer.split_heads(value)
                        attn_output = attn_layer._attn(query, key, value, attention_mask=None, head_mask=None)[0]
                        try:
                            attn_output = attn_layer._merge_heads(attn_output,attn_layer.num_heads,attn_layer.head_dim)
                        except AttributeError:
                            attn_output = attn_layer.merge_heads(attn_output)
                        z_reps.append(attn_output)

            tr_tokens = self.tokenizer.convert_ids_to_tokens(tr_token_ids.numpy())
            tr_tokens_array.append(tr_tokens)

            if self.verbose and i % 100 == 0:
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

            glove = [nlp(tr).vector]
            tr_glove_array.append([])
            # there is only one 'layer' for glove
            for layer in glove:
                tr_glove_array[-1].append(layer)

        self.stimulus_df["activations"] = tr_activations_array
        self.stimulus_df["z_reps"] = tr_z_reps_array
        self.stimulus_df["glove"] = tr_glove_array

        # Forward-fill our activations, but *not* the tokens-in-TR
        self.stimulus_df["activations"].ffill(inplace=True)
        self.stimulus_df["z_reps"].ffill(inplace=True)
        self.stimulus_df["glove"].ffill(inplace=True)

        self.stimulus_df["transformer_tokens_in_tr"] = tr_tokens_array
        self.stimulus_df["n_transformer_tokens_in_tr"] = list(map(lambda x: len(x) if x else 0, tr_tokens_array))

        print("Processed {} TRs for activations.".format(len(tr_activations_array)))

    def process_stimulus_attentions(self, num_window_tokens=20):
        """Return window_tokens x tr_tokens x num_heads attention matrix"""

        tr_chunked_tokens = self.stimulus_df.tokens.values
        tr_attentions_array = []
        '''
        tr_attention_rollouts_array = []
        tr_attention_flows_array = []
        '''
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
                #tr_attention_rollouts_array.append(None)
                #tr_attention_flows_array.append(None)
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

                '''
                # calcuate atttention including residual effects and take average across heads within the same layer
                full_attentions = [0.5*layer+0.5*torch.eye(layer.shape[-1])[None,...] for layer in squeezed]
                ave_attentions = [layer.mean(dim=0) for layer in full_attentions]

                # calculate attention rollout
                attention_rollout = [full_attentions[0]]
                ave_attention_rollout = [ave_attentions[0]]
                for layer,ave_layer in zip(full_attentions[1:],ave_attentions[1:]):
                    layer_rollout = torch.tensor([list((head@ave_attention_rollout[-1]).detach().numpy()) for head in layer])
                    attention_rollout.append(layer_rollout)
                    ave_attention_rollout.append(ave_layer@ave_attention_rollout[-1])
                tr_attention_rollouts_array.append(attention_rollout)

                # calculate attentino flow
                # create graph
                num_layers = len(ave_attentions)
                seq_len = ave_attentions[0].shape[0]
                adj_mat = np.zeros(((num_layers+1)*seq_len, (num_layers+1)*seq_len))
                for layer_id in range(1,num_layers+1):
                    for pos_from in range(seq_len):
                        for pos_to in range(seq_len):
                            adj_mat[layer_id*seq_len+pos_from][(layer_id-1)*seq_len+pos_to] = ave_attentions[layer_id-1][pos_from][pos_to]
                G=nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph())
                for i in range(adj_mat.shape[0]):
                    for j in range(adj_mat.shape[1]):
                        nx.set_edge_attributes(G, {(i,j): adj_mat[i,j]}, 'capacity')

                # calculate maximum flow
                max_flows = []
                for layer_id in range(1,num_layers+1):
                    max_flow_layer = np.zeros((seq_len,seq_len))
                    for pos in range(seq_len):
                        for input_node in range(seq_len):
                            max_flow_layer[pos,input_node] = nx.maximum_flow_value(G,layer_id*seq_len+pos,input_node, flow_func=nx.algorithms.flow.edmonds_karp)
                    max_flows.append(torch.from_numpy(max_flow_layer).float())
                normed_max_flows = [layer/layer.sum(dim=1)[...,None] for layer in max_flows]
                for layer in normed_max_flows:
                    assert torch.allclose(layer.sum(dim=1),torch.ones_like(layer.sum(dim=1)))

                attention_flow = [full_attentions[0]]
                for layer, ave_layer in zip(full_attentions[1:],normed_max_flows[:-1]):
                    layer_flow = torch.tensor([list((head@ave_layer).detach().numpy()) for head in layer])
                    attention_flow.append(layer_flow)
                tr_attention_flows_array.append(attention_flow)
                '''

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
        '''
        self.stimulus_df["attention_rollouts"] = tr_attention_rollouts_array
        self.stimulus_df["attention_flows"] = tr_attention_flows_array
        '''
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