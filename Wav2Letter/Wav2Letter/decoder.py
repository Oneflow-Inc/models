import pickle

import oneflow as flow
import Levenshtein as Lev


class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (string): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 28.
    """

    def __init__(self, blank_index=0):
        with open('./speech_data/int_encoder.pkl', 'rb') as f:
            self.int_to_char = pickle.load(f)['index2char']
        self.blank_index = blank_index


    def wer(self, real_strings, pred_strings):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        assert len(real_strings) == len(pred_strings)

        wer = 0
        for i in range(len(real_strings)):
            wer += Lev.distance(real_strings[i], pred_strings[i])

        return wer/len(real_strings)


class GreedyDecoder(Decoder):
    def __init__(self, blank_index=0):
        super(GreedyDecoder, self).__init__(blank_index)

    def convert_to_strings(self, sequences, sizes=6, remove_repetitions=True, return_offsets=True):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []

        for i in range(sequences.size(0)):
            string, string_offsets = self.process_string(sequences[i], remove_repetitions)
            strings.append(string)  # We only return one path
            if return_offsets:
                if i == 0:
                    offsets = string_offsets.unsqueeze(0)
                else:
                    offsets = flow.cat((offsets, string_offsets.unsqueeze(0)), 0)
        if return_offsets:
            return strings, offsets.to('cuda')
        else:
            return strings


    def process_string(self, sequence, remove_repetitions=True):
        string = ""
        offsets = []
        
        for i in range(sequence.size(0)):

            char = self.int_to_char[sequence[i].numpy().item()]
            if char != self.int_to_char[self.blank_index]:
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].numpy().item()]:
                    pass
                elif sequence[i].numpy().item() != 1 and len(offsets) < 6:
                    string = string + char
                    offsets.append(sequence[i].numpy().item())
                elif len(offsets) < 6:
                    offsets.append(sequence[i].numpy().item())
        if len(offsets) < 6:
            offsets += [1]*(6-len(offsets))

        return string, flow.tensor(offsets, dtype=flow.int)


    def decode(self, ctc_matrix):
        top = flow.topk(ctc_matrix, k=1, dim=1)

        new_top = top[1][0].detach()
        for i in range(1, top[1].size(0)):
            cur = top[1][i].detach()
            new_top = flow.cat((new_top, cur), 0)

        return new_top
