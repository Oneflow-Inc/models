import oneflow.nn as nn


class Seq2Seq(nn.Module):
    """Sequence-to-Sequence architecture with configurable encoder and decoder.
    """

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, padded_input, input_lengths, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, _ = self.encoder(padded_input)
        loss = self.decoder(padded_target, encoder_padded_outputs)
        return loss

    def recognize(self, input, input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam

        Returns:
            nbest_hyps: 1
        """
        encoder_outputs, _ = self.encoder(input.unsqueeze(0))
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
                                                 char_list,
                                                 args)
        return nbest_hyps
