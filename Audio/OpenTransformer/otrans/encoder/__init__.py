from otrans.encoder.transformer import TransformerEncoder
from otrans.encoder.conformer import ConformerEncoder


BuildEncoder = {
    'transformer': TransformerEncoder,
    'conformer': ConformerEncoder,
}
