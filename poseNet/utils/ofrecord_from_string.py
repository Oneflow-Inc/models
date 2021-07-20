import oneflow.core.record.record_pb2 as ofrecord
import struct

with open("../ofrecord/train/part-0", "rb") as f:
    for loop in range(0, 3):
        length = struct.unpack("q", f.read(8))
        serializedBytes = f.read(length[0])
        ofrecord_features = ofrecord.OFRecord.FromString(serializedBytes)

        image = ofrecord_features.feature["encoded"].bytes_list.value
        label = ofrecord_features.feature["class/label"].int32_list.value

        print(image, label, end="\n\n")
