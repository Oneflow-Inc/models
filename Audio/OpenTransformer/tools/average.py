import oneflow as flow
import os
import shutil
import sys

rootdir = sys.argv[1]
st = sys.argv[2]
ed = sys.argv[3]


def average_chkpt(datadir, start, end):
    id_chkpt = [str(i) for i in range(int(start), int(end) + 1)]
    print("Average these number %s models" % ",".join(id_chkpt))

    chkpts = ["model.epoch.%s.pt" % idx for idx in id_chkpt]

    params_dict = {}
    params_keys = {}
    new_state = None
    num_models = len(chkpts)

    for chkpt in chkpts:
        frontend_state = flow.load(os.path.join(datadir, chkpt, "frontend.pt"))
        encoder_state = flow.load(os.path.join(datadir, chkpt, "encoder.pt"))
        decoder_state = flow.load(os.path.join(datadir, chkpt, "decoder.pt"))
        state = {
            "frontend": frontend_state,
            "encoder": encoder_state,
            "decoder": decoder_state,
        }
        if new_state is None:
            new_state = state

        for key, value in state.items():

            if key != "model":
                continue

            model_params = value
            model_params_keys = list(model_params.keys())

            if key not in params_keys:
                params_keys[key] = model_params_keys

            if key not in params_dict:
                params_dict[key] = {}

            for k in params_keys[key]:
                p = model_params[k]

                if k not in params_dict[key]:
                    params_dict[key][k] = p.clone()
                    # NOTE: clone() is needed in case of p is a shared parameter
                else:
                    params_dict[key][k] += p

    averaged_params = {}
    for key, states in params_dict.items():
        averaged_params[key] = {}
        for k, v in states.items():
            averaged_params[key][k] = v
            try:
                averaged_params[key][k].div_(num_models)
            except:
                if "batch_norm.num_batches_tracked" in k:
                    averaged_params[key][k] = flow.div(
                        averaged_params[key][k], num_models
                    ).long()
                else:
                    print("Key: %s  Tensor: %s" % (key, k))
                    raise ValueError

        new_state[key] = averaged_params[key]

    flow.save(
        new_state["frontend"],
        os.path.join(
            datadir, "model.average.from%sto%s.pt" % (start, end), "frontend.pt"
        ),
    )
    flow.save(
        new_state["encoder"],
        os.path.join(
            datadir, "model.average.from%sto%s.pt" % (start, end), "encoder.pt"
        ),
    )
    flow.save(
        new_state["decoder"],
        os.path.join(
            datadir, "model.average.from%sto%s.pt" % (start, end), "decoder.pt"
        ),
    )
    print(
        "Save the average checkpoint as %s"
        % os.path.join(datadir, "model.average.from%sto%s.pt" % (start, end))
    )
    print("Done!")


shutil.copytree(
    os.path.join(rootdir, "model.epoch.%s.pt" % st, "params.tar"),
    os.path.join(rootdir, "model.average.from%sto%s.pt" % (st, ed)),
)

average_chkpt(rootdir, st, ed)
