# from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


# writer = SummaryWriter(comment='_Unet')
# for i in range(10):
#     writer.add_scalar('var', i**2, global_step=i)
#
# writer.close()


def plot_picture(filename):
    with open(filename, "r") as f:
        train_loss = f.readlines()
        train_loss = list(map(lambda x: float(x.strip()), train_loss))
    x = range(len(train_loss))
    y = train_loss
    plt.plot(
        x,
        y,
        label="train loss",
        linewidth=2,
        color="r",
        marker="o",
        markerfacecolor="r",
        markersize=5,
    )
    plt.xlabel("epoch")
    plt.ylabel("loss value")
    plt.legend()
    plt.show()
