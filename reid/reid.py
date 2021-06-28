# -*- coding:utf-8 -*-

import sys
import argparse
from data_loader import Market1501, RandomIdentitySampler, ImageDataset
import oneflow.experimental as flow
from bisect import bisect_right
import os
import os.path as osp
import numpy as np
from utils.loggers import Logger
from utils.distance import compute_distance_matrix
from loss import TripletLoss, CrossEntropyLossLS
from model import ResReid
from lr_scheduler import WarmupMultiStepLR


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_devices', type=str, default='0')
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    parser.add_argument("--eval_batch_size", type=int,
                        default=64, required=False)
    parser.add_argument("--num_classes", type=int, default=751, required=False)
    parser.add_argument("--lr", type=float, default=3.5e-04, required=False)
    parser.add_argument("--max_epoch", type=int, default=120, required=False)
    parser.add_argument("--step-size", type=list,
                        default=[40, 70], required=False)
    parser.add_argument("--weight_t", type=float, default=0.5, required=False)
    parser.add_argument("--margin", type=float, default=0.3, required=False)
    parser.add_argument("--weight_decay", type=float,
                        default=5e-4, required=False)
    parser.add_argument("--adam_beta1", type=float,
                        default=0.9, required=False)
    parser.add_argument("--adam_beta2", type=float,
                        default=0.999, required=False)
    parser.add_argument("--gamma", type=float, default=0.1,
                        required=False, help='learning rate decay multiplier')
    parser.add_argument("--warmup", action='store_true',
                        default=True, help="warm up lr scheduler")
    parser.add_argument("--warmup_factor", type=float,
                        default=0.1, required=False)
    parser.add_argument("--warmup_iters", type=int, default=10, required=False)
    parser.add_argument("--epsilon", type=float, default=0.1, required=False)
    parser.add_argument("--data_dir", type=str, default='./dataset',
                        required=False, help="dataset directory")
    parser.add_argument("--image_height",  type=int,
                        default=256, required=False)
    parser.add_argument("--image_width", type=int, default=128, required=False)
    parser.add_argument("--evaluate", action='store_true',
                        default=False, help="train or eval")
    parser.add_argument("--eval_freq", type=int, default=20, required=False)
    parser.add_argument("--dist_metric", type=str,
                        default='euclidean', help="euclidean or cosine")
    parser.add_argument('--rerank', type=bool, default=False)
    parser.add_argument("--load_weights", type=str,
                        default='./resnet50_pretrained_model', help="model load directory")
    parser.add_argument("--log_dir", type=str, default="./output",
                        required=False, help="log info save directory")
    parser.add_argument("--flow_weight", type=str, default="./output/flow_weight",
                        required=False, help="log info save directory")
    parser.add_argument("--num_instances", type=int, default=4)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')
    return parser.parse_args()


def main(args):
    flow.enable_eager_execution()

    # log setting
    log_name = 'log_test.log' if args.evaluate else 'log_train.log'
    sys.stdout = Logger(osp.join(args.log_dir, log_name))

    # cuda setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    print("Currently using GPU {}".format(os.environ['CUDA_VISIBLE_DEVICES']))

    print('Building re-id model ')
    model = ResReid(args.num_classes)

    if args.load_weights:
        pretrain_models = flow.load(args.load_weights)
        model.load_state_dict(pretrain_models, strict=False)

    model = model.to('cuda')

    print('=> init dataset')
    dataset = Market1501(root=args.data_dir)

    if args.evaluate:
        evaluate(model, dataset)
    else:
        optimizer = flow.optim.Adam(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    betas=(args.adam_beta1, args.adam_beta2)
                                    )

        # lr scheduler
        if args.warmup:
            scheduler = WarmupMultiStepLR(optimizer, milestones=args.step_size, gamma=args.gamma,
                                          warmup_factor=args.warmup_factor, warmup_iters=args.warmup_iters)
        else:
            def lambda1(
                epoch): return args.lr ** bisect_right(args.step_size, epoch)
            scheduler = flow.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda1
            )

        train(model, dataset, args.num_classes, optimizer, scheduler)


def train(model, dataset, num_classes, optimizer, scheduler):

    batch_size = args.batch_size

    is_best = False
    best_rank = 0
    print('=> Start training')

    # loss
    criterion_t = TripletLoss(margin=args.margin).to("cuda")
    criterion_x = CrossEntropyLossLS(
        num_classes=num_classes, epsilon=args.epsilon).to("cuda")
    weight_t = args.weight_t
    weight_x = 1.0 - args.weight_t

    _, train_id, _ = map(list, zip(*dataset.train))
    train_dataset = ImageDataset(dataset.train, flag='train', process_size=(
        args.image_height, args.image_width))
    #*****training*******#
    for epoch in range(0, args.max_epoch):
        # shift to train
        model.train()
        indicies = [x for x in RandomIdentitySampler(
            train_id, batch_size, args.num_instances)]
        for i in range(len(indicies) // batch_size):
            try:
                # train_batch[0,1,2] are [imgs, pid, cam_id]
                imgs, pids, _ = train_dataset.__getbatch__(
                    indicies[i * batch_size:(i + 1) * batch_size])
            except:
                imgs, pids, _ = train_dataset.__getbatch__(
                    indicies[-batch_size:])
            imgs = flow.Tensor(np.array(imgs)).to('cuda')
            pids = flow.Tensor(np.array(pids), dtype=flow.int32).to('cuda')
            outputs, features = model(imgs)
            loss_t = compute_loss(criterion_t, features, pids)
            loss_x = compute_loss(criterion_x, outputs, pids)

            loss = weight_t * loss_t + weight_x * loss_x
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        print('epoch:', epoch+1, 'loss_t:', loss_t.numpy()[0], 'loss_x:', loss_x.numpy()[
              0], 'loss:', loss.numpy()[0],   'lr:', optimizer.param_groups[0]['lr'])

        #*****testing********#
        if (epoch + 1) % args.eval_freq == 0 and (epoch + 1) != args.max_epoch:
            rank1, mAP = evaluate(
                model,  dataset
            )
            if (rank1 + mAP) / 2.0 > best_rank:
                is_best = True
            else:
                is_best = False
            if is_best:
                flow.save(model.state_dict(), args.flow_weight+'_'+str(epoch))
    print('=> End training')

    print('=> Final test')
    rank1, _ = evaluate(model, dataset)
    flow.save(model.state_dict(), args.flow_weight)


def compute_loss(criterion, outputs, targets):
    if isinstance(outputs, (tuple, list)):
        loss = DeepSupervision(criterion, outputs, targets)
    else:
        loss = criterion(outputs, targets)
    return loss


def DeepSupervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss


def evaluate(model, dataset):
    query_dataset = ImageDataset(dataset.query, flag='test', process_size=(
        args.image_height, args.image_width))
    gallery_dataset = ImageDataset(dataset.gallery, flag='test', process_size=(
        args.image_height, args.image_width))
    eval_batch = args.eval_batch_size
    model.eval()
    dist_metric = args.dist_metric  # distance metric, ['euclidean', 'cosine']
    rerank = args.rerank  # use person re-ranking

    save_dir = args.log_dir
    print('Extracting features from query set ...')
    # query features, query person IDs and query camera IDs
    qf, q_pids, q_camids = [], [], []
    q_ind = list(range(len(query_dataset)))
    for i in range((len(query_dataset) // eval_batch)):
        imgs, pids, camids = query_dataset.__getbatch__(
            q_ind[i * eval_batch:(i + 1) * eval_batch])

        imgs = flow.Tensor(np.array(imgs)).to('cuda')
        with flow.no_grad():
            features = model(imgs)

        qf.append(features.numpy())
        q_pids.extend(pids)
        q_camids.extend(camids)

    qf = np.concatenate(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Done, obtained {}-by-{} matrix'.format(qf.shape[0], qf.shape[1]))

    print('Extracting features from gallery set ...')
    # gallery features, gallery person IDs and gallery camera IDs
    gf, g_pids, g_camids = [], [], []
    g_ind = list(range(len(gallery_dataset)))
    for i in range((len(gallery_dataset) // eval_batch)):
        imgs, pids, camids = gallery_dataset.__getbatch__(
            g_ind[i * eval_batch:(i + 1) * eval_batch])

        imgs = flow.Tensor(np.array(imgs)).to('cuda')

        with flow.no_grad():
            features = model(imgs)
        gf.append(features.numpy())
        g_pids.extend(pids)
        g_camids.extend(camids)

    gf = np.concatenate(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print('Done, obtained {}-by-{} matrix'.format(gf.shape[0], gf.shape[1]))

    print('Computing distance matrix with metric={} ...'.format(dist_metric))
    distmat = compute_distance_matrix(qf, gf, dist_metric)

    if rerank:
        print('Applying person re-ranking ...')
        distmat_qq = compute_distance_matrix(qf, qf, dist_metric)
        distmat_gg = compute_distance_matrix(gf, gf, dist_metric)
        distmat = re_ranking(distmat, distmat_qq, distmat_gg)

    print('Computing CMC and mAP ...')
    cmc, mAP = _eval(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids
    )

    print("=".ljust(30, "=") + " Result " + "=".ljust(30, "="))
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in [1, 5, 10]:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print("=".ljust(66, "="))

    return cmc[0], mAP


def _eval(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1)
        ],
        axis=0
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(
        1. * original_dist / np.max(original_dist, axis=0)
    )
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[
            0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[
                candidate, :int(np.around(k1 / 2.)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, :int(np.around(k1 / 2.)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate
            )[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[
                fi_candidate]
            if len(
                    np.intersect1d(candidate_k_reciprocal_index,
                                   k_reciprocal_index)
            ) > 2. / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

  # get jaccard_dist
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]  # q_i's k-reciprocal index
        indImages = [invIndex[ind] for ind in indNonZero]  #
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j],
                                       indNonZero[j]]  # V_pigj, V_gigj
            )
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + \
        original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


if __name__ == '__main__':
    args = _parse_args()
    main(args)
