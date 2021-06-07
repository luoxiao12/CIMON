import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torchvision.transforms as transforms
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from loguru import logger
import torch.nn.functional as F
from model_loader import load_model
from evaluate import mean_average_precision
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from torch.nn import Parameter
def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

def train(train_dataloader,
          query_dataloader,
          retrieval_dataloader,
          multi_labels,
          code_length,
          num_features,
          alpha,
          beta,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          num_class,
          threshold,
          eta,
          temperature,
          evaluate_interval,
          ):
    """
    Training model.

    Args
        train_dataloader(torch.evaluate.data.DataLoader): Training data loader.
        query_dataloader(torch.evaluate.data.DataLoader): Query data loader.
        retrieval_dataloader(torch.evaluate.data.DataLoader): Retrieval data loader.
        multi_labels(bool): True, if dataset is multi-labels.
        code_length(int): Hash code length.
        num_features(int): Number of features.
        alpha, beta(float): Hyper-parameters.
        max_iter(int): Number of iterations.
        arch(str): Model name.
        lr(float): Learning rate.
        device(torch.device): GPU or CPU.
        verbose(bool): Print log.
        evaluate_interval(int): Interval of evaluation.
        snapshot_interval(int): Interval of snapshot.
        topk(int): Calculate top k data points map.
        checkpoint(str, optional): Paht of checkpoint.

    Returns
        None
    """
    # Model, optimizer, criterion
    model = load_model(arch, code_length)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = SEM_CON_Loss()
    criterion_aug = nn.MSELoss()

    # Extract features
    for i in range(1):
        features_1, features_2 = extract_features(model, train_dataloader, num_features, device, verbose)
        S_1, W_1 = generate_similarity_weight_matrix(features_1, alpha, beta, threshold=threshold, k_positive=400, k_negative=2500, Classes=num_class)
        S_1 = S_1.to(device)
        W_1 = W_1.to(device)
        S_2, W_2 = generate_similarity_weight_matrix(features_2, alpha, beta, threshold=threshold, k_positive=400, k_negative=2500, Classes=num_class)
        S_2 = S_2.to(device)
        W_2 = W_2.to(device)
        

        # Training
        model.train()
        for epoch in range(max_iter):
            n_batch = len(train_dataloader)
            for i, (data, data_aug,_, index) in enumerate(train_dataloader):


                data = data.to(device)
                batch_size = data.shape[0]
                data_aug = data_aug.to(device)

                optimizer.zero_grad()

                v= model(data)
                v_aug= model(data_aug)

                out = torch.cat([F.normalize(v), F.normalize(v_aug)], dim=0)
                sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
                mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
                sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
                pos_sim = torch.exp(torch.sum(F.normalize(v) * F.normalize(v_aug), dim=-1) / temperature)
               
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                
                nce_loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
                # nce_loss = -torch.mean(torch.sum(v*v_aug/code_length, dim=1)) 
                H = v @ v.t()/code_length
                H_aug = v_aug @ v_aug.t()/code_length
                targets_1 = S_1[index, :][:, index]
                targets_2 = S_2[index, :][:, index]
                weights_1 = W_1[index, :][:, index]
                weights_2 = W_2[index, :][:, index]
                
                loss =  (criterion(H, weights_2, targets_2)+  criterion(H_aug, weights_1, targets_1)) + (criterion(H_aug, weights_2, targets_2) +  criterion(H, weights_1, targets_1))+eta * nce_loss 
               

                loss.backward()
                optimizer.step()

                # Print log
                if verbose:
                    logger.info('[iter:{}/{}][loss:{:.4f}]'.format(epoch+1, max_iter, loss.item()))

                # Evaluate
            if (epoch % evaluate_interval == evaluate_interval-1):
                mAP = evaluate(model,
                                query_dataloader,
                                retrieval_dataloader,
                                code_length,
                                device,
                                topk,
                                multi_labels,
                                )
                logger.info('[iter:{}/{}][map:{:.4f}]'.format(
                    epoch+1,
                    max_iter,
                    mAP,
                ))

    # Evaluate and save 
    mAP = evaluate(model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   multi_labels,
                   )
    torch.save({'iteration': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join('checkpoints', 'resume_{}.t'.format(code_length)))
    logger.info('Training finish, [iteration:{}][map:{:.4f}]'.format(epoch+1, mAP))


def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk, multi_labels):
    model.eval()

    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)

    # One-hot encode targets
    if multi_labels:
        onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
    else:
        onehot_query_targets = query_dataloader.dataset.get_onehot_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)
    

    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )
    np.save("./code/query_code_{}_mAP_{}".format(code_length, mAP), query_code.cpu().detach().numpy())

    np.save("./code/retrieval_code_{}_mAP_{}".format(code_length, mAP), retrieval_code.cpu().detach().numpy())

    np.save("./code/query_target_{}_mAP_{}".format(code_length, mAP), onehot_query_targets.cpu().detach().numpy())

    np.save("./code/retrieval_target_{}_mAP_{}".format(code_length, mAP), onehot_retrieval_targets.cpu().detach().numpy())
    
    model.train()

    return mAP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, _, index in dataloader:
            data = data.to(device)
            outputs= model(data)
            code[index, :] = outputs.sign().cpu()

    return code


def generate_similarity_weight_matrix(features, alpha, beta, threshold, k_positive, k_negative, Classes):
    """
    Generate similarity and confidence matrix.

    Args
        features(torch.Tensor): Features.
        alpha, beta(float): Hyper-parameters.

    Returns
        S(torch.Tensor): Similarity matrix.
    """
    # # Cosine similarity
    cos_dist = squareform(pdist(features.numpy(), 'cosine'))
    features = features.numpy()

    # Construct similarity matrix
    S = (cos_dist <= threshold) * 1.0 + (cos_dist > threshold ) * -1.0
    
    # weight according to similarity

    # find the up and down extreme
    # Find maximum count of cosine distance
    max_cnt, max_cos = 0, 0
    interval = 1. / 100
    cur = 0
    for i in range(100):
        cur_cnt = np.sum((cos_dist > cur) & (cos_dist < cur + interval))
        if max_cnt < cur_cnt:
            max_cnt = cur_cnt
            max_cos = cur
        cur += interval



    # Split features into two parts
    flat_cos_dist = cos_dist.reshape((-1, 1))
    left = flat_cos_dist[np.where(flat_cos_dist <= max_cos)[0]]
    right = flat_cos_dist[np.where(flat_cos_dist > max_cos)[0]]

    # Reconstruct gaussian distribution
    left = np.concatenate([left, 2 * max_cos - left])
    right = np.concatenate([2 * max_cos - right, right])

    # Model data using gaussian distribution
    left_mean, left_std = norm.fit(left)
    right_mean, right_std = norm.fit(right)

    dist_up = right_mean + beta * right_std
    
    dist_down = left_mean - alpha * left_std

    def weight_norm(x, threshold):
        weight_f = (x>threshold)* (norm.cdf((x-right_mean)/right_std)-norm.cdf((threshold-right_mean)/right_std))/(1-norm.cdf((threshold-right_mean)/right_std)) + \
         (x<=threshold) * (norm.cdf((threshold-left_mean)/left_std)-norm.cdf((x-left_mean)/left_std) )/ (norm.cdf((threshold-left_mean)/left_std))
        return weight_f
    
    weight_1 = np.clip(weight_norm(cos_dist, threshold), 0, 1)


    # weight according to clustering
    features_norm = (features.T/ np.linalg.norm(features,axis=1)).T
    sp_cluster = SpectralClustering(n_clusters=Classes, random_state=0, assign_labels="discretize").fit(features_norm)
    A = sp_cluster.labels_[np.newaxis, :] #label vector
    # kmeans = KMeans(n_clusters=Classes, random_state=0, init='k-means++').fit(features_norm)
    # A = kmeans.labels_[np.newaxis, :] #label vector
    weight_2 =  ((((A - A.T) == 0)-1/2)*2* S +1)/2
    W = weight_1 * weight_2
    return torch.FloatTensor(S), torch.FloatTensor(W)


def extract_features(model, dataloader, num_features, device, verbose):
    """
    Extract features.
    """
    model.eval()
    model.set_extract_features(True)
    features_1 = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    features_2 = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    with torch.no_grad():
        N = len(dataloader)
        for i, (data_1, data_2 ,_, index) in enumerate(dataloader):
            if verbose:
                logger.debug('[Batch:{}/{}]'.format(i+1, N))
            data_1 = data_1.to(device)
            data_2 = data_2.to(device)
            features_1[index, :] = model(data_1).cpu()
            features_2[index, :] = model(data_2).cpu()

    model.set_extract_features(False)
    model.train()
    return features_1, features_2

class SEM_CON_Loss(nn.Module):
    def __init__(self):
        super(SEM_CON_Loss, self).__init__()

    def forward(self, H, W, S):
        loss = (W * S.abs() * (H - S).pow(2)).sum() / (H.shape[0] ** 2)
        return loss





    


    
