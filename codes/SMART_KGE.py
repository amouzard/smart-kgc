import os
import logging
import math
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import BatchType, ModeType, TestDataset

# from euclidean import givens_rotations, givens_reflection
# from hyperbolic import mobius_add, expmap0, logmap0, expmapX, logmapX, project, hyp_distance_multi_c, similarity_score

class KGEModel(nn.Module, ABC):
    """
    Must define
        `self.entity_embedding`
        `self.relation_embedding`
    in the subclasses.
    """

    @abstractmethod
    def func(self, head, rel, tail, batch_type, relation_choice):
        """
        Different tensor shape for different batch types.
        BatchType.SINGLE:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]
            relation_choice: [batch_size, hidden_dim]

        BatchType.HEAD_BATCH:
            head: [batch_size, negative_sample_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.TAIL_BATCH:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, negative_sample_size, hidden_dim]
        """
        ...

    def forward(self, sample, batch_type=BatchType.SINGLE):
        """
        Given the indexes in `sample`, extract the corresponding embeddings,
        and call func().

        Args:
            batch_type: {SINGLE, HEAD_BATCH, TAIL_BATCH},
                - SINGLE: positive samples in training, and all samples in validation / testing,
                - HEAD_BATCH: (?, r, t) tasks in training,
                - TAIL_BATCH: (h, r, ?) tasks in training.

            sample: different format for different batch types.
                - SINGLE: tensor with shape [batch_size, 3]
                - {HEAD_BATCH, TAIL_BATCH}: (positive_sample, negative_sample)
                    - positive_sample: tensor with shape [batch_size, 3]
                    - negative_sample: tensor with shape [batch_size, negative_sample_size]
        """
        if batch_type == BatchType.SINGLE:
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            relation_choice = torch.index_select(
                self.choice_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.HEAD_BATCH:
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            relation_choice = torch.index_select(
                self.choice_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.TAIL_BATCH:
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            relation_choice = torch.index_select(
                self.choice_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('batch_type %s not supported!'.format(batch_type))

        return self.func(head, relation, tail, batch_type, relation_choice), (head, tail)

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

        # negative scores
        negative_score, _ = model((positive_sample, negative_sample), batch_type=batch_type)

        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score, ent = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization:
            # Use regularization
            regularization = args.regularization * (
                ent[0].norm(p=2)**2 +
                ent[1].norm(p=2)**2
            ) / ent[0].shape[0]
            loss = loss + regularization
        else:
            regularization = torch.tensor([0])

        loss.backward()

        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
            'regularization': regularization.item()
        }

        return log

    @staticmethod
    def test_step(model, data_reader, mode, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.HEAD_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.TAIL_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []
        logs_rel = defaultdict(list)  # logs for every relation

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, batch_type in test_dataset:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score, _ = model((positive_sample, negative_sample), batch_type)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if batch_type == BatchType.HEAD_BATCH:
                        positive_arg = positive_sample[:, 0]
                    elif batch_type == BatchType.TAIL_BATCH:
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        rel = positive_sample[i][1].item()

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()

                        log = {
                            '*************  Model '+args.model + '*****************': 1,
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        }

                        logs.append(log)
                        logs_rel[rel].append(log)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... ({}/{})'.format(step, total_steps))
                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        metrics_rel = defaultdict(dict)
        for rel in logs_rel:
            for metric in logs_rel[rel][0].keys():
                metrics_rel[rel][metric] = sum([log[metric] for log in logs_rel[rel]]) / len(logs_rel[rel])

        return metrics, metrics_rel

    def Translation(self, re_head, im_head, re_tail, im_tail, re_rot, im_rot, re_ref, im_ref, trans_x, trans_y, scaling,
                    select, score_func=True):
        # re_head, im_head, re_tail, im_tail, trans_x, trans_y, select):
        # Translation of the head
        re_rhead, im_rhead = re_head + trans_x, im_head + trans_y
        sel = select[:, :, 0].view((select.shape[0], -1, 1))
        if score_func:
            score_1 = torch.stack([(re_rhead - re_tail) * sel, (im_rhead - im_tail) * sel], dim=0)
            return score_1.norm(dim=0, p=2).sum(dim=2)/torch.maximum(torch.tensor([1]).cuda(), torch.max(score_1)), re_rhead, im_rhead
        else:
            score_1 = torch.stack([(re_rhead * re_tail) * sel, (im_rhead * im_tail) * sel], dim=0)
            return score_1.sum(dim=0).sum(dim=2)/torch.maximum(torch.tensor([1]).cuda(), torch.max(score_1)), re_rhead, im_rhead

    def Scaling(self, re_head, im_head, re_tail, im_tail, re_rot, im_rot, re_ref, im_ref, trans_x, trans_y, scaling,
                select, score_func=True):
        # , re_head, im_head, re_tail, im_tail, scaling, select):
        # Scaling of the head
        re_rhead, im_rhead = re_head * scaling, im_head * scaling
        sel = select[:, :, 1].view((select.shape[0], -1, 1))
        if score_func:
            score_1 = torch.stack([(re_rhead - re_tail) * sel, (im_rhead - im_tail) * sel], dim=0)
            return score_1.norm(dim=0, p=2).sum(dim=2)/torch.maximum(torch.tensor([1]).cuda(), torch.max(score_1)), re_rhead, im_rhead
        else:
            score_1 = torch.stack([(re_rhead * re_tail) * sel, (im_rhead * im_tail) * sel], dim=0)
            return score_1.sum(dim=0).sum(dim=2)/torch.maximum(torch.tensor([1]).cuda(), torch.max(score_1)), re_rhead, im_rhead

    def Rotation(self, re_head, im_head, re_tail, im_tail, re_rot, im_rot, re_ref, im_ref, trans_x, trans_y, scaling,
                 select, score_func=True):
        # , re_head, im_head, re_tail, im_tail, re_rot, im_rot, select):
        # Rotation of the head
        re_rhead = re_rot * re_head - im_rot * im_head
        im_rhead = im_rot * re_head + re_rot * im_head
        sel = select[:, :, 2].view((select.shape[0], -1, 1))
        if score_func:
            score_1 = torch.stack([(re_rhead - re_tail) * sel, (im_rhead - im_tail) * sel], dim=0)
            return score_1.norm(dim=0, p=2).sum(dim=2)/torch.maximum(torch.tensor([1]).cuda(), torch.max(score_1)), re_rhead, im_rhead
        else:
            score_1 = torch.stack([(re_rhead * re_tail) * sel, (im_rhead * im_tail) * sel], dim=0)
            return score_1.sum(dim=0).sum(dim=2)/torch.maximum(torch.tensor([1]).cuda(), torch.max(score_1)), re_rhead, im_rhead

    def Reflection(self, re_head, im_head, re_tail, im_tail, re_rot, im_rot, re_ref, im_ref, trans_x, trans_y, scaling,
                   select, score_func=True):
        # , re_head, im_head, re_tail, im_tail, re_ref, im_ref, select):
        # Reflection of the head
        re_rhead = re_ref * re_head + im_ref * im_head
        im_rhead = im_ref * re_head - re_ref * im_head
        sel = select[:, :, 3].view((select.shape[0], -1, 1))
        if score_func:
            score_1 = torch.stack([(re_rhead - re_tail) * sel, (im_rhead - im_tail) * sel], dim=0)
            return score_1.norm(dim=0, p=2).sum(dim=2)/torch.maximum(torch.tensor([1]).cuda(), torch.max(score_1)), re_rhead, im_rhead
        else:
            score_1 = torch.stack([(re_rhead * re_tail) * sel, (im_rhead * im_tail) * sel], dim=0)
            return score_1.sum(dim=0).sum(dim=2)/torch.maximum(torch.tensor([1]).cuda(), torch.max(score_1)), re_rhead, im_rhead

    def Bulk_transf(self, re_head, im_head, re_tail, im_tail, re_rot, im_rot, re_ref, im_ref, trans_x, trans_y, scaling,
                    select, score_func=True):
        # # Translation
        score, _, _ = self.Translation(re_head, im_head, re_tail, im_tail, re_rot, im_rot, re_ref, im_ref, trans_x,
                                       trans_y, scaling, select, score_func=score_func)
        # # Scaling
        score1, _, _ = self.Scaling(re_head, im_head, re_tail, im_tail, re_rot, im_rot, re_ref, im_ref, trans_x,
                                    trans_y, scaling, select, score_func=score_func)
        score = score/torch.max(score) + score1/torch.maximum(torch.tensor([1]).cuda(), torch.max(score1))
        # # Rotation
        score1, _, _ = self.Rotation(re_head, im_head, re_tail, im_tail, re_rot, im_rot, re_ref, im_ref, trans_x,
                                     trans_y, scaling, select, score_func=score_func)
        score = score + score1/torch.maximum(torch.tensor([1]).cuda(), torch.max(score1))
        # # Reflection
        score1, _, _ = self.Reflection(re_head, im_head, re_tail, im_tail, re_rot, im_rot, re_ref, im_ref, trans_x,
                                       trans_y, scaling, select, score_func=score_func)
        score = score + score1/torch.maximum(torch.tensor([1]).cuda(), torch.max(score1))
        return score

class smart(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, p_norm):
        super().__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.p = p_norm

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 5))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # self.choice_embedding = nn.Parameter(torch.ones(num_relation, 4))
        # nn.init.uniform_seed(
        #    tensor=self.choice_embedding,
        #    a=0, #-self.embedding_range.item(),
        #    b=1 #self.embedding_range.item()
        # )
        # self.choice_embedding = nn.Parameter(torch.ones(num_relation, 4))

        self.load_choice_embs = False  # If True, SMART load pre-trained relation weights
        self.wnrr_embds = "choice_embeddings_wnrr_avg.npy"  # choice_embedding
        self.fb237_embds = "choice_embedding_fb237.npy"  # tud fb237
        if self.load_choice_embs:
            # self.load_wieghts_from_embs = torch.Tensor(np.load("choice_embedding.npy")).cuda()
            self.choice_embedding = nn.Parameter(torch.Tensor(np.load(self.wnrr_embds)), requires_grad=False) #.cuda()
        else:
            self.choice_embedding = nn.Parameter(torch.ones(num_relation, 4))

        self.pi = 3.14159262358979323846

        self.arg_sort_desc = True

        self.freeze = False
        self.start = False
        self.initiate = True
        self.step = 0

        self.dist_score = True
    def func(self, head, rel, tail, batch_type, relation_choice):
        # This model let each relation learn for parameters that characterized how appropriate the corresponding
        # transformation (trans, scaling, rot, ref) adhere to the graph structure if it represents the relation.
        # bash runs.sh train smart wn18rr 0 0 1024 256 32 12.0 .5 0.001 0 16 0 2 120000 50000 80000
        '''
        This model uses translation, rotation, reflection, and homothety as basis of relation embeddings.
        Each relation r_i learns its optimal embedding through the 4 values of relation_choice[i] which is softmaxed.
        '''
        #TODO: This model does not reproduce the initial results of SMART
        # I run this with the config above. And running smarter with the same config.
        # Now, I am going uniformize the score wrt the max score of a batch and rerun smart
        # Next, I will run sheet.
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        trans_x, trans_y, rot_angle, ref_angle, scaling = torch.chunk(rel, 5, dim=2)

        if self.initiate:
            select = torch.ones_like(relation_choice)
        if self.start:
            self.initiate = False
            if self.load_choice_embs:
                select = relation_choice
            else:
                select = torch.sqrt(torch.softmax(relation_choice, dim=-1))# print(self.choice_embedding[0].data, '\n', torch.softmax(relation_choice, dim=-1)[0].data)
        if self.freeze:
            self.initiate = False
            self.start = False
            if self.load_choice_embs:
                select = relation_choice
            else:
                select = torch.sqrt(torch.softmax(relation_choice, dim=-1))
            save, _ = select.max(dim=2, keepdim=True)
            select = torch.where(select < save, torch.tensor([0]).cuda(), torch.tensor([1]).cuda())

        rot_angle = rot_angle / (self.embedding_range.item() / pi)
        ref_angle = ref_angle / (self.embedding_range.item() / pi)

        re_rot = torch.cos(rot_angle)
        im_rot = torch.sin(rot_angle)

        re_ref = torch.cos(ref_angle)
        im_ref = torch.sin(ref_angle)

        # Translation of the head
        score_1, re_rhead, im_rhead = self.Translation(re_head, im_head, re_tail, im_tail, re_rot, im_rot, re_ref, im_ref, trans_x, trans_y, scaling, select, score_func=self.dist_score)
        score = score_1 #.norm(dim=0, p=2).sum(dim=2)

        # Scaling of the head
        score_1, re_rhead, im_rhead = self.Scaling(re_head, im_head, re_tail, im_tail, re_rot, im_rot, re_ref, im_ref, trans_x, trans_y, scaling, select, score_func=self.dist_score)
        score = score + score_1 #.norm(dim=0, p=2).sum(dim=2)

        # Rotation of the head
        score_1, re_rhead, im_rhead = self.Rotation(re_head, im_head, re_tail, im_tail, re_rot, im_rot, re_ref, im_ref, trans_x, trans_y, scaling, select, score_func=self.dist_score)
        score = score + score_1

        # Reflection of the head
        score_1, re_rhead, im_rhead = self.Reflection(re_head, im_head, re_tail, im_tail, re_rot, im_rot, re_ref, im_ref, trans_x, trans_y, scaling, select, score_func=self.dist_score)
        score = score + score_1

        score = self.gamma.item() - score
        return score
    def func0(self, head, rel, tail, batch_type, relation_choice):
        # This model let each relation learn for parameters that characterized how appropriate the corresponding
        # transformation (trans, scaling, rot, ref) adhere to the graph structure if it represents the relation.

        # bash runs.sh train smart wn18rr 0 0 1024 256 32 12.0 .5 0.001 0 16 0 2 120000 50000 80000
        '''
        This model uses translation, rotation, reflection, and homothety as basis of relation embeddings.
        Each relation r_i learns its optimal embedding through the 4 values of relation_choice[i] which is softmaxed.
        '''
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        trans_x, trans_y, rot_angle, ref_angle, scaling = torch.chunk(rel, 5, dim=2)

        # select = torch.sqrt(torch.softmax(relation_choice, dim=-1))
        if self.initiate:
            select = torch.ones_like(relation_choice)
        if self.start:
            self.initiate = False
            # select = torch.sqrt(torch.softmax(relation_choice, dim=-1))
            if self.load_choice_embs:
                select = relation_choice
            else:
                select = torch.sqrt(torch.softmax(relation_choice, dim=-1))
            # print(self.choice_embedding[0].data, '\n', torch.softmax(relation_choice, dim=-1)[0].data)
        if self.freeze:
            self.initiate = False
            self.start = False
            if self.load_choice_embs:
                select = relation_choice
            else:
                select = torch.sqrt(torch.softmax(relation_choice, dim=-1))
            save, _ = select.max(dim=2, keepdim=True)
            select = torch.where(select < save, torch.tensor([0]).cuda(), torch.tensor([1]).cuda())

        rot_angle = rot_angle / (self.embedding_range.item() / pi)
        ref_angle = ref_angle / (self.embedding_range.item() / pi)

        # rot_angle = rot_angle * rot_sel
        re_rot = torch.cos(rot_angle)
        im_rot = torch.sin(rot_angle)

        # ref_angle = ref_angle * ref_sel
        re_ref = torch.cos(ref_angle)
        im_ref = torch.sin(ref_angle)

        # Translation of the head
        # trans_x, trans_y = trans_x * trans_sel, trans_y * trans_sel
        re_rhead, im_rhead = re_head + trans_x, im_head + trans_y

        sel = select[:, :, 0] .view((select.shape[0], -1, 1))


        score_1 = torch.stack([(re_rhead - re_tail) * sel, (im_rhead - im_tail) * sel], dim=0)

        score = score_1.norm(dim=0, p=2).sum(dim=2)

        # Scaling of the head
        # scaling = scaling * scaling_sel
        re_rhead, im_rhead = re_head * scaling, im_head * scaling
        sel = select[:, :, 1].view((select.shape[0], -1, 1))

        score_1 = torch.stack([(re_rhead - re_tail) * sel, (im_rhead - im_tail) * sel], dim=0)
        score = score + score_1.norm(dim=0, p=2).sum(dim=2)

        # Rotation of the head
        re_rhead = re_rot * re_head - im_rot * im_head
        im_rhead = im_rot * re_head + re_rot * im_head
        sel = select[:, :, 2].view((select.shape[0], -1, 1))
        score_1 = torch.stack([(re_rhead - re_tail) * sel, (im_rhead - im_tail) * sel], dim=0)
        score_1 = score_1.norm(dim=0, p=2).sum(dim=2)
        score = score + score_1

        # Reflection of the head
        re_rhead = re_ref * re_head + im_ref * im_head
        im_rhead = im_ref * re_head - re_ref * im_head
        sel = select[:, :, 3].view((select.shape[0], -1, 1))
        score_1 = torch.stack([(re_rhead - re_tail) * sel, (im_rhead - im_tail) * sel], dim=0)
        score_1 = score_1.norm(dim=0, p=2).sum(dim=2)
        score = score + score_1

        # score = score_trans + score_rot + score_ref + score_sc #+ score_inv*inv_sel

        score = self.gamma.item() - score
        return score

