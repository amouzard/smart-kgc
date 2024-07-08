import os
import json
import logging
import argparse
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import KGEModel, Rotate3D, RotatE
from SMART_KGE import smart

from data import TrainDataset, BatchType, ModeType, DataReader
from data import BidirectionalOneShotIterator

models_list=[Rotate3D, RotatE]
models_list += [smart]

models_list_str = [str(name).strip().split('.')[-1][:-2] for name in models_list]

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='runs.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)

    # parser.add_argument('--choice', default=50000, type=int, help='Step to select optimal geometric transformation')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('--disable_adv', action='store_true', help='disable the adversarial negative sampling')
    # parser.add_argument('--enable_load_embs', action='store_true', help='enable choice embeddings loading')
    parser.add_argument('--enable_increase_order', action='store_true', help='enable composing in increasing order')
    parser.add_argument('-b', '--batch_size', default=512, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('-reg', '--regularization', default=0, type=float)
    parser.add_argument('-p', '--p_norm', default=2, type=int, help='p-norm for computing the score')

    parser.add_argument('-lr', '--learning_rate', default=0.00005, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=4, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=200000, type=int)
    parser.add_argument('-p_steps', '--pretraining_steps', default=10, type=int)
    parser.add_argument('-t_steps', '--training_steps', default=300, type=int)
    parser.add_argument('-f_steps', '--finetuning_steps', default=100, type=int)

    parser.add_argument('--save_checkpoint_steps', default=5000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')


    print(args)

    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as f:
        args_dict = json.load(f)

    args.model = args_dict['model']
    args.data_path = args_dict['data_path']
    args.hidden_dim = args_dict['hidden_dim']
    args.test_batch_size = args_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args, step=10000):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    args_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.module.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.module.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )
    if args.model in ['smart']:
        L = list(np.arange(args.pretraining_steps, args.max_steps-1, 2000))
        if (step in L) and (step > args.max_steps):#step <= args.pretraining_steps + args.training_steps
            index = str(L.index(step))
            # logging.info('************************** Intermediate saving at step '+str(step)+' with index '+index)
        else:
            index = ''
        entity_embedding = model.module.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'entity_embedding_'+index),
            entity_embedding
        )
        choice_embedding = model.module.choice_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'relation_embedding_'+index),
            choice_embedding
        )

        np.save(
            os.path.join(args.save_path, 'choice_embedding'),
            choice_embedding
        )


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def log_metrics_rel(mode, step, metrics_rel, relation_dict, args=None, arg={'model':'smart'}):
    '''
    Print the evaluation logs for every relation
    '''
    print('#'*50, '\n', '#'*50, '\nPrint the evaluation logs for every relation:')
    for rel in metrics_rel:
        logging.info('%s: ' % relation_dict[rel])
        for metric in metrics_rel[rel]:
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics_rel[rel][metric]))
        if arg.model in ['smart']:
            adherence_ = args.module.choice_embedding.detach().cpu()
            if args.module.load_choice_embs:
                adherence = adherence_
            else:
                adherence = torch.softmax(adherence_, dim=-1)
            #adherence = torch.softmax(adherence_, dim=-1)
            logging.info('Relation index  and Logmax adherence to geom-trans : %d %f %f %f %f'
                         % (rel, adherence[int(rel), 0], adherence[int(rel), 1], adherence[int(rel), 2], adherence[int(rel), 3]))


def main(args):
    args.max_steps = args.pretraining_steps + args.training_steps + args.finetuning_steps
    # torch.manual_seed(100)
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be chosen.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be chosen.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    data_reader = DataReader(args.data_path)
    num_entity = len(data_reader.entity_dict)
    num_relation = len(data_reader.relation_dict)

    # id2name
    relation_dict = {data_reader.relation_dict[key]:key for key in data_reader.relation_dict}

    logging.info('Model: {}'.format(args.model))
    logging.info('Data Path: {}'.format(args.data_path))
    logging.info('Num Entity: {}'.format(num_entity))
    logging.info('Num Relation: {}'.format(num_relation))

    logging.info('Num Train: {}'.format(len(data_reader.train_data)))
    logging.info('Num Valid: {}'.format(len(data_reader.valid_data)))
    logging.info('Num Test: {}'.format(len(data_reader.test_data)))

    if args.model in models_list_str:
        kge_model = models_list[models_list_str.index(args.model)](num_entity, num_relation, args.hidden_dim, args.gamma, args.p_norm)
    else:
        raise RuntimeError(f"Model {args.model} is not defined!")

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    kge_model = torch.nn.DataParallel(kge_model)  
    kge_model = kge_model.cuda()

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(data_reader, args.negative_sample_size, BatchType.HEAD_BATCH),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(data_reader, args.negative_sample_size, BatchType.TAIL_BATCH),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )

        warm_up_steps = args.warm_up_steps#.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Randomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %f' % current_learning_rate)
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %d' % args.gamma)
    logging.info('regularization = %f' % args.regularization)
    logging.info('adversarial_temperature = %.3f' % args.adversarial_temperature)
    logging.info('negative_sample_size = %d' % args.negative_sample_size)
    logging.info('p_norm = %d' % args.p_norm)
    logging.info('disable_adv = %s' % args.disable_adv)
    logging.info('Maximum steps = %d' % args.max_steps)
    if args.enable_increase_order:
        kge_model.module.arg_sort_desc = False
    # if args.enable_load_embs:
    #     kge_model.module.load_choice_embs = True
    if args.model in ['smart']:
        logging.info('Pretraining steps = %d' % args.pretraining_steps)
        logging.info('Training steps = %d' % args.training_steps)
        logging.info('Fine-tuning steps = %d' % args.finetuning_steps)
        logging.info('Enable choice embeddings to be loaded = %s' % kge_model.module.load_choice_embs)
    if args.do_train:
        training_logs = []

        # Training Loop
        for step in range(init_step, args.max_steps):

            log = kge_model.module.train_step(kge_model, optimizer, train_iterator, args)

            training_logs.append(log)

            if args.model in ['smart']:
                if step > args.pretraining_steps and (step <= args.pretraining_steps + args.training_steps):
                    kge_model.module.start = True
                elif step > args.pretraining_steps + args.training_steps:
                    kge_model.module.start = False
                    # print('Step higher')
                    kge_model.module.step = step
                    kge_model.module.freeze = True
                else:
                    pass

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
                if args.disable_adv:
                    args.adversarial_temperature = 0

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args, step=step)
            save_model(kge_model, optimizer, save_variable_list, args, step=step)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics, metrics_rel = kge_model.module.test_step(kge_model, data_reader, ModeType.VALID, args)
                log_metrics('Valid', step, metrics)


            if args.do_test and step % args.valid_steps == 0:
                logging.info('Evaluating on Test Dataset...')
                metrics, metrics_rel = kge_model.module.test_step(kge_model, data_reader, ModeType.TEST, args)
                log_metrics('Test', step, metrics)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args, step=step)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics, metrics_rel = kge_model.module.test_step(kge_model, data_reader, ModeType.VALID, args)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics, metrics_rel = kge_model.module.test_step(kge_model, data_reader, ModeType.TEST, args)
        log_metrics('Test', step, metrics)
        log_metrics_rel('Test', step, metrics_rel, relation_dict, args=kge_model, arg=args)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics, metrics_rel = kge_model.test_step(kge_model, data_reader, ModeType.TRAIN, args)
        log_metrics('Train', step, metrics)
        log_metrics_rel('Train', step, metrics_rel, relation_dict, args=kge_model, arg=args)



if __name__ == '__main__':
    main(parse_args())
