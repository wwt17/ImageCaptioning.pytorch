from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from criterions.expectedBLEUave import mBLEU

from texar.evals import sentence_bleu

import tensorflow as tf

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def train(opt):
    opt.use_att = utils.if_use_att(opt.caption_model)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    def ids_to_sents(ids):
        return utils.decode_sequence(loader, ids)

    tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)

    def load_infos(dir=opt.start_from, suffix=''):
        # open old infos and check if models are compatible
        with open(os.path.join(dir, 'infos_{}{}.pkl'.format(opt.id, suffix))) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert getattr(saved_model_opt, checkme) == getattr(opt, checkme), "Command line argument and saved model disagree on '%s'" % checkme
        return infos

    def load_histories(dir=opt.start_from, suffix=''):
        path = os.path.join(dir, 'histories_{}{}.pkl'.format(opt.id, suffix))
        if os.path.isfile(path):
            with open(path) as f:
                histories = cPickle.load(f)
        return histories

    infos = {}
    histories = {}
    if opt.start_from is not None:
        infos = load_infos()
        histories = load_histories()

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt)
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()

    crit_ce = utils.LanguageModelCriterion()
    crit_mb = mBLEU(4)

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    def eval_model():
        model.eval()

        eval_kwargs = {'split': 'val',
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))
        val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit_ce, loader, eval_kwargs)

        # Write validation result into summary
        if tf is not None:
            add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
            for k,v in lang_stats.items():
                add_summary_value(tf_summary_writer, k, v, iteration)
            tf_summary_writer.flush()

        model.train()

        return val_loss, predictions, lang_stats

    eval_model()

    opt.current_teach_mask_prefix_length = opt.teach_mask_prefix_length

    while True:
        if update_lr_flag:
            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            # Assign the teach mask prefix length
            if epoch > opt.teach_mask_prefix_length_increase_start:
                frac = (epoch - opt.teach_mask_prefix_length_increase_start) // opt.teach_mask_prefix_length_increase_every
                opt.current_teach_mask_prefix_length = opt.teach_mask_prefix_length + frac * opt.teach_mask_prefix_length_increase_steps
            update_lr_flag = False
 
        verbose = (iteration % opt.verbose_iters == 0)

        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        if iteration % opt.print_iters == 0:
            print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        tmp = [torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks = tmp
        
        optimizer.zero_grad()
        teach_mask = utils.make_teach_mask(labels.size(1), opt)
        enable_ce = (opt.bleu_w != 1)
        enable_mb = (opt.bleu_w != 0)
        if enable_ce:
            enable_xe = (opt.xe_w != 0)
            enable_pg = (opt.pg_w != 0)
            if enable_xe:
                logits = model(fc_feats, att_feats, labels, teach_mask=(teach_mask if opt.teach_ce and not opt.teach_all_input else None))
                if opt.teach_ce:
                    decode_length = logits.shape[1] + 1
                    teach_mask = teach_mask[:decode_length]
                    onehot = utils.to_onehot(labels[:, :decode_length], logits.shape[-1], dtype=torch.float)
                    probs = torch.exp(logits)
                    probs = torch.cat([onehot[:, :1], probs], 1)
                    probs = utils.mask_probs(probs, onehot, teach_mask)
                    if verbose:
                        verbose_probs = probs
                        verbose_probs.retain_grad()
                    logits = torch.log(1. - (1. - 1e-6) * (1. - probs))[:, 1:]
                loss_xe = crit_ce(logits, labels[:, 1:], masks[:, 1:])
            else:
                loss_xe = 0.
            if enable_pg:
                ids_sample, logprobs_sample = model.sample(fc_feats, att_feats, opt={'sample_max': 0})
                ids_greedy, logprobs_greedy = model.sample(fc_feats, att_feats, opt={'sample_max': 1})
                seq_sample = utils.tolist(ids_sample)
                seq_greedy = utils.tolist(ids_greedy)
                seq_target = utils.tolist(labels[:, 1:])
                rewards = [ sentence_bleu([t], s, smooth=True)
                           -sentence_bleu([t], g, smooth=True)
                           for s, g, t in zip(seq_sample, seq_greedy, seq_target)]
                rewards = torch.tensor(rewards, device='cuda')
                mask_sample = torch.ne(ids_sample, torch.tensor(0, device='cuda')).float()
                loss_pg = (rewards * (logprobs_sample * mask_sample).sum(1)).mean()
            else:
                loss_pg = 0.
            loss_ce = opt.xe_w * loss_xe + opt.pg_w * loss_pg
        else:
            loss_ce = 0.
        if enable_mb:
            logits = model(fc_feats, att_feats, labels, teach_mask=(teach_mask if not opt.teach_all_input else None))
            decode_length = logits.shape[1] + 1
            teach_mask = teach_mask[:decode_length]
            onehot = utils.to_onehot(labels[:, :decode_length], logits.shape[-1], dtype=torch.float)
            probs = torch.exp(logits)
            probs = torch.cat([onehot[:, :1], probs], 1) # pad bos
            probs = utils.mask_probs(probs, onehot, teach_mask)
            if verbose:
                verbose_probs = probs
                verbose_probs.retain_grad()
            mask = masks[:, :decode_length]
            mask = torch.cat([mask[:, :1], mask], 1)
            loss_mb = crit_mb(probs, labels[:, :decode_length], mask, min_fn=opt.min_fn, min_c=opt.min_c, verbose=verbose)
        else:
            loss_mb = 0.
        loss = loss_ce * (1-opt.bleu_w) + loss_mb * opt.bleu_w
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip) #TODO: examine clip method and record grad

        if verbose and 'verbose_probs' in locals():
            max_grads, max_ids = verbose_probs.grad.topk(opt.verbose_topk, -1, largest=False)
            max_probs = torch.gather(verbose_probs, -1, max_ids)
            max_sents = ids_to_sents(max_ids[:, :, 0])
            for sample_i in range(min(opt.samples, verbose_probs.shape[0])):
                l = len(max_sents[sample_i]) + 1
                print('max:\n{}'.format(max_sents[sample_i]))
                print('max probs:\n{}'.format(max_probs[sample_i][:l]))
                print('max grads:\n{}'.format(max_grads[sample_i][:l]))

        optimizer.step()
        train_loss = float(loss)
        torch.cuda.synchronize()
        end = time.time()
        if iteration % opt.print_iters == 0:
            print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, train_loss, end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                tf_summary_writer.flush()

            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            val_loss, predictions, lang_stats = eval_model()
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            if True: # if true
                best_flag = False
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history

                def save_model(suffix=''):
                    model_path = os.path.join(opt.checkpoint_path, 'model{}.pth'.format(suffix))
                    torch.save(model.state_dict(), model_path)
                    print("model saved to {}".format(model_path))
                    optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer{}.pth'.format(suffix))
                    torch.save(optimizer.state_dict(), optimizer_path)

                    with open(os.path.join(opt.checkpoint_path, 'infos_{}{}.pkl'.format(opt.id, suffix)), 'wb') as f:
                        cPickle.dump(infos, f)
                    with open(os.path.join(opt.checkpoint_path, 'histories_{}{}.pkl'.format(opt.id, suffix)), 'wb') as f:
                        cPickle.dump(histories, f)

                save_model()
                save_model(".iter{}".format(iteration))

                if best_flag:
                    save_model(".best")

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)
